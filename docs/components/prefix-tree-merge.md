# Prefix Tree Merge

Prefix Tree Merge 是 Claw-R1 针对 multi-step agent 训练场景的计算效率优化。它通过构建前缀树（Trie）将同一 prompt group 下共享前缀的多条序列合并为一次前向计算，消除重复的 prefix 计算开销。

---

## 为什么需要 Prefix Tree Merge

### 问题：Multi-step Agent 中的前缀冗余

在 agentic RL 训练中，同一个 prompt 会采样多条 trajectory，每条 trajectory 包含多个 step。这些 step 天然共享大量相同的 token 前缀：

```
Prompt: "Solve: what is 2+3?"

Trajectory 1, Step 0: [prompt] + [response_0a]
Trajectory 1, Step 1: [prompt] + [response_0a] + [tool_output_0a] + [response_1a]
Trajectory 2, Step 0: [prompt] + [response_0b]
Trajectory 2, Step 1: [prompt] + [response_0b] + [tool_output_0b] + [response_1b]
```

如果独立处理每条序列，`[prompt]` 部分被重复计算了 4 次。当 prompt 很长（几千 token）且采样数量较多（n=5~16）时，这种冗余会显著浪费 GPU 算力。

### 解决方案

Prefix Tree Merge 将共享前缀的序列构建为一棵压缩前缀树，然后将树展平为一条紧凑的 packed sequence。通过自定义的 tree-structured attention mask，在单次前向计算中同时处理所有序列，每个共享前缀只计算一次。

---

## 工作原理

### 1. 建树与压缩

给定一组 token 序列，`PrefixTree.build()` 执行三个阶段：

1. **插入**：将所有序列逐 token 插入一棵标准 Trie
2. **压缩**：将单孩子链合并为一个节点（如 `[1]-[2]-[3]` 压缩为 `[1,2,3]`），只在分叉点保留节点边界
3. **定位**：按前序 DFS 为每个节点分配在 packed sequence 中的位置（`start_pos`, `end_pos`）

```
输入序列:
  seq0: [1, 2, 3, 4, 5]
  seq1: [1, 2, 3, 6, 7]
  seq2: [1, 2, 3, 6, 7, 8, 9]

压缩前缀树:
  (root)
    Node 0: [1,2,3]  seqs=[0,1,2]  pos=[0,2]     ← 共享前缀
      Node 1: [4,5]  seqs=[0]      pos=[3,4]     ← seq0 的分支
      Node 2: [6,7]  seqs=[1,2]    pos=[5,6]     ← seq1,seq2 共享
        Node 3: [8,9] seqs=[2]     pos=[7,8]     ← seq2 独有

Packed sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9]  (9 tokens vs 原始 17 tokens)
Token ratio: 9/17 = 52.9%
```

### 2. Attention Mask 构建

Packed sequence 中不同序列的 token 交错排列，标准的 causal attention mask 不再适用。`PrefixTree.pack()` 生成一个 `(T, T)` 的 dense attention mask，保证：

- 每条序列只能看到**自己路径上**的 token（从根到叶）
- 路径内的 token 遵循**因果约束**（只能 attend 到路径中位置在自己之前的 token）
- 不同分支的 token 互不可见

```
seq0 路径: pos [0,1,2,3,4]     — Node 0 + Node 1
seq1 路径: pos [0,1,2,5,6]     — Node 0 + Node 2
seq2 路径: pos [0,1,2,5,6,7,8] — Node 0 + Node 2 + Node 3

Attention mask (简化示意，1=可见):
     pos: 0 1 2 3 4 5 6 7 8
seq0:     1 1 1 1 1 0 0 0 0
seq1:     1 1 1 0 0 1 1 0 0
seq2:     1 1 1 0 0 1 1 1 1
```

### 3. FlexAttention 集成

标准的 Flash Attention 2 不支持任意的 2-D attention mask。Claw-R1 使用 PyTorch 的 **FlexAttention**（`torch.nn.attention.flex_attention`）来高效处理 tree-structured mask：

1. Dense mask 被转换为 `BlockMask`（稀疏块表示），大幅减少内存占用
2. 通过 monkey-patch 替换 HuggingFace transformers 的 `_flash_attention_forward`
3. 当检测到 tree-packed 输入时自动路由到 FlexAttention，否则透明回退到原始 FA2 路径

### 4. Log Probability 恢复

前向计算在 packed sequence 上产生 logits，但训练需要每条原始序列独立的 log probability。`logprob_utils` 模块沿每条序列在树中的路径 gather 对应位置的 logits，计算 log P(next_token | context)，并通过**节点级缓存**避免对共享前缀的重复计算。

---

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    DataPool                              │
│  Steps grouped by prompt_uid                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              TreeVerlBackend                             │
│  1. Group steps by prompt_uid                            │
│  2. Build PrefixTree per group                           │
│  3. Pack into merged tensors                             │
│  4. Attach tree_attention_masks to DataProto             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           TreeDataParallelPPOActor                       │
│  1. Extract per-row tree masks from non_tensor_batch     │
│  2. For each tree row:                                   │
│     a. Create BlockMask from dense mask                  │
│     b. Forward with tree_attention_context                │
│  3. For standard rows: delegate to parent class           │
└─────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 路径 | 职责 |
|---|---|---|
| `PrefixTree` | `claw_r1/tree_utils/prefix_tree.py` | 建树、压缩、打包、位置分配 |
| `attention_patch` | `claw_r1/tree_utils/attention_patch.py` | FlexAttention monkey-patch，BlockMask 创建 |
| `tree_actor` | `claw_r1/tree_utils/tree_actor.py` | 继承 verl actor，支持 tree-packed 前向 |
| `logprob_utils` | `claw_r1/tree_utils/logprob_utils.py` | 从 packed logits 恢复 per-sequence log prob |
| `_worker_factory` | `claw_r1/tree_utils/_worker_factory.py` | 动态生成 tree-aware worker 子类 |
| `TreeVerlBackend` | `claw_r1/data_pool/training_backend_prefix_tree.py` | 数据准备：Steps → prefix tree → DataProto |

---

## 使用方式

在配置中启用：

```yaml
async_training:
  enable_prefix_tree_merge: true
```

或在启动脚本中传入：

```bash
python3 -m claw_r1.async_main \
    async_training.enable_prefix_tree_merge=true \
    ...
```

完整示例见 `example/test_async_bb_prefix_tree.sh`。

---

## 性能收益

Token ratio（packed tokens / 原始 tokens 总和）取决于前缀共享程度：

| 场景 | n_steps | prefix_len | branch_len | Token ratio |
|---|---|---|---|---|
| 短 prompt，少分支 | 4 | 256 | 128 | ~53% |
| 长 prompt，多分支 | 8 | 1024 | 256 | ~35% |
| 长 prompt，少分支 | 4 | 1024 | 128 | ~28% |

Token ratio 越低，前向计算的节省越大。对于典型的 agent 训练场景（长 prompt + 多条采样），可以节省 50-70% 的前缀计算量。

---

## 状态

> **实验阶段**：Prefix Tree Merge 目前在 [`prefix-tree-merge`](https://github.com/AgentR1/Claw-R1/tree/prefix-tree-merge) 分支上测试中。数值正确性已通过 benchmark 验证（`tests/benchmark_tree_attention.py`），正在进行端到端训练测试。
