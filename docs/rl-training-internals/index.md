# RL Training Internals

本文档以 **PPO（Proximal Policy Optimization）** 训练流程为主线，系统讲解强化学习训练中的核心概念、算法细节和关键 Metrics。同时覆盖 GRPO 等无 Critic 算法的差异，并解释 Claw-R1 在 Agentic RL 场景下的设计改进及其合理性。

---

## 1. 训练流程全景

PPO 训练两个模型——**Actor（策略网络）** 和 **Critic（价值网络）**，它们各自有独立的训练目标：

- **Actor 的目标**：让好动作的概率变大，坏动作的概率变小。衡量好坏的标准就是 **Advantage**
- **Critic 的目标**：学会预测"从当前状态开始，未来总共能拿到多少回报"。预测值叫 **Value**，预测目标叫 **Return**

对应的两个 loss：

- **Actor Loss**：$-\text{clip}(r(\theta), 1\pm\epsilon) \cdot A$，用 advantage 加权的策略梯度。Advantage > 0 → 增大该动作概率；< 0 → 减小
- **Critic Loss**：$(V_\theta(s) - R_t)^2$，让 value 预测逼近 return（回归任务）

一次训练 step 的数据流：

```
Rollout（生成 response）
    │
    ▼
Reward（环境打分）──► 原始信号，RL 的唯一外部输入
    │
    ├──► KL Penalty（可选，融入 reward）
    │
    ▼
Compute Values（Critic 前向推理）──► V(s)：Critic 对"未来累积回报"的预测
    │
    ▼
Compute Advantage & Return ◄── GAE / GRPO 算法
    │
    │   Advantage = 这个动作比平均好多少（由 reward 和 value 推算）
    │   Return    = 未来累积回报的估计值（由 reward 和 value 推算）
    │
    ├──► Actor Loss：让好动作概率变大 ──► Actor 梯度更新
    │       权重 = Advantage
    │
    └──► Critic Loss：让预测更准 ──► Critic 梯度更新
            目标 = Return，预测 = Value
```

核心量之间的关系：

- **Reward**：环境给的原始信号，是 RL 系统从外部获得的唯一输入
- **Value**：Critic 的预测——"从这个状态开始，未来总共能拿到多少回报"
- **Return**：对 Value 真实值的更好估计。理想情况下，Return 就是把当前和未来所有 reward 加起来：$R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$。但我们不可能等到 trajectory 完全结束才训练，所以实际做法是：**走几步拿到真实 reward，剩下的部分用 Critic 当前的 Value 预测来补上**。这个"用预测补全未来"的操作就叫 **bootstrap**。例如只走一步的情况：$R_t \approx r_t + \gamma V(s_{t+1})$——第一项是真实获得的 reward，第二项是 Critic 对后续所有回报的预测。GAE 通过混合不同步数的 bootstrap 估计来平衡偏差和方差（详见 [4.1 节](#41-gaegeneralized-advantage-estimationppo-默认)）。**Critic 的训练目标就是让 Value 逼近 Return**
- **Advantage**：这个动作比该状态下的"平均表现"好了多少。数值上 $A = R - V$，即 return 减去 value baseline。**Actor 的梯度权重就是 Advantage**

> 注意：在 GAE 的实现中，代码先算 advantage，再通过 $R = A + V$ 反推 return。这只是计算顺序的问题，概念上 Return 是独立定义的量（未来累积折扣回报的估计），不依赖于 Advantage。

---

## 2. MDP 建模：Token-level vs Step-level

RL 训练的第一个设计决策是：**什么是一个 "action"？**

### 传统 RLHF（Token-level）

```
状态  s_t = prompt + 已生成的 token_0..t-1
动作  a_t = 生成第 t 个 token
V(s_t) = 从 s_t 开始到序列结束的期望回报
```

每个 token 是一个独立的决策点。Critic 需要在每个 token 位置输出一个 value，GAE 在 token 维度逐步递推。

### Claw-R1 Agentic RL（Step-level）

```
状态  S_t = 到第 t 步之前的所有交互历史（prompt + 前几轮 response/tool output）
动作  A_t = 第 t 步 LLM 生成的整段 response
V(S_t) = 从 S_t 开始到 trajectory 结束的期望回报
```

**整段 response 是一个原子动作**。Critic 只需预测一个标量 V(S_t)，GAE 在 step 维度递推。

为什么 step-level 建模更适合 agent 场景？详见 [第 7 节](#7-为什么-claw-r1-的实现是合理的)。

---

## 3. 核心概念详解

### 3.1 Reward

Reward 是环境对 agent 行为的即时反馈，是 RL 训练的原始信号来源。

**数学定义：**

$$r_t = \text{RewardFn}(s_t, a_t)$$

**在代码中的体现：**

Reward 首先以 token-level 的形式存在（`token_level_scores`），通常只有最后一个 token 有非零值（outcome reward）。在 Claw-R1 中，token-level reward 会被聚合为 step-level：

```python
# claw_r1/core_algos.py
rewards = (token_level_rewards * response_mask).sum(dim=1)  # step-level 标量
```

**KL Penalty（可选）：** 可以将 KL 惩罚融入 reward，防止策略偏离 reference policy 太远：

$$r_t^{\text{shaped}} = r_t - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

KL 惩罚有多种估计方式（k1, k2, k3 等），详见 [5.4 节](#54-kl-penalty)。

### 3.2 Value（Critic 输出）

Value function（价值函数）是 Critic 模型的输出，预测从当前状态开始、遵循当前策略能获得的期望累积回报。

**数学定义：**

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s \right]$$

**传统实现（verl 原版）：** Critic 在每个 response token 位置输出一个 value，得到 shape 为 `(batch_size, response_length)` 的张量。

**Claw-R1 实现：** 只使用第 1 个 token 位置的 value：

```python
# claw_r1/core_algos.py — compute_gae_advantage_return
values = values[:, 0]
```

这里 `values[:, 0]` 对应的是 "response 生成前的状态价值"。Critic 模型内部的切片逻辑（`dp_critic.py`）将 `values[:, -response_length-1:-1]` 作为输出，所以 `[:, 0]` 正好对应 prompt 最后一个 token 之后的位置——即状态 $S_t$ 的 value。

### 3.3 Advantage

Advantage 衡量的是：在状态 $s$ 下执行动作 $a$，比"平均水平"好了多少。

**数学定义：**

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

直觉上，如果 advantage > 0，说明这个动作比该状态下的平均表现更好，应该增大其概率；反之则减小。Advantage 是 actor 策略梯度的核心权重：

$$\nabla_\theta J \propto \mathbb{E} \left[ A(s, a) \cdot \nabla_\theta \log \pi_\theta(a|s) \right]$$

减去 baseline $V(s)$ 不会引入偏差，但可以大幅降低梯度估计的方差。

### 3.4 Return

Return 是对"从当前状态开始，未来总共能拿到多少累积折扣回报"的估计值。它是 **Critic 的训练 target**——Critic 的目标就是让自己的预测 $V_\theta(s)$ 尽可能接近 Return。

**理想定义（蒙特卡洛 return）：**

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

如果能等到 trajectory 完全结束，把所有 reward 按折扣加起来，就得到了最精确的 return。但这样方差太大（每次 rollout 的 reward 序列差异很大），而且需要等整条 trajectory 跑完。

**实际做法（Bootstrap）：**

Bootstrap 的意思是"用 Critic 当前的预测来补全我们看不到的未来"。不同步数的 bootstrap 给出不同的 return 估计：

| 估计方式 | 公式 | 含义 |
|---|---|---|
| 1 步 bootstrap | $R_t \approx r_t + \gamma V(s_{t+1})$ | 真实走 1 步，剩下的让 Critic 猜 |
| 2 步 bootstrap | $R_t \approx r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$ | 真实走 2 步，剩下的让 Critic 猜 |
| $\infty$ 步（MC） | $R_t = r_t + \gamma r_{t+1} + \cdots$ | 全用真实 reward，不猜 |

步数越少，偏差越大（因为 Critic 的预测可能不准），但方差越小（因为依赖的随机 reward 更少）；步数越多则反之。GAE 通过 $\lambda$ 参数混合所有步数的 bootstrap，实现偏差-方差的折中（详见 [4.1 节](#41-gaegeneralized-advantage-estimationppo-默认)）。

**代码中的计算方式：**

实践中，先通过 GAE 算出 advantage $A_t$，再利用 $R_t = A_t + V_{\text{old}}(s_t)$ 反推 return。这里 $V_{\text{old}}$ 是 Critic 在本轮更新前的旧预测。这不是循环论证——GAE 内部的递推过程已经把真实 reward 和 bootstrap value 融合好了，加回旧 $V$ 只是还原计算过程。**随着训练推进，Critic 会越来越准，bootstrap 的质量也会越来越高，形成正向循环**。

**三者关系：**

| 量 | 含义 | 谁使用它 | 用途 |
|---|---|---|---|
| Reward $r_t$ | 即时奖励（外部输入） | GAE / GRPO | 作为计算的原始输入 |
| Return $R_t$ | 未来累积回报的估计 | **Critic** | 回归 target：$\mathcal{L} = (V_\theta - R_t)^2$ |
| Advantage $A_t$ | 动作相对好坏 | **Actor** | 策略梯度权重：$\nabla \propto A_t \nabla \log \pi$ |

---

## 4. Advantage 估计算法

### 4.1 GAE：Generalized Advantage Estimation（PPO 默认）

GAE 是 PPO 的默认 advantage 估计方法，通过指数加权多步 TD 误差来平衡偏差与方差。

**TD 误差（一步 advantage 估计）：**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**GAE（多步加权平均）：**

$$A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

**$\gamma$ 和 $\lambda$ 的作用：**

- $\gamma$（折扣因子）：控制对未来 reward 的重视程度。$\gamma \to 1$ 时更关注长期回报
- $\lambda$（GAE 参数）：控制偏差-方差折中
    - $\lambda = 0$：只看一步 TD 误差，低方差但高偏差
    - $\lambda = 1$：退化为蒙特卡洛估计，低偏差但高方差

**传统实现（verl 原版）：** 在 token 维度逐步递推，每个 batch row 独立：

```python
# verl/trainer/ppo/core_algos.py — 传统 GAE
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
```

**Claw-R1 改进 — Step-level GAE：**

在 Claw-R1 中，reward 和 value 都是 step-level 的标量。GAE 递推在 step 维度进行，并通过 `trajectory_uids` 和 `step_indices` 将同一条 trajectory 的多个 step 关联起来：

```python
# claw_r1/core_algos.py — compute_gae_advantage_return
rewards = (token_level_rewards * response_mask).sum(dim=1)  # 聚合为 step-level
values = values[:, 0]                                        # 只取第 1 个 token

# 按 trajectory 分组，在 step 维度做 GAE
rewards_map[traj_inv, step_ids] = rewards
values_map[traj_inv, step_ids] = values

for t in reversed(range(max_step)):
    nextvalues = values_map[:, t + 1] if t < max_step - 1 else 0.0
    delta = rewards_map[:, t] + gamma * nextvalues - values_map[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
```

最终 advantage 先在 step-level 做 whitening，再广播回 token 维度：

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
advantages = advantages.unsqueeze(1) * response_mask  # 广播到 token 维度
returns = returns.unsqueeze(1) * response_mask
```

**Claw-R1 还提供了 Token-level 跨 Step GAE**（`compute_token_gae_advantage_return`），它在 token 维度做 GAE 递推，但能跨越多个 step——通过 `response_mask` 跳过 tool/padding tokens 而不中断递推状态。这比 step-level GAE 更精细，但计算开销更大。

### 4.2 GRPO：Group Relative Policy Optimization（无 Critic）

GRPO 是一种不需要 Critic 模型的 advantage 估计方法。核心思想是：对同一个 prompt 采样多条 response，用组内统计量作为 baseline。

**计算方式：**

$$A_i = \frac{\text{score}_i - \text{mean}(\text{scores in group})}{\text{std}(\text{scores in group}) + \epsilon}$$

其中 group 由同一个 prompt 的多条 response 构成。

**Dr.GRPO 变体：** 不除以 std，避免 std 过小时放大噪声：

$$A_i = \text{score}_i - \text{mean}(\text{scores in group})$$

**GRPO vs PPO 的核心区别在于 advantage 来源不同**，而 actor loss 的 clipped objective 形式是共通的。GRPO 不需要 Critic 模型，因此没有 critic loss 和 value 计算步骤。

**传统实现（verl 原版）：** 每个 response 独立计算 score：

```python
# verl/trainer/ppo/core_algos.py — 传统 GRPO
scores = token_level_rewards.sum(dim=-1)  # 每行直接求和
# 按 index（prompt group）分组做 mean/std 归一化
```

**Claw-R1 改进 — Trajectory-level GRPO：**

Agent 场景中，一条 trajectory 包含多个 step。Outcome reward（如解题是否成功）属于整条 trajectory 而非单个 step。因此 Claw-R1 先按 `trajectory_uids` 聚合为 trajectory-level 的总 score，再做 group normalize：

```python
# claw_r1/core_algos.py — compute_grpo_outcome_advantage
step_scores = (token_level_rewards * response_mask).sum(dim=-1)

# 1) 按 trajectory 累加 step reward → trajectory-level outcome
for i in range(bsz):
    traj_uid = trajectory_uids[i]
    traj2total_score[traj_uid] += step_scores[i]

# 2) 按 prompt group 做 GRPO 归一化
for traj_uid, total_score in traj2total_score.items():
    advantage = (total_score - group_mean) / (group_std + epsilon)

# 3) 同一条 trajectory 的所有 step 共享同一个 advantage
for i in range(bsz):
    scores[i] = traj2adv[trajectory_uids[i]]
```

---

## 5. 训练 Loss

### 5.1 Actor（Policy）Loss — PPO Clipped Objective

PPO 的核心创新是 clipped surrogate objective，通过限制策略更新幅度来保证训练稳定性。

**概率比（importance sampling ratio）：**

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$$

**Clipped objective：**

$$\mathcal{L}^{\text{CLIP}} = -\mathbb{E} \left[ \min \left( r_t \cdot A_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t \right) \right]$$

直觉：

- 当 $A_t > 0$（好动作）：ratio 被限制不超过 $1+\epsilon$，防止过度增大概率
- 当 $A_t < 0$（坏动作）：ratio 被限制不低于 $1-\epsilon$，防止过度减小概率

**Dual-clip 扩展：** 当 $A_t < 0$ 且 $r_t$ 过大时（策略在远离旧策略的方向上增大了坏动作的概率），标准 PPO clip 可能不够保守。Dual-clip 引入下界 $c$（通常 $c=3$）：

$$\mathcal{L}^{\text{dual}} = \begin{cases} \min\left(\max(L_1, L_2), \; -c \cdot A_t\right) & \text{if } A_t < 0 \\ \max(L_1, L_2) & \text{otherwise} \end{cases}$$

**GRPO 的 actor loss 与 PPO 完全相同**——区别仅在于 $A_t$ 的来源不同（GRPO 用 group statistics，PPO 用 GAE）。

代码位置：`verl/trainer/ppo/core_algos.py` 中的 `compute_policy_loss_vanilla`。

### 5.2 Critic（Value）Loss

Critic 的训练目标是让 $V_\theta(s)$ 逼近 return $R_t$，使用 clipped MSE loss。

**Clipped value loss：**

$$V_{\text{clip}} = \text{clip}(V_\theta, V_{\text{old}} - \epsilon_v, V_{\text{old}} + \epsilon_v)$$

$$\mathcal{L}^{\text{VF}} = \frac{1}{2} \max \left( (V_\theta - R_t)^2, \; (V_{\text{clip}} - R_t)^2 \right)$$

Clipping 防止 value function 一次更新变化太大，与 policy clipping 的思路一致。

**Loss 聚合：** 只在 `response_mask=1` 的位置计算 loss，通过 `agg_loss` 函数聚合为标量。聚合模式包括：

- `token-mean`：所有有效 token 的平均（默认）
- `seq-mean-token-sum`：先对每条序列做 token 求和，再对序列求平均
- `seq-mean-token-mean`：先对每条序列做 token 平均，再对序列求平均

**Claw-R1 改进 — 仅第 1 个 token 位置训练 Critic：**

```python
# claw_r1/async_trainer.py — _process_batch
response_mask = batch.batch["response_mask"]
value_mask = torch.zeros_like(response_mask)
value_mask[:, 0] = 1                          # 只有第 1 个位置为 1
batch.batch["response_mask"] = value_mask      # 临时替换
critic_output = self.critic_wg.update_critic(batch)
batch.batch["response_mask"] = response_mask   # 立即恢复
```

这使得 critic loss 只在第 1 个 token 位置聚合，其他位置不参与梯度回传。配合 step-level 的 MDP 建模——V(s) 只需一个标量，训练其他位置既无意义也浪费算力。

代码位置：`verl/trainer/ppo/core_algos.py` 中的 `compute_value_loss`。

### 5.3 Entropy Loss

Entropy loss 鼓励策略保持一定的探索性，防止过早坍缩到某个确定性动作。

$$\mathcal{L}^{\text{entropy}} = -H(\pi_\theta) = \sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

Entropy 越大，策略越"均匀"（探索越多）。将 entropy loss 加入总 loss 中（带负号），相当于鼓励高 entropy。

代码位置：`verl/trainer/ppo/core_algos.py` 中的 `compute_entropy_loss`。

### 5.4 KL Penalty

KL penalty 约束当前策略不偏离 reference policy（通常是 SFT 模型）太远，防止 reward hacking。

**常见 KL 估计方式：**

| 名称 | 公式 | 特点 |
|---|---|---|
| k1 (kl) | $\log \pi_\theta - \log \pi_{\text{ref}}$ | 最简单，单侧 |
| k2 (mse) | $\frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$ | 梯度无偏 |
| k3 (low_var_kl) | $e^{\log \pi_{\text{ref}} - \log \pi_\theta} - (\log \pi_{\text{ref}} - \log \pi_\theta) - 1$ | 低方差 |
| abs | $\|\log \pi_\theta - \log \pi_{\text{ref}}\|$ | 对称惩罚 |

**两种使用方式：**

1. **Reward shaping：** 将 KL 惩罚融入 reward，$r_t^{\text{shaped}} = r_t - \beta \cdot \text{KL}_t$
2. **独立 loss：** 作为额外的 loss term 加入 actor 总 loss

代码位置：`verl/trainer/ppo/core_algos.py` 中的 `kl_penalty` 和 `kl_penalty_forward`。

---

## 6. 关键 Metrics 解读

### Actor 指标

| Metric | 含义 | 健康范围 |
|---|---|---|
| `actor/ppo_kl` | 当前策略与旧策略的近似 KL 散度 | 通常 < 0.1；过大说明更新过激 |
| `actor/pg_clipfrac` | 被 PPO clip 截断的 ratio 比例 | 通常 0.1-0.3；过高说明策略变化太大 |
| `actor/pg_clipfrac_lower` | dual-clip 生效的比例（仅 $A<0$ 时） | 应较小 |

### Critic 指标

| Metric | 含义 | 健康范围 |
|---|---|---|
| `critic/vf_loss` | Value function 的 MSE loss | 应随训练下降 |
| `critic/vf_explained_var` | $1 - \text{Var}(R - V) / \text{Var}(R)$，Critic 拟合质量 | 趋近 1.0 为好；< 0 说明 Critic 预测比均值还差 |
| `critic/vf_clipfrac` | value clipping 截断比例 | 类似 actor clipfrac |
| `critic/vpred_mean` | Critic 预测值的均值 | 应与 return 均值接近 |

### 数据指标

| Metric | 含义 |
|---|---|
| `critic/score/mean` | 平均 reward score |
| `critic/rewards/mean` | 平均 shaped reward（含 KL penalty） |
| `critic/advantages/mean` | advantage 均值（whitening 后应接近 0） |
| `critic/returns/mean` | return 均值 |
| `response_length/mean` | 平均 response token 数 |
| `response/aborted_ratio` | response 长度为 0 的比例（被截断/放弃的 response） |

---

## 7. 为什么 Claw-R1 的实现是合理的

Claw-R1 的每个设计决策都可以从 agent 场景的交互结构中自然推导出来，形成一条自洽的逻辑链。

### 7.1 整段 response 作为原子 action

**论据：**

- 在 agent 场景中，环境（tool、sandbox、API）无法对单个 token 给出反馈——只有在整段 response 生成完毕后，环境才会返回 observation
- Token 之间不存在独立的"状态转移"。给定前缀后，LLM 的自回归采样是一次性完成的，中间没有环境介入
- 因此，将整段 response 视为一个原子 action 符合 agent 交互的真实结构

**推论：** MDP 的 action 粒度从 token 提升到 sequence（step）。

### 7.2 Critic 只在第 1 个 token 位置训练和推理

**由 7.1 推导：**

- 既然 action 粒度是整段 response，$V(S_t)$ 只需要一个标量——表示"在执行第 $t$ 步动作之前，未来的期望回报"
- 第 1 个 response token 位置对应的正是"prompt 结束后、response 开始前"的状态，恰好是 $V(S_t)$ 的定义位置
- 其他 token 位置的 value 预测不参与 advantage 计算，训练它们既无意义也浪费算力
- 因此 `value_mask[:, 0] = 1` 的设计是合理且高效的

### 7.3 GAE 在 step 级别递推

**由 7.1 + 7.2 推导：**

- Reward 是 step-level 的（token reward 聚合为 step 标量）
- Value 是 step-level 的（只有第 1 个 token 的 value）
- GAE 的递推公式 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 中的每个量都是 step-level 的
- 因此 GAE 递推自然应在 step 维度进行

`trajectory_uids` 和 `step_indices` 的引入使得同一条 trajectory 的多个 step 可以被关联起来，跨 step 的 bootstrap $V(S_{t+1})$ 提供了 multi-step credit assignment 的能力。

### 7.4 GRPO 按 trajectory 聚合

**由 agent 场景的 reward 结构推导：**

- Agent 的 outcome reward（如解题是否成功）属于整条 trajectory 而非单个 step
- 如果按 step 独立计分，中间 step 的 reward 通常为 0，无法获得有效的 advantage signal
- 先按 `trajectory_uids` 聚合为 trajectory-level 总 score，再做 group normalize，才能正确地将 outcome credit 分配到每个 step

### 7.5 Actor loss 仍按 token-level 计算

这看似与 step-level MDP 矛盾，实则不然：

- Advantage 虽然是 step-level 标量，但通过 `* response_mask` 广播到 token 维度后，同一 step 内的每个 token 共享相同的 advantage
- Actor policy loss 仍逐 token 计算 $\log \pi_\theta(a_t|s_t)$ 的 ratio——因为策略 $\pi(A|S)$ 的对数概率本身就是 token-level 的自回归分解：$\log \pi(A|S) = \sum_t \log \pi(a_t|s, a_{<t})$
- Step-level advantage 乘以 token-level 梯度，效果等价于对整段 response 的 REINFORCE 梯度：

$$\nabla_\theta J = A(S, A) \cdot \nabla_\theta \sum_t \log \pi_\theta(a_t | s, a_{<t}) = A(S, A) \cdot \nabla_\theta \log \pi_\theta(A|S)$$

因此 token-level 的 policy loss 计算与 step-level MDP 是完全一致的。

---

## 8. 传统 vs Claw-R1 对比总结

| 维度 | 传统 RLHF / PPO (verl) | Claw-R1 Agentic RL |
|---|---|---|
| MDP action 粒度 | 单个 token | 整段 response（一个 step） |
| Critic V(s) | 逐 token 输出，`(bs, resp_len)` | 仅第 1 个 token，标量 |
| Critic 训练 mask | 全部 response token | 仅第 1 个 token 位置 |
| GAE 递推维度 | token 级别，batch 行独立 | step 级别，按 trajectory 分组 |
| GRPO reward 聚合 | 每个 response 独立 sum | 先按 trajectory 累加，再 group normalize |
| 额外元数据 | 无 | `trajectory_uids`, `step_indices` |
| Advantage 广播 | 天然是 token-level | step-level 标量广播到 token 维度 |
| Actor loss | 逐 token，与 advantage 同维度 | 逐 token（不变），advantage 通过广播对齐 |
| 适用场景 | 单轮对话 / RLHF | 多步 agent 交互 / tool use |

---

## 代码索引

| 模块 | 路径 | 说明 |
|---|---|---|
| Step-level GAE | `claw_r1/core_algos.py` `compute_gae_advantage_return` | 跨 trajectory 的 step-level GAE |
| Token-level 跨 Step GAE | `claw_r1/core_algos.py` `compute_token_gae_advantage_return` | 跨 step 的精细 token-level GAE |
| Trajectory-level GRPO | `claw_r1/core_algos.py` `compute_grpo_outcome_advantage` | 按 trajectory 聚合的 GRPO |
| Critic mask 替换 | `claw_r1/async_trainer.py` `_process_batch` | 仅第 1 个 token 位置训练 critic |
| PPO Policy Loss | `verl/trainer/ppo/core_algos.py` `compute_policy_loss_vanilla` | Clipped surrogate + dual-clip |
| Value Loss | `verl/trainer/ppo/core_algos.py` `compute_value_loss` | Clipped MSE regression |
| Entropy Loss | `verl/trainer/ppo/core_algos.py` `compute_entropy_loss` | 策略熵 |
| KL Penalty | `verl/trainer/ppo/core_algos.py` `kl_penalty` | 多种 KL 估计方式 |
| Training Metrics | `verl/trainer/ppo/metric_utils.py` `compute_data_metrics` | 训练指标计算 |
