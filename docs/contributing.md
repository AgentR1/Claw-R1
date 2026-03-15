# Contributing

感谢你对 Claw-R1 的关注！欢迎贡献代码、文档和想法。

## 项目结构

```
claw_r1/
├── agent_flow/           # Agent 执行框架（白盒 + 管理器）
├── blackbox_agent/       # 黑盒 Agent 系统（Flow + Agent 实现）
├── config/               # Hydra 配置文件
├── data_pool/            # DataPool（Ray Actor + Training Backend）
├── gateway/              # Gateway Server（FastAPI）
├── async_main.py         # 异步训练入口
├── async_rollouter.py    # AsyncRollouter（Rollout GPU Pool）
├── async_trainer.py      # AsyncTrainer（Training GPU Pool）
├── param_sync.py         # ParameterSynchronizer
├── detach_workers.py     # 分离式 Actor/Rollout Worker
├── core_algos.py         # PPO/GAE/GRPO 核心算法
├── reward_loop.py        # RewardLoopWorker
├── metric_utils.py       # 指标聚合
├── ray_agent_trainer.py  # 同步 Ray PPO Trainer
└── main_agent_ppo.py     # 同步训练入口
```

## 代码风格

- 使用 [Ruff](https://docs.astral.sh/ruff/) 进行 lint 和格式化
- 遵循 PEP 8
- 类型注解（Python 3.10+ 语法）

```bash
# 安装 pre-commit hooks
pip install pre-commit
pre-commit install

# 手动检查
ruff check .
ruff format .
```

## 贡献方向

### 高优先级

- 新的黑盒 Agent 实现（参考 `blackbox_agent/gsm8k_agent.py`）
- 新的 Reward 函数
- 性能优化（DataPool 吞吐、Gateway 延迟）

### 文档

- 教程和示例
- API 文档补充
- 中英文翻译

### 研究

- 新的 advantage 计算算法
- 在线学习策略
- 多 Agent 协作训练

## PR 流程

1. Fork 仓库
2. 创建 feature branch：`git checkout -b feature/my-feature`
3. 编写代码和测试
4. 确保 `ruff check .` 通过
5. 提交 PR，描述改动内容和动机

## 本地构建文档

```bash
pip install mkdocs-material
mkdocs serve
# 访问 http://localhost:8000
```

## 联系

- GitHub Issues: [AgentR1/Claw-R1](https://github.com/AgentR1/Claw-R1/issues)
