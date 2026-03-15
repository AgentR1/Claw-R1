# Getting Started

<div class="grid cards" markdown>

-   :material-download: **Installation**

    ---

    环境配置、依赖安装和验证。

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch: **Quick Start**

    ---

    5 分钟内运行你的第一个异步训练实验。

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

</div>

## 前置条件

| 依赖 | 最低版本 |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 12.1+ |
| Ray | 2.10+ |
| GPU | 3 张（2 训练 + 1 推理） |

## 架构一览

```
┌─────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   Agent     │────►│ Gateway  │────►│ DataPool │────►│ Trainer  │
│ (黑盒/白盒) │◄────│ (:8100)  │     │          │     │          │
└─────────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                           │ 权重同步
                                                           ▼
                                                     ┌──────────┐
                                                     │  vLLM    │
                                                     └──────────┘
```
