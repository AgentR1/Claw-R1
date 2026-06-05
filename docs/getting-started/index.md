# Getting Started

Use this section to set up the runtime environment and launch the first asynchronous training workflow.

<div class="grid cards" markdown>

-   **Installation**

    ---

    Prepare the `verl`-compatible Python environment used by Claw-R1.

    [Installation](installation.md)

-   **Quick Start**

    ---

    Run a black-box GSM8K training example and inspect it with the dashboard.

    [Quick Start](quickstart.md)

</div>

## Requirements

| Dependency | Recommended baseline |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 12.1+ |
| Ray | 2.10+ |
| GPU | At least 3 GPUs for the small async example |

## Runtime Shape

```text
Agent -> Gateway -> DataPool -> Trainer
                     ^             |
                     |             v
                  Dashboard <- Parameter Synchronizer
```

The Gateway receives agent traffic, DataPool stores step-level data, the Trainer consumes curated batches, and the dashboard monitors the live data lifecycle.
