# Training-data preparation

Each directory creates `train.parquet` and `test.parquet` directly in the
nested Arrow/Parquet form consumed by Verl's `RLHFDataset` and Claw-R1. Every
row has `uid`, `data_source`, chat-format `prompt`, and `reward_model`; task
adapters receive any additional columns as metadata.

```bash
python examples/data/gsm8k/prepare.py --output-dir data/gsm8k
python examples/data/alfworld/prepare.py --games-root /path/to/alfworld/games --output-dir data/alfworld
python examples/data/swegym/prepare.py --input /path/to/tasks.jsonl --output-dir data/swegym
```

`swegym/prepare.py` also accepts `--dataset <huggingface-id>`. Its source rows
must expose a task prompt plus a grader/test command; optional files and setup
commands are preserved as recipe metadata.
