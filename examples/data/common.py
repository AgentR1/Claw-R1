"""Shared helpers for producing Verl/Claw-R1 RLHFDataset Parquet files."""
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {"uid", "data_source", "prompt", "reward_model"}

def write_parquet(rows: list[dict[str, Any]], output: str | Path) -> None:
    """Write nested Arrow data and verify the RL dataset contract."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Data preparation requires `datasets` and `pyarrow`.") from exc
    if not rows:
        raise ValueError("refusing to write an empty training split")
    for index, row in enumerate(rows):
        missing = REQUIRED_COLUMNS - row.keys()
        if missing:
            raise ValueError(f"row {index} missing required columns: {sorted(missing)}")
        if not isinstance(row["prompt"], list) or not row["prompt"]:
            raise ValueError(f"row {index} prompt must be a non-empty chat-message list")
        if not isinstance(row["reward_model"], dict):
            raise ValueError(f"row {index} reward_model must be a dict")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    columns = set(Dataset.from_parquet(str(output)).column_names)
    if missing := REQUIRED_COLUMNS - columns:
        raise RuntimeError(f"Parquet verification failed; missing columns: {sorted(missing)}")
