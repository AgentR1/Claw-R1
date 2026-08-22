"""Download/convert GSM8K into train.parquet and test.parquet for Claw-R1."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_parquet

def row(example: dict, split: str, index: int) -> dict:
    answer = str(example["answer"])
    return {"uid": f"gsm8k-{split}-{index}", "data_source": "gsm8k", "prompt": [{"role": "user", "content": str(example["question"])}], "reward_model": {"style": "rule", "ground_truth": answer, "answer": answer.rsplit("####", 1)[-1].strip()}}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="data/gsm8k"); parser.add_argument("--dataset", default="openai/gsm8k"); parser.add_argument("--config", default="main"); parser.add_argument("--cache-dir", default=None); args = parser.parse_args()
    from datasets import load_dataset
    dataset = load_dataset(args.dataset, args.config, cache_dir=args.cache_dir)
    write_parquet([row(x, "train", i) for i, x in enumerate(dataset["train"])], Path(args.output_dir) / "train.parquet")
    write_parquet([row(x, "test", i) for i, x in enumerate(dataset["test"])], Path(args.output_dir) / "test.parquet")

if __name__ == "__main__": main()
