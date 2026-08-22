"""Prepare SWE-Gym-style tasks as validated Claw-R1 train/test Parquet."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_parquet

def load_records(args):
    if args.input:
        text = Path(args.input).read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()] if args.input.endswith("jsonl") else json.loads(text)
    from datasets import load_dataset
    return [dict(row) for row in load_dataset(args.dataset, args.config, split=args.split)]

def convert(row, index):
    prompt = row.get("prompt") or row.get("problem_statement") or row.get("instruction")
    grader = row.get("grader_command") or row.get("test_command")
    if not prompt or not grader: raise ValueError(f"SWE-Gym row {index} requires prompt/problem_statement and grader_command/test_command")
    return {"uid": str(row.get("instance_id") or row.get("id") or index), "data_source": "swegym", "prompt": [{"role": "user", "content": str(prompt)}], "reward_model": {"style": "external_grader", "ground_truth": "overall_score"}, "files": row.get("files", []), "workspace": row.get("workspace", "/workspace"), "before_agent_commands": row.get("setup_commands", []), "after_agent_commands": row.get("after_agent_commands", []), "grader_command": str(grader), "backend_options": row.get("backend_options", {})}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input"); parser.add_argument("--dataset"); parser.add_argument("--config", default=None); parser.add_argument("--split", default="train"); parser.add_argument("--output-dir", default="data/swegym"); parser.add_argument("--test-ratio", type=float, default=0.1); args = parser.parse_args()
    if bool(args.input) == bool(args.dataset): parser.error("provide exactly one of --input or --dataset")
    rows = [convert(row, i) for i, row in enumerate(load_records(args))]
    count = max(1, round(len(rows) * args.test_ratio)); write_parquet(rows[count:], Path(args.output_dir) / "train.parquet"); write_parquet(rows[:count], Path(args.output_dir) / "test.parquet")

if __name__ == "__main__": main()
