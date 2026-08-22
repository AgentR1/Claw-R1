"""Build reproducible ALFWorld train/test Parquet from installed game files."""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_parquet

def make_row(game: Path, split: str, index: int) -> dict:
    return {"uid": f"alfworld-{split}-{index}", "data_source": "alfworld", "prompt": [{"role": "user", "content": "Solve the ALFWorld household task using env_step."}], "reward_model": {"style": "rule", "ground_truth": "task_success"}, "game_file": str(game.resolve()), "max_steps": 50}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--games-root", required=True); parser.add_argument("--output-dir", default="data/alfworld"); parser.add_argument("--test-ratio", type=float, default=0.1); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    games = sorted(Path(args.games_root).rglob("*.z8"))
    if not games: raise FileNotFoundError(f"no .z8 games under {args.games_root}")
    random.Random(args.seed).shuffle(games); test_count = max(1, round(len(games) * args.test_ratio))
    write_parquet([make_row(g, "train", i) for i, g in enumerate(games[test_count:])], Path(args.output_dir) / "train.parquet")
    write_parquet([make_row(g, "test", i) for i, g in enumerate(games[:test_count])], Path(args.output_dir) / "test.parquet")

if __name__ == "__main__": main()
