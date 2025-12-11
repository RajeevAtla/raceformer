"""
Command-line entrypoint for raceformer.

Supports running pretraining or RL finetuning given a TOML config.
"""

import argparse
import tomllib

from src.training.pretrainer import run_pretraining
from src.training.rl_agent import run_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raceformer trainer")
    parser.add_argument("--mode", choices=["pretrain", "rl"], required=True, help="Which training pipeline to run")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "pretrain":
        run_pretraining(config)
    else:
        run_ppo(config)


if __name__ == "__main__":
    main()
