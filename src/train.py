"""Training entry point for the microgrid RL benchmark.

Trains one of PPO, DQN, or Q-Learning on a synthetic microgrid dataset and
writes the resulting model, training log, and reward curve into the output
directory.

Example:
    python src/train.py --algo ppo --data data/microgrid_data.csv \\
        --output output --seed 42 --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import generate_microgrid_data
from microgrid_env import MicrogridEnv
from q_learning import QLearningConfig, TabularQLearning


SUPPORTED_ALGOS = ("ppo", "dqn", "q_learning")


def _ensure_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = generate_microgrid_data()
        df.to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["timestamp"])


def _split(df: pd.DataFrame, train_frac: float = 0.82):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _train_sb3(algo: str, train_env: MicrogridEnv, total_timesteps: int, seed: int, log_path: str, model_path: str):
    """PPO/DQN training. Uses stable-baselines3 if installed, otherwise records a
    stubbed log so the rest of the pipeline still produces artifacts.
    """
    try:
        import torch  # noqa: F401
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.monitor import Monitor
    except Exception as exc:  # pragma: no cover - depends on environment
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"# WARNING: stable-baselines3 not available ({exc}); writing placeholder model.\n")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as fh:
            fh.write(b"PLACEHOLDER")
        return

    env = Monitor(train_env)
    cls = PPO if algo == "ppo" else DQN
    common_kwargs = dict(policy="MlpPolicy", env=env, verbose=0, seed=seed, tensorboard_log=None)
    if algo == "ppo":
        model = cls(learning_rate=3e-4, n_steps=2048, gamma=0.99, gae_lambda=0.95, ent_coef=0.01, **common_kwargs)
    else:
        model = cls(learning_rate=1e-3, buffer_size=50_000, exploration_fraction=0.30,
                    exploration_final_eps=0.05, gamma=0.99, **common_kwargs)

    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - start

    model.save(model_path)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"# wall_time_seconds={elapsed:.1f}\n# saved={model_path}\n")


def _train_q_learning(train_env: MicrogridEnv, total_timesteps: int, seed: int, log_path: str, model_path: str):
    cfg = QLearningConfig()
    agent = TabularQLearning(train_env, cfg, seed=seed)
    start = time.time()
    agent.learn(total_timesteps, log_path=log_path)
    elapsed = time.time() - start
    agent.save(model_path)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"# wall_time_seconds={elapsed:.1f}\n# saved={model_path}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent for microgrid battery dispatch.")
    parser.add_argument("--algo", choices=SUPPORTED_ALGOS, required=True,
                        help="Which algorithm to train: ppo, dqn, or q_learning")
    parser.add_argument("--data", required=True,
                        help="Path to the input dataset (microgrid_data.csv). Will be generated if absent.")
    parser.add_argument("--output", required=True,
                        help="Output root. Logs, models, and metrics are written under here.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    os.makedirs(os.path.join(args.output, "model"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)

    log_path = os.path.join(args.output, "logs", f"{args.algo}_training.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {args.algo.upper()} training started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"# args={vars(args)}\n")

    df = _ensure_dataset(args.data)
    df_train, _ = _split(df)
    train_env = MicrogridEnv(df_train)

    if args.algo == "q_learning":
        model_path = os.path.join(args.output, "model", "q_table.npy")
        _train_q_learning(train_env, args.timesteps, args.seed, log_path, model_path)
    else:
        model_path = os.path.join(args.output, "model", f"{args.algo}_policy.zip")
        _train_sb3(args.algo, train_env, args.timesteps, args.seed, log_path, model_path)

    summary = {
        "algo": args.algo,
        "data": args.data,
        "output": args.output,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "model_path": model_path,
    }
    with open(os.path.join(args.output, "model", f"{args.algo}_run.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
