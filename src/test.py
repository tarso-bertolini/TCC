"""Evaluation entry point: loads a trained policy and replays it over a test
slice of the microgrid dataset to produce metrics, action traces, and logs.

Example:
    python src/test.py --weights output/model/ppo_policy.zip \\
        --datadir data --output output --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microgrid_env import MicrogridEnv
from q_learning import TabularQLearning


def _detect_algo(weights_path: str) -> str:
    base = os.path.basename(weights_path).lower()
    if base.startswith("ppo"):
        return "ppo"
    if base.startswith("dqn"):
        return "dqn"
    if base.startswith("q_table"):
        return "q_learning"
    raise ValueError(f"Cannot infer algorithm from weights file: {weights_path}")


def _load_policy(algo: str, weights_path: str, env: MicrogridEnv):
    if algo == "q_learning":
        return TabularQLearning.load(env, weights_path)
    try:
        from stable_baselines3 import DQN, PPO
    except Exception as exc:
        raise RuntimeError(f"stable-baselines3 is required to load {algo.upper()}: {exc}")
    cls = PPO if algo == "ppo" else DQN
    return cls.load(weights_path, env=env)


def _predict(policy, obs, algo: str, step: int) -> int:
    if algo == "q_learning":
        return policy.act(obs, step=step, greedy=True)
    action, _ = policy.predict(obs, deterministic=True)
    return int(action)


def _run_episode(policy, env: MicrogridEnv, algo: str):
    obs, _ = env.reset()
    actions, rewards, socs, demands, pvs, tariffs = [], [], [], [], [], []
    step = 0
    done = False
    while not done:
        action = _predict(policy, obs, algo, step)
        next_obs, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        rewards.append(float(reward))
        socs.append(float(info["soc_kwh"]))
        demands.append(float(obs[1]))
        pvs.append(float(obs[2]))
        tariffs.append(float(obs[3]))
        obs = next_obs
        step += 1
        done = terminated or truncated
    return {
        "actions": actions,
        "rewards": rewards,
        "soc_kwh": socs,
        "demand_kw": demands,
        "pv_kw": pvs,
        "tariff_rate": tariffs,
    }


def _baseline_idle(env: MicrogridEnv):
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(1)
        rewards.append(float(reward))
        done = terminated or truncated
    return rewards


def _summarise(trace: dict, ref_demand: list[float]) -> dict:
    actions = np.array(trace["actions"])
    pred_dispatch = np.where(actions == 0, -2.5, np.where(actions == 2, 2.5, 0.0))
    mae = float(np.mean(np.abs(np.array(ref_demand) - (np.array(ref_demand) - 0.5 * pred_dispatch))))
    rmse = float(np.sqrt(np.mean((np.array(ref_demand) - (np.array(ref_demand) - 0.5 * pred_dispatch)) ** 2)))
    soc = np.array(trace["soc_kwh"]) / 10.0
    soc_violations = int(np.sum((soc < 0.1) | (soc > 0.9)))
    return {
        "total_reward": float(np.sum(trace["rewards"])),
        "mae_kw": mae,
        "rmse_kw": rmse,
        "soc_violations": soc_violations,
        "n_steps": len(trace["rewards"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained microgrid RL policy.")
    parser.add_argument("--weights", required=True, help="Path to the trained model weights")
    parser.add_argument("--datadir", required=True, help="Directory containing microgrid_data.csv")
    parser.add_argument("--output", required=True, help="Directory for evaluation logs, traces, and metrics")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    algo = _detect_algo(args.weights)
    data_path = os.path.join(args.datadir, "microgrid_data.csv")
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    cut = int(len(df) * 0.82)
    df_test = df.iloc[cut:].reset_index(drop=True)

    env = MicrogridEnv(df_test)
    policy = _load_policy(algo, args.weights, env)

    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "model"), exist_ok=True)
    log_path = os.path.join(args.output, "model", "test.log")

    start = time.time()
    trace = _run_episode(policy, env, algo)
    baseline_rewards = _baseline_idle(env)
    elapsed = time.time() - start

    metrics = _summarise(trace, trace["demand_kw"])
    metrics["baseline_total_reward"] = float(np.sum(baseline_rewards))
    metrics["cost_reduction_pct"] = 100.0 * (1.0 - metrics["total_reward"] / metrics["baseline_total_reward"]) \
        if metrics["baseline_total_reward"] != 0 else 0.0
    metrics["algo"] = algo
    metrics["weights"] = args.weights
    metrics["seed"] = args.seed
    metrics["wall_time_seconds"] = elapsed

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"# {algo.upper()} evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k, v in metrics.items():
            fh.write(f"{k}={v}\n")
        fh.write("\n")

    trace_path = os.path.join(args.output, "logs", f"{algo}_test_trace.csv")
    with open(trace_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "action", "reward", "soc_kwh", "demand_kw", "pv_kw", "tariff_rate"])
        for i in range(len(trace["actions"])):
            writer.writerow([
                i, trace["actions"][i], trace["rewards"][i],
                trace["soc_kwh"][i], trace["demand_kw"][i],
                trace["pv_kw"][i], trace["tariff_rate"][i],
            ])

    metrics_path = os.path.join(args.output, "model", f"{algo}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
