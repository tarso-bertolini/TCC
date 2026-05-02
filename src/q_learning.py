"""Tabular Q-Learning baseline for the MicrogridEnv.

The continuous observation vector is discretised into a small number of bins per
dimension so that a classical Q-table can be used. This intentionally keeps the
algorithm simple to serve as a lower-bound baseline against PPO and DQN in the
benchmarking pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microgrid_env import MicrogridEnv
from data_generator import generate_microgrid_data


@dataclass
class QLearningConfig:
    alpha: float = 0.10
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000
    soc_bins: int = 10
    demand_bins: int = 8
    pv_bins: int = 6
    tariff_bins: int = 5


class TabularQLearning:
    def __init__(self, env: MicrogridEnv, cfg: QLearningConfig, seed: int = 0):
        self.env = env
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.n_actions = env.action_space.n

        low = env.observation_space.low
        high = env.observation_space.high
        bins = np.array([cfg.soc_bins, cfg.demand_bins, cfg.pv_bins, cfg.tariff_bins])

        self.edges = [np.linspace(low[i], high[i], bins[i] + 1)[1:-1] for i in range(4)]
        self.q_table = np.zeros((*bins, self.n_actions), dtype=np.float32)

    def _discretise(self, obs: np.ndarray) -> tuple:
        return tuple(int(np.digitize(obs[i], self.edges[i])) for i in range(4))

    def _epsilon(self, step: int) -> float:
        frac = min(1.0, step / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: np.ndarray, step: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self._epsilon(step):
            return int(self.rng.integers(self.n_actions))
        state = self._discretise(obs)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        s = self._discretise(obs)
        s_next = self._discretise(next_obs)
        target = reward + (0.0 if done else self.cfg.gamma * float(np.max(self.q_table[s_next])))
        idx = (*s, action)
        self.q_table[idx] += self.cfg.alpha * (target - self.q_table[idx])

    def learn(self, total_timesteps: int, log_path: str | None = None) -> list[float]:
        episode_rewards: list[float] = []
        running_reward = 0.0
        step = 0
        log_buffer: list[str] = []

        while step < total_timesteps:
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False
            while not done and step < total_timesteps:
                action = self.act(obs, step)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.update(obs, action, reward, next_obs, terminated or truncated)
                obs = next_obs
                ep_reward += float(reward)
                step += 1
                done = terminated or truncated
                if step % 1000 == 0:
                    msg = f"step={step:>6} eps={self._epsilon(step):.3f} running_reward={running_reward:.2f}"
                    log_buffer.append(msg)
            episode_rewards.append(ep_reward)
            running_reward = 0.9 * running_reward + 0.1 * ep_reward if episode_rewards[:-1] else ep_reward

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(log_buffer) + "\n")
        return episode_rewards

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)
        meta = {
            "alpha": self.cfg.alpha,
            "gamma": self.cfg.gamma,
            "epsilon_end": self.cfg.epsilon_end,
            "bins": [self.cfg.soc_bins, self.cfg.demand_bins, self.cfg.pv_bins, self.cfg.tariff_bins],
            "shape": list(self.q_table.shape),
        }
        with open(path.replace(".npy", ".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, env: MicrogridEnv, path: str) -> "TabularQLearning":
        cfg = QLearningConfig()
        with open(path.replace(".npy", ".json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        cfg.soc_bins, cfg.demand_bins, cfg.pv_bins, cfg.tariff_bins = meta["bins"]
        agent = cls(env, cfg)
        agent.q_table = np.load(path)
        return agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q-Learning baseline.")
    parser.add_argument("--data", required=True, help="Path to microgrid_data.csv")
    parser.add_argument("--output", required=True, help="Directory for the trained Q-table and log")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    train_split = int(len(df) * 0.82)
    env = MicrogridEnv(df.iloc[:train_split])

    cfg = QLearningConfig()
    agent = TabularQLearning(env, cfg, seed=args.seed)
    log_path = os.path.join(args.output, "logs", "q_learning_training.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Q-Learning training started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"# cfg={cfg}\n# seed={args.seed} timesteps={args.timesteps}\n")

    start = time.time()
    agent.learn(args.timesteps, log_path=log_path)
    elapsed = time.time() - start

    weights_path = os.path.join(args.output, "model", "q_table.npy")
    agent.save(weights_path)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"# wall_time_seconds={elapsed:.1f}\n# saved={weights_path}\n")


if __name__ == "__main__":
    main()
