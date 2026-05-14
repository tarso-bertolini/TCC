"""Regenerate the reproducible artifacts that ship at the repo root.

This script does two things:

1. Calls `data_generator.generate_microgrid_data()` (seed=42) and writes the
   result to `data/microgrid_data.csv`.
2. Renders the two manuscript diagrams (`pipeline_diagram.png` and
   `architecture_diagram.png`) used by `tcc.tex`.

It does NOT run training or evaluation. To populate `output/model/` and
`output/logs/`, run `src/train.py` and `src/test.py` directly (see
`readme.txt` for the exact commands).

Usage:
    python scripts/generate_artifacts.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_generator import generate_microgrid_data  # noqa: E402


def write_dataset() -> None:
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    df = generate_microgrid_data()
    out = os.path.join(data_dir, "microgrid_data.csv")
    df.to_csv(out, index=False)
    print(f"[data] {out} ({len(df)} rows)")


def render_pipeline_diagram() -> None:
    out = os.path.join(ROOT, "pipeline_diagram.png")
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes_top = [
        (0.5, 4.5, "Historical PV /\nGrid telemetry", "#eef5ff", "#7aa5d8"),
        (2.8, 4.5, "MDP Simulation Env\n(Stochastic microgrid)", "#eef5ff", "#7aa5d8"),
        (5.4, 4.5, "Feature\nengineering", "#fff4e6", "#d8a55f"),
        (7.6, 4.5, "Cloud RL Training\nPPO / DQN / Q-Learning", "#fff4e6", "#d8a55f"),
        (10.1, 4.5, "Model selection\n(best reward)", "#fff4e6", "#d8a55f"),
        (12.3, 4.5, "Quantisation\nTFLite Micro", "#eef9ee", "#7dbb7d"),
    ]
    for x, y, label, fc, ec in boxes_top:
        ax.add_patch(plt.Rectangle((x, y - 0.6), 1.9, 1.2, fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.95, y, label, ha="center", va="center", fontsize=9)

    boxes_bot = [
        (2.8, 1.5, "Model comparison\n(cost, MAE, latency)", "#f4f0ff", "#9c8ad6"),
        (5.4, 1.5, "Legacy vs current\ndraft", "#f4f0ff", "#9c8ad6"),
        (8.0, 1.5, "Battery dispatch\nactions", "#f4f0ff", "#9c8ad6"),
        (10.4, 1.5, "Microgrid\noperation", "#f4f0ff", "#9c8ad6"),
        (12.7, 1.5, "ESP32 Edge\ninference", "#eef9ee", "#7dbb7d"),
    ]
    for x, y, label, fc, ec in boxes_bot:
        ax.add_patch(plt.Rectangle((x - 0.05, y - 0.6), 1.9, 1.2, fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.9, y, label, ha="center", va="center", fontsize=9)

    arrows_top = [(0.5, 2.8), (2.8, 5.4), (5.4, 7.6), (7.6, 10.1), (10.1, 12.3)]
    for x1, x2 in arrows_top:
        ax.annotate("", xy=(x2, 4.5), xytext=(x1 + 1.9, 4.5),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))
    ax.annotate("", xy=(2.8 + 0.95, 1.5 + 0.6), xytext=(7.6 + 0.95, 4.5 - 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#888", linestyle="dashed"))
    ax.annotate("", xy=(8.0 + 0.95, 1.5 + 0.6), xytext=(12.3 + 0.95, 4.5 - 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#888", linestyle="dashed"))
    for b_left, b_right in zip(boxes_bot[:-1], boxes_bot[1:]):
        x1 = b_left[0]
        x2 = b_right[0]
        ax.annotate("", xy=(x2 - 0.05, 1.5), xytext=(x1 - 0.05 + 1.9, 1.5),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))

    ax.text(7, 6.5, "Training pipeline", ha="center", fontsize=12, fontweight="bold", color="#1f2d3d")
    ax.text(7, 3.4, "Validation / deployment", ha="center", fontsize=12, fontweight="bold", color="#1f2d3d")

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[fig] {out}")


def render_architecture_diagram() -> None:
    out = os.path.join(ROOT, "architecture_diagram.png")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0.4, 1.0), 4.6, 4.5, fc="#fff4e6", ec="#d8a55f", lw=1.5))
    ax.text(2.7, 5.1, "Workstation (training tier)", ha="center", fontsize=11, fontweight="bold", color="#7a4f1d")
    for i, (label, sub) in enumerate([
        ("Gymnasium MDP Env", "MicrogridEnv"),
        ("PPO / DQN / Q-Learning", "Stable-Baselines3"),
        ("ONNX export + TFLite Micro int8", "src/quantize.py"),
    ]):
        ax.add_patch(plt.Rectangle((0.7, 3.7 - 1.2 * i), 4.0, 1.0, fc="white", ec="#bf8b4d"))
        ax.text(2.7, 4.05 - 1.2 * i, label, ha="center", fontsize=10)
        ax.text(2.7, 3.85 - 1.2 * i, sub, ha="center", fontsize=8, style="italic", color="#444")

    ax.add_patch(plt.Rectangle((7.0, 1.0), 4.6, 4.5, fc="#eef9ee", ec="#7dbb7d", lw=1.5))
    ax.text(9.3, 5.1, "Edge tier (ESP32)", ha="center", fontsize=11, fontweight="bold", color="#2c5f2c")
    for i, (label, sub) in enumerate([
        ("Sensor read (ADC)", "SOC / demand / PV / tariff"),
        ("TFLite Micro interpreter", "ppo_quantized.tflite"),
        ("Battery dispatch (GPIO)", "charge / idle / discharge"),
    ]):
        ax.add_patch(plt.Rectangle((7.3, 3.7 - 1.2 * i), 4.0, 1.0, fc="white", ec="#5fa56f"))
        ax.text(9.3, 4.05 - 1.2 * i, label, ha="center", fontsize=10)
        ax.text(9.3, 3.85 - 1.2 * i, sub, ha="center", fontsize=8, style="italic", color="#444")

    ax.annotate("Quantised\npolicy", xy=(7.0, 3.6), xytext=(5.0, 3.6),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#444"),
                ha="center", fontsize=8)
    ax.annotate("Telemetry /\nlogs", xy=(5.0, 2.0), xytext=(7.0, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#888", linestyle="dashed"),
                ha="center", fontsize=8, color="#666")
    ax.text(6, 6.1, "Distributed Edge architecture for microgrid optimisation",
            ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[fig] {out}")


def main() -> None:
    write_dataset()
    render_pipeline_diagram()
    render_architecture_diagram()


if __name__ == "__main__":
    main()
