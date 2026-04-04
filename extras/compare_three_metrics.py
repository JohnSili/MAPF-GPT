"""
Run MAPF-GPT on the same scenario for three checkpoints (2M, trimmed L3, int8),
print metrics and save a bar-chart figure.

Usage (from repository root):
  python extras/compare_three_metrics.py
  python extras/compare_three_metrics.py --map_name validation-mazes-seed-000 --num_agents 32 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.run_episode import run_episode
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
from mapf_gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

RUNS = [
    ("2M (float)", "weights/model-2M.pt", "cuda"),
    ("2M-L3 (trim)", "weights/model-2M-L3.pt", "cuda"),
    ("2M-int8 (quant)", "weights/model-2M-int8.pt", "cpu"),
]


def register_maps():
    for maps_file in Path("eval_configs").rglob("maps.yaml"):
        with open(maps_file, "r") as f:
            ToolboxRegistry.register_maps(yaml.safe_load(f))


def run_one(
    label: str,
    weights: str,
    device: str,
    map_name: str,
    num_agents: int,
    seed: int,
    max_episode_steps: int,
) -> dict:
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    env_cfg = Environment(
        with_animation=False,
        observation_type="MAPF",
        on_target="nothing",
        map_name=map_name,
        max_episode_steps=max_episode_steps,
        num_agents=num_agents,
        seed=seed,
        obs_radius=5,
        collision_system="soft",
    )

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    env = create_eval_env(env_cfg)
    algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=str(path), device=device))
    algo.reset_states()
    results = run_episode(env, algo)
    return {"label": label, "weights": weights, **dict(results)}


def plot_metrics(rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.2

    isr = [float(r["ISR"]) for r in rows]
    csr = [float(r["CSR"]) for r in rows]
    runtime = [float(r["runtime"]) for r in rows]
    soc = [float(r["SoC"]) for r in rows]
    ep_len = [float(r["ep_length"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("MAPF-GPT: три варианта весов, одна сцена", fontsize=13)

    ax = axes[0, 0]
    ax.bar(x - w / 2, isr, w, label="ISR", color="#2ecc71")
    ax.bar(x + w / 2, csr, w, label="CSR", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("доля")
    ax.set_title("ISR / CSR")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(labels, runtime, color="#e74c3c")
    ax.set_ylabel("сек")
    ax.set_title("runtime (эпизод)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(labels, soc, color="#9b59b6")
    ax.set_ylabel("SoC")
    ax.set_title("Sum of costs")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(labels, ep_len, color="#f39c12")
    ax.set_ylabel("шаги")
    ax.set_title("ep_length")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_name", default="validation-mazes-seed-000")
    parser.add_argument("--num_agents", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=128)
    parser.add_argument("-o", "--output", type=Path, default=Path("svg/compare-three-metrics.png"))
    args = parser.parse_args()

    register_maps()

    rows = []
    for label, weights, dev in RUNS:
        r = run_one(
            label,
            weights,
            dev,
            args.map_name,
            args.num_agents,
            args.seed,
            args.max_episode_steps,
        )
        rows.append(r)
        print(
            f"{label}: ISR={r['ISR']:.4f} CSR={r['CSR']:.4f} "
            f"ep_length={r['ep_length']} SoC={r['SoC']:.1f} "
            f"runtime={r['runtime']:.3f}s"
        )

    plot_metrics(rows, args.output)
    print(f"\nГрафик сохранён: {args.output.resolve()}")


if __name__ == "__main__":
    main()
