"""
Visualise a trained agent.

Two modes
---------
1. Live render (opens a window):
   python3 visualize.py --exp_name cql_halfcheetah_medium

2. Save an MP4 (headless, works on Colab / remote servers):
   python3 visualize.py --exp_name cql_halfcheetah_medium --save_video

3. Compare all algorithms side-by-side as a plot:
   python3 visualize.py --plot_curves --env halfcheetah --dataset medium

4. Plot the alpha ablation:
   python3 visualize.py --plot_ablation
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "results"

GYM_ENVS = {
    "halfcheetah": "HalfCheetah-v5",
    "hopper":      "Hopper-v5",
}

ALGO_STYLES = {
    "bc":    dict(color="#4C72B0", linestyle="--",  linewidth=2, label="BC"),
    "td3bc": dict(color="#DD8452", linestyle="-.",  linewidth=2, label="TD3+BC"),
    "cql":   dict(color="#55A868", linestyle="-",   linewidth=2, label="CQL"),
}

CQL_ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]


# Load agent from a results directory

def load_agent(exp_dir: str, device: torch.device):
    """Reconstruct agent from config.json + best.pt in exp_dir."""
    with open(os.path.join(exp_dir, "config.json")) as f:
        cfg = json.load(f)

    algo       = cfg["algo"]
    hidden     = (cfg["hidden"], cfg["hidden"])
    state_dim, action_dim = _infer_dims(cfg["env"])
    max_action = 1.0

    if algo == "bc":
        from bc import BehaviorCloning
        agent = BehaviorCloning(state_dim, action_dim, hidden_dims=hidden, device=device)

    elif algo == "td3bc":
        from td3_bc import TD3BC
        agent = TD3BC(state_dim, action_dim, hidden_dims=hidden,
                      discount=cfg["discount"], tau=cfg["tau"],
                      policy_noise=cfg["policy_noise"], noise_clip=cfg["noise_clip"],
                      policy_freq=cfg["policy_freq"], alpha=cfg["td3bc_alpha"],
                      device=device)

    elif algo == "cql":
        from cql import CQL
        agent = CQL(state_dim, action_dim, hidden_dims=hidden,
                    discount=cfg["discount"], tau=cfg["tau"],
                    cql_alpha=cfg["cql_alpha"], cql_n_actions=cfg["cql_n_actions"],
                    device=device)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    ckpt_path = os.path.join(exp_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_dir, "final.pt")
    agent.load(ckpt_path)
    print(f"Loaded {algo.upper()} from {ckpt_path}")
    return agent, cfg


def _infer_dims(env_name: str):
    """Return (state_dim, action_dim) without constructing the full env."""
    dims = {"halfcheetah": (17, 6), "hopper": (11, 3)}
    return dims[env_name.lower()]


# Live / video rollout

def run_rollout(agent, cfg, n_episodes: int, device: torch.device,
                render_mode: str = "human", video_dir: str = None):
    import gymnasium as gym
    from utils import get_normalized_score

    gym_env_id = GYM_ENVS[cfg["env"]]

    if video_dir:
        from gymnasium.wrappers import RecordVideo
        env = gym.make(gym_env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=video_dir,
                          episode_trigger=lambda ep: True,
                          name_prefix=cfg.get("exp_name", "agent"))
        print(f"Saving video to: {video_dir}/")
    else:
        env = gym.make(gym_env_id, render_mode=render_mode)

    scores = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        norm = get_normalized_score(cfg["env"], ep_reward)
        scores.append(norm)
        print(f"  Episode {ep + 1}: reward={ep_reward:.1f}  normalised={norm:.1f}")

    env.close()
    print(f"\nMean normalised score over {n_episodes} episodes: {np.mean(scores):.1f}")


# Learning curve plots

def plot_curves(env: str, dataset: str, out_path: str = None, smooth: int = 3):
    import pandas as pd

    fig, ax = plt.subplots(figsize=(8, 5))

    found_any = False
    for algo, style in ALGO_STYLES.items():
        exp_name   = f"{algo}_{env}_{dataset}"
        curve_path = os.path.join(RESULTS_DIR, exp_name, "learning_curve.csv")
        if not os.path.exists(curve_path):
            continue
        df = pd.read_csv(curve_path)
        y  = df["norm_score"].rolling(smooth, min_periods=1).mean()
        ax.plot(df["step"] / 1e6, y, **style)
        ax.fill_between(df["step"] / 1e6,
                        df["norm_score"].rolling(smooth, min_periods=1).min(),
                        df["norm_score"].rolling(smooth, min_periods=1).max(),
                        alpha=0.15, color=style["color"])
        found_any = True

    if not found_any:
        print(f"No results found for {env}/{dataset}. Run training first.")
        plt.close(fig)
        return

    ax.set_xlabel("Training Steps (×10⁶)", fontsize=12)
    ax.set_ylabel("D4RL Normalised Score", fontsize=12)
    ax.set_title(f"{env.capitalize()} — {dataset}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = out_path or f"results/figures/curve_{env}_{dataset}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_all_curves():
    """4-panel grid: one subplot per (env, dataset) combo."""
    import pandas as pd

    combos = [
        ("halfcheetah", "medium"),
        ("halfcheetah", "medium-replay"),
        ("hopper",      "medium"),
        ("hopper",      "medium-replay"),
    ]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    for idx, (env, dataset) in enumerate(combos):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        found_any = False
        for algo, style in ALGO_STYLES.items():
            exp_name   = f"{algo}_{env}_{dataset}"
            curve_path = os.path.join(RESULTS_DIR, exp_name, "learning_curve.csv")
            if not os.path.exists(curve_path):
                continue
            df = pd.read_csv(curve_path)
            y  = df["norm_score"].rolling(3, min_periods=1).mean()
            ax.plot(df["step"] / 1e6, y, **style)
            found_any = True

        title = f"{env.capitalize()} — {dataset}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Steps (×10⁶)", fontsize=9)
        ax.set_ylabel("Norm. Score", fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9)
        if not found_any:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    save_path = "results/figures/all_curves.png"
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_ablation():
    """Bar + line chart for the CQL α ablation."""
    import pandas as pd

    data = []
    for alpha in CQL_ALPHAS:
        exp_name   = f"cql_halfcheetah_medium_alpha{alpha}"
        curve_path = os.path.join(RESULTS_DIR, exp_name, "learning_curve.csv")
        if not os.path.exists(curve_path):
            continue
        df = pd.read_csv(curve_path)
        data.append({"alpha": alpha, "best": df["norm_score"].max(),
                     "final": df["norm_score"].iloc[-1]})

    if not data:
        print("No ablation results found. Run: python3 run_experiments.py --group ablation")
        return

    df_ab = pd.DataFrame(data).sort_values("alpha")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: line plot
    ax = axes[0]
    ax.plot(df_ab["alpha"], df_ab["best"],  marker="o", color="#55A868",
            linewidth=2, markersize=8, label="Best score")
    ax.plot(df_ab["alpha"], df_ab["final"], marker="s", color="#55A868",
            linewidth=2, markersize=8, linestyle="--", label="Final score")
    ax.set_xscale("log")
    ax.set_xlabel("CQL α  (log scale)", fontsize=12)
    ax.set_ylabel("D4RL Normalised Score", fontsize=12)
    ax.set_title("CQL α Ablation — HalfCheetah-medium", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: learning curves coloured by α
    ax2 = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_ab)))
    for (_, row), color in zip(df_ab.iterrows(), cmap):
        exp_name   = f"cql_halfcheetah_medium_alpha{row['alpha']}"
        curve_path = os.path.join(RESULTS_DIR, exp_name, "learning_curve.csv")
        if not os.path.exists(curve_path):
            continue
        df = pd.read_csv(curve_path)
        y  = df["norm_score"].rolling(3, min_periods=1).mean()
        ax2.plot(df["step"] / 1e6, y, color=color,
                 linewidth=2, label=f"α={row['alpha']}")
    ax2.set_xlabel("Steps (×10⁶)", fontsize=12)
    ax2.set_ylabel("Norm. Score", fontsize=12)
    ax2.set_title("Learning Curves by α", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = "results/figures/ablation_alpha.png"
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_final_bar_chart():
    """Grouped bar chart of final best scores for all 12 experiments."""
    import pandas as pd

    combos = [
        ("halfcheetah", "medium"),
        ("halfcheetah", "medium-replay"),
        ("hopper",      "medium"),
        ("hopper",      "medium-replay"),
    ]
    algos  = ["bc", "td3bc", "cql"]
    labels = ["BC", "TD3+BC", "CQL"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x     = np.arange(len(combos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (algo, label, color) in enumerate(zip(algos, labels, colors)):
        scores = []
        for env, dataset in combos:
            exp_name   = f"{algo}_{env}_{dataset}"
            curve_path = os.path.join(RESULTS_DIR, exp_name, "learning_curve.csv")
            if os.path.exists(curve_path):
                df = pd.read_csv(curve_path)
                scores.append(df["norm_score"].max())
            else:
                scores.append(0.0)
        bars = ax.bar(x + i * width, scores, width, label=label, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{e}\n{d}" for e, d in combos], fontsize=10)
    ax.set_ylabel("Best D4RL Normalised Score", fontsize=12)
    ax.set_title("Algorithm Comparison Across Tasks", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, None)
    fig.tight_layout()

    save_path = "results/figures/bar_comparison.png"
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


# CLI

def parse_args():
    p = argparse.ArgumentParser()

    # Agent rollout
    p.add_argument("--exp_name",   type=str, default=None,
                   help="Experiment name to load (e.g. cql_halfcheetah_medium)")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--save_video", action="store_true",
                   help="Save MP4 instead of opening a live window")
    p.add_argument("--video_dir",  type=str, default=None,
                   help="Where to save videos (default: results/<exp>/videos)")

    # Plot modes
    p.add_argument("--plot_curves",   action="store_true",
                   help="Plot learning curves for one (env, dataset)")
    p.add_argument("--plot_all",      action="store_true",
                   help="4-panel learning curve grid for all tasks")
    p.add_argument("--plot_ablation", action="store_true",
                   help="CQL alpha ablation plot")
    p.add_argument("--plot_bar",      action="store_true",
                   help="Grouped bar chart comparing all algorithms")

    p.add_argument("--env",     type=str, default="halfcheetah",
                   choices=["halfcheetah", "hopper"])
    p.add_argument("--dataset", type=str, default="medium",
                   choices=["medium", "medium-replay"])

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")

    # ---- Agent rollout ---------------------------------------------------- #
    if args.exp_name:
        exp_dir = os.path.join(RESULTS_DIR, args.exp_name)
        if not os.path.isdir(exp_dir):
            print(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)

        agent, cfg = load_agent(exp_dir, device)

        video_dir = None
        if args.save_video:
            video_dir = args.video_dir or os.path.join(exp_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

        render_mode = "rgb_array" if args.save_video else "human"
        run_rollout(agent, cfg, args.n_episodes, device, render_mode, video_dir)

    # ---- Plot modes -------------------------------------------------------- #
    if args.plot_all:
        plot_all_curves()

    if args.plot_curves:
        plot_curves(args.env, args.dataset)

    if args.plot_ablation:
        plot_ablation()

    if args.plot_bar:
        plot_final_bar_chart()

    if not any([args.exp_name, args.plot_all, args.plot_curves,
                args.plot_ablation, args.plot_bar]):
        print(__doc__)


if __name__ == "__main__":
    main()
