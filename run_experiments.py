import argparse
import subprocess
import sys
import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Experiment grid

MAIN_EXPERIMENTS = [
    # (algo, env, dataset, extra_flags)
    ("bc",    "halfcheetah", "medium",        []),
    ("bc",    "halfcheetah", "medium-replay", []),
    ("bc",    "hopper",      "medium",        []),
    ("bc",    "hopper",      "medium-replay", []),

    ("td3bc", "halfcheetah", "medium",        []),
    ("td3bc", "halfcheetah", "medium-replay", []),
    ("td3bc", "hopper",      "medium",        []),
    ("td3bc", "hopper",      "medium-replay", []),

    ("cql",   "halfcheetah", "medium",        ["--cql_alpha", "5.0"]),
    ("cql",   "halfcheetah", "medium-replay", ["--cql_alpha", "5.0"]),
    ("cql",   "hopper",      "medium",        ["--cql_alpha", "5.0"]),
    ("cql",   "hopper",      "medium-replay", ["--cql_alpha", "5.0"]),
]

# α ablation on HalfCheetah-medium
CQL_ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0]

ABLATION_EXPERIMENTS = [
    ("cql", "halfcheetah", "medium",
     ["--cql_alpha", str(a)],
     f"cql_halfcheetah_medium_alpha{a}")
    for a in CQL_ALPHAS
]


# Helper: run a single training job as a subprocess

def run_job(algo, env, dataset, extra_flags=None, exp_name=None,
            n_steps=1_000_000, seed=0, skip_existing=True):
    name = exp_name or f"{algo}_{env}_{dataset}"
    curve_path = os.path.join("results", name, "learning_curve.csv")
    if skip_existing and os.path.exists(curve_path):
        print(f"[SKIP] {name}  (learning_curve.csv already exists)")
        return None

    cmd = [sys.executable, "train.py",
           "--algo",    algo,
           "--env",     env,
           "--dataset", dataset,
           "--n_steps", str(n_steps),
           "--seed",    str(seed)]
    if exp_name:
        cmd += ["--exp_name", exp_name]
    if extra_flags:
        cmd += [str(f) for f in extra_flags]

    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


# Analysis helpers

def load_results(results_dir="results"):
    rows = []
    for cfg_path in glob.glob(os.path.join(results_dir, "*", "config.json")):
        exp_dir = os.path.dirname(cfg_path)
        curve_path = os.path.join(exp_dir, "learning_curve.csv")
        if not os.path.exists(curve_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        df = pd.read_csv(curve_path)
        best_score = df["norm_score"].max()
        rows.append({
            "algo":    cfg["algo"],
            "env":     cfg["env"],
            "dataset": cfg["dataset"],
            "exp_name": cfg.get("exp_name") or f"{cfg['algo']}_{cfg['env']}_{cfg['dataset']}",
            "best_score": best_score,
        })
    return pd.DataFrame(rows)


def make_summary_table(results_df, output_path="results/summary_table.csv"):
    pivot = results_df.pivot_table(
        index=["env", "dataset"],
        columns="algo",
        values="best_score",
        aggfunc="max",
    )
    pivot = pivot.round(1)
    pivot.to_csv(output_path)
    print("\n" + "="*60)
    print("Summary Table (D4RL Normalised Score)")
    print("="*60)
    print(pivot.to_string())
    print("="*60 + "\n")
    return pivot


def plot_learning_curves(results_dir="results", output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)

    # Group by (env, dataset) and plot all algos together
    combos = [
        ("halfcheetah", "medium"),
        ("halfcheetah", "medium-replay"),
        ("hopper",      "medium"),
        ("hopper",      "medium-replay"),
    ]
    algo_styles = {
        "bc":    dict(color="tab:blue",   linestyle="--",  label="BC"),
        "td3bc": dict(color="tab:orange", linestyle="-.",  label="TD3+BC"),
        "cql":   dict(color="tab:green",  linestyle="-",   label="CQL"),
    }

    for env, dataset in combos:
        fig, ax = plt.subplots(figsize=(7, 4))
        any_data = False
        for algo, style in algo_styles.items():
            exp_name = f"{algo}_{env}_{dataset}"
            curve_path = os.path.join(results_dir, exp_name, "learning_curve.csv")
            if not os.path.exists(curve_path):
                continue
            df = pd.read_csv(curve_path)
            ax.plot(df["step"] / 1e6, df["norm_score"], **style)
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Training Steps (×10⁶)")
        ax.set_ylabel("D4RL Normalised Score")
        ax.set_title(f"{env.capitalize()} — {dataset}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"curve_{env}_{dataset}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved {fname}")


def plot_alpha_ablation(results_dir="results", output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    data = []
    for alpha in CQL_ALPHAS:
        # prefer dedicated ablation folder, fall back to main experiment folder
        exp_name = f"cql_halfcheetah_medium_alpha{alpha}"
        curve_path = os.path.join(results_dir, exp_name, "learning_curve.csv")
        if not os.path.exists(curve_path):
            fallback = os.path.join(results_dir, "cql_halfcheetah_medium", "learning_curve.csv")
            cfg_path  = os.path.join(results_dir, "cql_halfcheetah_medium", "config.json")
            if os.path.exists(fallback) and os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if cfg.get("cql_alpha") == alpha:
                    curve_path = fallback
                else:
                    continue
            else:
                continue
        df = pd.read_csv(curve_path)
        data.append({"alpha": alpha, "best_score": df["norm_score"].max()})

    if not data:
        print("No ablation results found yet.")
        return

    df_ab = pd.DataFrame(data).sort_values("alpha")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_ab["alpha"], df_ab["best_score"], marker="o", color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("CQL α (log scale)")
    ax.set_ylabel("Best D4RL Normalised Score")
    ax.set_title("CQL α Ablation — HalfCheetah-medium")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "ablation_cql_alpha.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)


def plot_alpha_learning_curves(results_dir="results", output_dir="results/figures"):
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.viridis([0.1, 0.3, 0.55, 0.75, 0.95])
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for i, alpha in enumerate(CQL_ALPHAS):
        exp_name  = f"cql_halfcheetah_medium_alpha{alpha}"
        curve_path = os.path.join(results_dir, exp_name, "learning_curve.csv")
        if not os.path.exists(curve_path):
            # fall back to main experiment folder if alpha matches
            fallback = os.path.join(results_dir, "cql_halfcheetah_medium", "learning_curve.csv")
            cfg_path  = os.path.join(results_dir, "cql_halfcheetah_medium", "config.json")
            if os.path.exists(fallback) and os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if cfg.get("cql_alpha") == alpha:
                    curve_path = fallback
                else:
                    continue
            else:
                continue
        df = pd.read_csv(curve_path)
        ax.plot(df["step"] / 1e6, df["norm_score"],
                color=colors[i], label=f"α={alpha}")
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.set_xlabel("Training Steps (×10⁶)")
    ax.set_ylabel("D4RL Normalised Score")
    ax.set_title("CQL α Ablation — Learning Curves — HalfCheetah-medium")
    ax.legend(title="CQL α")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "ablation_cql_alpha_curves.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")
    print(f"Saved {fname}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["main", "ablation", "analyse"],
                        default="main",
                        help="Which experiment group to run, or 'analyse' to just plot.")
    parser.add_argument("--n_steps", type=int, default=1_000_000)
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.group == "main":
        for algo, env, dataset, flags in MAIN_EXPERIMENTS:
            run_job(algo, env, dataset, extra_flags=flags,
                    n_steps=args.n_steps, seed=args.seed, skip_existing=True)

    elif args.group == "ablation":
        for algo, env, dataset, flags, exp_name in ABLATION_EXPERIMENTS:
            run_job(algo, env, dataset, extra_flags=flags, exp_name=exp_name,
                    n_steps=args.n_steps, seed=args.seed, skip_existing=True)

    # Always analyse whatever results exist
    results_df = load_results()
    if not results_df.empty:
        make_summary_table(results_df)
        plot_learning_curves()
        plot_alpha_ablation()
        plot_alpha_learning_curves()
    else:
        print("No completed results found yet.")


if __name__ == "__main__":
    main()
