"""
Main training script.

Usage examples
--------------
# Behavior Cloning on HalfCheetah medium
python train.py --algo bc --env halfcheetah --dataset medium

# TD3+BC on Hopper medium-replay
python train.py --algo td3bc --env hopper --dataset medium-replay

# CQL on HalfCheetah medium with alpha=5.0
python train.py --algo cql --env halfcheetah --dataset medium --cql_alpha 5.0

# CQL alpha ablation (runs one config — loop over alphas in run_experiments.py)
python train.py --algo cql --env halfcheetah --dataset medium --cql_alpha 0.1 --exp_name cql_alpha0.1
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from utils import load_dataset_to_buffer, evaluate_policy



# Minari dataset IDs


DATASET_IDS = {
    ("halfcheetah", "medium"):        "mujoco/halfcheetah/medium-v0",
    ("halfcheetah", "medium-replay"): "mujoco/halfcheetah/simple-v0",
    ("hopper",      "medium"):        "mujoco/hopper/medium-v0",
    ("hopper",      "medium-replay"): "mujoco/hopper/simple-v0",
}

GYM_ENVS = {
    "halfcheetah": "HalfCheetah-v5",
    "hopper":      "Hopper-v5",
}



# Argument parsing


def parse_args():
    p = argparse.ArgumentParser()

    # Experiment identity
    p.add_argument("--algo",    type=str, required=True, choices=["bc", "td3bc", "cql"])
    p.add_argument("--env",     type=str, required=True, choices=["halfcheetah", "hopper"])
    p.add_argument("--dataset", type=str, required=True, choices=["medium", "medium-replay"])
    p.add_argument("--exp_name", type=str, default=None,
                   help="Override experiment name (default: <algo>_<env>_<dataset>)")
    p.add_argument("--seed",    type=int, default=0)

    # Training
    p.add_argument("--n_steps",      type=int,   default=1_000_000)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--eval_freq",    type=int,   default=5_000)
    p.add_argument("--n_eval_ep",    type=int,   default=10)
    p.add_argument("--normalise",    action="store_true", default=True)

    # Shared network / optimisation
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--discount",     type=float, default=0.99)

    # TD3+BC
    p.add_argument("--td3bc_alpha",  type=float, default=2.5)
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip",   type=float, default=0.5)
    p.add_argument("--policy_freq",  type=int,   default=2)
    p.add_argument("--tau",          type=float, default=5e-3)

    # CQL
    p.add_argument("--cql_alpha",    type=float, default=1.0)
    p.add_argument("--cql_n_actions",type=int,   default=10)

    # Output
    p.add_argument("--results_dir",  type=str,   default="results")

    return p.parse_args()



# Agent factory


def build_agent(args, state_dim, action_dim, device):
    hidden = (args.hidden, args.hidden)

    if args.algo == "bc":
        from bc import BehaviorCloning
        return BehaviorCloning(state_dim, action_dim,
                               hidden_dims=hidden, lr=args.lr, device=device)

    elif args.algo == "td3bc":
        from td3_bc import TD3BC
        return TD3BC(state_dim, action_dim,
                     hidden_dims=hidden,
                     discount=args.discount, tau=args.tau,
                     policy_noise=args.policy_noise, noise_clip=args.noise_clip,
                     policy_freq=args.policy_freq,
                     alpha=args.td3bc_alpha,
                     actor_lr=args.lr, critic_lr=args.lr,
                     device=device)

    elif args.algo == "cql":
        from cql import CQL
        return CQL(state_dim, action_dim,
                   hidden_dims=hidden,
                   discount=args.discount, tau=args.tau,
                   cql_alpha=args.cql_alpha,
                   cql_n_actions=args.cql_n_actions,
                   actor_lr=args.lr, critic_lr=args.lr,
                   device=device)

    else:
        raise ValueError(f"Unknown algo: {args.algo}")



# Main training loop


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    # Experiment name and output dir
    exp_name = args.exp_name or f"{args.algo}_{args.env}_{args.dataset}"
    out_dir  = os.path.join(args.results_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset — returns (buffer, mean, std)
    dataset_id = DATASET_IDS[(args.env, args.dataset)]
    buffer, state_mean, state_std = load_dataset_to_buffer(
        dataset_id, normalise=args.normalise)

    # Build agent — infer dims from buffer
    state_dim  = buffer.states.shape[1]
    action_dim = buffer.actions.shape[1]
    agent = build_agent(args, state_dim, action_dim, device)

    gym_env = GYM_ENVS[args.env]

    # Logging
    log_rows = []
    best_score = -np.inf
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  {exp_name}  |  {device}  |  {buffer.size:,} transitions")
    print(f"{'='*60}\n")

    pbar = tqdm(range(1, args.n_steps + 1), desc=exp_name,
                unit="step", dynamic_ncols=True)

    for step in pbar:
        info = agent.train_step(buffer, args.batch_size)

        if step % args.eval_freq == 0:
            mean_reward, norm_score = evaluate_policy(
                agent, gym_env,
                n_episodes=args.n_eval_ep,
                device=device,
                state_mean=state_mean,
                state_std=state_std,
                seed=args.seed,
            )

            elapsed = time.time() - t0
            pbar.set_postfix({
                "score": f"{norm_score:.1f}",
                "best":  f"{best_score:.1f}",
                "elapsed": f"{elapsed:.0f}s",
                **{k: f"{v:.4f}" for k, v in info.items()},
            })

            row = {"step": step, "norm_score": norm_score, "mean_reward": mean_reward,
                   "elapsed_s": elapsed, **info}
            log_rows.append(row)

            # Save best checkpoint
            if norm_score > best_score:
                best_score = norm_score
                agent.save(os.path.join(out_dir, "best.pt"))

    # Save final checkpoint and curves
    agent.save(os.path.join(out_dir, "final.pt"))
    df = pd.DataFrame(log_rows)
    df.to_csv(os.path.join(out_dir, "learning_curve.csv"), index=False)

    print(f"\nBest normalised score: {best_score:.1f}")
    print(f"Results saved to: {out_dir}")
    return df, best_score


if __name__ == "__main__":
    args = parse_args()
    train(args)
