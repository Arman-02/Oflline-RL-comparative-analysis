# Offline RL Comparative Analysis

A comparative study of three offline reinforcement learning algorithms — **Behavior Cloning (BC)**, **TD3+BC**, and **Conservative Q-Learning (CQL)** — evaluated on continuous control benchmarks from the D4RL suite.

---

## Algorithms

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| **BC** | Imitation Learning | Supervised regression from observations to actions |
| **TD3+BC** | Policy Constraint | TD3 with a behavior cloning regularization term on the actor |
| **CQL** | Conservative RL | SAC with a conservative penalty that lower-bounds Q-values on out-of-distribution actions |

> CQL reference: Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*, NeurIPS 2020. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)

---

## Environments & Datasets

Experiments run on **MuJoCo** locomotion tasks loaded via [Minari](https://minari.farama.org/):

| Environment | Datasets |
|-------------|----------|
| HalfCheetah-v5 | `medium`, `medium-replay` |
| Hopper-v5 | `medium`, `medium-replay` |

- **medium** — data collected by a medium-quality policy
- **medium-replay** — full replay buffer of a medium policy (more diverse, lower quality)

---

## Results

Best D4RL normalised scores (higher is better, 100 ≈ expert-level):

| Environment | Dataset | BC | TD3+BC | CQL |
|---|---|---|---|---|
| HalfCheetah | medium | 126.5 | 120.4 | 58.5 |
| HalfCheetah | medium-replay | 61.0 | 65.3 | 60.0 |
| Hopper | medium | 112.9 | 112.7 | 111.4 |
| Hopper | medium-replay | 99.6 | 99.0 | 98.3 |

Learning curves and CQL alpha ablation plots are saved under `results/figures/`.

---

## Project Structure

```
.
├── train.py              # Main training script
├── run_experiments.py    # Batch experiment runner + analysis + plotting
├── visualize.py          # Standalone visualization utilities
├── bc.py                 # Behavior Cloning implementation
├── td3_bc.py             # TD3+BC implementation
├── cql.py                # CQL implementation
├── networks.py           # Shared network architectures (GaussianActor, TwinCritic)
├── utils.py              # Dataset loading, policy evaluation
├── requirements.txt      # Python dependencies
└── results/
    ├── figures/          # Learning curve and ablation plots
    ├── summary_table.csv # Aggregated best scores
    └── <exp_name>/       # Per-experiment checkpoints and curves
        ├── best.pt
        ├── final.pt
        ├── learning_curve.csv
        └── config.json
```

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, MuJoCo 3.x, PyTorch 2.x

---

## Usage

### Train a single experiment

```bash
# Behavior Cloning on HalfCheetah-medium
python train.py --algo bc --env halfcheetah --dataset medium

# TD3+BC on Hopper medium-replay
python train.py --algo td3bc --env hopper --dataset medium-replay

# CQL on HalfCheetah-medium with custom alpha
python train.py --algo cql --env halfcheetah --dataset medium --cql_alpha 5.0
```

### Run all experiments

```bash
# Run the full main experiment grid (BC + TD3+BC + CQL across all envs/datasets)
python run_experiments.py --group main

# Run CQL alpha ablation (α ∈ {0.1, 0.5, 1.0, 5.0, 10.0} on HalfCheetah-medium)
python run_experiments.py --group ablation

# Re-generate plots and summary table from existing results
python run_experiments.py --group analyse
```

Already-completed runs are automatically skipped (checks for `learning_curve.csv`).

### Key training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | required | `bc`, `td3bc`, or `cql` |
| `--env` | required | `halfcheetah` or `hopper` |
| `--dataset` | required | `medium` or `medium-replay` |
| `--n_steps` | 1,000,000 | Total gradient steps |
| `--batch_size` | 256 | Minibatch size |
| `--lr` | 3e-4 | Learning rate |
| `--cql_alpha` | 1.0 | CQL conservative penalty weight |
| `--seed` | 0 | Random seed |

---

## Hardware

Training auto-detects and uses **CUDA** > **Apple MPS** > **CPU** in that order.
