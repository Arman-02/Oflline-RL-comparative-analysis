"""
Shared utilities: replay buffer, dataset loading, evaluation, D4RL score normalisation.
"""

import numpy as np
import torch
import gymnasium as gym

# D4RL reference scores for normalisation
# (random_score, expert_score)
D4RL_REFERENCE_SCORES = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper":      (20.272305,  3234.3),
}


def get_normalized_score(env_name: str, score: float) -> float:
    key = env_name.lower()
    for k, (rand, expert) in D4RL_REFERENCE_SCORES.items():
        if k in key:
            return (score - rand) / (expert - rand) * 100.0
    raise ValueError(f"No reference scores found for env '{env_name}'")



# Replay Buffer


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 2_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx],      device=device),
            torch.tensor(self.actions[idx],     device=device),
            torch.tensor(self.rewards[idx],     device=device),
            torch.tensor(self.next_states[idx], device=device),
            torch.tensor(self.dones[idx],       device=device),
        )

    def normalise_states(self):
        """
        Normalise observations to zero mean / unit std (in-place).
        Returns (mean, std) so they can be reused at evaluation time.
        """
        mean = self.states[:self.size].mean(axis=0)
        std  = self.states[:self.size].std(axis=0) + 1e-3
        self.states[:self.size]      = (self.states[:self.size]      - mean) / std
        self.next_states[:self.size] = (self.next_states[:self.size] - mean) / std
        return mean, std



# Load a minari dataset into a ReplayBuffer


def load_dataset_to_buffer(dataset_id: str,
                           normalise: bool = True,
                           max_size: int = 2_000_000):
    """
    Load a minari offline dataset and return a filled ReplayBuffer plus
    normalisation statistics.

    Parameters
    ----------
    dataset_id : str
        e.g. 'd4rl_halfcheetah-medium-v2'
    normalise : bool
        If True, normalise observations to zero mean / unit std.

    Returns
    -------
    buf : ReplayBuffer
    state_mean : np.ndarray or None
    state_std  : np.ndarray or None
    """
    import minari

    dataset = minari.load_dataset(dataset_id, download=True)

    # Infer dims from first episode
    first = next(iter(dataset.iterate_episodes()))
    state_dim  = first.observations.shape[-1]
    action_dim = first.actions.shape[-1]

    buf = ReplayBuffer(state_dim, action_dim, max_size)

    for ep in dataset.iterate_episodes():
        obs   = ep.observations   # (T+1, obs_dim)
        acts  = ep.actions        # (T, act_dim)
        rews  = ep.rewards        # (T,)
        terms = ep.terminations   # (T,)

        T = len(acts)
        for t in range(T):
            done = float(terms[t])
            buf.add(obs[t], acts[t], rews[t], obs[t + 1], done)

    state_mean = state_std = None
    if normalise:
        state_mean, state_std = buf.normalise_states()

    print(f"Loaded {buf.size:,} transitions from '{dataset_id}'")
    return buf, state_mean, state_std



# Policy evaluation


def evaluate_policy(policy, env_name: str, n_episodes: int = 10,
                    device: torch.device = torch.device("cpu"),
                    state_mean=None, state_std=None,
                    seed: int = 0):
    """
    Roll out `policy` for `n_episodes` episodes, return mean normalised score.

    policy must implement:  action = policy.select_action(state_tensor)
    """
    env = gym.make(env_name)
    total_reward = 0.0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        ep_reward = 0.0
        while not done:
            if state_mean is not None:
                obs = (obs - state_mean) / state_std
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward

    env.close()
    mean_reward = total_reward / n_episodes
    norm_score  = get_normalized_score(env_name, mean_reward)
    return mean_reward, norm_score
