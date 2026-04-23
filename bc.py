"""
Behavior Cloning (BC) — supervised learning on (state, action) pairs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networks import BCPolicy


class BehaviorCloning:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: tuple = (256, 256),
                 lr: float = 3e-4,
                 device: torch.device = torch.device("cpu")):

        self.device = device
        self.max_action = max_action

        self.policy = BCPolicy(state_dim, action_dim, hidden_dims, max_action).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_step(self, replay_buffer, batch_size: int = 256) -> dict:
        states, actions, *_ = replay_buffer.sample(batch_size, self.device)

        pred_actions = self.policy(states)
        loss = nn.functional.mse_loss(pred_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"bc_loss": loss.item()}

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action = self.policy(state.to(self.device))
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
