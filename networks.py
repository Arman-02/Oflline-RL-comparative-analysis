"""
Shared neural network architectures for BC, TD3+BC, and CQL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU):
    layers = []
    dims = [input_dim] + list(hidden_dims)
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


# Deterministic Actor  (used by TD3+BC)

class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), max_action=1.0):
        super().__init__()
        self.net = mlp(state_dim, hidden_dims, action_dim)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))


# Stochastic Actor  (used by CQL / SAC)

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), max_action=1.0):
        super().__init__()
        self.net = mlp(state_dim, hidden_dims, action_dim * 2)
        self.max_action = max_action

    def forward(self, state):
        out = self.net(state)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()                          # reparameterised sample
        action = torch.tanh(x)
        # log-prob with tanh squashing correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action * self.max_action, log_prob

    def log_prob(self, state, action):
        """Log-prob of a given (state, action) pair — used for BC term."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        # invert tanh to get pre-squash value
        action_clipped = (action / self.max_action).clamp(-1 + 1e-6, 1 - 1e-6)
        x = torch.atanh(action_clipped)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(x) - torch.log(1 - action_clipped.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


# BC Policy  (simple deterministic MLP, trained with MSE)

class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), max_action=1.0):
        super().__init__()
        self.net = mlp(state_dim, hidden_dims, action_dim)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.net(state))


# Twin Critic  (shared by TD3+BC and CQL)

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, hidden_dims, 1)
        self.q2 = mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
