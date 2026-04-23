"""
TD3+BC — "A Minimalist Approach to Offline Reinforcement Learning"
Fujimoto & Gu, NeurIPS 2021  (https://arxiv.org/abs/2106.06860)

Actor loss:  -λ·Q(s, π(s))  +  ||π(s) - a||²
where λ = α / (1/N Σ|Q(s_i, a_i)|)  and α = 2.5 by default.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import DeterministicActor, TwinCritic


class TD3BC:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: tuple = (256, 256),
                 # TD3 hyperparameters
                 discount: float = 0.99,
                 tau: float = 5e-3,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 # BC regularisation
                 alpha: float = 2.5,
                 # optimisation
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 device: torch.device = torch.device("cpu")):

        self.device      = device
        self.max_action  = max_action
        self.discount    = discount
        self.tau         = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip   = noise_clip   * max_action
        self.policy_freq = policy_freq
        self.alpha       = alpha

        self.actor  = DeterministicActor(state_dim, action_dim, hidden_dims, max_action).to(device)
        self.critic = TwinCritic(state_dim, action_dim, hidden_dims).to(device)

        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._step = 0

    def train_step(self, replay_buffer, batch_size: int = 256) -> dict:
        self._step += 1
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)

        
        # Critic update
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + (1.0 - dones) * self.discount * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        info = {"critic_loss": critic_loss.item()}

        
        # Delayed actor update
        
        if self._step % self.policy_freq == 0:
            pi = self.actor(states)
            q_pi = self.critic.q1_value(states, pi)

            # Normalisation factor λ
            lam = self.alpha / (q_pi.abs().mean().detach() + 1e-8)

            actor_loss = -lam * q_pi.mean() + nn.functional.mse_loss(pi, actions)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft-update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            info["actor_loss"] = actor_loss.item()

        return info

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(state.to(self.device))
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
