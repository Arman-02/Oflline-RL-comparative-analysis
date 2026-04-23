"""
Conservative Q-Learning (CQL) for continuous action spaces.
Kumar et al., NeurIPS 2020  (https://arxiv.org/abs/2006.04779)

Built on top of SAC with an added conservative penalty on the Q-function.

CQL critic loss (per Q-network):
  L_CQL = α · [ log Σ_a exp Q(s,a) - E_{a~β}[Q(s,a)] ]  +  standard Bellman error

For continuous actions the logsumexp is approximated by sampling:
  - K actions from Uniform[-1,1]
  - K actions from the current policy π
and computing log-mean-exp over the 2K samples.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import GaussianActor, TwinCritic


class CQL:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: tuple = (256, 256),
                 # SAC hyperparameters
                 discount: float = 0.99,
                 tau: float = 5e-3,
                 init_temperature: float = 1.0,
                 target_entropy: float = None,    # None → -action_dim
                 # CQL hyperparameters
                 cql_alpha: float = 1.0,          # weight on conservative penalty
                 cql_n_actions: int = 10,         # # samples for logsumexp
                 cql_lagrange: bool = False,       # auto-tune cql_alpha via Lagrange
                 cql_target_action_gap: float = -1.0,  # target for Lagrange
                 # optimisation
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 device: torch.device = torch.device("cpu")):

        self.device      = device
        self.max_action  = max_action
        self.discount    = discount
        self.tau         = tau
        self.action_dim  = action_dim

        # CQL settings
        self.cql_alpha              = cql_alpha
        self.cql_n_actions          = cql_n_actions
        self.cql_lagrange           = cql_lagrange
        self.cql_target_action_gap  = cql_target_action_gap

        # SAC temperature
        self.log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32, requires_grad=True, device=device)
        self.target_entropy = -action_dim if target_entropy is None else target_entropy

        # Networks
        self.actor  = GaussianActor(state_dim, action_dim, hidden_dims, max_action).to(device)
        self.critic = TwinCritic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimisers
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.temp_optimizer   = optim.Adam([self.log_alpha],          lr=temp_lr)

        # Optional Lagrange multiplier for CQL α
        if cql_lagrange:
            self.log_cql_alpha = torch.tensor(np.log(cql_alpha), dtype=torch.float32, requires_grad=True, device=device)
            self.cql_alpha_optimizer = optim.Adam([self.log_cql_alpha], lr=temp_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample_actions_for_cql(self, states, n):
        """
        Returns tensors of shape (B, n, action_dim) for:
          - random actions uniformly in [-max_action, max_action]
          - actions sampled from current policy
        Also returns log-probs of shape (B, n, 1) for policy actions.
        """
        B = states.shape[0]

        # Random uniform actions
        rand_actions = (torch.rand(B, n, self.action_dim, device=self.device) * 2 - 1) * self.max_action
        rand_log_prob = torch.log(torch.tensor(0.5 ** self.action_dim, device=self.device))  # log(1/2^d) per action

        # Policy actions — repeat state n times for efficiency
        states_rep = states.unsqueeze(1).repeat(1, n, 1).view(B * n, -1)
        pi_actions, pi_log_probs = self.actor.sample(states_rep)
        pi_actions   = pi_actions.view(B, n, self.action_dim)
        pi_log_probs = pi_log_probs.view(B, n, 1)

        return rand_actions, rand_log_prob, pi_actions, pi_log_probs

    def _cql_loss(self, states, actions):
        """
        Compute CQL penalty for one Q-network (applied to both Q1 and Q2).

        Returns scalar conservative loss.
        """
        B = states.shape[0]
        n = self.cql_n_actions

        rand_actions, rand_log_prob, pi_actions, pi_log_probs = \
            self._sample_actions_for_cql(states, n)

        # Q-values for random and policy actions — shape (B, n, 1)
        def get_q_values(critic_fn, sampled_actions):
            s_rep = states.unsqueeze(1).repeat(1, n, 1).view(B * n, -1)
            a_rep = sampled_actions.view(B * n, self.action_dim)
            sa = torch.cat([s_rep, a_rep], dim=-1)
            return critic_fn(sa).view(B, n, 1)

        q1_rand = get_q_values(self.critic.q1, rand_actions)   # (B, n, 1)
        q2_rand = get_q_values(self.critic.q2, rand_actions)
        q1_pi   = get_q_values(self.critic.q1, pi_actions)
        q2_pi   = get_q_values(self.critic.q2, pi_actions)

        # Importance-weighted logsumexp:  log E[exp(Q - log π)]
        # For random actions the IS weight is 1/(1/2)^d → subtract log(1/2^d)
        rand_log_prob_t = torch.full((B, n, 1), rand_log_prob.item(), device=self.device)

        cat_q1 = torch.cat([q1_rand - rand_log_prob_t, q1_pi - pi_log_probs], dim=1)  # (B, 2n, 1)
        cat_q2 = torch.cat([q2_rand - rand_log_prob_t, q2_pi - pi_log_probs], dim=1)

        # logsumexp over the 2n samples
        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1) - np.log(2 * n)  # (B, 1)
        logsumexp_q2 = torch.logsumexp(cat_q2, dim=1) - np.log(2 * n)

        # Q-values for dataset actions
        q1_data, q2_data = self.critic(states, actions)   # (B, 1)

        cql1 = (logsumexp_q1 - q1_data).mean()
        cql2 = (logsumexp_q2 - q2_data).mean()

        return cql1, cql2

    def train_step(self, replay_buffer, batch_size: int = 256) -> dict:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)

       
        # Critic update  (Bellman error + CQL penalty)
       
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1.0 - dones) * self.discount * q_next

        q1, q2 = self.critic(states, actions)
        bellman_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        cql1, cql2 = self._cql_loss(states, actions)

        if self.cql_lagrange:
            cql_alpha = self.log_cql_alpha.exp().clamp(0, 1e6)
        else:
            cql_alpha = self.cql_alpha

        critic_loss = bellman_loss + cql_alpha * (cql1 + cql2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Lagrange update for CQL α
        if self.cql_lagrange:
            cql_alpha_loss = -self.log_cql_alpha.exp() * (
                ((cql1 + cql2) / 2.0).detach() - self.cql_target_action_gap
            )
            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss.backward()
            self.cql_alpha_optimizer.step()

       
        # Actor update  (SAC policy gradient)
       
        pi_actions, log_probs = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

       
        # Temperature update  (entropy auto-tuning)
       
        temp_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.temp_optimizer.step()

        # Soft-update target critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "critic_loss":  bellman_loss.item(),
            "cql_loss":    (cql1 + cql2).item() / 2.0,
            "actor_loss":   actor_loss.item(),
            "temperature":  self.alpha.item(),
        }

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actor.sample(state.to(self.device))
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.log_alpha = ckpt["log_alpha"]
        self.critic_target = copy.deepcopy(self.critic)
