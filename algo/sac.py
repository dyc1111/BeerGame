from __future__ import annotations

import os
import random
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .base import ActionResult, BaseAgent, Pathish, Transition

Experience = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        action = transition.raw_action
        if action is None:
            action = torch.as_tensor(transition.action).long() - 1
        self.buffer.append(
            (
                transition.obs.detach().clone(),
                torch.as_tensor(action).long().reshape(()).detach().clone(),
                transition.reward.detach().clone(),
                transition.next_obs.detach().clone(),
                transition.done,
            )
        )

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def action_probs(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class DiscreteQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class SACAgent(BaseAgent):
    algo_name = "sac"

    def __init__(
        self,
        state_size: int,
        action_size: int,
        firm_id: int,
        max_order: int = 20,
        hidden_size: int = 64,
        buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        tau: float = 0.005,
        update_every: int = 1,
        gradient_steps: int = 1,
        min_replay_size: int = 1000,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: float | None = None,
        device: torch.device | str = "cpu",
        **_: Any,
    ) -> None:
        self.device = torch.device(device)
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.gradient_steps = gradient_steps
        self.min_replay_size = min_replay_size
        self.auto_entropy_tuning = auto_entropy_tuning
        self.alpha_lr = alpha_lr
        self.target_entropy = (
            0.98
            * torch.log(torch.tensor(float(action_size), device=self.device)).item()
            if target_entropy is None
            else target_entropy
        )

        self.actor = DiscretePolicyNetwork(state_size, action_size, hidden_size).to(
            self.device
        )
        self.critic_1 = DiscreteQNetwork(state_size, action_size, hidden_size).to(
            self.device
        )
        self.critic_2 = DiscreteQNetwork(state_size, action_size, hidden_size).to(
            self.device
        )
        self.target_critic_1 = DiscreteQNetwork(
            state_size, action_size, hidden_size
        ).to(self.device)
        self.target_critic_2 = DiscreteQNetwork(
            state_size, action_size, hidden_size
        ).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = (
            torch.tensor(float(alpha), device=self.device).log().detach().clone()
        )
        self.log_alpha.requires_grad_(True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0
        self.pending_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_entropy_tuning:
            return self.log_alpha.exp()
        return self.log_alpha.exp().detach()

    def act(self, state: Any, mode: str = "train") -> ActionResult:
        state = (
            torch.as_tensor(state, dtype=torch.float32, device=self.device)
            .flatten()
            .unsqueeze(0)
        )

        self.actor.eval()
        with torch.no_grad():
            probs, log_probs = self.actor.action_probs(state)
            if mode == "train":
                distribution = Categorical(probs=probs)
                raw_action = distribution.sample()
                log_prob = distribution.log_prob(raw_action)
            else:
                raw_action = torch.argmax(probs, dim=-1)
                log_prob = log_probs.gather(1, raw_action.unsqueeze(1)).squeeze(1)
        self.actor.train()

        raw_action = raw_action.reshape(())
        env_action = (raw_action.long().clamp(0, self.max_order - 1) + 1).float()
        return ActionResult(
            env_action=env_action,
            raw_action=raw_action.detach().clone(),
            log_prob=log_prob.reshape(()).detach().clone(),
        )

    def observe(self, transition: Transition) -> None:
        self.memory.add(transition)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.min_replay_size:
            self.pending_updates += 1

    def ready_to_update(self) -> bool:
        return self.pending_updates > 0

    def update(self) -> dict[str, float]:
        if not self.ready_to_update():
            return {}

        self.pending_updates -= 1
        metrics: dict[str, float] = {}
        for _ in range(self.gradient_steps):
            metrics = self._update_once()
        return metrics

    def _update_once(self) -> dict[str, float]:
        states, actions, rewards, next_states, dones = self._sample_batch()

        with torch.no_grad():
            next_probs, next_log_probs = self.actor.action_probs(next_states)
            target_q_1 = self.target_critic_1(next_states)
            target_q_2 = self.target_critic_2(next_states)
            min_target_q = torch.minimum(target_q_1, target_q_2)
            next_values = (
                next_probs * (min_target_q - self.alpha.detach() * next_log_probs)
            ).sum(dim=1, keepdim=True)
            q_targets = rewards + self.gamma * (1.0 - dones) * next_values

        q_1 = self.critic_1(states).gather(1, actions)
        q_2 = self.critic_2(states).gather(1, actions)
        critic_1_loss = F.mse_loss(q_1, q_targets)
        critic_2_loss = F.mse_loss(q_2, q_targets)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs, log_probs = self.actor.action_probs(states)
        min_q = torch.minimum(self.critic_1(states), self.critic_2(states))
        actor_loss = (
            (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_entropy_tuning:
            alpha_loss = self.log_alpha * (entropy - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy.item(),
        }

    def _sample_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return (
            torch.stack(states).float().to(self.device),
            torch.stack(actions).long().unsqueeze(1).to(self.device),
            torch.stack(rewards).float().unsqueeze(1).to(self.device),
            torch.stack(next_states).float().to(self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filename: Pathish) -> None:
        filename = os.fspath(filename)
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_1_state_dict": self.critic_1.state_dict(),
                "critic_2_state_dict": self.critic_2.state_dict(),
                "target_critic_1_state_dict": self.target_critic_1.state_dict(),
                "target_critic_2_state_dict": self.target_critic_2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
                "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load(self, filename: Pathish) -> bool:
        filename = os.fspath(filename)
        if os.path.isfile(filename):
            checkpoint = torch.load(
                filename, map_location=self.device, weights_only=True
            )
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
            self.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
            self.target_critic_1.load_state_dict(
                checkpoint["target_critic_1_state_dict"]
            )
            self.target_critic_2.load_state_dict(
                checkpoint["target_critic_2_state_dict"]
            )
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_1_optimizer.load_state_dict(
                checkpoint["critic_1_optimizer_state_dict"]
            )
            self.critic_2_optimizer.load_state_dict(
                checkpoint["critic_2_optimizer_state_dict"]
            )
            self.log_alpha = checkpoint["log_alpha"].to(self.device).detach().clone()
            self.log_alpha.requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer_state_dict"]
            )
            print(f"Loaded model from {filename}")
            return True
        return False
