from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base import ActionResult, BaseAgent, Pathish, Transition

TensorBatch = dict[str, torch.Tensor]


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, action_size)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        policy_output = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_output, values

    def distribution(self, states: torch.Tensor) -> tuple[Any, torch.Tensor]:
        policy_output, values = self.forward(states)
        return Categorical(logits=policy_output), values


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


class PPOAgent(BaseAgent):
    algo_name = "ppo"

    def __init__(
        self,
        state_size: int,
        action_size: int,
        firm_id: int,
        max_order: int = 20,
        hidden_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        rollout_steps: int = 100,
        minibatch_size: int = 64,
        ppo_epochs: int = 4,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        device: torch.device | str = "cpu",
        **_: Any,
    ) -> None:
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.firm_id: int = firm_id
        self.max_order: int = max_order
        self.gamma: float = gamma
        self.rollout_steps: int = rollout_steps
        self.minibatch_size: int = minibatch_size
        self.ppo_epochs: int = ppo_epochs
        self.clip_coef: float = clip_coef
        self.value_coef: float = value_coef
        self.entropy_coef: float = entropy_coef
        self.gae_lambda: float = gae_lambda
        self.max_grad_norm: float = max_grad_norm
        self.normalize_advantages: bool = normalize_advantages
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(state_size, action_size, hidden_size)
        self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.rollout = RolloutBuffer()
        self.pending_update: bool = False

    def act(self, obs: Any, mode: str = "train") -> ActionResult:
        obs = (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            .flatten()
            .unsqueeze(0)
        )

        with torch.no_grad():
            distribution, value = self.network.distribution(obs)
            if mode == "train":
                raw_action = distribution.sample()
            else:
                raw_action = torch.argmax(distribution.logits, dim=-1)

            log_prob = distribution.log_prob(raw_action)

        raw_action = raw_action.reshape(())
        env_action = (raw_action.long().clamp(0, self.max_order - 1) + 1).float()
        return ActionResult(
            env_action=env_action,
            raw_action=raw_action.detach().clone(),
            log_prob=log_prob.reshape(()).detach().clone(),
            value=value.reshape(()).detach().clone(),
        )

    def observe(self, transition: Transition) -> None:
        self.rollout.add(transition)
        if len(self.rollout) >= self.rollout_steps:
            self.pending_update = True

    def ready_to_update(self) -> bool:
        return self.pending_update and len(self.rollout) > 0

    def update(self) -> dict[str, float]:
        if not self.ready_to_update():
            return {}

        batch = self._build_batch()
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []

        batch_size = batch["states"].shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                metrics = self._update_minibatch(batch, mb_idx)
                losses.append(metrics["loss"])
                policy_losses.append(metrics["policy_loss"])
                value_losses.append(metrics["value_loss"])
                entropy_losses.append(metrics["entropy"])

        self.rollout.clear()
        self.pending_update = False

        return {
            "loss": float(torch.tensor(losses, device=self.device).mean().item()),
            "policy_loss": float(
                torch.tensor(policy_losses, device=self.device).mean().item()
            ),
            "value_loss": float(
                torch.tensor(value_losses, device=self.device).mean().item()
            ),
            "entropy": float(
                torch.tensor(entropy_losses, device=self.device).mean().item()
            ),
        }

    def on_episode_end(self) -> None:
        if len(self.rollout) > 0:
            self.pending_update = True

    def _build_batch(self) -> TensorBatch:
        transitions = self.rollout.transitions
        states = torch.stack(
            [
                torch.as_tensor(t.obs, dtype=torch.float32, device=self.device).flatten()
                for t in transitions
            ]
        )
        next_states = torch.stack(
            [
                torch.as_tensor(
                    t.next_obs, dtype=torch.float32, device=self.device
                ).flatten()
                for t in transitions
            ]
        )
        rewards = torch.stack(
            [
                torch.as_tensor(
                    t.reward, dtype=torch.float32, device=self.device
                ).reshape(())
                for t in transitions
            ]
        )
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.stack(
            [
                torch.as_tensor(
                    t.log_prob, dtype=torch.float32, device=self.device
                ).reshape(())
                for t in transitions
            ]
        )
        actions = torch.stack([self._training_action(t) for t in transitions])

        with torch.no_grad():
            _, values = self.network.distribution(states)
            _, next_values = self.network.distribution(next_states)

        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.tensor(0.0, device=self.device)
        for step in reversed(range(len(transitions))):
            next_value = (
                next_values[step]
                if step == len(transitions) - 1
                else values[step + 1]
            )
            non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_value * non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[step] = gae

        returns = advantages + values
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages.detach(),
            "returns": returns.detach(),
        }

    def _training_action(self, transition: Transition) -> torch.Tensor:
        if transition.raw_action is not None:
            action = torch.as_tensor(transition.raw_action, device=self.device)
        else:
            action = torch.as_tensor(transition.action, device=self.device).long() - 1

        return action.long().reshape(())

    def _update_minibatch(
        self, batch: TensorBatch, indices: torch.Tensor
    ) -> dict[str, float]:
        states = batch["states"][indices]
        actions = batch["actions"][indices]
        old_log_probs = batch["old_log_probs"][indices]
        advantages = batch["advantages"][indices]
        returns = batch["returns"][indices]

        distribution, values = self.network.distribution(states)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        ratio = torch.exp(log_probs - old_log_probs)
        unclipped_policy_loss = -advantages * ratio
        clipped_policy_loss = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
        )
        policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_bonus = entropy.mean()
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy_bonus
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
        }

    def save(self, filename: Pathish) -> None:
        filename = os.fspath(filename)
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load(self, filename: Pathish) -> bool:
        filename = os.fspath(filename)
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, weights_only=True, map_location=self.device)
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded model from {filename}")
            return True
        return False
