from __future__ import annotations

import os
from dataclasses import dataclass
from os import PathLike
from typing import Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base import ActionResult

Pathish = Union[str, PathLike[str]]
TensorBatch = dict[str, torch.Tensor]


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def distribution(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(states))

    def add_action_bias(self, action_index: int, bias: float) -> None:
        output_layer = self.net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("PolicyNetwork output layer must be linear")
        with torch.no_grad():
            output_layer.bias[action_index] += bias


class CentralizedValueNetwork(nn.Module):
    def __init__(self, global_state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        return self.net(global_states).squeeze(-1)


@dataclass
class MAPPOTransition:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_global_obs: torch.Tensor
    done: bool
    log_prob: torch.Tensor


class MAPPORolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[MAPPOTransition] = []

    def add(self, transition: MAPPOTransition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


class MAPPO:
    algo_name = "mappo"

    def __init__(
        self,
        num_firms: int,
        firm_ids: list[int],
        state_size: int,
        action_size: int,
        max_order: int = 20,
        hidden_size: int = 128,
        critic_hidden_size: int = 256,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        rollout_steps: int = 1000,
        minibatch_size: int = 256,
        ppo_epochs: int = 6,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        normalize_advantages: bool = True,
        initial_order: int | None = None,
        initial_order_bias: float = 0.0,
        device: torch.device | str = "cpu",
        **_: Any,
    ) -> None:
        self.num_firms = num_firms
        self.firm_ids = [int(firm_id) for firm_id in firm_ids]
        self.state_size = state_size
        self.global_state_size = num_firms * state_size
        self.action_size = action_size
        self.max_order = max_order
        self.gamma = gamma
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.device = torch.device(device)

        self.actors = nn.ModuleDict(
            {
                str(firm_id): PolicyNetwork(state_size, action_size, hidden_size)
                for firm_id in self.firm_ids
            }
        ).to(self.device)
        self.critics = nn.ModuleDict(
            {
                str(firm_id): CentralizedValueNetwork(
                    self.global_state_size, critic_hidden_size
                )
                for firm_id in self.firm_ids
            }
        ).to(self.device)

        self.actor_optimizers = {
            firm_id: optim.Adam(self.actors[str(firm_id)].parameters(), lr=actor_lr)
            for firm_id in self.firm_ids
        }
        self.critic_optimizers = {
            firm_id: optim.Adam(self.critics[str(firm_id)].parameters(), lr=critic_lr)
            for firm_id in self.firm_ids
        }
        self.rollouts = {firm_id: MAPPORolloutBuffer() for firm_id in self.firm_ids}

        if initial_order is not None and initial_order_bias != 0.0:
            action_index = int(initial_order) - 1
            if action_index < 0 or action_index >= action_size:
                raise ValueError("initial_order must be in the action space")
            for actor in self.actors.values():
                actor.add_action_bias(action_index, float(initial_order_bias))

    def has_firm(self, firm_id: int) -> bool:
        return int(firm_id) in self.rollouts

    def act(self, firm_id: int, obs: Any, mode: str = "train") -> ActionResult:
        actor = self.actors[str(int(firm_id))]
        obs = (
            torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            .flatten()
            .unsqueeze(0)
        )

        with torch.no_grad():
            distribution = actor.distribution(obs)
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
        )

    def observe(
        self,
        firm_id: int,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_global_obs: torch.Tensor,
        done: bool,
        log_prob: torch.Tensor,
    ) -> None:
        firm_id = int(firm_id)
        self.rollouts[firm_id].add(
            MAPPOTransition(
                local_obs=local_obs.detach().clone(),
                global_obs=global_obs.detach().clone(),
                action=action.detach().clone(),
                reward=reward.detach().clone(),
                next_global_obs=next_global_obs.detach().clone(),
                done=done,
                log_prob=log_prob.detach().clone(),
            )
        )

    def ready_to_update(self) -> bool:
        return all(len(buffer) >= self.rollout_steps for buffer in self.rollouts.values())

    def has_rollout_data(self) -> bool:
        return any(len(buffer) > 0 for buffer in self.rollouts.values())

    def update(self, force: bool = False) -> dict[str, float]:
        if not force and not self.ready_to_update():
            return {}

        metrics_by_firm: dict[int, dict[str, float]] = {}
        for firm_id in self.firm_ids:
            if len(self.rollouts[firm_id]) == 0:
                continue
            batch = self._build_batch(firm_id)
            metrics_by_firm[firm_id] = self._update_firm(firm_id, batch)
            self.rollouts[firm_id].clear()

        return self._average_metrics(metrics_by_firm)

    def _build_batch(self, firm_id: int) -> TensorBatch:
        transitions = self.rollouts[firm_id].transitions
        local_obs = torch.stack(
            [
                torch.as_tensor(t.local_obs, dtype=torch.float32, device=self.device)
                .flatten()
                for t in transitions
            ]
        )
        global_obs = torch.stack(
            [
                torch.as_tensor(t.global_obs, dtype=torch.float32, device=self.device)
                .flatten()
                for t in transitions
            ]
        )
        next_global_obs = torch.stack(
            [
                torch.as_tensor(
                    t.next_global_obs, dtype=torch.float32, device=self.device
                ).flatten()
                for t in transitions
            ]
        )
        actions = torch.stack(
            [
                torch.as_tensor(t.action, dtype=torch.long, device=self.device).reshape(
                    ()
                )
                for t in transitions
            ]
        )
        rewards = torch.stack(
            [
                torch.as_tensor(t.reward, dtype=torch.float32, device=self.device).reshape(
                    ()
                )
                for t in transitions
            ]
        )
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.stack(
            [
                torch.as_tensor(t.log_prob, dtype=torch.float32, device=self.device).reshape(
                    ()
                )
                for t in transitions
            ]
        )

        critic = self.critics[str(firm_id)]
        with torch.no_grad():
            values = critic(global_obs)
            next_values = critic(next_global_obs)

        advantages = torch.zeros_like(rewards, device=self.device)
        gae = torch.tensor(0.0, device=self.device)
        for step in reversed(range(len(transitions))):
            non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + self.gamma * next_values[step] * non_terminal
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[step] = gae

        returns = advantages + values
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "local_obs": local_obs,
            "global_obs": global_obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages.detach(),
            "returns": returns.detach(),
        }

    def _update_firm(self, firm_id: int, batch: TensorBatch) -> dict[str, float]:
        actor = self.actors[str(firm_id)]
        critic = self.critics[str(firm_id)]
        actor_optimizer = self.actor_optimizers[firm_id]
        critic_optimizer = self.critic_optimizers[firm_id]

        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        batch_size = batch["local_obs"].shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                metrics = self._update_minibatch(
                    actor,
                    critic,
                    actor_optimizer,
                    critic_optimizer,
                    batch,
                    mb_idx,
                )
                losses.append(metrics["loss"])
                policy_losses.append(metrics["policy_loss"])
                value_losses.append(metrics["value_loss"])
                entropies.append(metrics["entropy"])

        return {
            "loss": self._mean(losses),
            "policy_loss": self._mean(policy_losses),
            "value_loss": self._mean(value_losses),
            "entropy": self._mean(entropies),
        }

    def _update_minibatch(
        self,
        actor: PolicyNetwork,
        critic: CentralizedValueNetwork,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        batch: TensorBatch,
        indices: torch.Tensor,
    ) -> dict[str, float]:
        local_obs = batch["local_obs"][indices]
        global_obs = batch["global_obs"][indices]
        actions = batch["actions"][indices]
        old_log_probs = batch["old_log_probs"][indices]
        advantages = batch["advantages"][indices]
        returns = batch["returns"][indices]

        distribution = actor.distribution(local_obs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        ratio = torch.exp(log_probs - old_log_probs)

        unclipped_policy_loss = -advantages * ratio
        clipped_policy_loss = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
        )
        policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()
        value_loss = nn.functional.mse_loss(critic(global_obs), returns)
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
        actor_optimizer.step()
        critic_optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def save(self, filename: Pathish) -> None:
        filename = os.fspath(filename)
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "firm_ids": self.firm_ids,
                "actors_state_dict": self.actors.state_dict(),
                "critics_state_dict": self.critics.state_dict(),
                "actor_optimizer_state_dicts": {
                    firm_id: optimizer.state_dict()
                    for firm_id, optimizer in self.actor_optimizers.items()
                },
                "critic_optimizer_state_dicts": {
                    firm_id: optimizer.state_dict()
                    for firm_id, optimizer in self.critic_optimizers.items()
                },
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load(self, filename: Pathish) -> bool:
        filename = os.fspath(filename)
        if not os.path.isfile(filename):
            return False
        checkpoint = torch.load(filename, weights_only=True, map_location=self.device)
        self.actors.load_state_dict(checkpoint["actors_state_dict"])
        self.critics.load_state_dict(checkpoint["critics_state_dict"])
        for firm_id, state_dict in checkpoint["actor_optimizer_state_dicts"].items():
            self.actor_optimizers[int(firm_id)].load_state_dict(state_dict)
        for firm_id, state_dict in checkpoint["critic_optimizer_state_dicts"].items():
            self.critic_optimizers[int(firm_id)].load_state_dict(state_dict)
        print(f"Loaded model from {filename}")
        return True

    def _average_metrics(
        self, metrics_by_firm: dict[int, dict[str, float]]
    ) -> dict[str, float]:
        metric_values: dict[str, list[float]] = {}
        for metrics in metrics_by_firm.values():
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(value)
        return {name: self._mean(values) for name, values in metric_values.items()}

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(torch.tensor(values, dtype=torch.float32).mean().item())
