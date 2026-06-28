from __future__ import annotations

import os
import random
from os import PathLike
from typing import Any, Union

import torch
import torch.nn as nn
import torch.optim as optim

from .base import ActionResult
from .mappo import (
    CentralizedValueNetwork,
    MAPPOTransition,
    MAPPORolloutBuffer,
    PolicyNetwork,
)

Pathish = Union[str, PathLike[str]]
TensorBatch = dict[str, torch.Tensor]


class HAPPO:
    algo_name = "happo"

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
        individual_reward_weight: float = 0.5,
        randomize_update_order: bool = True,
        use_happo_factor: bool = True,
        initial_orders: Any | None = None,
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
        self.individual_reward_weight = float(individual_reward_weight)
        self.randomize_update_order = bool(randomize_update_order)
        self.use_happo_factor = bool(use_happo_factor)
        self.initial_order_bias = float(initial_order_bias)
        self.device = torch.device(device)
        self.firm_feature_size = 1

        self.actors = nn.ModuleDict(
            {
                str(firm_id): PolicyNetwork(state_size, action_size, hidden_size)
                for firm_id in self.firm_ids
            }
        ).to(self.device)
        self.critic = CentralizedValueNetwork(
            self.global_state_size + self.firm_feature_size, critic_hidden_size
        ).to(self.device)

        self.actor_optimizers = {
            firm_id: optim.Adam(self.actors[str(firm_id)].parameters(), lr=actor_lr)
            for firm_id in self.firm_ids
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.rollouts = {firm_id: MAPPORolloutBuffer() for firm_id in self.firm_ids}

        if initial_order_bias != 0.0:
            self._apply_initial_order_bias(initial_order, initial_orders)

    def has_firm(self, firm_id: int) -> bool:
        return int(firm_id) in self.rollouts

    def _apply_initial_order_bias(
        self, initial_order: int | None, initial_orders: Any | None
    ) -> None:
        if initial_orders is None:
            if initial_order is None:
                return
            orders_by_firm = {firm_id: int(initial_order) for firm_id in self.firm_ids}
        elif isinstance(initial_orders, dict):
            orders_by_firm = {
                int(firm_id): int(order) for firm_id, order in initial_orders.items()
            }
        else:
            orders = [int(order) for order in initial_orders]
            if len(orders) == self.num_firms:
                orders_by_firm = {firm_id: orders[firm_id] for firm_id in self.firm_ids}
            elif len(orders) == len(self.firm_ids):
                orders_by_firm = dict(zip(self.firm_ids, orders))
            else:
                raise ValueError(
                    "initial_orders must have length num_firms or len(firm_ids)"
                )

        for firm_id in self.firm_ids:
            order = orders_by_firm.get(firm_id)
            if order is None:
                continue
            action_index = int(order) - 1
            if action_index < 0 or action_index >= self.action_size:
                raise ValueError("initial_order values must be in the action space")
            self.actors[str(firm_id)].add_action_bias(
                action_index, float(self.initial_order_bias)
            )

    def act(self, firm_id: int, obs: Any, mode: str = "train") -> ActionResult:
        firm_id = int(firm_id)
        actor = self.actors[str(firm_id)]
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
        return all(
            len(buffer) >= self.rollout_steps for buffer in self.rollouts.values()
        )

    def has_rollout_data(self) -> bool:
        return any(len(buffer) > 0 for buffer in self.rollouts.values())

    def update(self, force: bool = False) -> dict[str, float]:
        if not force and not self.ready_to_update():
            return {}

        batches: dict[int, TensorBatch] = {}
        for firm_id in self.firm_ids:
            if len(self.rollouts[firm_id]) == 0:
                continue
            batches[firm_id] = self._build_batch(firm_id)
            self.rollouts[firm_id].clear()

        if not batches:
            return {}

        critic_metrics = self._update_critic(self._merge_batches(list(batches.values())))
        actor_metrics = self._update_actors_sequential(batches)
        return {**critic_metrics, **actor_metrics}

    def _firm_feature_value(self, firm_id: int) -> float:
        if self.num_firms <= 1:
            return 0.0
        return float(firm_id) / float(self.num_firms - 1)

    def _firm_features(self, firm_id: int, batch_size: int) -> torch.Tensor:
        value = self._firm_feature_value(firm_id)
        return torch.full(
            (batch_size, self.firm_feature_size),
            value,
            dtype=torch.float32,
            device=self.device,
        )

    def _append_firm_feature(self, inputs: torch.Tensor, firm_id: int) -> torch.Tensor:
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        return torch.cat([inputs, self._firm_features(firm_id, inputs.shape[0])], dim=1)

    def _build_batch(self, firm_id: int) -> TensorBatch:
        transitions = self.rollouts[firm_id].transitions
        local_obs = torch.stack(
            [
                torch.as_tensor(
                    t.local_obs, dtype=torch.float32, device=self.device
                ).flatten()
                for t in transitions
            ]
        )
        global_obs = torch.stack(
            [
                torch.as_tensor(
                    t.global_obs, dtype=torch.float32, device=self.device
                ).flatten()
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

        critic_obs = self._append_firm_feature(global_obs, firm_id)
        next_critic_obs = self._append_firm_feature(next_global_obs, firm_id)
        with torch.no_grad():
            values = self.critic(critic_obs)
            next_values = self.critic(next_critic_obs)

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
            "global_obs": critic_obs,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages.detach(),
            "returns": returns.detach(),
        }

    def _merge_batches(self, batches: list[TensorBatch]) -> TensorBatch:
        keys = batches[0].keys()
        return {
            key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys
        }

    def _update_critic(self, batch: TensorBatch) -> dict[str, float]:
        value_losses: list[float] = []
        batch_size = batch["global_obs"].shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                global_obs = batch["global_obs"][mb_idx]
                returns = batch["returns"][mb_idx]
                value_loss = nn.functional.mse_loss(self.critic(global_obs), returns)
                loss = self.value_coef * value_loss

                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                value_losses.append(value_loss.item())

        return {"value_loss": self._mean(value_losses)}

    def _update_actors_sequential(
        self, batches: dict[int, TensorBatch]
    ) -> dict[str, float]:
        policy_losses: list[float] = []
        entropies: list[float] = []
        firm_ids = list(batches)

        for _ in range(self.ppo_epochs):
            update_order = firm_ids.copy()
            if self.randomize_update_order:
                random.shuffle(update_order)

            first_batch = batches[update_order[0]]
            happo_factor = torch.ones(
                first_batch["actions"].shape[0],
                dtype=torch.float32,
                device=self.device,
            )

            for firm_id in update_order:
                batch = batches[firm_id]
                if batch["actions"].shape[0] != happo_factor.shape[0]:
                    raise RuntimeError(
                        "HAPPO requires aligned rollout lengths across trained firms"
                    )
                metrics = self._update_actor_for_firm(
                    firm_id,
                    batch,
                    happo_factor.detach(),
                )
                policy_losses.extend(metrics["policy_losses"])
                entropies.extend(metrics["entropies"])

                if self.use_happo_factor:
                    happo_factor = happo_factor * self._full_batch_ratio(
                        firm_id, batch
                    ).detach()

        return {
            "policy_loss": self._mean(policy_losses),
            "entropy": self._mean(entropies),
        }

    def _update_actor_for_firm(
        self,
        firm_id: int,
        batch: TensorBatch,
        happo_factor: torch.Tensor,
    ) -> dict[str, list[float]]:
        actor = self.actors[str(firm_id)]
        optimizer = self.actor_optimizers[firm_id]
        policy_losses: list[float] = []
        entropies: list[float] = []
        batch_size = batch["local_obs"].shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)

        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            local_obs = batch["local_obs"][mb_idx]
            actions = batch["actions"][mb_idx]
            old_log_probs = batch["old_log_probs"][mb_idx]
            advantages = batch["advantages"][mb_idx]
            factor = happo_factor[mb_idx]

            distribution = actor.distribution(local_obs)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()
            ratio = torch.exp(log_probs - old_log_probs)

            unclipped_policy_loss = -factor * advantages * ratio
            clipped_policy_loss = -factor * advantages * torch.clamp(
                ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
            )
            policy_loss = torch.max(
                unclipped_policy_loss, clipped_policy_loss
            ).mean()
            loss = policy_loss - self.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            entropies.append(entropy.item())

        return {"policy_losses": policy_losses, "entropies": entropies}

    def _full_batch_ratio(self, firm_id: int, batch: TensorBatch) -> torch.Tensor:
        actor = self.actors[str(firm_id)]
        with torch.no_grad():
            distribution = actor.distribution(batch["local_obs"])
            log_probs = distribution.log_prob(batch["actions"])
            return torch.exp(log_probs - batch["old_log_probs"])

    def save(self, filename: Pathish) -> None:
        filename = os.fspath(filename)
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "firm_ids": self.firm_ids,
                "actors_state_dict": self.actors.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dicts": {
                    firm_id: optimizer.state_dict()
                    for firm_id, optimizer in self.actor_optimizers.items()
                },
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
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
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        for firm_id, state_dict in checkpoint["actor_optimizer_state_dicts"].items():
            self.actor_optimizers[int(firm_id)].load_state_dict(state_dict)
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Loaded model from {filename}")
        return True

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(torch.tensor(values, dtype=torch.float32).mean().item())
