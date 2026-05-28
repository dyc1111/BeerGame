from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from .base import ActionAdapter, ActionResult, BaseAgent, Pathish, Transition

TensorBatch = dict[str, torch.Tensor]


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int,
        action_type: str,
    ) -> None:
        super().__init__()
        self.action_type = action_type
        self.body = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        if action_type == "discrete":
            self.output = nn.Linear(hidden_size, action_size)
            self.log_std = None
        else:
            self.output = nn.Linear(hidden_size, 1)
            self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(states))

    def distribution(self, states: torch.Tensor) -> Any:
        output = self.forward(states)
        if self.action_type == "discrete":
            return Categorical(logits=output)

        if self.log_std is None:
            raise RuntimeError("Continuous policy is missing log_std")
        std = self.log_std.exp().expand_as(output)
        return Normal(output, std)


class ValueNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


class TRPOAgent(BaseAgent):
    algo_name = "trpo"

    def __init__(
        self,
        state_size: int,
        action_size: int,
        firm_id: int,
        max_order: int = 20,
        hidden_size: int = 64,
        gamma: float = 0.99,
        critic_lr: float = 1e-3,
        action_type: str = "discrete",
        rollout_steps: int = 100,
        gae_lambda: float = 0.95,
        max_kl: float = 1e-2,
        damping: float = 1e-2,
        cg_iters: int = 10,
        line_search_iters: int = 10,
        line_search_decay: float = 0.5,
        value_epochs: int = 10,
        normalize_advantages: bool = True,
        **_: Any,
    ) -> None:
        if action_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown action type: {action_type}")

        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.gamma = gamma
        self.action_type = action_type
        self.rollout_steps = rollout_steps
        self.gae_lambda = gae_lambda
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.line_search_iters = line_search_iters
        self.line_search_decay = line_search_decay
        self.value_epochs = value_epochs
        self.normalize_advantages = normalize_advantages

        self.action_adapter = ActionAdapter(action_type, max_order)
        self.actor = PolicyNetwork(state_size, action_size, hidden_size, action_type)
        self.critic = ValueNetwork(state_size, hidden_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_loss = nn.MSELoss()
        self.rollout = RolloutBuffer()
        self.pending_update = False

    def act(self, obs: Any, mode: str = "train") -> ActionResult:
        obs = torch.as_tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)

        with torch.no_grad():
            distribution = self.actor.distribution(obs)
            value = self.critic(obs)
            if mode == "train":
                raw_action = distribution.sample()
            elif self.action_type == "discrete":
                raw_action = torch.argmax(distribution.logits, dim=-1)
            else:
                raw_action = distribution.mean

            log_prob = distribution.log_prob(raw_action)
            if self.action_type == "continuous":
                raw_action = raw_action.squeeze(-1)
                log_prob = log_prob.squeeze(-1)

        raw_action = raw_action.reshape(())
        env_action = self.action_adapter.to_env_action(raw_action)
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
        critic_loss = self._update_critic(batch)
        policy_loss, kl, accepted_step = self._update_actor(batch)

        self.rollout.clear()
        self.pending_update = False

        return {
            "policy_loss": policy_loss,
            "value_loss": critic_loss,
            "kl": kl,
            "accepted_step": float(accepted_step),
        }

    def on_episode_end(self) -> None:
        if len(self.rollout) > 0:
            self.pending_update = True

    def _build_batch(self) -> TensorBatch:
        transitions = self.rollout.transitions
        states = torch.stack(
            [torch.as_tensor(t.obs, dtype=torch.float32).flatten() for t in transitions]
        )
        next_states = torch.stack(
            [
                torch.as_tensor(t.next_obs, dtype=torch.float32).flatten()
                for t in transitions
            ]
        )
        rewards = torch.stack(
            [
                torch.as_tensor(t.reward, dtype=torch.float32).reshape(())
                for t in transitions
            ]
        )
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        old_log_probs = torch.stack(
            [
                torch.as_tensor(t.log_prob, dtype=torch.float32).reshape(())
                for t in transitions
            ]
        )
        actions = torch.stack([self._training_action(t) for t in transitions])

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)
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
            action = torch.as_tensor(transition.raw_action)
        elif self.action_type == "discrete":
            action = torch.as_tensor(transition.action).long() - 1
        else:
            action = torch.as_tensor(transition.action).float()

        if self.action_type == "discrete":
            return action.long().reshape(())
        return action.float().reshape(1)

    def _update_critic(self, batch: TensorBatch) -> float:
        states = batch["states"]
        returns = batch["returns"]
        loss = torch.tensor(0.0)

        for _ in range(self.value_epochs):
            values = self.critic(states)
            loss = self.value_loss(values, returns)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

        return loss.item()

    def _update_actor(self, batch: TensorBatch) -> tuple[float, float, bool]:
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]

        with torch.no_grad():
            old_policy_output = self.actor(states).detach()
            old_log_std = (
                self.actor.log_std.detach().clone()
                if self.action_type == "continuous" and self.actor.log_std is not None
                else None
            )
            old_surrogate = self._surrogate_loss(
                states, actions, old_log_probs, advantages
            ).detach()

        surrogate = self._surrogate_loss(states, actions, old_log_probs, advantages)
        policy_grads = torch.autograd.grad(surrogate, self.actor.parameters())
        flat_grads = torch.cat([grad.reshape(-1) for grad in policy_grads]).detach()

        if torch.norm(flat_grads) <= 1e-12:
            return -old_surrogate.item(), 0.0, False

        step_direction = self._conjugate_gradient(
            lambda vector: self._hessian_vector_product(
                states, old_policy_output, old_log_std, vector
            ),
            flat_grads,
        )
        hvp_step = self._hessian_vector_product(
            states, old_policy_output, old_log_std, step_direction
        )
        curvature = torch.dot(step_direction, hvp_step)
        if curvature <= 0:
            return -old_surrogate.item(), 0.0, False

        scale = torch.sqrt(2.0 * self.max_kl / (curvature + 1e-8))
        full_step = scale * step_direction
        accepted, final_surrogate, final_kl = self._line_search(
            states,
            actions,
            old_log_probs,
            advantages,
            old_policy_output,
            old_log_std,
            old_surrogate,
            full_step,
        )
        return -final_surrogate, final_kl, accepted

    def _surrogate_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self.actor.distribution(states)
        log_probs = distribution.log_prob(actions)
        if self.action_type == "continuous":
            log_probs = log_probs.squeeze(-1)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantages)

    def _mean_kl(
        self,
        states: torch.Tensor,
        old_policy_output: torch.Tensor,
        old_log_std: torch.Tensor | None,
    ) -> torch.Tensor:
        new_policy_output = self.actor(states)
        if self.action_type == "discrete":
            old_dist = Categorical(logits=old_policy_output)
            new_dist = Categorical(logits=new_policy_output)
            return kl_divergence(old_dist, new_dist).mean()

        if old_log_std is None or self.actor.log_std is None:
            raise RuntimeError("Continuous TRPO KL requires log_std tensors")
        old_std = old_log_std.exp().expand_as(old_policy_output)
        new_std = self.actor.log_std.exp().expand_as(new_policy_output)
        old_dist = Normal(old_policy_output, old_std)
        new_dist = Normal(new_policy_output, new_std)
        return kl_divergence(old_dist, new_dist).sum(dim=-1).mean()

    def _hessian_vector_product(
        self,
        states: torch.Tensor,
        old_policy_output: torch.Tensor,
        old_log_std: torch.Tensor | None,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        kl = self._mean_kl(states, old_policy_output, old_log_std)
        kl_grads = torch.autograd.grad(
            kl, self.actor.parameters(), create_graph=True, retain_graph=True
        )
        flat_kl_grads = torch.cat([grad.reshape(-1) for grad in kl_grads])
        grad_vector_product = torch.dot(flat_kl_grads, vector)
        hvp = torch.autograd.grad(grad_vector_product, self.actor.parameters())
        flat_hvp = torch.cat([grad.reshape(-1) for grad in hvp]).detach()
        return flat_hvp + self.damping * vector

    def _conjugate_gradient(
        self,
        matvec: Any,
        b: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rs_old = torch.dot(r, r)

        for _ in range(self.cg_iters):
            ap = matvec(p)
            alpha = rs_old / (torch.dot(p, ap) + 1e-8)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < 1e-10:
                break
            p = r + (rs_new / (rs_old + 1e-8)) * p
            rs_old = rs_new
        return x

    def _line_search(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_policy_output: torch.Tensor,
        old_log_std: torch.Tensor | None,
        old_surrogate: torch.Tensor,
        full_step: torch.Tensor,
    ) -> tuple[bool, float, float]:
        old_params = parameters_to_vector(self.actor.parameters()).detach()

        for step_idx in range(self.line_search_iters):
            step_frac = self.line_search_decay**step_idx
            new_params = old_params + step_frac * full_step
            vector_to_parameters(new_params, self.actor.parameters())

            with torch.no_grad():
                surrogate = self._surrogate_loss(
                    states, actions, old_log_probs, advantages
                )
                kl = self._mean_kl(states, old_policy_output, old_log_std)

            if torch.isfinite(surrogate) and torch.isfinite(kl):
                if surrogate > old_surrogate and kl <= self.max_kl:
                    return True, surrogate.item(), kl.item()

        vector_to_parameters(old_params, self.actor.parameters())
        return False, old_surrogate.item(), 0.0

    def save(self, filename: Pathish) -> None:
        filename = os.fspath(filename)
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load(self, filename: Pathish) -> bool:
        filename = os.fspath(filename)
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, weights_only=True)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
            print(f"Loaded model from {filename}")
            return True
        return False
