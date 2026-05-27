import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from .base import ActionAdapter, ActionResult, BaseAgent


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, action_type):
        super().__init__()
        self.action_type = action_type
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        if action_type == "discrete":
            self.policy_head = nn.Linear(hidden_size, action_size)
            self.log_std = None
        else:
            self.policy_head = nn.Linear(hidden_size, 1)
            self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        features = self.shared(states)
        policy_output = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_output, values

    def distribution(self, states):
        policy_output, values = self.forward(states)
        if self.action_type == "discrete":
            return Categorical(logits=policy_output), values

        std = self.log_std.exp().expand_as(policy_output)
        return Normal(policy_output, std), values


class RolloutBuffer:
    def __init__(self):
        self.transitions = []

    def add(self, transition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


class PPOAgent(BaseAgent):
    algo_name = "ppo"

    def __init__(
        self,
        state_size,
        action_size,
        firm_id,
        max_order=20,
        hidden_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        action_type="discrete",
        rollout_steps=100,
        minibatch_size=64,
        ppo_epochs=4,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        normalize_advantages=True,
        **_,
    ):
        if action_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown action type: {action_type}")

        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.gamma = gamma
        self.action_type = action_type
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.ppo_epochs = ppo_epochs
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages

        self.action_adapter = ActionAdapter(action_type, max_order)
        self.network = ActorCriticNetwork(
            state_size, action_size, hidden_size, action_type
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.rollout = RolloutBuffer()
        self.pending_update = False

    def act(self, obs, mode="train"):
        obs = torch.as_tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)

        with torch.no_grad():
            distribution, value = self.network.distribution(obs)
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

    def observe(self, transition):
        self.rollout.add(transition)
        if len(self.rollout) >= self.rollout_steps:
            self.pending_update = True

    def ready_to_update(self):
        return self.pending_update and len(self.rollout) > 0

    def update(self):
        if not self.ready_to_update():
            return {}

        batch = self._build_batch()
        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []

        batch_size = batch["states"].shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size)
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
            "loss": float(torch.tensor(losses).mean().item()),
            "policy_loss": float(torch.tensor(policy_losses).mean().item()),
            "value_loss": float(torch.tensor(value_losses).mean().item()),
            "entropy": float(torch.tensor(entropy_losses).mean().item()),
        }

    def on_episode_end(self):
        if len(self.rollout) > 0:
            self.pending_update = True

    def _build_batch(self):
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
            [torch.as_tensor(t.reward, dtype=torch.float32).reshape(()) for t in transitions]
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
            _, values = self.network.distribution(states)
            _, next_values = self.network.distribution(next_states)

        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0)
        for step in reversed(range(len(transitions))):
            next_value = next_values[step] if step == len(transitions) - 1 else values[step + 1]
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

    def _training_action(self, transition):
        if transition.raw_action is not None:
            action = torch.as_tensor(transition.raw_action)
        elif self.action_type == "discrete":
            action = torch.as_tensor(transition.action).long() - 1
        else:
            action = torch.as_tensor(transition.action).float()

        if self.action_type == "discrete":
            return action.long().reshape(())
        return action.float().reshape(1)

    def _update_minibatch(self, batch, indices):
        states = batch["states"][indices]
        actions = batch["actions"][indices]
        old_log_probs = batch["old_log_probs"][indices]
        advantages = batch["advantages"][indices]
        returns = batch["returns"][indices]

        distribution, values = self.network.distribution(states)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.action_type == "continuous":
            log_probs = log_probs.squeeze(-1)
            entropy = entropy.squeeze(-1)

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

    def save(self, filename):
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

    def load(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, weights_only=True)
            self.network.load_state_dict(checkpoint["network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded model from {filename}")
            return True
        return False
