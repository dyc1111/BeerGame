from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import random
from collections import deque
import os
from typing import Any

from .base import ActionAdapter, ActionResult, BaseAgent, Pathish, Transition

Experience = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        """
        Initialize the Q-network.

        :param state_size: State-space dimension
        :param action_size: Action-space dimension
        :param hidden_size: hidden dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param state: Input state
        :return: Q-values for each action
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        """
        Initialize a dueling Q-network with separate value and advantage streams.

        :param state_size: State-space dimension
        :param action_size: Action-space dimension
        :param hidden_size: Hidden dimension
        """
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param state: Input state
        :return: Q-values for each action
        """
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + advantages - advantages.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        """
        Initialize the experience replay buffer.

        :param capacity: Buffer capacity
        """
        self.buffer: deque[Experience] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        """
        Add an experience to the buffer.

        :param transition: Transition object
        """
        self.buffer.append(
            (
                transition.obs.detach().clone(),
                transition.action.detach().clone(),
                transition.reward.detach().clone(),
                transition.next_obs.detach().clone(),
                transition.done,
            )
        )

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Batch size
        :return: A batch of experiences (state, action, reward, next_state, done)
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent):
    algo_name = "dqn"

    def __init__(
        self,
        state_size: int,
        action_size: int,
        firm_id: int,
        max_order: int = 20,
        buffer_size: int = 10000,
        batch_size: int = 64,
        hidden_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        tau: float = 1e-3,
        update_every: int = 4,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: float = 0.995,
        action_type: str = "discrete",
    ) -> None:
        """
        Initialize the DQN agent.

        :param state_size: State-space dimension
        :param action_size: Action-space dimension
        :param firm_id: Firm ID indicating which firm is trained
        :param max_order: Maximum order quantity for the discrete action space
        :param buffer_size: Replay buffer size
        :param batch_size: Batch size
        :param hidden_size: Hidden dimension for QNet
        :param gamma: Discount factor
        :param learning_rate: Learning rate
        :param tau: Soft-update parameter
        :param update_every: Target network update frequency
        """
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.firm_id: int = firm_id
        self.max_order: int = max_order
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.tau: float = tau
        self.update_every: int = update_every
        self.learning_step: int = 0
        self.epsilon: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay
        self.action_adapter = ActionAdapter(action_type, max_order)

        self.q_network = self._build_network(state_size, action_size, hidden_size)
        self.target_network = self._build_network(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size)
        self.t_step: int = 0
        self.pending_updates: int = 0

    def _build_network(
        self, state_size: int, action_size: int, hidden_size: int
    ) -> nn.Module:
        return QNetwork(state_size, action_size, hidden_size)

    def observe(self, transition: Transition) -> None:
        """
        Add an experience to the replay buffer and schedule learning when ready.

        :param transition: Transition object
        """
        self.memory.add(transition)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.pending_updates += 1

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Backward-compatible wrapper for the old DQN-specific training loop.
        """
        self.observe(Transition(state, action, reward, next_state, done))
        if self.ready_to_update():
            self.update()

    def act(self, state: Any, mode: str = "train") -> ActionResult:
        """
        Choose an action from the current state.

        :param state: Current state
        :param mode: "train" enables epsilon-greedy exploration, "eval" is greedy
        :return: Selected action
        """
        state = torch.as_tensor(state, dtype=torch.float32).flatten()

        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state.unsqueeze(0))
        self.q_network.train()

        epsilon = self.epsilon if mode == "train" else 0.0
        if random.random() > epsilon:
            raw_action = torch.argmax(action_values).reshape(())
        else:
            raw_action = torch.randint(0, self.max_order, ()).reshape(())

        env_action = self.action_adapter.to_env_action(raw_action)
        return ActionResult(env_action=env_action, raw_action=raw_action)

    def ready_to_update(self) -> bool:
        return self.pending_updates > 0

    def update(self) -> dict[str, float]:
        if not self.ready_to_update():
            return {}
        self.pending_updates -= 1
        experiences = self.memory.sample(self.batch_size)
        loss = self.learn(experiences)
        return {"loss": loss, "epsilon": self.epsilon}

    def on_episode_end(self) -> None:
        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

    def learn(self, experiences: list[Experience]) -> float:
        """
        Learn from a batch of experiences.

        :param experiences: Tuple of (state, action, reward, next_state, done)
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.stack(states).float()
        actions = torch.stack(actions).long().unsqueeze(1) - 1
        rewards = torch.stack(rewards).float().unsqueeze(1)
        next_states = torch.stack(next_states).float()
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        Q_targets = self._compute_q_targets(rewards, next_states, dones)
        Q_expected = torch.gather(self.q_network(states), 1, actions)
        loss = self.loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learning_step += 1
        if self.learning_step % self.update_every == 0:
            old_param = parameters_to_vector(self.target_network.parameters())
            local_param = parameters_to_vector(self.q_network.parameters())
            new_param = self.tau * local_param + (1.0 - self.tau) * old_param
            vector_to_parameters(new_param, self.target_network.parameters())

        return loss.item()

    def _compute_q_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        Q_targets_next = torch.max(self.target_network(next_states), 1)[0].unsqueeze(1)
        return rewards + (self.gamma * Q_targets_next * (1 - dones))

    def save(self, filename: Pathish) -> None:
        """
        Save model parameters.

        :param filename: File name
        """
        filename = os.fspath(filename)
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )
        print(f"Model saved to {filename}")

    def load(self, filename: Pathish) -> bool:
        """
        Load model parameters.

        :param filename: File name
        """
        filename = os.fspath(filename)
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, weights_only=True)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded model from {filename}")
            return True
        return False


class DoubleDQNAgent(DQNAgent):
    algo_name = "double_dqn"

    def _compute_q_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        best_actions = torch.argmax(self.q_network(next_states), dim=1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, best_actions)
        return rewards + (self.gamma * next_q_values * (1 - dones))


class DuelingDQNAgent(DQNAgent):
    algo_name = "dueling_dqn"

    def _build_network(
        self, state_size: int, action_size: int, hidden_size: int
    ) -> nn.Module:
        return DuelingQNetwork(state_size, action_size, hidden_size)
