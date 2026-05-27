import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import random
from collections import deque
import os

from .base import ActionAdapter, ActionResult, BaseAgent, Transition


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
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

    def forward(self, state):
        """
        Forward pass.

        :param state: Input state
        :return: Q-values for each action
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the experience replay buffer.

        :param capacity: Buffer capacity
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
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

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Batch size
        :return: A batch of experiences (state, action, reward, next_state, done)
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    algo_name = "dqn"

    def __init__(
        self,
        state_size,
        action_size,
        firm_id,
        max_order=20,
        buffer_size=10000,
        batch_size=64,
        hidden_size=64,
        gamma=0.99,
        learning_rate=1e-3,
        tau=1e-3,
        update_every=4,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        action_type="discrete",
    ):
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
        self.state_size = state_size
        self.action_size = action_size
        self.firm_id = firm_id
        self.max_order = max_order
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.learning_step = 0
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.action_adapter = ActionAdapter(action_type, max_order)

        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0
        self.pending_updates = 0

    def observe(self, transition):
        """
        Add an experience to the replay buffer and schedule learning when ready.

        :param transition: Transition object
        """
        self.memory.add(transition)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.pending_updates += 1

    def step(self, state, action, reward, next_state, done):
        """
        Backward-compatible wrapper for the old DQN-specific training loop.
        """
        self.observe(Transition(state, action, reward, next_state, done))
        if self.ready_to_update():
            self.update()

    def act(self, state, mode="train"):
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

    def ready_to_update(self):
        return self.pending_updates > 0

    def update(self):
        if not self.ready_to_update():
            return {}
        self.pending_updates -= 1
        experiences = self.memory.sample(self.batch_size)
        loss = self.learn(experiences)
        return {"loss": loss, "epsilon": self.epsilon}

    def on_episode_end(self):
        self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)

    def learn(self, experiences):
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

    def _compute_q_targets(self, rewards, next_states, dones):
        Q_targets_next = torch.max(self.target_network(next_states), 1)[0].unsqueeze(1)
        return rewards + (self.gamma * Q_targets_next * (1 - dones))

    def save(self, filename):
        """
        Save model parameters.

        :param filename: File name
        """
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

    def load(self, filename):
        """
        Load model parameters.

        :param filename: File name
        """
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded model from {filename}")
            return True
        return False


class DoubleDQNAgent(DQNAgent):
    algo_name = "double_dqn"

    def _compute_q_targets(self, rewards, next_states, dones):
        best_actions = torch.argmax(self.q_network(next_states), dim=1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, best_actions)
        return rewards + (self.gamma * next_q_values * (1 - dones))
