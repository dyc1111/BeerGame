from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ActionResult:
    env_action: torch.Tensor
    raw_action: Optional[torch.Tensor] = None
    log_prob: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: bool
    raw_action: Optional[torch.Tensor] = None
    log_prob: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


class ActionAdapter:
    def __init__(self, action_type="discrete", max_order=20):
        if action_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown action type: {action_type}")
        self.action_type = action_type
        self.max_order = max_order

    def to_env_action(self, action):
        action = torch.as_tensor(action)
        if self.action_type == "discrete":
            action = action.long().clamp(0, self.max_order - 1) + 1
            return action.float().reshape(())
        action = action.float().round().clamp(1, self.max_order)
        return action.reshape(())


class BaseAgent:
    algo_name = "base"

    def act(self, obs, mode="train"):
        raise NotImplementedError

    def observe(self, transition):
        raise NotImplementedError

    def ready_to_update(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def on_episode_end(self):
        pass

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError
