from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Optional, Union

import torch

Pathish = Union[str, PathLike[str]]


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
    def __init__(self, action_type: str = "discrete", max_order: int = 20) -> None:
        if action_type not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown action type: {action_type}")
        self.action_type = action_type
        self.max_order = max_order

    def to_env_action(self, action: Any) -> torch.Tensor:
        action = torch.as_tensor(action)
        if self.action_type == "discrete":
            action = action.long().clamp(0, self.max_order - 1) + 1
            return action.float().reshape(())
        action = action.float().round().clamp(1, self.max_order)
        return action.reshape(())


class BaseAgent:
    algo_name = "base"
    firm_id: int

    def act(self, obs: Any, mode: str = "train") -> ActionResult:
        raise NotImplementedError

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError

    def ready_to_update(self) -> bool:
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        raise NotImplementedError

    def on_episode_end(self) -> None:
        pass

    def save(self, filename: Pathish) -> None:
        raise NotImplementedError

    def load(self, filename: Pathish) -> bool:
        raise NotImplementedError
