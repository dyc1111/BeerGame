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


class BaseAgent:
    algo_name = "base"
    firm_id: int
    device: torch.device

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
