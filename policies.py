from __future__ import annotations

from typing import Any, Protocol

import torch


class OpponentPolicy(Protocol):
    def act(self, obs: torch.Tensor, firm_id: int) -> torch.Tensor: ...


class RandomOrderPolicy:
    def __init__(self, max_order: int, device: torch.device) -> None:
        self.max_order: int = max_order
        self.device: torch.device = device

    def act(self, obs: torch.Tensor, firm_id: int) -> torch.Tensor:
        return torch.randint(1, self.max_order + 1, (), device=self.device).float()


class ConstantOrderPolicy:
    def __init__(self, order: float, device: torch.device) -> None:
        self.order: float = float(order)
        self.device: torch.device = device

    def act(self, obs: torch.Tensor, firm_id: int) -> torch.Tensor:
        return torch.tensor(self.order, dtype=torch.float32, device=self.device)


class AgentObservationTransform:
    def __init__(self, config: Any, device: torch.device) -> None:
        self.enabled: bool = bool(config["train"].get("normalize_observations", True))
        env_config = config["env"]
        self.scale = torch.tensor(
            [
                max(float(env_config["max_order"]), 1.0),
                max(float(env_config["max_order"]), 1.0),
                max(float(env_config["initial_inventory"]), 1.0),
            ],
            dtype=torch.float32,
            device=device,
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.scale.device)
        if not self.enabled:
            return obs
        return obs / self.scale


def build_opponent_policy(config: Any, device: torch.device) -> OpponentPolicy:
    policy_name = config["opponents"].get("policy", "random")
    if policy_name == "random":
        return RandomOrderPolicy(config["env"]["max_order"], device)
    if policy_name == "constant":
        return ConstantOrderPolicy(
            config["opponents"].get("constant_order", 10), device
        )
    raise ValueError(f"Unknown opponent policy: {policy_name}")
