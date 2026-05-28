from __future__ import annotations

from typing import Any, TypeAlias

from .base import BaseAgent
from .dqn import DQNAgent, DoubleDQNAgent, DuelingDQNAgent
from .ppo import PPOAgent
from .sac import SACAgent
from .trpo import TRPOAgent

AgentClass: TypeAlias = type[BaseAgent]

ALGORITHMS: dict[str, AgentClass] = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "ppo": PPOAgent,
    "trpo": TRPOAgent,
    "sac": SACAgent,
}


def build_agent(name: str, **kwargs: Any) -> BaseAgent:
    if name not in ALGORITHMS:
        known = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {known}")
    return ALGORITHMS[name](**kwargs)


__all__ = ["ALGORITHMS", "build_agent"]
