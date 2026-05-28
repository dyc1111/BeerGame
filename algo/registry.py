from __future__ import annotations

from typing import Any, TypeAlias

from .base import BaseAgent
from .dqn import DQNAgent, DoubleDQNAgent
from .not_implemented import SACAgent, TRPOAgent
from .ppo import PPOAgent

AgentClass: TypeAlias = type[BaseAgent]

ALGORITHMS: dict[str, AgentClass] = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "ppo": PPOAgent,
    "trpo": TRPOAgent,
    "sac": SACAgent,
}


def build_agent(name: str, **kwargs: Any) -> BaseAgent:
    if name not in ALGORITHMS:
        known = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {known}")

    agent_cls = ALGORITHMS[name]
    if agent_cls in {TRPOAgent, SACAgent}:
        kwargs["requested_algo"] = name
    return agent_cls(**kwargs)
