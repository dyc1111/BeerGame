from .dqn import DQNAgent, DoubleDQNAgent
from .not_implemented import SACAgent, TRPOAgent
from .ppo import PPOAgent

ALGORITHMS = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "ppo": PPOAgent,
    "trpo": TRPOAgent,
    "sac": SACAgent,
}


def build_agent(name, **kwargs):
    if name not in ALGORITHMS:
        known = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {known}")

    agent_cls = ALGORITHMS[name]
    if agent_cls in {TRPOAgent, SACAgent}:
        kwargs["requested_algo"] = name
    return agent_cls(**kwargs)
