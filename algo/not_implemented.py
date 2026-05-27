from .base import BaseAgent


class AlgorithmNotImplementedAgent(BaseAgent):
    algo_name = "not_implemented"

    def __init__(self, *args, **kwargs):
        requested = kwargs.pop("requested_algo", self.algo_name)
        raise NotImplementedError(
            f"Algorithm '{requested}' is registered but not implemented yet. "
            "Add an agent with the BaseAgent API before selecting it in CONFIG."
        )


class PPOAgent(AlgorithmNotImplementedAgent):
    algo_name = "ppo"


class TRPOAgent(AlgorithmNotImplementedAgent):
    algo_name = "trpo"


class SACAgent(AlgorithmNotImplementedAgent):
    algo_name = "sac"
