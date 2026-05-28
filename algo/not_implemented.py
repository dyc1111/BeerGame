from __future__ import annotations

from typing import Any

from .base import BaseAgent


class AlgorithmNotImplementedAgent(BaseAgent):
    algo_name = "not_implemented"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        requested = kwargs.pop("requested_algo", self.algo_name)
        raise NotImplementedError(
            f"Algorithm '{requested}' is registered but not implemented yet. "
            "Add an agent with the BaseAgent API before selecting it in CONFIG."
        )


class SACAgent(AlgorithmNotImplementedAgent):
    algo_name = "sac"
