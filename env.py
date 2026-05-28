from __future__ import annotations

from typing import Sequence

import torch


class Env:
    def __init__(
        self,
        num_firms: int,
        p: Sequence[float],
        h: float,
        c: float,
        initial_inventory: float,
        poisson_lambda: float = 10,
        max_steps: int = 100,
    ) -> None:
        """
        Initialize the supply chain management simulation environment.

        :param num_firms: Number of firms
        :param p: Price list for each firm
        :param h: Inventory holding cost
        :param c: Lost-sales cost
        :param initial_inventory: Initial inventory for each firm
        :param poisson_lambda: Mean of the downstream firm's Poisson demand
        :param max_steps: Maximum number of steps per episode
        """
        self.num_firms: int = num_firms
        self.p = torch.tensor(list(p) + [0], dtype=torch.float32)
        self.firm_state_size: int = 3
        self.h: float = h
        self.c: float = c
        self.poisson_lambda: float = float(poisson_lambda)
        self.max_steps: int = max_steps
        self.initial_inventory: float = float(initial_inventory)
        self.inventory: torch.Tensor
        self.orders: torch.Tensor
        self.demand: torch.Tensor
        self.satisfied_demand: torch.Tensor
        self.current_step: int
        self.done: bool
        self.reset()

    def reset(self) -> torch.Tensor:
        """
        reset the environment
        """
        self.inventory = torch.full(
            (self.num_firms,), self.initial_inventory, dtype=torch.float32
        )
        self.orders = torch.zeros(self.num_firms, dtype=torch.float32)
        self.satisfied_demand = torch.zeros(self.num_firms, dtype=torch.float32)
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self) -> torch.Tensor:
        return torch.stack((self.orders, self.satisfied_demand, self.inventory), dim=1)

    def _generate_demand(self) -> torch.Tensor:
        demand = torch.zeros(self.num_firms, dtype=torch.float32)
        demand[0] = torch.poisson(torch.tensor(self.poisson_lambda))
        demand[1:] = self.orders[:-1]
        return demand

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Run one simulation time step and update the environment state from the
        given actions, where each action is a firm's order quantity.

        :param actions: Each firm's order quantity (shape: (num_firms,))
        :return: next_state, reward, done
        """
        self.orders = actions
        self.demand = self._generate_demand()
        self.satisfied_demand = torch.minimum(self.demand, self.inventory)
        self.inventory += self.orders - self.satisfied_demand
        loss_sales = torch.clamp(self.demand - self.satisfied_demand, min=0.0)
        rewards = (
            self.p[:-1] * self.satisfied_demand
            - self.p[1:] * self.orders
            - self.h * self.inventory
            - self.c * loss_sales
        )
        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        return self._get_observation(), rewards, self.done
