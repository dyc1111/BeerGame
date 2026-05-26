import torch


class Env:
    def __init__(
        self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100
    ):
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
        self.num_firms = num_firms
        self.p = torch.tensor(p + [0], dtype=torch.float32)
        self.firm_state_size = 3
        self.h = h
        self.c = c
        self.poisson_lambda = float(poisson_lambda)
        self.max_steps = max_steps
        self.initial_inventory = float(initial_inventory)
        self.reset()

    def reset(self):
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

    def _get_observation(self):
        return torch.stack((self.orders, self.satisfied_demand, self.inventory), dim=1)

    def _generate_demand(self):
        demand = torch.zeros(self.num_firms, dtype=torch.float32)
        demand[0] = torch.poisson(torch.tensor(self.poisson_lambda))
        demand[1:] = self.orders[:-1]
        return demand

    def step(self, actions):
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
