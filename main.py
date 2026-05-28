from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from algo import build_agent
from algo.base import ActionResult, BaseAgent, Transition
from env import Env

History = list[list[torch.Tensor]]
TestResult = tuple[list[torch.Tensor], History, History, History, History]


class OpponentPolicy(Protocol):
    def act(self, obs: torch.Tensor, firm_id: int) -> torch.Tensor:
        ...


class RandomOrderPolicy:
    def __init__(self, max_order: int) -> None:
        self.max_order: int = max_order

    def act(self, obs: torch.Tensor, firm_id: int) -> torch.Tensor:
        return torch.randint(1, self.max_order + 1, ()).float()


class ConstantOrderPolicy:
    def __init__(self, order: float) -> None:
        self.order: float = float(order)

    def act(self, obs: torch.Tensor, firm_id: int) -> torch.Tensor:
        return torch.tensor(self.order, dtype=torch.float32)


def build_opponent_policy(config: Any) -> OpponentPolicy:
    policy_name = config["opponents"].get("policy", "random")
    if policy_name == "random":
        return RandomOrderPolicy(config["env"]["max_order"])
    if policy_name == "constant":
        return ConstantOrderPolicy(config["opponents"].get("constant_order", 10))
    raise ValueError(f"Unknown opponent policy: {policy_name}")


def build_env(config: Any) -> Env:
    env_config = config["env"]
    return Env(
        env_config["num_firms"],
        list(env_config["p"]),
        env_config["h"],
        env_config["c"],
        env_config["initial_inventory"],
        env_config["poisson_lambda"],
        env_config["max_steps"],
    )


def build_configured_agent(config: Any) -> BaseAgent:
    env_config = config["env"]
    algo_config = config["algo"]
    agent_config = {
        **OmegaConf.to_container(algo_config["agent"], resolve=True),
        "max_order": env_config["max_order"],
    }
    return build_agent(algo_config["name"], **agent_config)


def build_output_dirs(config: Any) -> tuple[Path, Path]:
    algo_name = config["algo"]["name"]
    exp = str(config.get("exp", "debug"))
    model_root = config["output"].get("model_root", "models")
    figure_root = config["output"].get("figure_root", "figures")
    model_dir = Path(to_absolute_path(os.path.join(model_root, algo_name, exp)))
    figure_dir = Path(to_absolute_path(os.path.join(figure_root, algo_name, exp)))
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, figure_dir


def collect_actions(
    env: Env,
    agent: BaseAgent,
    opponent_policy: OpponentPolicy,
    state: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, ActionResult]:
    actions = torch.zeros(env.num_firms, dtype=torch.float32)
    action_result: ActionResult | None = None

    for firm_id in range(env.num_firms):
        if firm_id == agent.firm_id:
            action_result = agent.act(state[firm_id], mode=mode)
            actions[firm_id] = action_result.env_action
        else:
            actions[firm_id] = opponent_policy.act(state[firm_id], firm_id)

    if action_result is None:
        raise RuntimeError(f"No action was collected for firm_id={agent.firm_id}")
    return actions, action_result


def train(
    env: Env,
    agent: BaseAgent,
    opponent_policy: OpponentPolicy,
    train_config: Any,
    model_dir: Path,
) -> list[torch.Tensor]:
    """
    Train any agent that implements the BaseAgent API.
    """
    scores: list[torch.Tensor] = []
    num_episodes = train_config.get("num_episodes", 1000)
    max_t = train_config.get("max_t", env.max_steps)
    log_every = train_config.get("log_every", 100)
    checkpoint_every = train_config.get("checkpoint_every", 500)
    os.makedirs(model_dir, exist_ok=True)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = torch.tensor(0.0)
        last_update_metrics: dict[str, float] = {}

        for _ in range(max_t):
            actions, action_result = collect_actions(
                env, agent, opponent_policy, state, mode="train"
            )
            next_state, rewards, done = env.step(actions)

            transition = Transition(
                obs=state[agent.firm_id],
                action=actions[agent.firm_id],
                reward=rewards[agent.firm_id],
                next_obs=next_state[agent.firm_id],
                done=done,
                raw_action=action_result.raw_action,
                log_prob=action_result.log_prob,
                value=action_result.value,
            )
            agent.observe(transition)

            update_guard = 0
            while agent.ready_to_update():
                last_update_metrics = agent.update()
                update_guard += 1
                if update_guard > 1000:
                    raise RuntimeError(
                        f"{agent.algo_name}.update() did not clear update readiness"
                    )

            state = next_state
            score += rewards[agent.firm_id]

            if done:
                break

        agent.on_episode_end()
        while agent.ready_to_update():
            last_update_metrics = agent.update()
        scores.append(score.detach().clone())

        if i_episode % log_every == 0:
            average_score = torch.stack(scores[-log_every:]).mean().item()
            metric_text = ""
            if last_update_metrics:
                metric_text = " | " + " | ".join(
                    f"{name}: {value:.4f}"
                    for name, value in last_update_metrics.items()
                    if isinstance(value, (int, float))
                )
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Average Score: {average_score:.2f}{metric_text}"
            )

        if i_episode % checkpoint_every == 0:
            agent.save(
                os.path.join(
                    model_dir,
                    f"{agent.algo_name}_agent_firm_{agent.firm_id}_episode_{i_episode}.pth",
                )
            )

    agent.save(
        os.path.join(
            model_dir, f"{agent.algo_name}_agent_firm_{agent.firm_id}_final.pth"
        )
    )

    return scores


def test(
    env: Env,
    agent: BaseAgent,
    opponent_policy: OpponentPolicy,
    num_episodes: int = 10,
) -> TestResult:
    """
    Test any trained agent that implements the BaseAgent API.
    """
    scores: list[torch.Tensor] = []
    inventory_history: History = []
    orders_history: History = []
    demand_history: History = []
    satisfied_demand_history: History = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = torch.tensor(0.0)
        episode_inventory: list[torch.Tensor] = []
        episode_orders: list[torch.Tensor] = []
        episode_demand: list[torch.Tensor] = []
        episode_satisfied_demand: list[torch.Tensor] = []

        for _ in range(env.max_steps):
            actions, _ = collect_actions(
                env, agent, opponent_policy, state, mode="eval"
            )
            next_state, rewards, done = env.step(actions)

            episode_inventory.append(env.inventory[agent.firm_id].detach().clone())
            episode_orders.append(actions[agent.firm_id].detach().clone())
            episode_demand.append(env.demand[agent.firm_id].detach().clone())
            episode_satisfied_demand.append(
                env.satisfied_demand[agent.firm_id].detach().clone()
            )

            score = score + rewards[agent.firm_id]
            state = next_state

            if done:
                break

        scores.append(score.detach().clone())
        inventory_history.append(episode_inventory)
        orders_history.append(episode_orders)
        demand_history.append(episode_demand)
        satisfied_demand_history.append(episode_satisfied_demand)

        print(f"Test Episode {i_episode}/{num_episodes} | Score: {score.item():.2f}")

    return (
        scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )


def plot_training_results(
    scores: list[torch.Tensor],
    algo_name: str,
    figure_dir: Path,
    window_size: int = 100,
) -> None:
    """
    Plot training rewards.
    """

    score_tensor = torch.stack(
        [torch.as_tensor(score, dtype=torch.float32).reshape(()) for score in scores]
    )

    def moving_average(data: torch.Tensor, window_size: int) -> torch.Tensor:
        return torch.stack(
            [data[max(0, i - window_size) : i + 1].mean() for i in range(len(data))]
        )

    avg_scores = moving_average(score_tensor, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(
        torch.arange(len(score_tensor)),
        score_tensor,
        alpha=0.3,
        label="Raw Reward",
    )
    plt.plot(
        torch.arange(len(avg_scores)),
        avg_scores,
        label=f"{window_size}-Episode Moving Average",
    )
    plt.title(f"{algo_name.upper()} Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(figure_dir, "training_rewards.png"))
    plt.close()


def plot_test_results(
    scores: list[torch.Tensor],
    inventory_history: History,
    orders_history: History,
    demand_history: History,
    satisfied_demand_history: History,
    figure_dir: Path,
) -> None:
    """
    Plot test results.
    """

    def history_to_tensor(history: History) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(value, dtype=torch.float32).reshape(())
                        for value in episode
                    ]
                )
                for episode in history
            ]
        )

    avg_inventory = history_to_tensor(inventory_history).mean(dim=0)
    avg_orders = history_to_tensor(orders_history).mean(dim=0)
    avg_demand = history_to_tensor(demand_history).mean(dim=0)
    avg_satisfied_demand = history_to_tensor(satisfied_demand_history).mean(dim=0)
    score_tensor = torch.stack(
        [torch.as_tensor(score, dtype=torch.float32).reshape(()) for score in scores]
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title("Average Inventory")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Inventory")

    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title("Average Order Quantity")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Order Quantity")

    axs[1, 0].plot(avg_demand, label="Demand")
    axs[1, 0].plot(avg_satisfied_demand, label="Satisfied Demand")
    axs[1, 0].set_title("Average Demand vs. Satisfied Demand")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Quantity")
    axs[1, 0].legend()

    axs[1, 1].bar(torch.arange(len(score_tensor)), score_tensor)
    axs[1, 1].set_title("Test Episode Rewards")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Total Reward")

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "test_results.png"))
    plt.close()


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(config: Any) -> None:
    model_dir, figure_dir = build_output_dirs(config)
    env = build_env(config)
    agent = build_configured_agent(config)
    opponent_policy = build_opponent_policy(config)

    scores = train(env, agent, opponent_policy, config["train"], model_dir)

    plt.rcParams["axes.unicode_minus"] = False
    plot_training_results(scores, config["algo"]["name"], figure_dir)

    (
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    ) = test(
        env,
        agent,
        opponent_policy,
        num_episodes=config["test"]["num_episodes"],
    )

    plot_test_results(
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
        figure_dir,
    )


if __name__ == "__main__":
    main()
