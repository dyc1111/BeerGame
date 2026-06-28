from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch

from training import History


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
        [
            torch.as_tensor(score, dtype=torch.float32).detach().cpu().reshape(())
            for score in scores
        ]
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


def _score_tensor(scores: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [
            torch.as_tensor(score, dtype=torch.float32).detach().cpu().reshape(())
            for score in scores
        ]
    )


def _moving_average(data: torch.Tensor, window_size: int) -> torch.Tensor:
    return torch.stack(
        [data[max(0, i - window_size) : i + 1].mean() for i in range(len(data))]
    )


def plot_multi_agent_training_results(
    scores_by_firm: dict[int, list[torch.Tensor]],
    algo_name: str,
    figure_dir: Path,
    window_size: int = 100,
) -> None:
    """
    Plot independent learner rewards by firm and their mean curve.
    """
    firm_ids = sorted(scores_by_firm)
    score_tensors = {
        firm_id: _score_tensor(scores_by_firm[firm_id]) for firm_id in firm_ids
    }
    mean_scores = torch.stack([score_tensors[firm_id] for firm_id in firm_ids]).mean(
        dim=0
    )

    plt.figure(figsize=(12, 7))
    for firm_id in firm_ids:
        avg_scores = _moving_average(score_tensors[firm_id], window_size)
        plt.plot(avg_scores, alpha=0.45, label=f"Firm {firm_id}")
    plt.plot(
        _moving_average(mean_scores, window_size),
        color="black",
        linewidth=2.5,
        label="Mean",
    )
    plt.title(f"{algo_name.upper()} Multi-Agent Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "multi_agent_training_rewards.png"))
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
                        torch.as_tensor(value, dtype=torch.float32)
                        .detach()
                        .cpu()
                        .reshape(())
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
        [
            torch.as_tensor(score, dtype=torch.float32).detach().cpu().reshape(())
            for score in scores
        ]
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


def plot_multi_agent_test_results(
    scores_by_firm: dict[int, list[torch.Tensor]],
    orders_history: dict[int, History],
    figure_dir: Path,
) -> None:
    """
    Plot mean evaluation reward and order quantity for each trained firm.
    """
    firm_ids = sorted(scores_by_firm)
    mean_scores = [
        _score_tensor(scores_by_firm[firm_id]).mean().item() for firm_id in firm_ids
    ]
    mean_orders = []
    for firm_id in firm_ids:
        order_values = torch.stack(
            [
                torch.as_tensor(value, dtype=torch.float32).detach().cpu().reshape(())
                for episode in orders_history[firm_id]
                for value in episode
            ]
        )
        mean_orders.append(order_values.mean().item())

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].bar(firm_ids, mean_scores)
    axs[0].set_title("Mean Test Reward by Firm")
    axs[0].set_xlabel("Firm")
    axs[0].set_ylabel("Reward")

    axs[1].bar(firm_ids, mean_orders)
    axs[1].set_title("Mean Test Order by Firm")
    axs[1].set_xlabel("Firm")
    axs[1].set_ylabel("Order Quantity")

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "multi_agent_test_results.png"))
    plt.close()
