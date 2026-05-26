import matplotlib.pyplot as plt
import torch
import os
from env import Env
from algo.dqn import DQNAgent


def train(
    env,
    agent,
    num_episodes=1000,
    max_t=100,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):
    """
    Train the DQN agent.

    :param env: Environment
    :param agent: DQN agent
    :param num_episodes: Number of training episodes
    :param max_t: Maximum number of steps per episode
    :param eps_start: Starting epsilon value
    :param eps_end: Minimum epsilon value
    :param eps_decay: Epsilon decay rate
    :return: Rewards for all episodes
    """
    scores = []
    eps = eps_start

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = torch.tensor(0.0)

        for _ in range(max_t):
            actions = torch.zeros(env.num_firms, dtype=torch.float32)
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    firm_state = state[firm_id]
                    action = agent.act(firm_state, eps)
                    actions[firm_id] = action
                else:
                    actions[firm_id] = torch.randint(1, 21, ()).float()
            next_state, rewards, done = env.step(actions)

            agent.step(
                state[agent.firm_id],
                actions[agent.firm_id],
                rewards[agent.firm_id],
                next_state[agent.firm_id],
                done,
            )

            state = next_state
            score += rewards[agent.firm_id]

            if done:
                break

        eps = max(eps_end, eps_decay * eps)
        scores.append(score.detach().clone())

        if i_episode % 100 == 0:
            average_score = torch.stack(scores[-100:]).mean().item()
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Average Score: {average_score:.2f} | "
                f"Epsilon: {eps:.4f}"
            )

        if i_episode % 500 == 0:
            agent.save(f"models/dqn_agent_firm_{agent.firm_id}_episode_{i_episode}.pth")

    agent.save(f"models/dqn_agent_firm_{agent.firm_id}_final.pth")

    return scores


def test(env, agent, num_episodes=10):
    """
    Test the trained DQN agent.

    :param env: Environment
    :param agent: Trained DQN agent
    :param num_episodes: Number of test episodes
    :return: Rewards and details for all episodes
    """
    scores = []
    inventory_history = []
    orders_history = []
    demand_history = []
    satisfied_demand_history = []

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = torch.tensor(0.0)
        episode_inventory = []
        episode_orders = []
        episode_demand = []
        episode_satisfied_demand = []

        for _ in range(env.max_steps):
            actions = torch.zeros(env.num_firms, dtype=torch.float32)
            for firm_id in range(env.num_firms):
                if firm_id == agent.firm_id:
                    # Use the agent policy without exploration
                    firm_state = state[firm_id]
                    action = agent.act(firm_state, epsilon=0.0)
                    actions[firm_id] = action
                else:
                    actions[firm_id] = torch.randint(1, 21, ()).float()

            next_state, rewards, done = env.step(actions)

            episode_inventory.append(env.inventory[agent.firm_id].detach().clone())
            episode_orders.append(actions[agent.firm_id].detach().clone())
            episode_demand.append(env.demand[agent.firm_id].detach().clone())
            episode_satisfied_demand.append(
                env.satisfied_demand[agent.firm_id].detach().clone()
            )

            reward = rewards[agent.firm_id]
            score = score + reward

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


def plot_training_results(scores, window_size=100):
    """
    Plot training results.

    :param scores: Reward for each episode
    :param window_size: Moving average window size
    """

    score_tensor = torch.stack(
        [torch.as_tensor(score, dtype=torch.float32).reshape(()) for score in scores]
    )

    def moving_average(data, window_size):
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
    plt.title("DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("figures/training_rewards.png")
    plt.close()


def plot_test_results(
    scores, inventory_history, orders_history, demand_history, satisfied_demand_history
):
    """
    Plot test results.

    :param scores: Reward for each episode
    :param inventory_history: Inventory history for each episode
    :param orders_history: Order history for each episode
    :param demand_history: Demand history for each episode
    :param satisfied_demand_history: Satisfied-demand history for each episode
    """

    def history_to_tensor(history):
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

    # Inventory plot
    axs[0, 0].plot(avg_inventory)
    axs[0, 0].set_title("Average Inventory")
    axs[0, 0].set_xlabel("Time Step")
    axs[0, 0].set_ylabel("Inventory")

    # Order plot
    axs[0, 1].plot(avg_orders)
    axs[0, 1].set_title("Average Order Quantity")
    axs[0, 1].set_xlabel("Time Step")
    axs[0, 1].set_ylabel("Order Quantity")

    # Demand and satisfied-demand plot
    axs[1, 0].plot(avg_demand, label="Demand")
    axs[1, 0].plot(avg_satisfied_demand, label="Satisfied Demand")
    axs[1, 0].set_title("Average Demand vs. Satisfied Demand")
    axs[1, 0].set_xlabel("Time Step")
    axs[1, 0].set_ylabel("Quantity")
    axs[1, 0].legend()

    # Reward bar chart
    axs[1, 1].bar(torch.arange(len(score_tensor)), score_tensor)
    axs[1, 1].set_title("Test Episode Rewards")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Total Reward")

    plt.tight_layout()
    plt.savefig("figures/test_results.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    num_firms = 10
    p = [11 - i for i in range(10)]
    h = 0.5
    c = 2
    initial_inventory = 100
    poisson_lambda = 10
    max_steps = 100

    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)

    firm_id = 5
    state_size = 3
    action_size = 20

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        firm_id=firm_id,
        max_order=action_size,
    )

    scores = train(
        env,
        agent,
        num_episodes=2000,
        max_t=max_steps,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
    )

    plt.rcParams["axes.unicode_minus"] = False

    plot_training_results(scores)

    (
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    ) = test(env, agent, num_episodes=10)

    plot_test_results(
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )
