import os

# os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch

from algo import build_agent
from algo.base import Transition
from env import Env

CONFIG = {
    "algo": "dqn",
    "env": {
        "num_firms": 10,
        "p": [11 - i for i in range(10)],
        "h": 0.5,
        "c": 2,
        "initial_inventory": 100,
        "poisson_lambda": 10,
        "max_steps": 100,
        "max_order": 20,
    },
    "agent": {
        "firm_id": 5,
        "state_size": 3,
        "action_size": 20,
        "hidden_size": 64,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 1e-3,
        "tau": 1e-3,
        "update_every": 4,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "action_type": "discrete",
    },
    "train": {
        "num_episodes": 2000,
        "checkpoint_every": 500,
        "log_every": 100,
        "model_dir": "models",
    },
    "test": {
        "num_episodes": 10,
    },
    "opponents": {
        "policy": "random",
        "constant_order": 10,
    },
}


class RandomOrderPolicy:
    def __init__(self, max_order):
        self.max_order = max_order

    def act(self, *_):
        return torch.randint(1, self.max_order + 1, ()).float()


class ConstantOrderPolicy:
    def __init__(self, order):
        self.order = float(order)

    def act(self, *_):
        return torch.tensor(self.order, dtype=torch.float32)


def build_opponent_policy(config):
    policy_name = config["opponents"].get("policy", "random")
    if policy_name == "random":
        return RandomOrderPolicy(config["env"]["max_order"])
    if policy_name == "constant":
        return ConstantOrderPolicy(config["opponents"].get("constant_order", 10))
    raise ValueError(f"Unknown opponent policy: {policy_name}")


def build_env(config):
    env_config = config["env"]
    return Env(
        env_config["num_firms"],
        env_config["p"],
        env_config["h"],
        env_config["c"],
        env_config["initial_inventory"],
        env_config["poisson_lambda"],
        env_config["max_steps"],
    )


def build_configured_agent(config):
    env_config = config["env"]
    agent_config = {
        **config["agent"],
        "max_order": env_config["max_order"],
    }
    return build_agent(config["algo"], **agent_config)


def collect_actions(env, agent, opponent_policy, state, mode):
    actions = torch.zeros(env.num_firms, dtype=torch.float32)
    action_result = None

    for firm_id in range(env.num_firms):
        if firm_id == agent.firm_id:
            action_result = agent.act(state[firm_id], mode=mode)
            actions[firm_id] = action_result.env_action
        else:
            actions[firm_id] = opponent_policy.act(state[firm_id], firm_id)

    return actions, action_result


def train(env, agent, opponent_policy, train_config):
    """
    Train any agent that implements the BaseAgent API.
    """
    scores = []
    num_episodes = train_config.get("num_episodes", 1000)
    max_t = train_config.get("max_t", env.max_steps)
    log_every = train_config.get("log_every", 100)
    checkpoint_every = train_config.get("checkpoint_every", 500)
    model_dir = train_config.get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = torch.tensor(0.0)
        last_update_metrics = {}

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


def test(env, agent, opponent_policy, num_episodes=10):
    """
    Test any trained agent that implements the BaseAgent API.
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


def plot_training_results(scores, algo_name, window_size=100):
    """
    Plot training rewards.
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
    plt.title(f"{algo_name.upper()} Training Rewards")
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
    plt.savefig("figures/test_results.png")
    plt.close()


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    env = build_env(CONFIG)
    agent = build_configured_agent(CONFIG)
    opponent_policy = build_opponent_policy(CONFIG)

    scores = train(env, agent, opponent_policy, CONFIG["train"])

    plt.rcParams["axes.unicode_minus"] = False
    plot_training_results(scores, CONFIG["algo"])

    (
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    ) = test(env, agent, opponent_policy, num_episodes=CONFIG["test"]["num_episodes"])

    plot_test_results(
        test_scores,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )
