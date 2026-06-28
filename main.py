from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import hydra
import matplotlib.pyplot as plt

from experiment import (
    DEFAULT_DEVICE_NAME,
    build_configured_agent,
    build_configured_agents,
    build_configured_happo,
    build_configured_mappo,
    build_env,
    build_output_dirs,
    select_device,
    set_global_seed,
)
from plotting import (
    plot_multi_agent_test_results,
    plot_multi_agent_training_results,
    plot_test_results,
    plot_training_results,
)
from policies import AgentObservationTransform, build_opponent_policy
from training import (
    test,
    test_happo,
    test_independent_agents,
    test_mappo,
    train,
    train_happo,
    train_independent_agents,
    train_mappo,
)


@hydra.main(config_path="cfg", config_name="base", version_base=None)
def main(config: Any) -> None:
    set_global_seed(config.get("seed"))
    device = select_device(DEFAULT_DEVICE_NAME)
    print(f"Using device: {device}")
    model_dir, figure_dir = build_output_dirs(config)
    env = build_env(config, device)
    opponent_policy = build_opponent_policy(config, device)
    obs_transform = AgentObservationTransform(config, device)
    execution_mode = config.get("execution_mode", "single_agent")

    if execution_mode == "ctde_multi_agent":
        algo_name = config["algo"]["name"]
        if algo_name not in {"mappo", "happo"}:
            raise ValueError(
                "ctde_multi_agent execution_mode currently requires algo=mappo or algo=happo"
            )
        learner = (
            build_configured_happo(config, device)
            if algo_name == "happo"
            else build_configured_mappo(config, device)
        )
        print(
            f"Training {algo_name.upper()} CTDE learners for firms: "
            + ", ".join(str(firm_id) for firm_id in learner.firm_ids)
        )
        train_fn = train_happo if algo_name == "happo" else train_mappo
        test_fn = test_happo if algo_name == "happo" else test_mappo
        scores_by_firm = train_fn(
            env, learner, opponent_policy, obs_transform, config["train"], model_dir
        )

        plt.rcParams["axes.unicode_minus"] = False
        plot_multi_agent_training_results(
            scores_by_firm, config["algo"]["name"], figure_dir
        )

        (
            test_scores_by_firm,
            _inventory_history,
            orders_history,
            _demand_history,
            _satisfied_demand_history,
        ) = test_fn(
            env,
            learner,
            opponent_policy,
            obs_transform,
            num_episodes=config["test"]["num_episodes"],
        )

        plot_multi_agent_test_results(
            test_scores_by_firm,
            orders_history,
            figure_dir,
        )
        return

    if execution_mode == "independent_multi_agent":
        agents = build_configured_agents(config, device)
        print(
            "Training independent learners for firms: "
            + ", ".join(str(agent.firm_id) for agent in agents)
        )
        scores_by_firm = train_independent_agents(
            env, agents, opponent_policy, obs_transform, config["train"], model_dir
        )

        plt.rcParams["axes.unicode_minus"] = False
        plot_multi_agent_training_results(
            scores_by_firm, config["algo"]["name"], figure_dir
        )

        (
            test_scores_by_firm,
            _inventory_history,
            orders_history,
            _demand_history,
            _satisfied_demand_history,
        ) = test_independent_agents(
            env,
            agents,
            opponent_policy,
            obs_transform,
            num_episodes=config["test"]["num_episodes"],
        )

        plot_multi_agent_test_results(
            test_scores_by_firm,
            orders_history,
            figure_dir,
        )
        return

    if execution_mode != "single_agent":
        raise ValueError(f"Unknown execution_mode: {execution_mode}")

    agent = build_configured_agent(config, device)

    scores = train(
        env, agent, opponent_policy, obs_transform, config["train"], model_dir
    )

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
        obs_transform,
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
