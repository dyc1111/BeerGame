from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from algo.base import ActionResult, BaseAgent, Transition
from algo.happo import HAPPO
from algo.mappo import MAPPO
from env import Env
from policies import AgentObservationTransform, OpponentPolicy

History = list[list[torch.Tensor]]
TestResult = tuple[list[torch.Tensor], History, History, History, History]
AgentScores = dict[int, list[torch.Tensor]]
AgentHistories = dict[int, History]
IndependentTestResult = tuple[
    AgentScores,
    AgentHistories,
    AgentHistories,
    AgentHistories,
    AgentHistories,
]


def collect_actions(
    env: Env,
    agent: BaseAgent,
    opponent_policy: OpponentPolicy,
    state: torch.Tensor,
    mode: str,
    obs_transform: AgentObservationTransform,
) -> tuple[torch.Tensor, ActionResult]:
    actions = torch.zeros(env.num_firms, dtype=torch.float32, device=state.device)
    action_result: ActionResult | None = None

    for firm_id in range(env.num_firms):
        if firm_id == agent.firm_id:
            action_result = agent.act(obs_transform(state[firm_id]), mode=mode)
            actions[firm_id] = action_result.env_action
        else:
            actions[firm_id] = opponent_policy.act(state[firm_id], firm_id)

    if action_result is None:
        raise RuntimeError(f"No action was collected for firm_id={agent.firm_id}")
    return actions, action_result


def _agents_by_firm(agents: list[BaseAgent]) -> dict[int, BaseAgent]:
    agents_by_firm = {agent.firm_id: agent for agent in agents}
    if len(agents_by_firm) != len(agents):
        raise ValueError("Independent learners must have unique firm_id values")
    return agents_by_firm


def collect_independent_actions(
    env: Env,
    agents_by_firm: dict[int, BaseAgent],
    opponent_policy: OpponentPolicy,
    state: torch.Tensor,
    mode: str,
    obs_transform: AgentObservationTransform,
) -> tuple[torch.Tensor, dict[int, ActionResult]]:
    actions = torch.zeros(env.num_firms, dtype=torch.float32, device=state.device)
    action_results: dict[int, ActionResult] = {}

    for firm_id in range(env.num_firms):
        agent = agents_by_firm.get(firm_id)
        if agent is None:
            actions[firm_id] = opponent_policy.act(state[firm_id], firm_id)
            continue
        action_result = agent.act(obs_transform(state[firm_id]), mode=mode)
        actions[firm_id] = action_result.env_action
        action_results[firm_id] = action_result

    return actions, action_results


def _update_agent(agent: BaseAgent) -> dict[str, float]:
    last_update_metrics: dict[str, float] = {}
    update_guard = 0
    while agent.ready_to_update():
        last_update_metrics = agent.update()
        update_guard += 1
        if update_guard > 1000:
            raise RuntimeError(
                f"{agent.algo_name}.update() did not clear update readiness"
            )
    return last_update_metrics


def _format_metric_averages(metrics_by_firm: dict[int, dict[str, float]]) -> str:
    metric_values: dict[str, list[float]] = {}
    for metrics in metrics_by_firm.values():
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(name, []).append(float(value))
    if not metric_values:
        return ""
    return " | " + " | ".join(
        f"avg_{name}: {sum(values) / len(values):.4f}"
        for name, values in metric_values.items()
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return ""
    return " | " + " | ".join(
        f"{name}: {value:.4f}"
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    )


def _global_observation(
    state: torch.Tensor, obs_transform: AgentObservationTransform
) -> torch.Tensor:
    return obs_transform(state).flatten()


def train(
    env: Env,
    agent: BaseAgent,
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
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
    reward_scale = float(train_config.get("reward_scale", 1.0))
    os.makedirs(model_dir, exist_ok=True)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        score = torch.tensor(0.0, device=env.device)
        last_update_metrics: dict[str, float] = {}

        for _ in range(max_t):
            actions, action_result = collect_actions(
                env,
                agent,
                opponent_policy,
                state,
                mode="train",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)

            transition = Transition(
                obs=obs_transform(state[agent.firm_id]),
                action=actions[agent.firm_id],
                reward=rewards[agent.firm_id] * reward_scale,
                next_obs=obs_transform(next_state[agent.firm_id]),
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


def train_independent_agents(
    env: Env,
    agents: list[BaseAgent],
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
    train_config: Any,
    model_dir: Path,
) -> AgentScores:
    """
    Train independent learners in the same environment.

    Each learner owns its own policy, replay or rollout buffer, optimizer, and
    reward stream. There is no parameter sharing or centralized critic.
    """
    agents_by_firm = _agents_by_firm(agents)
    scores_by_firm: AgentScores = {agent.firm_id: [] for agent in agents}
    num_episodes = train_config.get("num_episodes", 1000)
    max_t = train_config.get("max_t", env.max_steps)
    log_every = train_config.get("log_every", 100)
    checkpoint_every = train_config.get("checkpoint_every", 500)
    reward_scale = float(train_config.get("reward_scale", 1.0))
    os.makedirs(model_dir, exist_ok=True)

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_scores = {
            agent.firm_id: torch.tensor(0.0, device=env.device) for agent in agents
        }
        last_metrics_by_firm: dict[int, dict[str, float]] = {}

        for _ in range(max_t):
            actions, action_results = collect_independent_actions(
                env,
                agents_by_firm,
                opponent_policy,
                state,
                mode="train",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)

            for agent in agents:
                firm_id = agent.firm_id
                action_result = action_results[firm_id]
                transition = Transition(
                    obs=obs_transform(state[firm_id]),
                    action=actions[firm_id],
                    reward=rewards[firm_id] * reward_scale,
                    next_obs=obs_transform(next_state[firm_id]),
                    done=done,
                    raw_action=action_result.raw_action,
                    log_prob=action_result.log_prob,
                    value=action_result.value,
                )
                agent.observe(transition)
                episode_scores[firm_id] += rewards[firm_id]

            for agent in agents:
                metrics = _update_agent(agent)
                if metrics:
                    last_metrics_by_firm[agent.firm_id] = metrics

            state = next_state
            if done:
                break

        for agent in agents:
            agent.on_episode_end()
        for agent in agents:
            metrics = _update_agent(agent)
            if metrics:
                last_metrics_by_firm[agent.firm_id] = metrics

        for firm_id, score in episode_scores.items():
            scores_by_firm[firm_id].append(score.detach().clone())

        if i_episode % log_every == 0:
            recent_means = [
                torch.stack(scores[-log_every:]).mean()
                for scores in scores_by_firm.values()
                if scores
            ]
            mean_agent_score = torch.stack(recent_means).mean().item()
            metric_text = _format_metric_averages(last_metrics_by_firm)
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Mean Agent Score: {mean_agent_score:.2f}{metric_text}"
            )

        if i_episode % checkpoint_every == 0:
            for agent in agents:
                agent.save(
                    os.path.join(
                        model_dir,
                        f"{agent.algo_name}_agent_firm_{agent.firm_id}_episode_{i_episode}.pth",
                    )
                )

    for agent in agents:
        agent.save(
            os.path.join(
                model_dir, f"{agent.algo_name}_agent_firm_{agent.firm_id}_final.pth"
            )
        )

    return scores_by_firm


def collect_mappo_actions(
    env: Env,
    mappo: MAPPO,
    opponent_policy: OpponentPolicy,
    state: torch.Tensor,
    mode: str,
    obs_transform: AgentObservationTransform,
) -> tuple[torch.Tensor, dict[int, ActionResult]]:
    actions = torch.zeros(env.num_firms, dtype=torch.float32, device=state.device)
    action_results: dict[int, ActionResult] = {}

    for firm_id in range(env.num_firms):
        if mappo.has_firm(firm_id):
            action_result = mappo.act(firm_id, obs_transform(state[firm_id]), mode=mode)
            actions[firm_id] = action_result.env_action
            action_results[firm_id] = action_result
        else:
            actions[firm_id] = opponent_policy.act(state[firm_id], firm_id)

    return actions, action_results


def train_mappo(
    env: Env,
    mappo: MAPPO,
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
    train_config: Any,
    model_dir: Path,
) -> AgentScores:
    """
    Train shared-parameter MAPPO with decentralized actors and a centralized critic.
    """
    scores_by_firm: AgentScores = {firm_id: [] for firm_id in mappo.firm_ids}
    num_episodes = train_config.get("num_episodes", 1000)
    max_t = train_config.get("max_t", env.max_steps)
    log_every = train_config.get("log_every", 100)
    checkpoint_every = train_config.get("checkpoint_every", 500)
    reward_scale = float(train_config.get("reward_scale", 1.0))
    os.makedirs(model_dir, exist_ok=True)
    last_update_metrics: dict[str, float] = {}

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_scores = {
            firm_id: torch.tensor(0.0, device=env.device) for firm_id in mappo.firm_ids
        }

        for _ in range(max_t):
            global_obs = _global_observation(state, obs_transform)
            actions, action_results = collect_mappo_actions(
                env,
                mappo,
                opponent_policy,
                state,
                mode="train",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)
            next_global_obs = _global_observation(next_state, obs_transform)

            for firm_id in mappo.firm_ids:
                action_result = action_results[firm_id]
                if action_result.raw_action is None or action_result.log_prob is None:
                    raise RuntimeError("MAPPO actions must include raw_action/log_prob")
                mappo.observe(
                    firm_id=firm_id,
                    local_obs=obs_transform(state[firm_id]),
                    global_obs=global_obs,
                    action=action_result.raw_action,
                    reward=rewards[firm_id] * reward_scale,
                    next_global_obs=next_global_obs,
                    done=done,
                    log_prob=action_result.log_prob,
                )
                episode_scores[firm_id] += rewards[firm_id]

            if mappo.ready_to_update():
                last_update_metrics = mappo.update()

            state = next_state
            if done:
                break

        for firm_id, score in episode_scores.items():
            scores_by_firm[firm_id].append(score.detach().clone())

        if i_episode % log_every == 0:
            recent_means = [
                torch.stack(scores[-log_every:]).mean()
                for scores in scores_by_firm.values()
                if scores
            ]
            mean_agent_score = torch.stack(recent_means).mean().item()
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Mean Agent Score: {mean_agent_score:.2f}"
                f"{_format_metrics(last_update_metrics)}"
            )

        if i_episode % checkpoint_every == 0:
            mappo.save(
                os.path.join(model_dir, f"{mappo.algo_name}_episode_{i_episode}.pth")
            )

    if mappo.has_rollout_data():
        last_update_metrics = mappo.update(force=True)
        if last_update_metrics:
            print("Final partial update" + _format_metrics(last_update_metrics))

    mappo.save(os.path.join(model_dir, f"{mappo.algo_name}_final.pth"))
    return scores_by_firm


def collect_happo_actions(
    env: Env,
    happo: HAPPO,
    opponent_policy: OpponentPolicy,
    state: torch.Tensor,
    mode: str,
    obs_transform: AgentObservationTransform,
) -> tuple[torch.Tensor, dict[int, ActionResult]]:
    actions = torch.zeros(env.num_firms, dtype=torch.float32, device=state.device)
    action_results: dict[int, ActionResult] = {}

    for firm_id in range(env.num_firms):
        if happo.has_firm(firm_id):
            action_result = happo.act(firm_id, obs_transform(state[firm_id]), mode=mode)
            actions[firm_id] = action_result.env_action
            action_results[firm_id] = action_result
        else:
            actions[firm_id] = opponent_policy.act(state[firm_id], firm_id)

    return actions, action_results


def train_happo(
    env: Env,
    happo: HAPPO,
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
    train_config: Any,
    model_dir: Path,
) -> AgentScores:
    """
    Train HAPPO with heterogeneous actors and a shared centralized critic.
    """
    scores_by_firm: AgentScores = {firm_id: [] for firm_id in happo.firm_ids}
    num_episodes = train_config.get("num_episodes", 1000)
    max_t = train_config.get("max_t", env.max_steps)
    log_every = train_config.get("log_every", 100)
    checkpoint_every = train_config.get("checkpoint_every", 500)
    reward_scale = float(train_config.get("reward_scale", 1.0))
    os.makedirs(model_dir, exist_ok=True)
    last_update_metrics: dict[str, float] = {}

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_scores = {
            firm_id: torch.tensor(0.0, device=env.device) for firm_id in happo.firm_ids
        }

        for _ in range(max_t):
            global_obs = _global_observation(state, obs_transform)
            actions, action_results = collect_happo_actions(
                env,
                happo,
                opponent_policy,
                state,
                mode="train",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)
            next_global_obs = _global_observation(next_state, obs_transform)
            mean_reward = rewards.mean()

            for firm_id in happo.firm_ids:
                action_result = action_results[firm_id]
                if action_result.raw_action is None or action_result.log_prob is None:
                    raise RuntimeError("HAPPO actions must include raw_action/log_prob")
                mixed_reward = (
                    happo.individual_reward_weight * rewards[firm_id]
                    + (1.0 - happo.individual_reward_weight) * mean_reward
                )
                happo.observe(
                    firm_id=firm_id,
                    local_obs=obs_transform(state[firm_id]),
                    global_obs=global_obs,
                    action=action_result.raw_action,
                    reward=mixed_reward * reward_scale,
                    next_global_obs=next_global_obs,
                    done=done,
                    log_prob=action_result.log_prob,
                )
                episode_scores[firm_id] += rewards[firm_id]

            if happo.ready_to_update():
                last_update_metrics = happo.update()

            state = next_state
            if done:
                break

        for firm_id, score in episode_scores.items():
            scores_by_firm[firm_id].append(score.detach().clone())

        if i_episode % log_every == 0:
            recent_means = [
                torch.stack(scores[-log_every:]).mean()
                for scores in scores_by_firm.values()
                if scores
            ]
            mean_agent_score = torch.stack(recent_means).mean().item()
            print(
                f"Episode {i_episode}/{num_episodes} | "
                f"Mean Agent Score: {mean_agent_score:.2f}"
                f"{_format_metrics(last_update_metrics)}"
            )

        if i_episode % checkpoint_every == 0:
            happo.save(
                os.path.join(model_dir, f"{happo.algo_name}_episode_{i_episode}.pth")
            )

    if happo.has_rollout_data():
        last_update_metrics = happo.update(force=True)
        if last_update_metrics:
            print("Final partial update" + _format_metrics(last_update_metrics))

    happo.save(os.path.join(model_dir, f"{happo.algo_name}_final.pth"))
    return scores_by_firm


def test(
    env: Env,
    agent: BaseAgent,
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
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
        score = torch.tensor(0.0, device=env.device)
        episode_inventory: list[torch.Tensor] = []
        episode_orders: list[torch.Tensor] = []
        episode_demand: list[torch.Tensor] = []
        episode_satisfied_demand: list[torch.Tensor] = []

        for _ in range(env.max_steps):
            actions, _ = collect_actions(
                env,
                agent,
                opponent_policy,
                state,
                mode="eval",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)

            episode_inventory.append(
                env.inventory[agent.firm_id].detach().cpu().clone()
            )
            episode_orders.append(actions[agent.firm_id].detach().cpu().clone())
            episode_demand.append(env.demand[agent.firm_id].detach().cpu().clone())
            episode_satisfied_demand.append(
                env.satisfied_demand[agent.firm_id].detach().cpu().clone()
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


def test_independent_agents(
    env: Env,
    agents: list[BaseAgent],
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
    num_episodes: int = 10,
) -> IndependentTestResult:
    """
    Evaluate independent learners acting together in one environment.
    """
    agents_by_firm = _agents_by_firm(agents)
    scores_by_firm: AgentScores = {agent.firm_id: [] for agent in agents}
    inventory_history: AgentHistories = {agent.firm_id: [] for agent in agents}
    orders_history: AgentHistories = {agent.firm_id: [] for agent in agents}
    demand_history: AgentHistories = {agent.firm_id: [] for agent in agents}
    satisfied_demand_history: AgentHistories = {agent.firm_id: [] for agent in agents}

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_scores = {
            agent.firm_id: torch.tensor(0.0, device=env.device) for agent in agents
        }
        episode_inventory: AgentHistories = {agent.firm_id: [] for agent in agents}
        episode_orders: AgentHistories = {agent.firm_id: [] for agent in agents}
        episode_demand: AgentHistories = {agent.firm_id: [] for agent in agents}
        episode_satisfied_demand: AgentHistories = {
            agent.firm_id: [] for agent in agents
        }

        for _ in range(env.max_steps):
            actions, _ = collect_independent_actions(
                env,
                agents_by_firm,
                opponent_policy,
                state,
                mode="eval",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)

            for agent in agents:
                firm_id = agent.firm_id
                episode_inventory[firm_id].append(
                    env.inventory[firm_id].detach().cpu().clone()
                )
                episode_orders[firm_id].append(actions[firm_id].detach().cpu().clone())
                episode_demand[firm_id].append(
                    env.demand[firm_id].detach().cpu().clone()
                )
                episode_satisfied_demand[firm_id].append(
                    env.satisfied_demand[firm_id].detach().cpu().clone()
                )
                episode_scores[firm_id] += rewards[firm_id]

            state = next_state
            if done:
                break

        for agent in agents:
            firm_id = agent.firm_id
            scores_by_firm[firm_id].append(episode_scores[firm_id].detach().clone())
            inventory_history[firm_id].append(episode_inventory[firm_id])
            orders_history[firm_id].append(episode_orders[firm_id])
            demand_history[firm_id].append(episode_demand[firm_id])
            satisfied_demand_history[firm_id].append(episode_satisfied_demand[firm_id])

        mean_score = torch.stack(list(episode_scores.values())).mean().item()
        print(
            f"Test Episode {i_episode}/{num_episodes} | "
            f"Mean Agent Score: {mean_score:.2f}"
        )

    return (
        scores_by_firm,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )


def test_mappo(
    env: Env,
    mappo: MAPPO,
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
    num_episodes: int = 10,
) -> IndependentTestResult:
    """
    Evaluate MAPPO actors with decentralized execution.
    """
    scores_by_firm: AgentScores = {firm_id: [] for firm_id in mappo.firm_ids}
    inventory_history: AgentHistories = {firm_id: [] for firm_id in mappo.firm_ids}
    orders_history: AgentHistories = {firm_id: [] for firm_id in mappo.firm_ids}
    demand_history: AgentHistories = {firm_id: [] for firm_id in mappo.firm_ids}
    satisfied_demand_history: AgentHistories = {
        firm_id: [] for firm_id in mappo.firm_ids
    }

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_scores = {
            firm_id: torch.tensor(0.0, device=env.device) for firm_id in mappo.firm_ids
        }
        episode_inventory: AgentHistories = {firm_id: [] for firm_id in mappo.firm_ids}
        episode_orders: AgentHistories = {firm_id: [] for firm_id in mappo.firm_ids}
        episode_demand: AgentHistories = {firm_id: [] for firm_id in mappo.firm_ids}
        episode_satisfied_demand: AgentHistories = {
            firm_id: [] for firm_id in mappo.firm_ids
        }

        for _ in range(env.max_steps):
            actions, _ = collect_mappo_actions(
                env,
                mappo,
                opponent_policy,
                state,
                mode="eval",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)

            for firm_id in mappo.firm_ids:
                episode_inventory[firm_id].append(
                    env.inventory[firm_id].detach().cpu().clone()
                )
                episode_orders[firm_id].append(actions[firm_id].detach().cpu().clone())
                episode_demand[firm_id].append(
                    env.demand[firm_id].detach().cpu().clone()
                )
                episode_satisfied_demand[firm_id].append(
                    env.satisfied_demand[firm_id].detach().cpu().clone()
                )
                episode_scores[firm_id] += rewards[firm_id]

            state = next_state
            if done:
                break

        for firm_id in mappo.firm_ids:
            scores_by_firm[firm_id].append(episode_scores[firm_id].detach().clone())
            inventory_history[firm_id].append(episode_inventory[firm_id])
            orders_history[firm_id].append(episode_orders[firm_id])
            demand_history[firm_id].append(episode_demand[firm_id])
            satisfied_demand_history[firm_id].append(episode_satisfied_demand[firm_id])

        mean_score = torch.stack(list(episode_scores.values())).mean().item()
        print(
            f"Test Episode {i_episode}/{num_episodes} | "
            f"Mean Agent Score: {mean_score:.2f}"
        )

    return (
        scores_by_firm,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )


def test_happo(
    env: Env,
    happo: HAPPO,
    opponent_policy: OpponentPolicy,
    obs_transform: AgentObservationTransform,
    num_episodes: int = 10,
) -> IndependentTestResult:
    """
    Evaluate HAPPO actors with decentralized execution.
    """
    scores_by_firm: AgentScores = {firm_id: [] for firm_id in happo.firm_ids}
    inventory_history: AgentHistories = {firm_id: [] for firm_id in happo.firm_ids}
    orders_history: AgentHistories = {firm_id: [] for firm_id in happo.firm_ids}
    demand_history: AgentHistories = {firm_id: [] for firm_id in happo.firm_ids}
    satisfied_demand_history: AgentHistories = {
        firm_id: [] for firm_id in happo.firm_ids
    }

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_scores = {
            firm_id: torch.tensor(0.0, device=env.device) for firm_id in happo.firm_ids
        }
        episode_inventory: AgentHistories = {firm_id: [] for firm_id in happo.firm_ids}
        episode_orders: AgentHistories = {firm_id: [] for firm_id in happo.firm_ids}
        episode_demand: AgentHistories = {firm_id: [] for firm_id in happo.firm_ids}
        episode_satisfied_demand: AgentHistories = {
            firm_id: [] for firm_id in happo.firm_ids
        }

        for _ in range(env.max_steps):
            actions, _ = collect_happo_actions(
                env,
                happo,
                opponent_policy,
                state,
                mode="eval",
                obs_transform=obs_transform,
            )
            next_state, rewards, done = env.step(actions)

            for firm_id in happo.firm_ids:
                episode_inventory[firm_id].append(
                    env.inventory[firm_id].detach().cpu().clone()
                )
                episode_orders[firm_id].append(actions[firm_id].detach().cpu().clone())
                episode_demand[firm_id].append(
                    env.demand[firm_id].detach().cpu().clone()
                )
                episode_satisfied_demand[firm_id].append(
                    env.satisfied_demand[firm_id].detach().cpu().clone()
                )
                episode_scores[firm_id] += rewards[firm_id]

            state = next_state
            if done:
                break

        for firm_id in happo.firm_ids:
            scores_by_firm[firm_id].append(episode_scores[firm_id].detach().clone())
            inventory_history[firm_id].append(episode_inventory[firm_id])
            orders_history[firm_id].append(episode_orders[firm_id])
            demand_history[firm_id].append(episode_demand[firm_id])
            satisfied_demand_history[firm_id].append(episode_satisfied_demand[firm_id])

        mean_score = torch.stack(list(episode_scores.values())).mean().item()
        print(
            f"Test Episode {i_episode}/{num_episodes} | "
            f"Mean Agent Score: {mean_score:.2f}"
        )

    return (
        scores_by_firm,
        inventory_history,
        orders_history,
        demand_history,
        satisfied_demand_history,
    )
