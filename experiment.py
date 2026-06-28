from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from algo import build_agent
from algo.base import BaseAgent
from algo.mappo import MAPPO
from env import Env

DEFAULT_DEVICE_NAME = "cuda:0"


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str = DEFAULT_DEVICE_NAME) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def build_env(config: Any, device: torch.device) -> Env:
    env_config = config["env"]
    return Env(
        env_config["num_firms"],
        list(env_config["p"]),
        env_config["h"],
        env_config["c"],
        env_config["initial_inventory"],
        env_config["poisson_lambda"],
        env_config["max_steps"],
        device=device,
    )


def build_configured_agent(config: Any, device: torch.device) -> BaseAgent:
    env_config = config["env"]
    algo_config = config["algo"]
    agent_config = {
        **OmegaConf.to_container(algo_config["agent"], resolve=True),
        "max_order": env_config["max_order"],
        "device": device,
    }
    return build_agent(algo_config["name"], **agent_config)


def build_configured_agents(config: Any, device: torch.device) -> list[BaseAgent]:
    env_config = config["env"]
    algo_config = config["algo"]
    multi_agent_config = config.get("multi_agent", {})
    firm_ids = multi_agent_config.get("firm_ids")
    if firm_ids is None:
        firm_ids = range(env_config["num_firms"])

    agents: list[BaseAgent] = []
    for firm_id in firm_ids:
        agent_config = {
            **OmegaConf.to_container(algo_config["agent"], resolve=True),
            "firm_id": int(firm_id),
            "max_order": env_config["max_order"],
            "device": device,
        }
        agents.append(build_agent(algo_config["name"], **agent_config))
    return agents


def build_configured_mappo(config: Any, device: torch.device) -> MAPPO:
    env_config = config["env"]
    algo_config = config["algo"]
    multi_agent_config = config.get("multi_agent", {})
    firm_ids = multi_agent_config.get("firm_ids")
    if firm_ids is None:
        firm_ids = range(env_config["num_firms"])

    agent_config = {
        **OmegaConf.to_container(algo_config["agent"], resolve=True),
        "num_firms": env_config["num_firms"],
        "firm_ids": [int(firm_id) for firm_id in firm_ids],
        "max_order": env_config["max_order"],
        "device": device,
    }
    return MAPPO(**agent_config)


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
