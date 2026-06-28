# Multiagent Beer Game RL

This project studies reinforcement learning policies for a serial beer-game
supply chain. The environment has `num_firms` firms ordered from downstream to
upstream. Firm `0` faces exogenous Poisson demand, firm `i > 0` faces demand
equal to firm `i - 1`'s order, and the last firm can replenish from an external
source.

The code supports single-agent training against fixed opponent policies,
independent multi-agent training, and centralized-training/decentralized-
execution training with MAPPO and HAPPO.

## Setup

There is no packaged environment file. Install the runtime dependencies in a
Python environment with PyTorch available:

```bash
pip install torch hydra-core omegaconf matplotlib
```

The entrypoint is `main.py` and all commands should be run from the repository
root.

## Project Structure

- `env.py`: beer-game environment. The local observation is
  `[last_order, last_satisfied_demand, inventory]`.
- `policies.py`: opponent policies and observation normalization. The normalizer
  scales inventory by `max(initial_inventory, max_order, poisson_lambda, 1)`.
- `training.py`: training and testing loops for single-agent, independent
  multi-agent, MAPPO, and HAPPO execution modes.
- `experiment.py`: Hydra config wiring, environment construction, agent
  construction, device selection, and output directory creation.
- `main.py`: experiment entrypoint.
- `algo/`: algorithm implementations.
  - `dqn.py`: DQN, Double DQN, and Dueling DQN.
  - `ppo.py`: single-agent PPO.
  - `trpo.py`: single-agent TRPO.
  - `sac.py`: discrete-action SAC.
  - `mappo.py`: shared-parameter MAPPO.
  - `happo.py`: HAPPO with per-firm actors and a shared centralized critic.
- `cfg/base.yaml`: default environment, training, testing, opponent, and output
  settings.
- `cfg/algo/*.yaml`: algorithm-specific hyperparameters.
- `models/{algo}/{exp}`: saved checkpoints.
- `figures/{algo}/{exp}`: training and test plots.
- `MAPPO.md`, `share.md`, `plan.md`, `survey.md`, `summary.md`: design notes,
  literature notes, implementation plan, and experiment summary.

## Configuration

Hydra composes `cfg/base.yaml` with an algorithm config from `cfg/algo`. The
default config uses `double_dqn` in `single_agent` mode.

Important base settings:

- `execution_mode`: one of `single_agent`, `independent_multi_agent`,
  `ctde_multi_agent`.
- `env.initial_inventory`: initial stock at every firm.
- `env.max_order`: action space is integer orders from `1` to `max_order`.
- `opponents.policy`: `random` or `constant`.
- `opponents.constant_order`: order used when `opponents.policy=constant`.
- `multi_agent.firm_ids`: firms controlled by multi-agent learners.
- `train.reward_scale`: scalar applied to rewards before learning updates.
- `exp`: experiment name used in output paths.

Hydra overrides are passed on the command line:

```bash
python3 main.py algo=ppo exp=ppo_firm5 train.num_episodes=2000
```

## Single-Agent Experiments

Single-agent mode trains one firm while every other firm follows the configured
opponent policy. The trained firm is selected by `algo.agent.firm_id`.

```bash
python3 main.py algo=dueling_dqn exp=firm5_init100 algo.agent.firm_id=5 env.initial_inventory=100
```

Run the same algorithm for different firms and inventory levels:

```bash
python3 main.py algo=dueling_dqn exp=firm0_init0 algo.agent.firm_id=0 env.initial_inventory=0
python3 main.py algo=dueling_dqn exp=firm5_init0 algo.agent.firm_id=5 env.initial_inventory=0
python3 main.py algo=dueling_dqn exp=firm8_init0 algo.agent.firm_id=8 env.initial_inventory=0
```

Supported single-agent algorithms:

```bash
python3 main.py algo=dqn exp=dqn_run
python3 main.py algo=double_dqn exp=double_dqn_run
python3 main.py algo=dueling_dqn exp=dueling_dqn_run
python3 main.py algo=ppo exp=ppo_run
python3 main.py algo=trpo exp=trpo_run
python3 main.py algo=sac exp=sac_run
```

## Independent Multi-Agent Experiments

Independent mode creates one separate learner per selected firm. Each learner
has its own policy, value/replay state, optimizer, and reward stream. There is
no centralized critic and no parameter sharing.

```bash
python3 main.py execution_mode=independent_multi_agent algo=dueling_dqn exp=independent_3firm multi_agent.firm_ids='[0,4,8]'
```

Firms not listed in `multi_agent.firm_ids` use the configured opponent policy.

## MAPPO Experiments

MAPPO is enabled with `execution_mode=ctde_multi_agent algo=mappo`. The current
implementation shares both the actor and critic across trained firms and appends
a normalized scalar firm index to both networks' inputs.

```bash
python3 main.py execution_mode=ctde_multi_agent algo=mappo exp=shared_3firms_300 \
  train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 \
  test.num_episodes=10 multi_agent.firm_ids='[0,4,8]' \
  algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

For all firms:

```bash
python3 main.py execution_mode=ctde_multi_agent algo=mappo exp=shared_10firms_300 \
  train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 \
  test.num_episodes=10 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' \
  algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

## HAPPO Experiments

HAPPO is enabled with `execution_mode=ctde_multi_agent algo=happo`. The current
implementation uses separate actors for each trained firm, a shared centralized
critic conditioned on firm index, sequential actor updates, and an optional
HAPPO correction factor.

The effective learning reward is mixed as:

$$
\tilde r_i = w r_i + (1 - w)\bar r,
$$

where `algo.agent.individual_reward_weight` is $w$ and $\bar r$ is the mean firm
reward at the step. Setting `w=1.0` gives purely individual rewards; setting
`w=0.0` gives a pure team-average reward.

Basic HAPPO run:

```bash
python3 main.py execution_mode=ctde_multi_agent algo=happo exp=happo_3firms_300 \
  train.num_episodes=300 train.log_every=10 train.checkpoint_every=999 \
  test.num_episodes=10 multi_agent.firm_ids='[0,4,8]' \
  algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

Per-firm action priors can be set with `algo.agent.initial_orders`. This was
useful for diagnosing the full 10-firm case:

```bash
python3 main.py execution_mode=ctde_multi_agent algo=happo exp=happo_10firms_stair_indiv_200 \
  train.num_episodes=200 train.log_every=20 train.checkpoint_every=999 \
  test.num_episodes=5 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' \
  algo.agent.individual_reward_weight=1.0 \
  algo.agent.initial_orders='[10,9,8,7,6,5,4,3,2,1]' \
  algo.agent.initial_order_bias=3.0 algo.agent.entropy_coef=0.01 \
  algo.agent.actor_lr=0.0001 algo.agent.critic_lr=0.0005 \
  algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=2
```

## Outputs and Checkpoints

Every run writes:

- checkpoints to `models/{algo}/{exp}`;
- plots to `figures/{algo}/{exp}`.

`main.py` always trains first and then tests the trained policy. Existing
checkpoints can be evaluated from Python by constructing the matching agent,
calling `agent.load(...)` or `learner.load(...)`, and then calling the
appropriate test function from `training.py`.

## Current Findings

The single-agent algorithms are working and can learn useful firm-level
policies against random opponents. Dueling DQN was the most useful algorithm for
the firm and initial-inventory comparisons.

The CTDE multi-agent code runs, but full 10-firm MAPPO/HAPPO is not yet a solved
global-control method in this environment. The main issue is an externality:
one firm's order is a cost to itself but creates demand and revenue for upstream
firms. With all firms learned jointly, low-order policies can become
self-reinforcing. See `summary.md` for the experiment ledger and analysis.
