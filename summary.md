# Experiment Summary

This file summarizes the current project state, the algorithms tried, the test
results, and the interpretation of the main trends. It is written so that a
formal report can be extracted from it with minimal restructuring.

## Main Conclusions

Single-agent RL is working. DQN-family methods, PPO, TRPO, and SAC all train and
produce usable policies for one controlled firm against random opponent firms.
Among the tested single-agent runs, Dueling DQN gave the strongest final
checkpoint result for the base firm-5 setting.

Multiagent RL with MAPPO and HAPPO is implemented and runnable, but it is not
fully working as a global optimizer for the full 10-firm beer game. The
3-controlled-firm setting can look reasonable, especially under
`initial_inventory=100`, but the full 10-firm setting often collapses to a
low-order policy or preserves only a manually injected role-specific order prior.

The inventory trend is now understood as an environment/economics effect rather
than only a training failure. With `initial_inventory=100`, the initial stock is
a valuable free asset, and downstream firms monetize it at higher sales prices.
With `initial_inventory=0` or `10`, the system starts supply-constrained, and
upstream firms have a structural advantage because they are closer to the
external replenishment source and face less immediate lost-sales pressure.

## Environment

The environment is a serial supply chain with firms indexed from downstream to
upstream. Firm `0` faces exogenous customer demand. Firm `i > 0` faces demand
equal to the previous downstream firm's order. The last firm replenishes from an
external source.

The local observation for firm `i` is:

$$
o_i = [a_i^{t-1}, s_i^{t-1}, I_i^t],
$$

where $a_i$ is the order, $s_i$ is satisfied demand, and $I_i$ is inventory.
Actions are discrete order quantities $a_i \in \{1,\dots,\texttt{max_order}\}$.

Demand is:

$$
d_0^t \sim \operatorname{Poisson}(\lambda), \qquad d_i^t = a_{i-1}^t \quad
\text{for } i > 0.
$$

Satisfied demand is $s_i^t = \min(d_i^t, I_i^t)$. Inbound replenishment is
$x_i^t = s_{i+1}^t$ for all non-last firms, while the last firm receives
$x_{n-1}^t = a_{n-1}^t$. Inventory evolves as:

$$
I_i^{t+1} = I_i^t + x_i^t - s_i^t.
$$

The code uses the reward:

$$
r_i^t = p_i s_i^t - p_{i+1} a_i^t - h I_i^{t+1}
- c(d_i^t - s_i^t)_+.
$$

Two details matter for interpretation. First, prices decrease upstream in the
base config: `p = [10,9,8,7,6,5,4,3,2,1]`, so firm `0` sells at the highest
price and firm `9` at the lowest. Second, the purchase/order cost is charged on
the submitted order $a_i^t$, not on the realized inbound shipment.

## Methods and Why They Were Tried

### Single-Agent RL

Single-agent training controls one firm while all other firms follow an opponent
policy, usually random ordering. This isolates whether standard RL algorithms
can learn the local inventory/order tradeoff before introducing multi-agent
nonstationarity and credit assignment.

DQN was used as the baseline value-based method. It learns $Q(o,a)$ with a
target network and replay buffer. The basic target is:

$$
y = r + \gamma \max_{a'} Q_{\text{target}}(o', a').
$$

Double DQN was tried because ordinary DQN can overestimate action values. It
selects the next action with the online network and evaluates it with the target
network:

$$
y = r + \gamma Q_{\text{target}}\left(o',
\arg\max_{a'} Q_{\text{online}}(o',a')\right).
$$

Dueling DQN was tried because inventory control contains states where the value
of being in the state can matter more than the precise action advantage. It
decomposes:

$$
Q(o,a) = V(o) + A(o,a) - \frac{1}{|\mathcal A|}\sum_{a'} A(o,a').
$$

PPO was tried as the main on-policy policy-gradient baseline. It optimizes a
clipped surrogate:

$$
L^{\text{PPO}} =
\mathbb E\left[
\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\right)
\right],
$$

where $\rho_t = \pi_\theta(a_t|o_t) / \pi_{\theta_{\text{old}}}(a_t|o_t)$.

TRPO was tried because it provides a more explicit trust-region update than PPO:

$$
\max_\theta \mathbb E[\rho_t A_t]
\quad \text{s.t.} \quad
\mathbb E[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}} || \pi_\theta)] \le \delta.
$$

SAC was tried as an entropy-regularized off-policy actor-critic method. In the
discrete action setting, its policy objective encourages both high value and
entropy:

$$
J_\pi =
\mathbb E_{o}\left[
\sum_a \pi(a|o)\left(\alpha \log \pi(a|o) - Q(o,a)\right)
\right].
$$

### Multiagent RL

Independent multi-agent training creates one normal single-agent learner per
controlled firm. It is simple, but each learner treats the other learners as
part of the environment, so the environment is nonstationary from each learner's
view.

MAPPO was tried because the beer game has strong cross-firm dependence. MAPPO
uses decentralized actors but a centralized critic over the full chain state.
The current implementation shares both the actor and critic across controlled
firms and passes a normalized firm index as an input feature. This gives a
single policy class that can still condition behavior on the role in the chain.

HAPPO was tried because simultaneous PPO updates can give poor joint-policy
improvement in heterogeneous cooperative systems. HAPPO updates agents
sequentially. In this project, HAPPO uses separate actors, a shared centralized
critic conditioned on firm index, and an optional HAPPO correction factor. We
also added a mixed reward:

$$
\tilde r_i = w r_i + (1-w)\bar r,
$$

where $w$ is `individual_reward_weight` and $\bar r$ is the mean reward across
firms at that step. The purpose was to interpolate between firm self-interest
and team-level optimization.

Per-firm initial action priors were introduced as a diagnostic tool. A prior
such as `[10,9,8,7,6,5,4,3,2,1]` biases downstream firms to order more and
upstream firms to order less at initialization. This was not meant to be the
final algorithmic solution; it was used to test whether the learning machinery
can preserve a better role-specific policy once placed near one.

## Experiment Ledger

Unless stated otherwise, experiments use the base environment
`num_firms=10`, `max_steps=100`, `max_order=20`,
`poisson_lambda=10`, `seed=0`, `reward_scale=0.01`, and random opponents for
uncontrolled firms. Test results are reported as undiscounted episode reward.
For multi-agent tests, "mean" means the mean over controlled firms and test
episodes.

### Single-Agent Algorithm Comparison

Command pattern:

```bash
python3 main.py algo=<algo> exp=<exp> algo.agent.firm_id=5 test.num_episodes=10
```

Final checkpoints were then evaluated with `training.test(...)` after loading
the checkpoint from `models/<algo>/<exp>/..._final.pth`.

| Algorithm | Experiment/checkpoint | Test result |
| --- | --- | --- |
| DQN | `algo=dqn exp=final_check_dqn` | mean `65.15`, min `-130.00`, max `239.00` |
| Double DQN | `algo=double_dqn exp=final_check_double_dqn` | mean `33.25`, min `-57.50`, max `147.00` |
| Dueling DQN | `algo=dueling_dqn exp=final_check_dueling_dqn` | mean `118.20`, min `-24.00`, max `248.00` |
| PPO | `algo=ppo exp=final_check_ppo_tuned` | mean `83.10`, min `-76.50`, max `171.50` |
| TRPO | `algo=trpo exp=final_check_trpo` | mean `57.05`, min `-80.00`, max `196.00` |
| SAC | `algo=sac exp=final_check_sac` | mean `90.70`, min `-23.50`, max `239.00` |

Interpretation: all six algorithms produced nontrivial policies, so the
single-agent pipeline is functional. Dueling DQN performed best in this
checkpoint comparison, consistent with the intuition that inventory states have
a strong state-value component and only some states require sharp action
distinctions. SAC and PPO were also competitive. Double DQN underperformed in
this run, which suggests that overestimation reduction alone was not the main
limiting factor for firm 5.

### Single-Agent Initial-Inventory and Firm-Position Trend

Command pattern for the trend study:

```bash
python3 main.py algo=dueling_dqn exp=<exp> algo.agent.firm_id=<firm_id> env.initial_inventory=<init> test.num_episodes=10
```

The `initial_inventory=10` checkpoints are stored in folders
`firm0`, `firm5`, and `firm8`. The `initial_inventory=0` and `100` checkpoints
are stored in folders with `_init0` and `_init100` suffixes.

| Initial inventory | Firm | Command override | Test result |
| --- | --- | --- | --- |
| `0` | `0` | `exp=firm0_init0 algo.agent.firm_id=0 env.initial_inventory=0` | mean `-1735.50`, min `-1826.50`, max `-1639.50` |
| `0` | `5` | `exp=firm5_init0 algo.agent.firm_id=5 env.initial_inventory=0` | mean `-1461.55`, min `-1615.50`, max `-1326.50` |
| `0` | `8` | `exp=firm8_init0 algo.agent.firm_id=8 env.initial_inventory=0` | mean `-1418.40`, min `-2971.00`, max `-596.00` |
| `10` | `0` | `exp=firm0 algo.agent.firm_id=0 env.initial_inventory=10` | mean `-420.60`, min `-707.00`, max `-224.00` |
| `10` | `5` | `exp=firm5 algo.agent.firm_id=5 env.initial_inventory=10` | mean `-513.15`, min `-763.50`, max `-316.50` |
| `10` | `8` | `exp=firm8 algo.agent.firm_id=8 env.initial_inventory=10` | mean `-174.15`, min `-403.00`, max `-45.50` |
| `100` | `0` | `exp=firm0_init100 algo.agent.firm_id=0 env.initial_inventory=100` | mean `701.30`, min `622.50`, max `817.50` |
| `100` | `5` | `exp=firm5_init100 algo.agent.firm_id=5 env.initial_inventory=100` | mean `64.25`, min `-155.50`, max `202.50` |
| `100` | `8` | `exp=firm8_init100 algo.agent.firm_id=8 env.initial_inventory=100` | mean `-142.85`, min `-262.50`, max `-77.50` |

Interpretation: `initial_inventory=100` favors downstream firms because the
initial stock is effectively free working capital. If a firm can satisfy demand
from this stock, downstream firm `0` earns price `10` per unit while firm `8`
earns price `2` per unit, but both pay the same holding cost rate. This creates
a much larger liquidation value for downstream inventory.

With `initial_inventory=0` or `10`, the system starts short of supply. Firm `0`
faces customer demand immediately, so stockouts and lost-sales penalties arrive
from the first step. More upstream firms are closer to the external source and
are less exposed to immediate exogenous demand. This reverses the trend: firm
`8` becomes much easier than firm `0`, especially for `initial_inventory=10`.
The `initial_inventory=10` result is not perfectly monotonic between firm `0`
and firm `5`, but it clearly favors the upstream firm `8`.

### MAPPO Results

MAPPO command for the main 3-firm run:

```bash
python3 main.py algo=mappo execution_mode=ctde_multi_agent exp=shared_3firms_300 train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,4,8]' algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

Result: final training mean `66.35`; test mean `200.88`, min `131.33`, max
`257.33`.

MAPPO command for the main 10-firm run:

```bash
python3 main.py algo=mappo execution_mode=ctde_multi_agent exp=shared_10firms_300 train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

Result: final training mean `-3900.90`; test mean `-4493.32`.

MAPPO tuned 10-firm run with lower actor LR, higher entropy, and order-10 bias:

```bash
python3 main.py algo=mappo execution_mode=ctde_multi_agent exp=shared_10firms_tuned_300 train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.rollout_steps=1000 algo.agent.minibatch_size=1024 algo.agent.ppo_epochs=2 algo.agent.actor_lr=0.0001 algo.agent.critic_lr=0.0005 algo.agent.entropy_coef=0.08 algo.agent.initial_order=10 algo.agent.initial_order_bias=1.0
```

Result: test mean `-4490.24`.

MAPPO stronger order-10 bias run:

```bash
python3 main.py algo=mappo execution_mode=ctde_multi_agent exp=shared_10firms_bias10_300 train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.rollout_steps=1000 algo.agent.minibatch_size=1024 algo.agent.ppo_epochs=2 algo.agent.actor_lr=0.00005 algo.agent.critic_lr=0.0005 algo.agent.entropy_coef=0.04 algo.agent.initial_order=10 algo.agent.initial_order_bias=3.0
```

Result: test mean `-4490.24`.

MAPPO conservative 3-firm tuned run:

```bash
python3 main.py algo=mappo execution_mode=ctde_multi_agent exp=shared_3firms_tuned_300 train.num_episodes=300 train.log_every=50 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,4,8]' algo.agent.rollout_steps=1000 algo.agent.minibatch_size=1024 algo.agent.ppo_epochs=2 algo.agent.actor_lr=0.0001 algo.agent.critic_lr=0.0005 algo.agent.entropy_coef=0.08 algo.agent.initial_order=10 algo.agent.initial_order_bias=1.0
```

Result: final training mean `-397.72`; test mean `-1011.02`.

Additional stored MAPPO checkpoint evaluations:

| Checkpoint evaluation target | Result |
| --- | --- |
| Load `models/mappo/3firm/mappo_final.pth`, `firm_ids=[0,4,8]`, `initial_inventory=100` | mean `274.22`, min `190.00`, max `344.50`; per-firm means `0: 796.10`, `4: 172.40`, `8: -145.85` |
| Load `models/mappo/3firm_init10/mappo_final.pth`, `firm_ids=[0,4,8]`, `initial_inventory=10` | mean `-340.05`, min `-483.17`, max `-212.33`; per-firm means `0: -398.55`, `4: -474.40`, `8: -147.20` |

MAPPO analysis: the 3-firm result is much better than the full 10-firm result
because uncontrolled random firms inject demand and break some feedback loops.
In the 10-firm run, all firms learn together. The shared actor can collapse to
low orders; once this happens, upstream demand disappears, upstream rewards
become poor, and the critic/actor updates reinforce the low-demand regime.
Simple global order-10 bias did not fix this because one shared actor cannot
represent the descending role-specific structure as easily, and because the
individual rewards still make orders look like local costs.

### HAPPO Results

HAPPO main 3-firm run:

```bash
python3 main.py algo=happo execution_mode=ctde_multi_agent exp=happo_3firms_300 train.num_episodes=300 train.log_every=10 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,4,8]' algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

Result: final training mean `-476.58`; test mean `-598.47`, min `-657.00`, max
`-548.00`.

HAPPO main 10-firm run:

```bash
PYTHONUNBUFFERED=1 python3 main.py algo=happo execution_mode=ctde_multi_agent exp=happo_10firms_300 train.num_episodes=300 train.log_every=10 train.checkpoint_every=999 test.num_episodes=10 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=4
```

Result: final training mean `-4094.10`; test mean `-4556.37`, min `-4563.10`,
max `-4547.55`. Greedy initial actions after training were mostly low:
`{0:1,1:1,2:1,3:1,4:1,5:3,6:1,7:2,8:1,9:1}`.

HAPPO pure team reward with order-10 prior:

```bash
PYTHONUNBUFFERED=1 python3 main.py algo=happo execution_mode=ctde_multi_agent exp=happo_10firms_team_200 train.num_episodes=200 train.log_every=20 train.checkpoint_every=999 test.num_episodes=5 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.individual_reward_weight=0.0 algo.agent.initial_order=10 algo.agent.initial_order_bias=2.0 algo.agent.entropy_coef=0.04 algo.agent.actor_lr=0.0002 algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=3
```

Result: test mean `-6257.74`, min `-6375.65`, max `-6086.65`. The greedy
policy stayed near all order `10` except one larger action. This was worse than
the low-order collapse, showing that pure team reward had too diffuse a credit
signal and that all-order-10 is not a good policy under `initial_inventory=100`.

HAPPO individual reward with order-5 prior:

```bash
PYTHONUNBUFFERED=1 python3 main.py algo=happo execution_mode=ctde_multi_agent exp=happo_10firms_indiv_order5_200 train.num_episodes=200 train.log_every=20 train.checkpoint_every=999 test.num_episodes=5 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.individual_reward_weight=1.0 algo.agent.initial_order=5 algo.agent.initial_order_bias=3.0 algo.agent.entropy_coef=0.01 algo.agent.actor_lr=0.0001 algo.agent.critic_lr=0.0005 algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=2
```

Result: test mean `-4052.58`; greedy actions all `5`.

HAPPO individual reward with order-9 prior:

```bash
PYTHONUNBUFFERED=1 python3 main.py algo=happo execution_mode=ctde_multi_agent exp=happo_10firms_indiv_order9_200 train.num_episodes=200 train.log_every=20 train.checkpoint_every=999 test.num_episodes=5 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.individual_reward_weight=1.0 algo.agent.initial_order=9 algo.agent.initial_order_bias=3.0 algo.agent.entropy_coef=0.01 algo.agent.actor_lr=0.0001 algo.agent.critic_lr=0.0005 algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=2
```

Result: test mean `-3799.05`; greedy actions all `9`.

HAPPO individual reward with descending per-firm order prior:

```bash
PYTHONUNBUFFERED=1 python3 main.py algo=happo execution_mode=ctde_multi_agent exp=happo_10firms_stair_indiv_200 train.num_episodes=200 train.log_every=20 train.checkpoint_every=999 test.num_episodes=5 multi_agent.firm_ids='[0,1,2,3,4,5,6,7,8,9]' algo.agent.individual_reward_weight=1.0 algo.agent.initial_orders='[10,9,8,7,6,5,4,3,2,1]' algo.agent.initial_order_bias=3.0 algo.agent.entropy_coef=0.01 algo.agent.actor_lr=0.0001 algo.agent.critic_lr=0.0005 algo.agent.rollout_steps=500 algo.agent.minibatch_size=512 algo.agent.ppo_epochs=2
```

Result: final training mean `-2513.75`; test mean `-1841.51`, min `-1987.20`,
max `-1669.50`. Greedy actions exactly matched the prior:
`{0:10,1:9,2:8,3:7,4:6,5:5,6:4,7:3,8:2,9:1}`.

One evaluated episode for the descending-prior policy had per-firm scores
`[-3694.5,-962.5,-1085.0,-1227.0,-1377.0,-1542.5,-1714.0,-1897.0,-2084.0,-2278.5]`
and mean score `-1786.20`. Mean orders were `[10,9,8,7,6,5,4,3,2,1]`; mean
demand was `[9.99,10,9,8,7,6,5,4,3,2]`.

Additional stored HAPPO checkpoint evaluations:

| Checkpoint evaluation target | Result |
| --- | --- |
| Load `models/happo/3firm_idcoef1/happo_final.pth`, `firm_ids=[0,4,8]`, `initial_inventory=100` | mean `297.25`, min `217.17`, max `354.17`; per-firm means `0: 787.70`, `4: 245.20`, `8: -141.15` |
| Load `models/happo/3firm_idcoef1_init10/happo_final.pth`, `firm_ids=[0,4,8]`, `initial_inventory=10` | mean `-482.28`, min `-589.83`, max `-385.00`; per-firm means `0: -718.05`, `4: -537.40`, `8: -191.40` |
| Load `models/happo/10firm_idcoef1/happo_final.pth`, `firm_ids=[0,1,2,3,4,5,6,7,8,9]`, `initial_inventory=100` | mean `-4490.81`, min `-4500.85`, max `-4483.35`; per-firm means `0: -808.10`, all other firms near `-4900.00` |
| Load `models/happo/10firm_idcoef1_init10/happo_final.pth`, `firm_ids=[0,1,2,3,4,5,6,7,8,9]`, `initial_inventory=10` | mean `-522.89`, min `-527.80`, max `-518.40`; per-firm means `0: -1628.90`, all other firms near `-400.00` |

HAPPO analysis: setting `individual_reward_weight=1.0` recovered good
performance in the 3-firm `initial_inventory=100` checkpoint, but it did not
solve the full 10-firm case. In the full chain, individual reward strongly
penalizes orders as local purchase costs, while the benefit of those orders
appears as upstream revenue. Thus each firm has an incentive to reduce its own
orders, even though the chain as a whole needs downstream orders to create
upstream demand and move inventory.

Pure team reward did not solve the problem either. It removes some selfish
incentive but makes credit assignment very diffuse: every actor receives nearly
the same noisy signal even though each action has a different role by position.
The team-reward run also showed that a uniform order-10 policy is not good when
the chain starts with 100 units of inventory per firm.

The descending-prior HAPPO run was the best full 10-firm MARL run, but it should
be interpreted carefully. The learned greedy policy mostly preserved the
provided prior. This proves that the environment can score much better than the
low-order trap with a role-specific policy, but it does not prove that HAPPO
discovered that policy from scratch.

### Static Policy Baselines

Static baselines were evaluated by replacing learned actors with fixed
deterministic orders in the environment.

| Policy | Test result |
| --- | --- |
| Uniform order `1` | mean about `-4490.08` |
| Uniform order `5` | mean about `-4054.54` |
| Uniform order `8` | mean about `-3783.31` |
| Uniform order `9` | mean about `-3778.12` |
| Uniform order `10` | mean about `-4010.50` |
| Descending `[10,9,8,7,6,5,4,3,2,1]` | mean about `-1767.58` |
| Descending `[9,8,7,6,5,4,3,2,1,1]` | mean about `-1836.73` |
| Shifted `[12,11,10,9,8,7,6,5,4,3]` | mean about `-2357.03` |

These baselines are important because they show that the `-4490` MAPPO/HAPPO
result is essentially a low-order trap, close to uniform order `1`. They also
show that a simple role-specific descending policy is much better than any
uniform constant order under `initial_inventory=100`.

## Analysis of the Main Trends

### Why `initial_inventory=100` Favors Downstream Firms

Large initial inventory changes the task from "build supply flow" to "monetize
existing stock without holding it too long." Since downstream firms sell at
higher prices, the same unit of starting stock has greater revenue potential
downstream. Holding cost is not price-scaled, so a unit held by firm `0` and a
unit held by firm `8` have the same holding-cost rate but very different sales
revenue.

This explains the Dueling DQN trend:

- firm `0`: mean `701.30`;
- firm `5`: mean `64.25`;
- firm `8`: mean `-142.85`.

It also explains why a descending order pattern can help the full-chain case.
With too much initial stock everywhere, a uniform high order keeps inventory
high. A descending order pattern creates downstream demand and upstream sales
while gradually reducing the excessive upstream stock burden.

### Why `initial_inventory=0` and `10` Favor Upstream Firms

With little or no initial inventory, the downstream side faces immediate
external demand but cannot satisfy it. That creates lost-sales penalties before
inventory has time to propagate. Upstream firms are closer to the external
source and are less exposed to exogenous customer demand at the beginning of an
episode.

The Dueling DQN results therefore reverse the `initial_inventory=100` ordering.
For `initial_inventory=0`, firm `8` is best among the three tested firms. For
`initial_inventory=10`, firm `8` is again much better than firm `0` or firm `5`.

The same mechanism appears in multi-agent checkpoint tests. Under
`initial_inventory=10`, a full 10-firm HAPPO individual-reward checkpoint had
mean score `-522.89`, much less negative than the corresponding
`initial_inventory=100` checkpoint mean `-4490.81`. This does not mean the
policy is globally optimal; it means the low-inventory regime changes the
baseline economics and reduces the huge holding-cost burden seen with 100 units
of initial inventory.

### Why Full 10-Firm MAPPO/HAPPO Is Mediocre

The central difficulty is the mismatch between individual incentives and chain
externalities. Firm `i`'s order $a_i$ enters its own reward as a cost
$-p_{i+1}a_i$. But that same order becomes demand for firm `i+1`, creating
upstream revenue. A locally selfish learner sees the cost immediately and does
not receive the upstream benefit directly.

In 3-firm experiments with `firm_ids=[0,4,8]`, uncontrolled random opponents
provide demand between controlled firms. This masks part of the externality
problem because the chain is not fully dependent on learned downstream orders.
In 10-firm experiments, every demand link is controlled by a learner. Once a
few policies reduce orders, upstream demand disappears, upstream firms receive
poor learning signals, and low-order behavior becomes self-reinforcing.

MAPPO's shared parameters help data efficiency but can hurt role specialization.
Although the firm index is included as an input, the policy still needs to learn
a strongly position-dependent order profile from sparse and nonstationary
feedback.

HAPPO improves the update structure by updating actors sequentially, but it
does not by itself solve reward design or credit assignment. The HAPPO factor
helps keep sequential policy updates consistent with the joint-policy update,
but the gradient is still driven by whatever reward signal is supplied. With
individual reward, the signal is selfish. With team reward, the signal is too
diffuse.

## What We Learned

Single-agent methods are suitable for studying local firm behavior and the
effect of environment parameters. Dueling DQN is a strong default for this
project's single-agent analyses.

MAPPO and HAPPO are implemented correctly enough to run controlled experiments,
but the current reward and exploration setup is not enough for full-chain
global optimization. The best 10-firm result came from injecting a descending
role-specific prior, not from discovering a globally good policy from scratch.

For a future formal study, the next algorithmic changes should target credit
assignment and global coordination directly. Candidates include difference
rewards, counterfactual baselines, value decomposition, explicit service-level
or inventory-balance shaping, curriculum learning from static baselines, or
behavior cloning/pretraining from good heuristic order profiles before MARL
fine-tuning.
