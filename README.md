# Project for 2026Spring Multiagent Course

## Configuration

Training is configured with Hydra. The base config lives at `cfg/base.yaml`, and
algorithm configs live under `cfg/algo`.

```bash
python main.py algo=ppo exp=my_run
```

`dqn`, `double_dqn`, `dueling_dqn`, `ppo`, `trpo`, and `sac` are implemented with
the same `BaseAgent` API, so `train()` and `test()` work across algorithms.

Models are saved to `models/{algo}/{exp}` and figures to `figures/{algo}/{exp}`.
