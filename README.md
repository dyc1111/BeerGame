# Project for 2026Spring Multiagent Course

## Configuration

Training is configured with Hydra. The base config lives at `cfg/base.yaml`, and
algorithm configs live under `cfg/algo`.

```bash
python main.py algo=ppo exp=my_run
```

`dqn`, `double_dqn`, `dueling_dqn`, `ppo`, and `trpo` are implemented. `sac` is
registered with an explicit placeholder so future implementations can plug into the
same `BaseAgent` API without changing `train()` or `test()`.

Models are saved to `models/{algo}/{exp}` and figures to `figures/{algo}/{exp}`.
