# Project for 2026Spring Multiagent Course

## Algorithm selection

Training is configured from `CONFIG` in `main.py`. Change the `algo` field to switch
between registered algorithms:

```python
CONFIG = {
    "algo": "dqn",  # dqn | double_dqn | ppo | trpo | sac
    ...
}
```

`dqn` and `double_dqn` are implemented. `ppo`, `trpo`, and `sac` are registered with
explicit placeholders so future implementations can plug into the same `BaseAgent`
API without changing `train()` or `test()`.
