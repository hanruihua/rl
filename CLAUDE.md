# CLAUDE.md

## Project Overview

This is the **TorchRL** source repository (pytorch/rl) — the official PyTorch reinforcement learning library. Version 0.11.0 (dev). MIT license.

**Purpose:** I'm reading this codebase to learn how to build custom TorchRL environments and training pipelines for my IR-SIM multi-robot project (see `/Users/han/tech/MMPC/`).

## Key Directories for Learning

```
torchrl/envs/common.py          # EnvBase — the base class I need to subclass
torchrl/envs/custom/            # Example custom envs (Pendulum, Chess, TicTacToe)
torchrl/envs/utils.py           # check_env_specs, make_composite_from_td
torchrl/envs/transforms/        # Transforms (normalization, frame stacking, etc.)

torchrl/objectives/ppo.py       # ClipPPOLoss — the PPO loss I'll use
torchrl/objectives/value/advantages.py  # GAE advantage estimation

torchrl/collectors/_single.py   # SyncDataCollector for gathering rollouts
torchrl/modules/tensordict_module/actors.py  # ProbabilisticActor, ValueOperator

sota-implementations/ppo/       # Complete PPO reference implementation
tutorials/                      # Jupyter notebook tutorials
```

## Building a Custom EnvBase

To create a custom environment, subclass `torchrl.envs.EnvBase` and implement:

| Method | Purpose |
|--------|---------|
| `_reset(self, tensordict)` | Reset env, return initial observation as TensorDict |
| `_step(self, tensordict)` | Take action, return next obs + reward + done as TensorDict |
| `_make_spec(self)` | Define observation_spec, action_spec, reward_spec, done_spec |
| `_set_seed(self, seed)` | Set RNG seed |

Validate with: `check_env_specs(env)` and `env.rollout(max_steps=100)`

Reference implementations:
- `torchrl/envs/custom/pendulum.py` — simplest example (stateless physics)
- `torchrl/envs/gym_like.py` — wrapping stateful simulators (closer to IR-SIM use case)

## PPO Training Pipeline Pattern

```
IRSimEnv(EnvBase)
    → SyncDataCollector (gather rollouts)
    → ReplayBuffer (store transitions)
    → ClipPPOLoss + GAE (compute loss)
    → optimizer.step()
```

See `sota-implementations/ppo/` for full working example with Hydra configs.

## Build & Run

```bash
# Dev install
pip install -e . --no-build-isolation

# Build C++ extensions
python setup.py build_ext --inplace

# Run tests
make test

# Quick smoke test
python test/smoke_test.py
```

## Dependencies

Core: `torch>=2.1.0`, `tensordict>=0.11.0,<0.12.0`, `numpy`, `cloudpickle`

Python >= 3.10 required (3.11+ recommended)

## Notes

- This is a **read-only reference** — I'm not modifying TorchRL source
- Focus on understanding `EnvBase` contract and PPO training loop
- My custom env code lives in `/Users/han/tech/MMPC/torchrl/`
