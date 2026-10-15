# RLMind

RLMind is sort of my personal take on [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak), just prioritising and fixing parts that were unfinished before the leak, along with a few changes of my own.

I liked GigaLearnCPP to a certain extent, but I thought it'd be nice to take it upon myself & modify the framework without losing what made it good in the first place. This goes beyond simple bug fixes and restructures quite a bit of the project while preserving lots of the original files.

The biggest change is moving from [RocketSim 2.2.1](https://github.com/ZealanL/RocketSim/releases/tag/v2.2.1) to [RocketSim v3-rust](https://github.com/ZealanL/RocketSim/tree/v3-rust).

## So what's different?

- This uses RocketSim v3-rust instead of 2.2.1
- Quite a fair few commonly known bug fixes
- [nanobind](https://github.com/wjakob/nanobind) instead of pybind11 for the Python/C++ bindings
- A much simpler project layout; Python for scripts/configs, and C++ for the *s p e e d !*
- CUDA graphs for policy inference
- A new RLBot implementation for loading trained checkpoints and playing through **RLBot v4** (not v5; I'm not the biggest fan here).

## Requirements

RLMind is currently only built for 64-bit Windows, so you'll need:

- Python 3.11
- VS2022 w/ C++ build tools
- Rust 1.95+
- CMake & Ninja
- An NVIDIA GPU for CUDA training (CPU also works, just a lot slower)

The Python packages are in `requirements.txt`. You'll also need Rocket League's `collision_meshes`, which you can get using [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper).

## Credits

Most of this project wouldn't exist without [ZealanL](https://github.com/ZealanL)'s original [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) and [RocketSim](https://github.com/ZealanL/RocketSim), along with [VirxEC](https://github.com/VirxEC)'s work on RocketSim.

The environment code also uses parts of [RLGym Sim](https://github.com/AechPro/rocket-league-gym-sim), and the thread pool is by Paul Tsouchlos.

RLMind is an unofficial personal rework, so it's not an official continuation of GGL.