# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XR teleoperation package for Isaac Lab — enables VR controller-driven robot teleoperation and demonstration collection for imitation learning. Supports different robot morphologies (Franka manipulator, G1 humanoid).

## Setup and Installation

Requires conda environment with Python 3.11, IsaacSim 5.1.0, and two git submodules:
```bash
git submodule update --init --recursive
conda create -n isaaclab_teleop python=3.11
conda activate isaaclab_teleop
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
cd submodules/IsaacLab && ./isaaclab.sh --install && cd ../..
cd submodules/XRoboToolkit-PC-Service-Pybind && bash setup_ubuntu.sh && cd ../..
pip install -e .
```

## Running Tasks

Record Franka cube stacking demos:
```bash
python -m isaaclab_xr_teleop.tasks.franka_stack.record \
    --task Isaac-Stack-Cube-Franka-IK-Rel-XR-v0 \
    --teleop_device xr_controller --enable_cameras --device cpu
```

Record G1 pick-place demos:
```bash
python -m isaaclab_xr_teleop.tasks.g1_pick_place.record \
    --task Isaac-PickPlace-G1-InspireFTP-XR-v0 \
    --teleop_device xr_controller --enable_cameras --device cpu
```

Common args: `--step_hz 50`, `--num_demos N` (0=infinite), `--num_success_steps 10`, `--dataset_file ./datasets/dataset.hdf5`.

## Architecture

**Data flow pipeline:** `XRControllerDevice → Retargeters → Robot Commands → Isaac Lab Environment`

### Key Modules

- **`devices/`** — `XRControllerDevice` interfaces with XRoboToolkit SDK to capture VR controller poses, trigger/grip values, and button states. Supports `right_hand`, `left_hand`, and `dual_hand` control modes.

- **`retargeters/`** — Transform raw XR input into robot commands:
  - `XRSe3AbsRetargeter` — Direct absolute pose mapping
  - `XRSe3RelRetargeter` — Delta-based relative control with exponential smoothing and activation gating
  - `XRGripperRetargeter` — Continuous or binary gripper control with hysteresis
  - `g1/mink_ik.py` — Full-body IK solver using Mink/MuJoCo for G1 humanoid dual-arm control (runs IK in background thread)
  - `g1/inspire_hand.py` — 24-joint Inspire Hand retargeter (12 per hand)
  - `g1/robot_cfg.py` — Single source of truth for G1 joint names and limits

- **`recording/`** — Demonstration collection infrastructure:
  - `loop.py` — Rate-limited recording loop with state sync, success tracking, and HDF5 export
  - `callbacks.py` — VR button → recording action mapping (A=START, B=SAVE, X=RESET, Y=PAUSE)
  - `rate_limiter.py` — Frequency-enforcing loop controller

- **`tasks/`** — Task environment configurations. Each task defines its env_cfg (scene, observations, actions, rewards) and a `record.py` entry point. Tasks requiring IK state sync provide custom `sync_fn`/`reset_fn` callbacks.

### Patterns

- **Config-driven:** All components use `@configclass` dataclasses (IsaacLab pattern). Cfg objects define structure; runtime instantiation follows factory pattern.
- **Retargeter composition:** Tasks compose multiple retargeters (e.g., Mink IK + Inspire Hand for G1). The recording loop iterates retargeters to build the full action vector.
- **State sync callbacks:** Tasks with IK solvers pass `sync_fn` to feed measured joint positions back to retargeters each step, and `reset_fn` for episode resets.
- **AppLauncher-first imports:** IsaacSim requires `AppLauncher` to initialize before any Omniverse imports. All `record.py` files follow this pattern: parse args → launch app → import remaining modules.

## Submodules

- `submodules/IsaacLab` — NVIDIA's robotics simulation framework (core environments, device APIs, action managers)
- `submodules/XRoboToolkit-PC-Service-Pybind` — Python bindings for VR/AR controller access
