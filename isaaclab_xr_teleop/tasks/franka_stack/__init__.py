# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka cube stack XR teleoperation task."""

import gymnasium as gym

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-XR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:FrankaCubeStackXREnvCfg"},
    disable_env_checker=True,
)
