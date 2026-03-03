# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Recording entry point for G1 pick-place XR teleoperation.

Example usage:
    python -m isaaclab_xr_teleop.tasks.g1_pick_place.record \
        --task Isaac-PickPlace-G1-InspireFTP-XR-v0 \
        --teleop_device xr_controller --enable_cameras --device cpu
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demonstrations for G1 pick-place with XR controller.")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-InspireFTP-XR-v0", help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="xr_controller", help="Device for teleoperation.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=50, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record (0=infinite).")
parser.add_argument("--num_success_steps", type=int, default=10, help="Number of continuous success steps.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

# Register our task environments
import isaaclab_xr_teleop.tasks.g1_pick_place  # noqa: F401

from isaaclab_xr_teleop.retargeters.g1.robot_cfg import G1_ARM_JOINT_NAMES
from isaaclab_xr_teleop.recording.loop import (
    create_environment_config,
    run_recording_loop,
    setup_output_directories,
)


def sync_fn(env, teleop_interface):
    """G1-specific per-step sync: feed measured arm joints to Mink IK retargeter."""
    robot = env.scene["robot"]

    arm_joint_ids = getattr(sync_fn, '_arm_joint_ids', None)
    if arm_joint_ids is None:
        try:
            arm_joint_ids = robot.find_joints(G1_ARM_JOINT_NAMES, preserve_order=True)[0]
            sync_fn._arm_joint_ids = arm_joint_ids
        except Exception:
            return

    measured_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
    if hasattr(teleop_interface, 'set_measured_joint_positions'):
        teleop_interface.set_measured_joint_positions(measured_joint_pos)

    # Copy raw gripper command onto env for the observation term
    for retargeter in teleop_interface._retargeters:
        if hasattr(retargeter, 'raw_gripper_command'):
            env.gripper_command = retargeter.raw_gripper_command.clone()
            break


def reset_fn(env, teleop_interface):
    """G1-specific reset: additional retargeter state sync if needed."""
    pass  # handle_reset in loop.py already syncs retargeters


def main():
    # Use isaaclab_tasks parse_env_cfg since it handles gym registration
    try:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    except ImportError:
        # Fallback: manual parse
        from isaaclab.envs import ManagerBasedRLEnvCfg

        def parse_env_cfg(task, device="cpu", num_envs=1):
            spec = gym.spec(task)
            cfg = spec.kwargs["env_cfg_entry_point"]()
            cfg.sim.device = device
            cfg.scene.num_envs = num_envs
            return cfg

    output_dir, output_file_name = setup_output_directories(args_cli)
    env_cfg, success_term = create_environment_config(args_cli, parse_env_cfg, output_dir, output_file_name)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Stash success_term on the loop function so it can access it
    run_recording_loop._success_term = success_term

    count = run_recording_loop(
        env=env,
        env_cfg=env_cfg,
        args_cli=args_cli,
        simulation_app=simulation_app,
        sync_fn=sync_fn,
        reset_fn=reset_fn,
        arm_joint_names=G1_ARM_JOINT_NAMES,
    )

    env.close()
    print(f"Recording session completed with {count} successful demonstrations")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
