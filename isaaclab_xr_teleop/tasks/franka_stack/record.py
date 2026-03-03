# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Recording entry point for Franka cube stack XR teleoperation.

Franka uses relative SE3 control — no Mink IK state sync needed.

Example usage:
    python -m isaaclab_xr_teleop.tasks.franka_stack.record \
        --task Isaac-Stack-Cube-Franka-IK-Rel-XR-v0 \
        --teleop_device xr_controller --enable_cameras --device cpu
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demonstrations for Franka stack with XR controller.")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-IK-Rel-XR-v0", help="Name of the task.")
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
import isaaclab_xr_teleop.tasks.franka_stack  # noqa: F401

from isaaclab_xr_teleop.recording.loop import (
    create_environment_config,
    run_recording_loop,
    setup_output_directories,
)


def main():
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    output_dir, output_file_name = setup_output_directories(args_cli)
    env_cfg, success_term = create_environment_config(args_cli, parse_env_cfg, output_dir, output_file_name)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Stash success_term on the loop function so it can access it
    run_recording_loop._success_term = success_term

    # Franka uses relative SE3 control — no joint position sync needed
    count = run_recording_loop(
        env=env,
        env_cfg=env_cfg,
        args_cli=args_cli,
        simulation_app=simulation_app,
        sync_fn=None,
        reset_fn=None,
        arm_joint_names=None,
    )

    env.close()
    print(f"Recording session completed with {count} successful demonstrations")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
