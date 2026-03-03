# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared recording loop parameterized by task-specific callbacks."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from .callbacks import RecordingState, make_recording_callbacks
from .rate_limiter import RateLimiter

# Optional InstructionDisplay with no-op fallback
try:
    from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions
    _HAS_INSTRUCTION_DISPLAY = True
except ImportError:
    _HAS_INSTRUCTION_DISPLAY = False

    class InstructionDisplay:
        def __init__(self, *a, **kw):
            pass

        def show_subtask(self, *a, **kw):
            pass

        def show_demo(self, *a, **kw):
            pass

        def set_labels(self, *a, **kw):
            pass

    def show_subtask_instructions(*a, **kw):
        pass


def get_robot_state_for_twist(robot: object, env_idx: int = 0) -> dict[str, np.ndarray]:
    """Get robot state dictionary for TWIST retargeter."""
    joint_pos = robot.data.joint_pos[env_idx].cpu().numpy()
    joint_vel = robot.data.joint_vel[env_idx].cpu().numpy()

    base_quat_wxyz = robot.data.root_quat_w[env_idx].cpu().numpy()
    base_quat_xyzw = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])

    base_ang_vel = robot.data.root_ang_vel_w[env_idx].cpu().numpy()

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "base_quat": base_quat_xyzw,
        "base_ang_vel": base_ang_vel,
    }


def sync_twist_retargeter_state(teleop_interface: object, robot: object, env_idx: int = 0):
    """Sync robot state to TWIST retargeter(s)."""
    if not hasattr(teleop_interface, '_retargeters'):
        return

    robot_state = get_robot_state_for_twist(robot, env_idx)

    for retargeter in teleop_interface._retargeters:
        if hasattr(retargeter, 'set_robot_state'):
            retargeter.set_robot_state(robot_state)


def setup_output_directories(args_cli: argparse.Namespace) -> tuple[str, str]:
    """Set up output directories for saving demonstrations."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"{output_file_name}_{timestamp}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Dataset will be saved as: {output_file_name}.hdf5")

    return output_dir, output_file_name


def create_environment_config(
    args_cli: argparse.Namespace,
    parse_env_cfg_fn: Callable,
    output_dir: str,
    output_file_name: str,
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration."""
    env_cfg = parse_env_cfg_fn(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Extract success term
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.sim.render.antialiasing_mode = "DLSS"
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def process_success_condition(
    env: gym.Env,
    success_term: object | None,
    success_step_count: int,
    num_success_steps: int,
) -> tuple[int, bool]:
    """Process the success condition for the current step."""
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env,
    instruction_display: InstructionDisplay,
    label_text: str,
    teleop_interface: object | None = None,
    robot: object | None = None,
    arm_joint_ids: list | None = None,
    has_twist_retargeter: bool = False,
) -> None:
    """Handle resetting the environment."""
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    instruction_display.show_demo(label_text)

    if teleop_interface is not None and robot is not None and arm_joint_ids is not None:
        measured_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
        if hasattr(teleop_interface, 'set_measured_joint_positions'):
            teleop_interface.set_measured_joint_positions(measured_joint_pos)
        if hasattr(teleop_interface, '_retargeters'):
            for retargeter in teleop_interface._retargeters:
                if hasattr(retargeter, 'reset'):
                    if hasattr(retargeter, 'set_robot_state'):
                        robot_state = get_robot_state_for_twist(robot, env_idx=0)
                        retargeter.reset(robot_state=robot_state)
                    else:
                        try:
                            retargeter.reset(joint_positions=measured_joint_pos.cpu().numpy())
                        except TypeError:
                            retargeter.reset()
                    print(f"  Reset {type(retargeter).__name__} with state sync")

    if has_twist_retargeter and teleop_interface is not None and robot is not None:
        sync_twist_retargeter_state(teleop_interface, robot, env_idx=0)
        print("  TWIST retargeter state synchronized")


def run_recording_loop(
    env: gym.Env,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    args_cli: argparse.Namespace,
    simulation_app: object,
    sync_fn: Callable | None = None,
    reset_fn: Callable | None = None,
    arm_joint_names: list[str] | None = None,
) -> int:
    """Run the main recording loop.

    Args:
        env: The gymnasium environment.
        env_cfg: The environment config (needed for teleop_devices).
        args_cli: Parsed CLI arguments.
        simulation_app: The Isaac Sim application object.
        sync_fn: Optional task-specific function called each step to sync retargeter state.
            Signature: sync_fn(env, teleop_interface)
        reset_fn: Optional task-specific function called on reset.
            Signature: reset_fn(env, teleop_interface)
        arm_joint_names: Optional list of arm joint names for Mink IK state sync.

    Returns:
        Number of successful demonstrations recorded.
    """
    rec_state = RecordingState()
    callbacks = make_recording_callbacks(env, rec_state)

    # Create teleop interface
    if not hasattr(env_cfg, "teleop_devices") or args_cli.teleop_device not in env_cfg.teleop_devices.devices:
        print(f"No '{args_cli.teleop_device}' found in environment config.")
        return 0

    teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
    if teleop_interface is None:
        print(f"Failed to create {args_cli.teleop_device} interface")
        return 0

    teleop_interface.add_callback("R", callbacks["R"])

    # Setup for state synchronization
    arm_joint_ids = None
    robot = None
    has_twist_retargeter = False
    if "robot" in env.scene.keys():
        robot = env.scene["robot"]
        if arm_joint_names:
            try:
                arm_joint_ids = robot.find_joints(arm_joint_names, preserve_order=True)[0]
                print(f"State synchronization enabled for {len(arm_joint_names)} arm joints")
            except Exception as e:
                print(f"Could not find arm joints for state sync: {e}. State sync disabled.")
                arm_joint_ids = None

        if hasattr(teleop_interface, '_retargeters'):
            for retargeter in teleop_interface._retargeters:
                if hasattr(retargeter, 'set_robot_state'):
                    has_twist_retargeter = True

    # Rate limiter
    rate_limiter = RateLimiter(hz=args_cli.step_hz)

    # Initial reset
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    # Sync device state on initial reset
    if arm_joint_ids is not None and robot is not None:
        measured_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
        if hasattr(teleop_interface, 'set_measured_joint_positions'):
            teleop_interface.set_measured_joint_positions(measured_joint_pos)
        if hasattr(teleop_interface, '_retargeters'):
            for retargeter in teleop_interface._retargeters:
                if hasattr(retargeter, 'reset'):
                    if hasattr(retargeter, 'set_robot_state'):
                        robot_state = get_robot_state_for_twist(robot, env_idx=0)
                        retargeter.reset(robot_state=robot_state)
                    else:
                        try:
                            retargeter.reset(joint_positions=measured_joint_pos.cpu().numpy())
                        except TypeError:
                            retargeter.reset()

    if has_twist_retargeter and robot is not None:
        sync_twist_retargeter_state(teleop_interface, robot, env_idx=0)

    current_recorded_demo_count = 0
    success_step_count = 0
    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."

    # Setup UI
    try:
        from isaaclab.envs.ui import EmptyWindow
        import omni.ui as ui
        instruction_display = InstructionDisplay(xr=False)
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)
    except Exception:
        instruction_display = InstructionDisplay()

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Task-specific sync (e.g., Mink IK state sync)
            if sync_fn is not None:
                sync_fn(env, teleop_interface)
            elif arm_joint_ids is not None and robot is not None:
                # Default sync: feed measured joint positions
                measured_joint_pos = robot.data.joint_pos[0, arm_joint_ids]
                if hasattr(teleop_interface, 'set_measured_joint_positions'):
                    teleop_interface.set_measured_joint_positions(measured_joint_pos)

            if has_twist_retargeter and robot is not None:
                sync_twist_retargeter_state(teleop_interface, robot, env_idx=0)

            action = teleop_interface.advance()

            if action is None:
                env.sim.render()
                continue

            actions = action.repeat(env.num_envs, 1)
            obv = env.step(actions)

            if not rec_state.is_running:
                env.recorder_manager.get_episode(0).data.clear()

            if subtasks is not None:
                if subtasks == {}:
                    subtasks = obv[0].get("subtask_terms")
                elif subtasks:
                    show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)

            success_step_count, success_reset_needed = process_success_condition(
                env, success_term=getattr(run_recording_loop, '_success_term', None),
                success_step_count=success_step_count,
                num_success_steps=args_cli.num_success_steps,
            )
            if success_reset_needed:
                rec_state.should_reset = True

            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    rate_limiter.sleep(env)
                break

            if rec_state.should_reset:
                handle_reset(
                    env, instruction_display, label_text,
                    teleop_interface, robot, arm_joint_ids, has_twist_retargeter
                )
                if reset_fn is not None:
                    reset_fn(env, teleop_interface)
                rec_state.should_reset = False
                success_step_count = 0

            if env.sim.is_stopped():
                break

            rate_limiter.sleep(env)

    return current_recorded_demo_count
