# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka Cube Stack environment with XRoboToolkit VR controller teleoperation."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_xr_teleop.devices import XRControllerDeviceCfg
from isaaclab_xr_teleop.retargeters import XRGripperRetargeterCfg, XRSe3RelRetargeterCfg

# Base config from isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeStackXREnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Configuration for Franka Cube Stack environment with XRoboToolkit VR controller teleoperation.

    This configuration uses XRoboToolkit SDK for VR controller input (Meta Quest, HTC Vive, etc.),
    enabling teleoperation with relative (delta-based) control for precise cube manipulation tasks.

    The controller uses:
    - Grip button: Activates end-effector pose control
    - Trigger button: Controls gripper open/close
    - Controller movement: Translates to robot end-effector deltas
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot with stiffer PD controller for IK tracking
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Configure teleoperation devices
        self.teleop_devices = DevicesCfg(
            devices={
                # XRoboToolkit VR controller device
                "xr_controller": XRControllerDeviceCfg(
                    control_mode="right_hand",
                    gripper_source="trigger",
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.0,
                    deadzone_threshold=0.01,
                    retargeters=[
                        XRSe3RelRetargeterCfg(
                            control_hand="right",
                            pos_scale_factor=10.0,
                            rot_scale_factor=10.0,
                            activation_source="grip",
                            activation_threshold=0.9,
                            alpha_pos=0.9,
                            alpha_rot=0.9,
                            zero_out_xy_rotation=False,
                            enable_visualization=False,
                            sim_device=self.sim.device,
                        ),
                        XRGripperRetargeterCfg(
                            control_hand="right",
                            input_source="trigger",
                            mode="continuous",
                            binary_threshold=0.5,
                            invert=True,
                            open_value=1.0,
                            closed_value=-1.0,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                ),
                # Keyboard as fallback
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
