# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Franka Cube Stack environment with XRoboToolkit VR controller teleoperation."""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Base config from isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_joint_pos_env_cfg

from isaaclab_xr_teleop.devices import XRControllerDeviceCfg
from isaaclab_xr_teleop.retargeters import XRGripperRetargeterCfg, XRSe3RelRetargeterCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state and camera terms."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        wrist_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


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

    observations: ObservationsCfg = ObservationsCfg()

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

        # Wrist camera (mounted on panda_hand)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
        )

        # Table overview camera (fixed)
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.35, 0.0, 0.75), rot=(0.32649, -0.62721, -0.62721, 0.32649), convention="ros"
            ),
        )

        # Render quality
        self.sim.render.antialiasing_mode = "DLAA"
        self.num_rerenders_on_reset = 3
