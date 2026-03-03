# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 Inspire pick place with XR controller teleoperation."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip

from isaaclab_xr_teleop.devices import XRControllerDeviceCfg
from isaaclab_xr_teleop.retargeters.g1 import XRG1MinkIKRetargeterCfg, XRInspireHandRetargeterCfg
from isaaclab_xr_teleop.retargeters.g1.robot_cfg import G1_ARM_JOINT_NAMES, G1_HAND_JOINT_NAMES

from . import mdp as pick_place_mdp


##
# Scene definition
##
@configclass
class G1XRSceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 XR teleoperation environment."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.9996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(
                mass=0.05,
            ),
        ),
    )

    # Humanoid robot with fixed root (manipulation task)
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 1.05),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # Arms at rest
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # Fixed body joints
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                # Hands open
                ".*_thumb_.*": 0.0,
                ".*_index_.*": 0.0,
                ".*_middle_.*": 0.0,
                ".*_ring_.*": 0.0,
                ".*_pinky_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Head camera (RealSense D435i mounted on torso/head)
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",
        update_period=0.0,
        height=240,
        width=424,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.88,
            focus_distance=400.0,
            horizontal_aperture=3.74,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.9848078, 0.0, -0.1736482, 0.0),
            convention="world",
        ),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for XR teleop - direct joint position control."""

    # Arm joint position control (14 joints)
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_ARM_JOINT_NAMES,
        preserve_order=True,
    )

    # Hand joint position control (24 joints)
    hand_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_HAND_JOINT_NAMES,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=pick_place_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=pick_place_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=pick_place_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=pick_place_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=pick_place_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=pick_place_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=pick_place_mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})

        raw_gripper_command = ObsTerm(func=pick_place_mdp.raw_gripper_command)

        head_cam_rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("head_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        object = ObsTerm(
            func=pick_place_mdp.object_obs,
            params={"left_eef_link_name": "left_wrist_yaw_link", "right_eef_link_name": "right_wrist_yaw_link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=pick_place_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=pick_place_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=pick_place_mdp.task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=pick_place_mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=pick_place_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.025, 0.025],
                "y": [-0.025, 0.025],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    randomize_arm_joints = EventTerm(
        func=pick_place_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINT_NAMES),
        },
    )


@configclass
class PickPlaceG1InspireFTPXREnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for G1 Inspire pick place with XR controller teleoperation."""

    # Scene settings
    scene: G1XRSceneCfg = G1XRSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Idle action to hold robot in default pose (38 joints: 14 arm + 24 hand)
    idle_action = torch.zeros(38)

    def __post_init__(self):
        """Post initialization - configure simulation and XR teleop devices."""
        # General settings
        self.decimation = 10
        self.episode_length_s = 20.0

        # Simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = 5

        # Configure XR controller teleoperation with Mink IK retargeter
        self.teleop_devices = DevicesCfg(
            devices={
                "xr_controller": XRControllerDeviceCfg(
                    control_mode="dual_hand",
                    pos_sensitivity=1.0,
                    rot_sensitivity=1.0,
                    deadzone_threshold=0.01,
                    retargeters=[
                        XRG1MinkIKRetargeterCfg(
                            headless=True,
                            reference_frame="trunk",
                            sim_device=self.sim.device,
                        ),
                        XRInspireHandRetargeterCfg(
                            hand_joint_names=G1_HAND_JOINT_NAMES,
                            mode="continuous",
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                ),
            }
        )
