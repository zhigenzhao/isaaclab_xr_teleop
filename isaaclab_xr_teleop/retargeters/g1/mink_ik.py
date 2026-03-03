# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter using Mink IK for G1 humanoid arm control."""

import numpy as np
import time
import torch
from dataclasses import dataclass
from importlib.resources import files
from scipy.spatial.transform import Rotation
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

from .robot_cfg import G1_ARM_JOINT_NAMES

# Default coordinate transformation from headset frame to world frame
R_HEADSET_TO_WORLD = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])

# Constants for state synchronization
QUAT_NORM_THRESHOLD = 1e-6  # Minimum quaternion norm for validation
SYNC_TIMEOUT_SECONDS = 0.1  # Timeout for waiting on state sync completion
SYNC_POLL_INTERVAL_SECONDS = 0.001  # Polling interval for sync completion check

# Import Mink IK dependencies
try:
    import mujoco as mj
    import mink as ik
    import threading
    from loop_rate_limiters import RateLimiter
    MINK_AVAILABLE = True
except ImportError:
    MINK_AVAILABLE = False
    print("Warning: mink, mujoco, or loop_rate_limiters not available. XRG1MinkIKRetargeter will not function.")


def _default_xml_path() -> str:
    """Get the default path to the bundled G1 upper body IK MuJoCo XML."""
    return str(files("isaaclab_xr_teleop.retargeters.g1") / "data" / "mjcf" / "scene_g1_upper_ik.xml")


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(mat)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    else:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = quat
    # Normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return mat


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


class XRG1MinkIKRetargeter(RetargeterBase):
    """Retargets XR controller poses to G1 humanoid arm joint positions using Mink IK.

    This retargeter creates a MuJoCo-based IK solver running in a separate thread
    that continuously solves for G1 arm joint positions to match XR controller target poses.

    The retargeter:
    - Takes left/right controller poses from XR device
    - Runs Mink IK solver to compute G1 arm joint angles
    - Returns arm joint positions for Isaac Lab simulation
    - Handles activation/deactivation via grip buttons

    Output format: 14-element tensor with arm joint positions
        - Elements 0-6: Left arm joints (shoulder_pitch, shoulder_roll, shoulder_yaw,
                        elbow, wrist_roll, wrist_pitch, wrist_yaw)
        - Elements 7-13: Right arm joints (same order as left)

    Hand control is handled separately by XRInspireHandRetargeter.
    """

    def __init__(self, cfg: "XRG1MinkIKRetargeterCfg"):
        """Initialize the G1 Mink IK retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        if not MINK_AVAILABLE:
            raise RuntimeError("Mink IK dependencies not available. Install mink, mujoco, and loop-rate-limiters.")

        self._sim_device = cfg.sim_device
        self._xml_path = cfg.xml_path
        self._headless = cfg.headless
        self._ik_rate_hz = cfg.ik_rate_hz

        # Initialize MuJoCo model
        self.mj_model = mj.MjModel.from_xml_path(self._xml_path)
        self.mj_data = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Initialize Mink IK configuration
        self.configuration = ik.Configuration(self.mj_model, self.mj_model.keyframe("home").qpos)

        # Define IK tasks for G1 robot
        # Build per-joint posture costs: higher for shoulders, zero for wrists.
        posture_costs = np.zeros(self.mj_model.nv)
        for i, jnt_name in enumerate(G1_ARM_JOINT_NAMES):
            jnt_id = self.mj_model.joint(jnt_name).id
            if "shoulder" in jnt_name:
                posture_costs[jnt_id] = cfg.posture_cost_shoulder
            elif "elbow" in jnt_name:
                posture_costs[jnt_id] = cfg.posture_cost_elbow
            else:  # wrist joints
                posture_costs[jnt_id] = cfg.posture_cost_wrist
        self.posture_task = ik.PostureTask(self.mj_model, cost=posture_costs)
        self.lh_task = ik.FrameTask(
            frame_name="left_hand",
            frame_type="site",
            position_cost=6.0,
            orientation_cost=2.0,
            lm_damping=0.03,
        )
        self.rh_task = ik.FrameTask(
            frame_name="right_hand",
            frame_type="site",
            position_cost=6.0,
            orientation_cost=2.0,
            lm_damping=0.03,
        )
        self.damping_task = ik.DampingTask(
            self.mj_model,
            cost=np.array([0.1] * self.mj_model.nv)
        )
        self.tasks = [self.posture_task, self.lh_task, self.rh_task, self.damping_task]

        # Define IK limits for G1 joints
        factor = cfg.velocity_limit_factor
        self.limits = [
            ik.ConfigurationLimit(self.mj_model, min_distance_from_limits=0.1),
            ik.VelocityLimit(
                self.mj_model,
                {
                    "left_shoulder_pitch_joint": 5.0 * factor,
                    "left_shoulder_roll_joint": 5.0 * factor,
                    "left_shoulder_yaw_joint": 5.0 * factor,
                    "left_elbow_joint": 5.0 * factor,
                    "left_wrist_roll_joint": 2.5,
                    "left_wrist_pitch_joint": 2.5,
                    "left_wrist_yaw_joint": 2.5,
                    "right_shoulder_pitch_joint": 5.0 * factor,
                    "right_shoulder_roll_joint": 5.0 * factor,
                    "right_shoulder_yaw_joint": 5.0 * factor,
                    "right_elbow_joint": 5.0 * factor,
                    "right_wrist_roll_joint": 2.5,
                    "right_wrist_pitch_joint": 2.5,
                    "right_wrist_yaw_joint": 2.5,
                }
            ),
        ]

        # Thread synchronization
        self.datalock = threading.RLock()
        self.is_running = False
        self.is_ready = False
        self.is_solving = True
        self.shutdown_requested = False

        # Viewer setup
        if not self._headless:
            import mujoco.viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            self.viewer = None

        # Mocap tracking state
        self.synced_mocap = {}
        self.lhold = False
        self.rhold = False

        # Measured joint positions from simulation (for state sync)
        self.measured_joint_positions = None
        self.force_sync = False
        self.sync_complete = True

        # Reference frame for relative control
        self._reference_frame = cfg.reference_frame

        # Motion tracker configuration and state
        self._motion_tracker_config = cfg.motion_tracker_config
        self._motion_tracker_task_weight = cfg.motion_tracker_task_weight
        self._arm_length_scale_factor = cfg.arm_length_scale_factor
        self.tracker_tasks = {}

        # Setup motion tracker tasks if configured
        if self._motion_tracker_config:
            print(f"[XRG1MinkIKRetargeter] Motion tracker config: {self._motion_tracker_config}")
            self._setup_motion_tracker_tasks()
        else:
            print(f"[XRG1MinkIKRetargeter] No motion tracker configured")

        # Start IK solver thread
        self._start_ik()

    def __del__(self):
        """Destructor to clean up IK solver thread."""
        self._stop_ik()

    def _setup_motion_tracker_tasks(self):
        """Setup Mink IK position tasks for motion trackers."""
        with self.datalock:
            for arm_name, tracker_config in self._motion_tracker_config.items():
                link_target = tracker_config["link_target"]
                serial = tracker_config["serial"]

                if arm_name == "left_arm":
                    mocap_name = "left_elbow_target"
                elif arm_name == "right_arm":
                    mocap_name = "right_elbow_target"
                else:
                    print(f"Warning: Unknown arm name '{arm_name}'. Skipping motion tracker setup.")
                    continue

                try:
                    self.mj_model.site(link_target).id
                except KeyError:
                    print(f"Warning: Motion tracker link site '{link_target}' not found in model. Skipping.")
                    continue

                ik.move_mocap_to_frame(self.mj_model, self.mj_data, mocap_name, link_target, "site")

                tracker_task = ik.FrameTask(
                    frame_name=link_target,
                    frame_type="site",
                    position_cost=self._motion_tracker_task_weight,
                    orientation_cost=0.0,
                    lm_damping=0.03,
                )

                mocap_se3 = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, mocap_name)
                tracker_task.set_target(mocap_se3)

                self.tracker_tasks[arm_name] = tracker_task
                self.tasks.append(tracker_task)

                print(f"Motion tracker task created: {arm_name} -> {link_target} (serial: {serial}, mocap: {mocap_name})")

    def _start_ik(self):
        """Start the IK solver thread."""
        if self.is_running:
            return
        self.is_running = True
        self.is_ready = False
        self.ik_thread = threading.Thread(target=self._solve_ik_loop, name="G1MinkIKThread")
        self.ik_thread.daemon = True
        self.ik_thread.start()

    def _stop_ik(self):
        """Stop the IK solver thread."""
        print("Stopping G1 Mink IK solver...")
        if hasattr(self, 'shutdown_requested'):
            self.shutdown_requested = True
        if hasattr(self, 'is_running'):
            self.is_running = False

        if hasattr(self, 'ik_thread') and self.ik_thread.is_alive():
            self.ik_thread.join(timeout=2.0)
            if self.ik_thread.is_alive():
                print("Warning: G1 Mink IK thread did not stop cleanly")

        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass

    def _solve_ik_loop(self):
        """Main IK solver loop running in separate thread."""
        mj.mj_forward(self.mj_model, self.mj_data)
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")

        rate = RateLimiter(self._ik_rate_hz)

        while self.is_running and not self.shutdown_requested:
            with self.datalock:
                if self.force_sync and self.measured_joint_positions is not None:
                    self.set_qpos_arm(self.measured_joint_positions)
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")
                    self.force_sync = False
                    self.sync_complete = True

                lh_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, "left_hand_target")
                rh_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, "right_hand_target")

                self.posture_task.set_target_from_configuration(self.configuration)
                self.lh_task.set_target(lh_T)
                self.rh_task.set_target(rh_T)

                if self._motion_tracker_config:
                    for arm_name in self.tracker_tasks:
                        if arm_name == "left_arm":
                            elbow_mocap_name = "left_elbow_target"
                        elif arm_name == "right_arm":
                            elbow_mocap_name = "right_elbow_target"
                        else:
                            continue

                        elbow_T = ik.SE3.from_mocap_name(self.mj_model, self.mj_data, elbow_mocap_name)
                        self.tracker_tasks[arm_name].set_target(elbow_T)

                if self.is_solving:
                    vel = ik.solve_ik(
                        self.configuration,
                        self.tasks,
                        rate.dt,
                        "daqp",
                        1e-2,
                        safety_break=True,
                        limits=self.limits,
                    )
                    self.configuration.integrate_inplace(vel, rate.dt)
                    self.mj_data.qpos[:] = self.configuration.q
                else:
                    self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])

                mj.mj_forward(self.mj_model, self.mj_data)

            if self.viewer is not None:
                with self.viewer.lock():
                    self.viewer.sync()

            self.is_ready = True
            rate.sleep()

        self.is_running = False
        self.is_ready = False

    def reframe_mocap(self, name: str, wxyz_xyz: np.ndarray, relative_site_name: str = "world"):
        """Establish reference frame for mocap target tracking."""
        if not self.is_ready:
            return

        pos = wxyz_xyz[4:]
        quat = wxyz_xyz[:4]

        site_id = self.mj_model.site(relative_site_name).id
        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        mocap_id = self.mj_model.body(name).mocapid[0]

        mocap_xpos_W = self.mj_data.mocap_pos[mocap_id].copy()
        mocap_quat_W = self.mj_data.mocap_quat[mocap_id].copy()

        site_quat = mat_to_quat(site_xmat)
        site_quat_inv = np.array([site_quat[0], -site_quat[1], -site_quat[2], -site_quat[3]])

        pos_rel_to_site = site_xmat.T @ (mocap_xpos_W - site_xpos)
        pos_offset = pos_rel_to_site - pos

        current_quat_rel_to_site = quat_multiply(site_quat_inv, mocap_quat_W)
        desired_quat_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        quat_offset = quat_multiply(desired_quat_inv, current_quat_rel_to_site)

        self.synced_mocap[name] = {
            "pos_offset": pos_offset,
            "quat_offset": quat_offset
        }

    def sync_mocap(self, name: str, wxyz_xyz: np.ndarray, relative_site_name: str = "world"):
        """Update mocap target with offset tracking."""
        if not self.is_ready:
            return

        if name not in self.synced_mocap:
            self.reframe_mocap(name, wxyz_xyz, relative_site_name)
            return

        mocap_id = self.mj_model.body(name).mocapid[0]
        site_id = self.mj_model.site(relative_site_name).id
        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        pos_offset = self.synced_mocap[name]["pos_offset"]
        quat_offset = self.synced_mocap[name]["quat_offset"]

        pos = wxyz_xyz[4:]
        quat = wxyz_xyz[:4]

        pos_corrected = pos + pos_offset
        relative_quat_corrected = quat_multiply(quat, quat_offset)

        mocap_xpos_W = site_xpos + site_xmat @ pos_corrected
        site_quat = mat_to_quat(site_xmat)
        mocap_quat_W = quat_multiply(site_quat, relative_quat_corrected)

        self.mj_data.mocap_pos[mocap_id] = mocap_xpos_W
        self.mj_data.mocap_quat[mocap_id] = mocap_quat_W

    def move_mocap_to(self, name: str, target_site_name: str):
        """Move mocap target to match a site's current pose."""
        ik.move_mocap_to_frame(self.mj_model, self.mj_data, name, target_site_name, "site")

    def _transform_world_to_site(self, wxyz_xyz_world: np.ndarray, site_name: str) -> np.ndarray:
        """Transform a pose from world frame to site-relative frame."""
        try:
            site_id = self.mj_model.site(site_name).id
        except KeyError:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model")

        site_xpos = self.mj_data.site_xpos[site_id]
        site_xmat = self.mj_data.site_xmat[site_id].reshape(3, 3)

        quat_world = wxyz_xyz_world[:4]
        pos_world = wxyz_xyz_world[4:]

        pos_rel = site_xmat.T @ (pos_world - site_xpos)

        site_quat = mat_to_quat(site_xmat)
        site_quat_inv = np.array([site_quat[0], -site_quat[1], -site_quat[2], -site_quat[3]])
        quat_rel = quat_multiply(site_quat_inv, quat_world)

        return np.concatenate([quat_rel, pos_rel])

    def _transform_xr_pose_to_reference_frame(self, xr_pose: np.ndarray) -> np.ndarray:
        """Transform XR controller pose from headset frame to reference frame."""
        pos_headset = xr_pose[:3]
        quat_xr = xr_pose[3:]  # [qx, qy, qz, qw]
        quat_headset = np.array([quat_xr[3], quat_xr[0], quat_xr[1], quat_xr[2]])  # [qw, qx, qy, qz]

        quat_norm = np.linalg.norm(quat_headset)
        if quat_norm < QUAT_NORM_THRESHOLD:
            quat_headset = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quat_headset = quat_headset / quat_norm

        pos_world = R_HEADSET_TO_WORLD @ pos_headset

        R_quat_scipy = Rotation.from_matrix(R_HEADSET_TO_WORLD).as_quat()  # [x, y, z, w]
        R_quat = np.array([R_quat_scipy[3], R_quat_scipy[0], R_quat_scipy[1], R_quat_scipy[2]])  # [w, x, y, z]
        R_quat_conj = np.array([R_quat[0], -R_quat[1], -R_quat[2], -R_quat[3]])
        quat_world = quat_multiply(quat_multiply(R_quat, quat_headset), R_quat_conj)

        pose_mj_world = np.concatenate([quat_world, pos_world])

        return self._transform_world_to_site(pose_mj_world, self._reference_frame)

    def get_qpos_arm(self) -> np.ndarray:
        """Get current G1 arm joint positions."""
        with self.datalock:
            res = np.zeros(14)
            for i, jnt_name in enumerate(G1_ARM_JOINT_NAMES):
                res[i] = self.mj_data.joint(jnt_name).qpos
            return res

    def set_qpos_arm(self, joint_positions: np.ndarray):
        """Update Mink internal state from measured joint positions."""
        if joint_positions is None or len(joint_positions) != 14:
            return

        with self.datalock:
            for i, jnt_name in enumerate(G1_ARM_JOINT_NAMES):
                self.mj_data.joint(jnt_name).qpos = joint_positions[i]

            self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])
            mj.mj_forward(self.mj_model, self.mj_data)

    def reset(self, joint_positions: np.ndarray | None = None):
        """Reset IK solver state to match current simulation state."""
        with self.datalock:
            if joint_positions is not None:
                for i, jnt_name in enumerate(G1_ARM_JOINT_NAMES):
                    self.mj_data.joint(jnt_name).qpos = joint_positions[i]
                self.configuration = ik.Configuration(self.mj_model, self.mj_data.qpos[:])
            else:
                mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
                self.configuration = ik.Configuration(self.mj_model, self.mj_model.keyframe("home").qpos)

            mj.mj_forward(self.mj_model, self.mj_data)
            ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_hand_target", "left_hand", "site")
            ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_hand_target", "right_hand", "site")

            if self._motion_tracker_config:
                if "left_arm" in self._motion_tracker_config:
                    link_target = self._motion_tracker_config["left_arm"]["link_target"]
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "left_elbow_target", link_target, "site")
                if "right_arm" in self._motion_tracker_config:
                    link_target = self._motion_tracker_config["right_arm"]["link_target"]
                    ik.move_mocap_to_frame(self.mj_model, self.mj_data, "right_elbow_target", link_target, "site")

            self.synced_mocap = {}
            self.lhold = False
            self.rhold = False
            self.force_sync = False

    def get_mocap_pose_b(self, name: str) -> np.ndarray:
        """Get mocap pose in body frame."""
        mocap_id = self.mj_model.body(name).mocapid[0]
        site = self.mj_data.site("trunk")
        g_wb = ik.SE3(np.concatenate([mat_to_quat(site.xmat.reshape(3, 3)), site.xpos]))
        g_wm = ik.SE3(np.concatenate([self.mj_data.mocap_quat[mocap_id], self.mj_data.mocap_pos[mocap_id]]))
        g_bm = g_wb.inverse().multiply(g_wm)
        return np.concatenate([g_bm.wxyz_xyz[4:], g_bm.wxyz_xyz[:4]])

    def _update_motion_tracker_tasks(self, motion_tracker_data: dict, left_active: bool, right_active: bool, data: dict[str, Any] = None):
        """Update elbow mocap bodies based on motion tracker data."""
        from isaaclab_xr_teleop.devices.xr_controller import XRControllerDevice

        if not self._motion_tracker_config or data is None:
            return

        arm_info = {
            "left_arm": {
                "active": left_active,
                "hand_mocap": "left_hand_target",
                "elbow_mocap": "left_elbow_target",
                "elbow_site": None
            },
            "right_arm": {
                "active": right_active,
                "hand_mocap": "right_hand_target",
                "elbow_mocap": "right_elbow_target",
                "elbow_site": None
            }
        }

        with self.datalock:
            for arm_name, tracker_config in self._motion_tracker_config.items():
                serial = tracker_config["serial"]
                link_target = tracker_config["link_target"]

                if arm_name not in arm_info:
                    continue

                arm_data = arm_info[arm_name]
                arm_data["elbow_site"] = link_target

                if not arm_data["active"]:
                    ik.move_mocap_to_frame(
                        self.mj_model, self.mj_data,
                        arm_data["elbow_mocap"], link_target, "site"
                    )
                    continue

                if serial not in motion_tracker_data:
                    ik.move_mocap_to_frame(
                        self.mj_model, self.mj_data,
                        arm_data["elbow_mocap"], link_target, "site"
                    )
                    continue

                tracker_pose_xr = motion_tracker_data[serial]["pose"]
                tracker_pose_mj = self._transform_xr_pose_to_reference_frame(tracker_pose_xr)
                tracker_xyz = tracker_pose_mj[4:]

                if arm_name == "left_arm":
                    controller_pose_xr = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_CONTROLLER.value)
                elif arm_name == "right_arm":
                    controller_pose_xr = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_CONTROLLER.value)
                else:
                    continue

                if controller_pose_xr is None:
                    continue

                controller_pose_mj = self._transform_xr_pose_to_reference_frame(controller_pose_xr)
                controller_xyz = controller_pose_mj[4:]

                offset = tracker_xyz - controller_xyz

                hand_mocap_id = self.mj_model.body(arm_data["hand_mocap"]).mocapid[0]
                hand_target_xyz = self.mj_data.mocap_pos[hand_mocap_id].copy()

                elbow_target_xyz = hand_target_xyz + offset * self._arm_length_scale_factor

                elbow_mocap_id = self.mj_model.body(arm_data["elbow_mocap"]).mocapid[0]
                self.mj_data.mocap_pos[elbow_mocap_id] = elbow_target_xyz

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert XR controller data to G1 humanoid arm joint commands."""
        from isaaclab_xr_teleop.devices.xr_controller import XRControllerDevice

        measured_positions = data.get("measured_joint_positions")
        if measured_positions is not None:
            if isinstance(measured_positions, torch.Tensor):
                measured_positions = measured_positions.cpu().numpy()
            with self.datalock:
                self.measured_joint_positions = measured_positions.copy()

        left_grip = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_GRIP.value, 0.0)
        right_grip = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_GRIP.value, 0.0)
        left_pose = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_CONTROLLER.value)
        right_pose = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_CONTROLLER.value)

        if left_pose is not None:
            left_pose_mj = self._transform_xr_pose_to_reference_frame(left_pose)

        if right_pose is not None:
            right_pose_mj = self._transform_xr_pose_to_reference_frame(right_pose)

        # Handle left hand activation
        if left_grip > 0.5 and left_pose is not None:
            if not self.lhold:
                if self.measured_joint_positions is not None:
                    self.sync_complete = False
                    self.force_sync = True
                    start_time = time.time()
                    while not self.sync_complete and (time.time() - start_time) < SYNC_TIMEOUT_SECONDS:
                        time.sleep(SYNC_POLL_INTERVAL_SECONDS)
                self.reframe_mocap("left_hand_target", left_pose_mj, relative_site_name=self._reference_frame)
                self.lhold = True
            self.sync_mocap("left_hand_target", left_pose_mj, relative_site_name=self._reference_frame)
        else:
            self.lhold = False
            self.move_mocap_to("left_hand_target", "left_hand")

        # Handle right hand activation
        if right_grip > 0.5 and right_pose is not None:
            if not self.rhold:
                if self.measured_joint_positions is not None:
                    self.sync_complete = False
                    self.force_sync = True
                    start_time = time.time()
                    while not self.sync_complete and (time.time() - start_time) < SYNC_TIMEOUT_SECONDS:
                        time.sleep(SYNC_POLL_INTERVAL_SECONDS)
                self.reframe_mocap("right_hand_target", right_pose_mj, relative_site_name=self._reference_frame)
                self.rhold = True
            self.sync_mocap("right_hand_target", right_pose_mj, relative_site_name=self._reference_frame)
        else:
            self.rhold = False
            self.move_mocap_to("right_hand_target", "right_hand")

        # Update motion tracker targets
        if self._motion_tracker_config:
            motion_tracker_data = data.get(XRControllerDevice.XRControllerDeviceValues.MOTION_TRACKERS.value, {})
            left_active = left_grip > 0.5 and left_pose is not None
            right_active = right_grip > 0.5 and right_pose is not None
            self._update_motion_tracker_tasks(motion_tracker_data, left_active, right_active, data)

        qpos_arm = self.get_qpos_arm()

        return torch.tensor(qpos_arm, dtype=torch.float32, device=self._sim_device)


@dataclass
class XRG1MinkIKRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit G1 Mink IK retargeter."""

    xml_path: str = ""
    """Path to MuJoCo XML model file for G1 upper body IK. Empty string uses bundled default."""

    headless: bool = True
    """If True, run without MuJoCo viewer visualization."""

    ik_rate_hz: float = 100.0
    """IK solver update rate in Hz."""

    collision_avoidance_distance: float = 0.04
    """Minimum distance from collisions (meters)."""

    collision_detection_distance: float = 0.10
    """Distance at which collision avoidance activates (meters)."""

    velocity_limit_factor: float = 0.7
    """Velocity limit scaling factor for joints."""

    reference_frame: str = "trunk"
    """Reference frame for relative control."""

    motion_tracker_config: dict[str, dict[str, str]] | None = None
    """Optional motion tracker configuration for additional IK constraints."""

    motion_tracker_task_weight: float = 0.8
    """Weight/priority for motion tracker position tasks in IK solver."""

    arm_length_scale_factor: float = 0.9
    """Scale factor for arm length when mapping tracker-to-controller offset to robot."""

    posture_cost_shoulder: float = 0.5
    """Posture regularization cost for shoulder joints."""

    posture_cost_elbow: float = 0.1
    """Posture regularization cost for elbow joints."""

    posture_cost_wrist: float = 0.0
    """Posture regularization cost for wrist joints."""

    retargeter_type: type[RetargeterBase] = XRG1MinkIKRetargeter

    def __post_init__(self):
        """Set default XML path from bundled assets if not provided."""
        if not self.xml_path:
            self.xml_path = _default_xml_path()
