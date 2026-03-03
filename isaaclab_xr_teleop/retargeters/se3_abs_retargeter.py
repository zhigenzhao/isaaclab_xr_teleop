# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter for absolute SE(3) control."""

import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab_xr_teleop.devices.xr_controller import XRControllerDevice
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Default coordinate transformation from headset frame to world frame
R_HEADSET_TO_WORLD = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
])


class XRSe3AbsRetargeter(RetargeterBase):
    """Retargets XR controller data to end-effector commands using absolute positioning.

    This retargeter maps controller poses directly to robot end-effector positions and
    orientations, rather than using relative movements.

    Features:
    - Direct pose mapping from controller to end-effector
    - Coordinate frame transformation support
    - Optional workspace bounds/clamping
    - Optional constraint to zero out X/Y rotations (keeping only Z-axis rotation)
    - Optional visualization of the target end-effector pose
    """

    def __init__(self, cfg: "XRSe3AbsRetargeterCfg"):
        """Initialize the absolute pose retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        # Store configuration
        if cfg.control_hand not in ["left", "right"]:
            raise ValueError("control_hand must be 'left' or 'right'")
        self._control_hand = cfg.control_hand
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation

        # Set up coordinate transformation
        if cfg.R_xr_to_world is not None:
            self._R_xr_to_world = cfg.R_xr_to_world
        else:
            self._R_xr_to_world = R_HEADSET_TO_WORLD

        # Set up position offset
        if cfg.position_offset is not None:
            self._position_offset = cfg.position_offset
        else:
            self._position_offset = np.zeros(3)

        # Set up workspace bounds
        self._workspace_bounds_min = cfg.workspace_bounds_min
        self._workspace_bounds_max = cfg.workspace_bounds_max

        # Initialize visualization if enabled
        self._enable_visualization = cfg.enable_visualization
        if cfg.enable_visualization:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/xr_ee_goal_abs"))
            self._goal_marker.set_visibility(True)

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert controller data to robot end-effector command.

        Args:
            data: Dictionary containing controller data from XRControllerDevice:
                - 'left_controller': [x, y, z, qx, qy, qz, qw] pose array
                - 'right_controller': [x, y, z, qx, qy, qz, qw] pose array

        Returns:
            torch.Tensor: 7D tensor containing position (xyz) and orientation (quaternion)
        """
        # Get controller pose based on control hand
        if self._control_hand == "left":
            controller_key = XRControllerDevice.XRControllerDeviceValues.LEFT_CONTROLLER.value
        else:
            controller_key = XRControllerDevice.XRControllerDeviceValues.RIGHT_CONTROLLER.value
        controller_pose = data.get(controller_key)

        if controller_pose is None:
            # Return default pose if no controller data
            default_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)  # [x, y, z, qw, qx, qy, qz]
            return torch.tensor(default_pose, dtype=torch.float32, device=self._sim_device)

        # Extract position and quaternion from controller pose
        # XRoboToolkit format: [x, y, z, qx, qy, qz, qw]
        controller_pos = controller_pose[:3]
        controller_quat = np.array([controller_pose[6], controller_pose[3], controller_pose[4], controller_pose[5]])  # Convert to [qw, qx, qy, qz]

        # Retarget to absolute pose
        ee_command_np = self._retarget_abs(controller_pos, controller_quat)

        # Convert to torch tensor
        ee_command = torch.tensor(ee_command_np, dtype=torch.float32, device=self._sim_device)

        return ee_command

    def _retarget_abs(self, controller_pos: np.ndarray, controller_quat: np.ndarray) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            controller_pos: Controller position [x, y, z]
            controller_quat: Controller quaternion [qw, qx, qy, qz]

        Returns:
            np.ndarray: 7D array containing position (xyz) and orientation (quaternion [qw, qx, qy, qz])
        """
        # Apply coordinate transformation
        position = self._R_xr_to_world @ controller_pos

        # Apply position offset
        position = position + self._position_offset

        # Apply workspace bounds if specified
        if self._workspace_bounds_min is not None:
            position = np.maximum(position, self._workspace_bounds_min)
        if self._workspace_bounds_max is not None:
            position = np.minimum(position, self._workspace_bounds_max)

        # Transform rotation
        # Convert controller quaternion to rotation matrix
        controller_rot = Rotation.from_quat([controller_quat[1], controller_quat[2], controller_quat[3], controller_quat[0]])  # scipy expects [qx, qy, qz, qw]

        # Apply coordinate frame transformation
        R_transform = Rotation.from_matrix(self._R_xr_to_world)
        final_rot = R_transform * controller_rot

        # Apply zero_out_xy_rotation if enabled
        if self._zero_out_xy_rotation:
            z, y, x = final_rot.as_euler("ZYX")
            y = 0.0  # Zero out rotation around y-axis
            x = 0.0  # Zero out rotation around x-axis
            final_rot = Rotation.from_euler("ZYX", [z, y, x])

        # Convert back to quaternion in [qw, qx, qy, qz] format
        quat_scipy = final_rot.as_quat()  # Returns [qx, qy, qz, qw]
        rotation = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # Convert to [qw, qx, qy, qz]

        # Update visualization if enabled
        if self._enable_visualization:
            self._update_visualization(position, rotation)

        return np.concatenate([position, rotation])

    def _update_visualization(self, position: np.ndarray, rotation: np.ndarray):
        """Update visualization markers with current pose.

        Args:
            position: Position [x, y, z]
            rotation: Quaternion [qw, qx, qy, qz]
        """
        if self._enable_visualization:
            trans = torch.tensor([position], dtype=torch.float32, device=self._sim_device)
            rot = torch.tensor([rotation], dtype=torch.float32, device=self._sim_device)
            self._goal_marker.visualize(translations=trans, orientations=rot)


@dataclass
class XRSe3AbsRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit absolute position retargeter."""

    control_hand: str = "right"
    """Which hand to use for control: 'left' or 'right'."""

    zero_out_xy_rotation: bool = False
    """If True, zero out rotation around x and y axes."""

    enable_visualization: bool = False
    """If True, visualize the target pose in the scene."""

    R_xr_to_world: np.ndarray | None = None
    """Rotation matrix to transform XR frame to world frame. If None, uses R_HEADSET_TO_WORLD."""

    position_offset: np.ndarray | None = None
    """Offset to apply to controller position [x, y, z]. If None, uses zeros."""

    workspace_bounds_min: np.ndarray | None = None
    """Minimum workspace bounds [x, y, z]. If None, no lower bound."""

    workspace_bounds_max: np.ndarray | None = None
    """Maximum workspace bounds [x, y, z]. If None, no upper bound."""

    retargeter_type: type[RetargeterBase] = XRSe3AbsRetargeter
