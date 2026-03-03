# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter for Inspire hand control."""

import torch
from dataclasses import dataclass, field
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab_xr_teleop.devices.xr_controller import XRControllerDevice

from .robot_cfg import INSPIRE_HAND_JOINT_LIMITS


class XRInspireHandRetargeter(RetargeterBase):
    """Retargets XR controller trigger inputs to Inspire hand joint commands.

    This retargeter maps the left and right trigger values from the XR controller
    to joint positions for the Inspire dexterous hand. It uses linear interpolation
    between the open (lower limit) and closed (upper limit) positions based on
    the trigger value.

    Features:
    - Maps left trigger to left hand joints (prefixed with "L_")
    - Maps right trigger to right hand joints (prefixed with "R_")
    - Supports continuous (analog) or binary (open/closed) modes
    - Joint limits automatically extracted from Inspire Hand URDF specifications
    - Optional inversion for different control conventions
    """

    def __init__(self, cfg: "XRInspireHandRetargeterCfg"):
        """Initialize the Inspire hand retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        # Validate configuration
        if cfg.mode not in ["continuous", "binary"]:
            raise ValueError("mode must be 'continuous' or 'binary'")
        if not cfg.hand_joint_names:
            raise ValueError("hand_joint_names must be provided")

        # Store configuration
        self._hand_joint_names = cfg.hand_joint_names
        self._mode = cfg.mode
        self._binary_threshold = cfg.binary_threshold
        self._invert = cfg.invert

        # Raw trigger values before retargeting [left_trigger, right_trigger]
        self.raw_gripper_command = torch.zeros(2, dtype=torch.float32, device=self._sim_device)

        # Build open/closed position arrays matching joint order
        self._open_positions: list[float] = []
        self._closed_positions: list[float] = []

        for name in self._hand_joint_names:
            joint_type = self._extract_joint_type(name)
            lower, upper = INSPIRE_HAND_JOINT_LIMITS.get(joint_type, (0.0, 1.7))
            self._open_positions.append(lower)
            self._closed_positions.append(upper)

    def _extract_joint_type(self, joint_name: str) -> str:
        """Extract the joint type from a full joint name.

        Converts joint names like "L_index_proximal_joint" or "R_thumb_distal_joint"
        to their base type like "index_proximal" or "thumb_distal".

        Args:
            joint_name: Full joint name with hand prefix and "_joint" suffix

        Returns:
            Base joint type that matches keys in INSPIRE_HAND_JOINT_LIMITS
        """
        # Remove hand prefix (L_ or R_)
        if joint_name.startswith("L_") or joint_name.startswith("R_"):
            joint_name = joint_name[2:]

        # Remove _joint suffix if present
        if joint_name.endswith("_joint"):
            joint_name = joint_name[:-6]

        return joint_name

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert controller trigger inputs to hand joint commands.

        Args:
            data: Dictionary containing controller data from XRControllerDevice:
                - 'left_trigger': float [0-1]
                - 'right_trigger': float [0-1]

        Returns:
            torch.Tensor: 1D tensor containing joint position commands for all
                hand joints in the order specified by hand_joint_names
        """
        # Get trigger values
        left_trigger = data.get(XRControllerDevice.XRControllerDeviceValues.LEFT_TRIGGER.value, 0.0)
        right_trigger = data.get(XRControllerDevice.XRControllerDeviceValues.RIGHT_TRIGGER.value, 0.0)

        # Store raw trigger values before any processing
        self.raw_gripper_command = torch.tensor(
            [left_trigger, right_trigger], dtype=torch.float32, device=self._sim_device
        )

        # Apply inversion if requested
        if self._invert:
            left_trigger = 1.0 - left_trigger
            right_trigger = 1.0 - right_trigger

        # Apply binary threshold if in binary mode
        if self._mode == "binary":
            left_trigger = 1.0 if left_trigger > self._binary_threshold else 0.0
            right_trigger = 1.0 if right_trigger > self._binary_threshold else 0.0

        # Compute joint values via linear interpolation
        joint_values: list[float] = []

        for i, name in enumerate(self._hand_joint_names):
            # Select trigger based on hand (L_ prefix = left, otherwise right)
            trigger = left_trigger if name.startswith("L_") else right_trigger

            # Linear interpolation: open + trigger * (closed - open)
            value = self._open_positions[i] + trigger * (self._closed_positions[i] - self._open_positions[i])
            joint_values.append(value)

        return torch.tensor(joint_values, dtype=torch.float32, device=self._sim_device)


@dataclass
class XRInspireHandRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit Inspire hand retargeter."""

    hand_joint_names: list[str] = field(default_factory=list)
    """List of hand joint names in action space order (24 joints: 12 per hand)."""

    mode: str = "continuous"
    """Hand control mode: 'continuous' (analog) or 'binary' (open/closed)."""

    binary_threshold: float = 0.5
    """Threshold for binary mode [0-1]. Above threshold = closed, below = open."""

    invert: bool = False
    """If True, invert the trigger mapping (0 = closed, 1 = open)."""

    retargeter_type: type[RetargeterBase] = XRInspireHandRetargeter
