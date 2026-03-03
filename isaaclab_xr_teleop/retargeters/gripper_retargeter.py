# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit controller retargeter for gripper control."""

import torch
from dataclasses import dataclass
from typing import Any

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab_xr_teleop.devices.xr_controller import XRControllerDevice


class XRGripperRetargeter(RetargeterBase):
    """Retargets XR controller input to gripper commands.

    This retargeter maps trigger or grip button values from the controller to gripper
    control commands. It supports both continuous (analog) and binary (open/closed) modes.

    Features:
    - Continuous mode: Direct mapping of trigger/grip value [0-1] to gripper position
    - Binary mode: Threshold-based open/closed control
    - Optional hysteresis to prevent oscillation in binary mode
    - Configurable inversion for different gripper conventions
    """

    def __init__(self, cfg: "XRGripperRetargeterCfg"):
        """Initialize the gripper retargeter.

        Args:
            cfg: Configuration for the retargeter
        """
        super().__init__(cfg)

        # Store configuration
        if cfg.control_hand not in ["left", "right"]:
            raise ValueError("control_hand must be 'left' or 'right'")
        if cfg.input_source not in ["trigger", "grip"]:
            raise ValueError("input_source must be 'trigger' or 'grip'")
        if cfg.mode not in ["continuous", "binary"]:
            raise ValueError("mode must be 'continuous' or 'binary'")

        self._control_hand = cfg.control_hand
        self._input_source = cfg.input_source
        self._mode = cfg.mode
        self._binary_threshold = cfg.binary_threshold
        self._invert = cfg.invert
        self._open_value = cfg.open_value
        self._closed_value = cfg.closed_value
        self._hysteresis = cfg.hysteresis

        # State for hysteresis in binary mode
        self._previous_state = "open"  # "open" or "closed"

    def retarget(self, data: dict[str, Any]) -> torch.Tensor:
        """Convert controller input to gripper command.

        Args:
            data: Dictionary containing controller data from XRControllerDevice:
                - 'left_trigger': float [0-1]
                - 'right_trigger': float [0-1]
                - 'left_grip': float [0-1]
                - 'right_grip': float [0-1]

        Returns:
            torch.Tensor: 1D tensor containing gripper command value
        """
        # Get input value based on control hand and source
        if self._control_hand == "left":
            if self._input_source == "trigger":
                input_key = XRControllerDevice.XRControllerDeviceValues.LEFT_TRIGGER.value
            else:
                input_key = XRControllerDevice.XRControllerDeviceValues.LEFT_GRIP.value
        else:
            if self._input_source == "trigger":
                input_key = XRControllerDevice.XRControllerDeviceValues.RIGHT_TRIGGER.value
            else:
                input_key = XRControllerDevice.XRControllerDeviceValues.RIGHT_GRIP.value
        input_value = data.get(input_key, 0.0)

        # Process based on mode
        if self._mode == "continuous":
            gripper_value = self._process_continuous(input_value)
        else:  # binary
            gripper_value = self._process_binary(input_value)

        # Convert to torch tensor
        gripper_command = torch.tensor([gripper_value], dtype=torch.float32, device=self._sim_device)

        return gripper_command

    def _process_continuous(self, input_value: float) -> float:
        """Process input in continuous mode.

        Args:
            input_value: Raw input value [0-1]

        Returns:
            float: Gripper command value mapped to [closed_value, open_value] range
        """
        # Invert if requested (flip the input range)
        if self._invert:
            input_value = 1.0 - input_value

        # Map from [0, 1] to [closed_value, open_value]
        # input_value=0.0 -> closed_value, input_value=1.0 -> open_value
        gripper_value = self._closed_value + input_value * (self._open_value - self._closed_value)

        return gripper_value

    def _process_binary(self, input_value: float) -> float:
        """Process input in binary mode with hysteresis.

        Args:
            input_value: Raw input value [0-1]

        Returns:
            float: Gripper command value (open_value or closed_value)
        """
        # Apply hysteresis
        if self._previous_state == "open":
            # Need to exceed threshold + hysteresis to close
            if input_value > (self._binary_threshold + self._hysteresis / 2):
                self._previous_state = "closed"
        else:  # previous_state == "closed"
            # Need to go below threshold - hysteresis to open
            if input_value < (self._binary_threshold - self._hysteresis / 2):
                self._previous_state = "open"

        # Return appropriate value based on current state
        if self._previous_state == "closed":
            return self._closed_value if not self._invert else self._open_value
        else:
            return self._open_value if not self._invert else self._closed_value

    @property
    def raw_gripper_command(self) -> torch.Tensor:
        """Get the raw gripper command value based on previous state.

        Returns:
            torch.Tensor: 1D tensor with gripper value (0.0 for open, 1.0 for closed)
        """
        if self._previous_state == "closed":
            return torch.tensor([1.0], dtype=torch.float32, device=self._sim_device)
        else:
            return torch.tensor([0.0], dtype=torch.float32, device=self._sim_device)


@dataclass
class XRGripperRetargeterCfg(RetargeterCfg):
    """Configuration for XRoboToolkit gripper retargeter."""

    control_hand: str = "right"
    """Which hand to use for control: 'left' or 'right'."""

    input_source: str = "trigger"
    """Input source for gripper control: 'trigger' or 'grip'."""

    mode: str = "continuous"
    """Gripper control mode: 'continuous' (analog) or 'binary' (open/closed)."""

    binary_threshold: float = 0.5
    """Threshold for binary mode [0-1]. Above threshold = closed, below = open."""

    invert: bool = False
    """If True, invert the gripper value (1.0 = open, 0.0 = closed)."""

    open_value: float = 1.0
    """Value to output when gripper is open (used in binary mode)."""

    closed_value: float = 0.0
    """Value to output when gripper is closed (used in binary mode)."""

    hysteresis: float = 0.0
    """Hysteresis band for binary mode to prevent oscillation [0-1]."""

    retargeter_type: type[RetargeterBase] = XRGripperRetargeter
