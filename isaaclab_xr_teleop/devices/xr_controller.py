# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XRoboToolkit XR controller for SE(3) control."""

import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

# Import XRoboToolkit SDK
try:
    import xrobotoolkit_sdk as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False
    print("Warning: xrobotoolkit_sdk not available. XRControllerDevice will not function properly.")


class XRControllerDevice(DeviceBase):
    """XRoboToolkit XR controller for sending SE(3) commands as delta poses.

    This class implements an XR controller interface using the XRoboToolkit SDK to provide
    commands to a robotic arm with a gripper. It tracks VR/AR controller poses and maps
    them to robot control commands via the _get_raw_data() method and retargeters.

    Raw data format (_get_raw_data output):
    * A dictionary containing controller poses, input states, and button states
    * Dictionary keys are XRControllerDeviceValues enum members
    * Controller poses as 7-element arrays: [x, y, z, qw, qx, qy, qz]
    * Input values (triggers, grips) as floats [0-1]
    * Button states as boolean values

    Control modes:
    * right_hand: Uses right controller for pose control
    * left_hand: Uses left controller for pose control
    * dual_hand: Uses both controllers

    Gripper sources:
    * trigger: Uses trigger value for gripper control
    * grip: Uses grip value for gripper control
    * button: Uses primary button for gripper toggle
    """

    class XRControllerDeviceValues(Enum):
        """Enum for XR controller device data keys.

        Provides type-safe keys for accessing device data in the dictionary returned
        by _get_raw_data(). This enables IDE autocomplete and prevents typos.
        """
        LEFT_CONTROLLER = "left_controller"      # Left controller pose [x, y, z, qx, qy, qz, qw]
        RIGHT_CONTROLLER = "right_controller"    # Right controller pose [x, y, z, qx, qy, qz, qw]
        HEADSET = "headset"                      # Headset pose [x, y, z, qx, qy, qz, qw]
        LEFT_TRIGGER = "left_trigger"            # Left trigger value [0-1]
        RIGHT_TRIGGER = "right_trigger"          # Right trigger value [0-1]
        LEFT_GRIP = "left_grip"                  # Left grip value [0-1]
        RIGHT_GRIP = "right_grip"                # Right grip value [0-1]
        BUTTONS = "buttons"                      # Dictionary of button states
        TIMESTAMP = "timestamp"                  # Timestamp in nanoseconds
        CONFIG = "config"                        # Device configuration dictionary
        MOTION_TRACKERS = "motion_trackers"      # Dictionary of motion tracker data {serial: {"pose": [x,y,z,qx,qy,qz,qw]}}

    def __init__(self, cfg: "XRControllerDeviceCfg", retargeters: list | None = None):
        """Initialize the XR controller device.

        Args:
            cfg: Configuration object for XR controller settings.
            retargeters: List of retargeter instances to transform raw data into robot commands.
        """
        super().__init__(retargeters)

        if not XRT_AVAILABLE:
            raise RuntimeError("xrobotoolkit_sdk is not available. Cannot initialize XRControllerDevice.")

        # Store configuration
        self.cfg = cfg
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.control_mode = cfg.control_mode
        self.gripper_source = cfg.gripper_source
        self.deadzone_threshold = cfg.deadzone_threshold
        self._sim_device = cfg.sim_device

        # Initialize XRoboToolkit SDK
        try:
            xrt.init()
            print("XRoboToolkit SDK initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize XRoboToolkit SDK: {e}")

        # Internal state for button tracking
        self._button_pressed_prev = {}

        # Callback dictionary
        self._additional_callbacks = dict()

        # Measured joint positions from simulation (for state sync with retargeters)
        self._measured_joint_positions = None

        print(f"XR Controller initialized with mode: {self.control_mode}, gripper source: {self.gripper_source}")

    def __del__(self):
        """Destructor for the class."""
        if XRT_AVAILABLE:
            try:
                xrt.close()
                print("XRoboToolkit SDK closed.")
            except Exception:
                pass

    def __str__(self) -> str:
        """Returns: A string containing the information of the XR controller."""
        msg = f"XRoboToolkit XR Controller: {self.__class__.__name__}\n"
        msg += f"\tControl Mode: {self.control_mode}\n"
        msg += f"\tGripper Source: {self.gripper_source}\n"
        msg += f"\tPosition Sensitivity: {self.pos_sensitivity}\n"
        msg += f"\tRotation Sensitivity: {self.rot_sensitivity}\n"
        msg += f"\tDeadzone Threshold: {self.deadzone_threshold}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tController Usage:\n"
        if self.control_mode == "right_hand":
            msg += "\t\tRight controller: Move to control robot end-effector\n"
        elif self.control_mode == "left_hand":
            msg += "\t\tLeft controller: Move to control robot end-effector\n"
        else:
            msg += "\t\tBoth controllers: Dual-hand control\n"

        if self.gripper_source == "trigger":
            msg += "\t\tTrigger: Squeeze to close gripper\n"
        elif self.gripper_source == "grip":
            msg += "\t\tGrip: Squeeze to close gripper\n"
        else:
            msg += "\t\tPrimary button: Press to toggle gripper\n"

        msg += "\t----------------------------------------------\n"
        msg += "\tButton Mappings for Demo Recording:\n"
        msg += "\t\tA button: START recording\n"
        msg += "\t\tB button: SAVE and reset (saves current episode)\n"
        msg += "\t\tX button: RESET environment (discards data)\n"
        msg += "\t\tY button: PAUSE recording\n"
        msg += "\t\tRight joystick click: DISCARD recording\n"
        return msg

    def reset(self):
        """Reset the internal state."""
        self._button_pressed_prev = {}
        self._measured_joint_positions = None

    def set_measured_joint_positions(self, joint_positions: torch.Tensor | np.ndarray):
        """Set measured joint positions from the simulation.

        This method allows the environment to provide the current robot joint positions
        to the device, which can then be passed to retargeters for state synchronization.

        Args:
            joint_positions: Tensor or array of upper body joint positions (16 elements)
        """
        self._measured_joint_positions = joint_positions

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to controller buttons.

        Args:
            key: The button to bind to. Supported: 'RESET', 'START', 'STOP'.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def _get_raw_data(self) -> dict[str, Any]:
        """Get raw controller data from XRoboToolkit.

        Returns:
            Dictionary containing:
                - left_controller: [x, y, z, qw, qx, qy, qz] pose array
                - right_controller: [x, y, z, qw, qx, qy, qz] pose array
                - headset: [x, y, z, qw, qx, qy, qz] pose array
                - left_trigger: float [0-1]
                - right_trigger: float [0-1]
                - left_grip: float [0-1]
                - right_grip: float [0-1]
                - buttons: dict with button states
                - config: dict with device configuration
                - motion_trackers: dict {serial: {"pose": [x,y,z,qx,qy,qz,qw]}}
        """
        try:
            # Get controller poses
            left_pose = np.array(xrt.get_left_controller_pose(), dtype=np.float32)
            right_pose = np.array(xrt.get_right_controller_pose(), dtype=np.float32)
            headset_pose = np.array(xrt.get_headset_pose(), dtype=np.float32)

            # Get input states
            left_trigger = xrt.get_left_trigger()
            right_trigger = xrt.get_right_trigger()
            left_grip = xrt.get_left_grip()
            right_grip = xrt.get_right_grip()

            # Get button states
            buttons = {
                'left_primary': xrt.get_X_button(),
                'right_primary': xrt.get_A_button(),
                'left_secondary': xrt.get_Y_button(),
                'right_secondary': xrt.get_B_button(),
                'left_menu': xrt.get_left_menu_button(),
                'right_menu': xrt.get_right_menu_button(),
                'left_axis_click': xrt.get_left_axis_click(),
                'right_axis_click': xrt.get_right_axis_click(),
            }

            # Handle button callbacks
            self._handle_button_callbacks(buttons)

            # Get motion tracker data
            motion_trackers = {}
            try:
                num_motion_data = xrt.num_motion_data_available()
                if num_motion_data > 0:
                    poses = xrt.get_motion_tracker_pose()
                    serial_numbers = xrt.get_motion_tracker_serial_numbers()

                    for i in range(num_motion_data):
                        serial = serial_numbers[i]
                        motion_trackers[serial] = {
                            "pose": np.array(poses[i], dtype=np.float32)
                        }
            except Exception:
                # Motion trackers are optional, don't fail if not available
                pass

            data = {
                self.XRControllerDeviceValues.LEFT_CONTROLLER.value: left_pose,
                self.XRControllerDeviceValues.RIGHT_CONTROLLER.value: right_pose,
                self.XRControllerDeviceValues.HEADSET.value: headset_pose,
                self.XRControllerDeviceValues.LEFT_TRIGGER.value: left_trigger,
                self.XRControllerDeviceValues.RIGHT_TRIGGER.value: right_trigger,
                self.XRControllerDeviceValues.LEFT_GRIP.value: left_grip,
                self.XRControllerDeviceValues.RIGHT_GRIP.value: right_grip,
                self.XRControllerDeviceValues.BUTTONS.value: buttons,
                self.XRControllerDeviceValues.TIMESTAMP.value: xrt.get_time_stamp_ns(),
                self.XRControllerDeviceValues.CONFIG.value: {
                    'pos_sensitivity': self.pos_sensitivity,
                    'rot_sensitivity': self.rot_sensitivity,
                    'control_mode': self.control_mode,
                    'gripper_source': self.gripper_source,
                    'deadzone_threshold': self.deadzone_threshold
                },
                self.XRControllerDeviceValues.MOTION_TRACKERS.value: motion_trackers
            }

            # Include measured joint positions if available (for state sync with retargeters)
            if self._measured_joint_positions is not None:
                data["measured_joint_positions"] = self._measured_joint_positions

            return data

        except Exception as e:
            print(f"Error getting XR controller data: {e}")
            # Return default values on error
            default_pose = np.zeros(7, dtype=np.float32)
            return {
                self.XRControllerDeviceValues.LEFT_CONTROLLER.value: default_pose,
                self.XRControllerDeviceValues.RIGHT_CONTROLLER.value: default_pose,
                self.XRControllerDeviceValues.LEFT_TRIGGER.value: 0.0,
                self.XRControllerDeviceValues.RIGHT_TRIGGER.value: 0.0,
                self.XRControllerDeviceValues.LEFT_GRIP.value: 0.0,
                self.XRControllerDeviceValues.RIGHT_GRIP.value: 0.0,
                self.XRControllerDeviceValues.BUTTONS.value: {k: False for k in ['left_primary', 'right_primary', 'left_secondary',
                                               'right_secondary', 'left_menu', 'right_menu',
                                               'left_axis_click', 'right_axis_click']},
                self.XRControllerDeviceValues.TIMESTAMP.value: 0,
                self.XRControllerDeviceValues.CONFIG.value: {
                    'pos_sensitivity': self.pos_sensitivity,
                    'rot_sensitivity': self.rot_sensitivity,
                    'control_mode': self.control_mode,
                    'gripper_source': self.gripper_source,
                    'deadzone_threshold': self.deadzone_threshold
                }
            }

    def _handle_button_callbacks(self, buttons: dict[str, bool]) -> None:
        """Handle button press callbacks.

        Args:
            buttons: Dictionary of current button states
        """
        # Check for button press events (rising edge)
        for button_name, is_pressed in buttons.items():
            was_pressed = self._button_pressed_prev.get(button_name, False)

            if is_pressed and not was_pressed:  # Rising edge
                # A button = START recording
                if button_name == 'right_primary' and 'START' in self._additional_callbacks:
                    self._additional_callbacks['START']()
                # B button = SAVE and reset
                elif button_name == 'right_secondary' and 'SAVE' in self._additional_callbacks:
                    self._additional_callbacks['SAVE']()
                # X button = RESET
                elif button_name == 'left_primary' and 'RESET' in self._additional_callbacks:
                    self._additional_callbacks['RESET']()
                # Y button = PAUSE recording
                elif button_name == 'left_secondary' and 'PAUSE' in self._additional_callbacks:
                    self._additional_callbacks['PAUSE']()
                # Right joystick click = DISCARD
                elif button_name == 'right_axis_click' and 'DISCARD' in self._additional_callbacks:
                    self._additional_callbacks['DISCARD']()

        # Update previous button states
        self._button_pressed_prev = buttons.copy()


@dataclass
class XRControllerDeviceCfg(DeviceCfg):
    """Configuration for XRoboToolkit XR controller devices."""

    pos_sensitivity: float = 0.4
    """Sensitivity for positional control (m/s)."""

    rot_sensitivity: float = 0.8
    """Sensitivity for rotational control (rad/s)."""

    control_mode: str = "right_hand"
    """Control mode: 'right_hand', 'left_hand', or 'dual_hand'."""

    gripper_source: str = "trigger"
    """Gripper control source: 'trigger', 'grip', or 'button'."""

    deadzone_threshold: float = 0.05
    """Minimum movement threshold to filter out noise."""

    class_type: type[DeviceBase] = XRControllerDevice
