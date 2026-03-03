# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 humanoid robot retargeters for XR teleoperation."""

from .robot_cfg import G1_ARM_JOINT_NAMES, G1_HAND_JOINT_NAMES, INSPIRE_HAND_JOINT_LIMITS
from .mink_ik import XRG1MinkIKRetargeter, XRG1MinkIKRetargeterCfg
from .inspire_hand import XRInspireHandRetargeter, XRInspireHandRetargeterCfg
