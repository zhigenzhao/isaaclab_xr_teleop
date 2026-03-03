# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 humanoid robot joint configuration — single source of truth.

Consolidates joint name constants and hand joint limits that were previously
duplicated across multiple files in IsaacLab.
"""

# G1 arm joint names (14 total: 7 per arm)
G1_ARM_JOINT_NAMES: list[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# G1 Inspire hand joint names (24 joints total: 12 per hand)
G1_HAND_JOINT_NAMES: list[str] = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]

# Default joint limits from Inspire Hand URDF
# Source: https://github.com/unitreerobotics/xr_teleoperate/tree/main/assets/inspire_hand
INSPIRE_HAND_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "thumb_proximal_yaw": (-0.1, 1.3),
    "thumb_proximal_pitch": (0.0, 0.5),
    "thumb_intermediate": (0.0, 0.8),
    "thumb_distal": (0.0, 1.2),
    "index_proximal": (0.0, 1.7),
    "index_intermediate": (0.0, 1.7),
    "middle_proximal": (0.0, 1.7),
    "middle_intermediate": (0.0, 1.7),
    "ring_proximal": (0.0, 1.7),
    "ring_intermediate": (0.0, 1.7),
    "pinky_proximal": (0.0, 1.7),
    "pinky_intermediate": (0.0, 1.7),
}
