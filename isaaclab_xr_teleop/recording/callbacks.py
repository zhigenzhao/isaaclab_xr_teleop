# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""VR button callback factory for demo recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import gymnasium as gym


class RecordingState:
    """Mutable state container shared between callbacks and the main loop."""

    def __init__(self):
        self.should_reset: bool = False
        self.is_running: bool = False


def make_recording_callbacks(
    env: gym.Env,
    state: RecordingState,
) -> dict[str, callable]:
    """Create the standard VR button -> recording action callback mapping.

    Button mapping: A=START, B=SAVE, X=RESET, Y=PAUSE, Right-stick=DISCARD

    Args:
        env: The gymnasium environment (needs recorder_manager).
        state: Shared mutable recording state.

    Returns:
        Dictionary of callback name -> function.
    """

    def reset_recording_instance():
        state.should_reset = True
        print("Recording instance reset requested")

    def start_recording_instance():
        state.is_running = True
        env.recorder_manager.record_post_reset([0])
        print("Recording started")

    def pause_recording_instance():
        state.is_running = False
        print("Recording paused")

    def save_recording_instance():
        if state.is_running or env.recorder_manager.get_episode(0).length() > 0:
            print("Saving current demonstration...")
            state.is_running = False
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Episode saved successfully!")
            state.should_reset = True
        else:
            print("No data to save - episode is empty")

    def discard_recording_instance():
        if state.is_running:
            print("Discarding current demonstration (not saved)")
            state.is_running = False
            state.should_reset = True
        else:
            print("Nothing to discard - not currently recording")

    return {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "SAVE": save_recording_instance,
        "RESET": reset_recording_instance,
        "PAUSE": pause_recording_instance,
        "DISCARD": discard_recording_instance,
    }
