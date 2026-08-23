# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure reward-phase gating helpers for the PX4-compatible M2 landing task."""

from __future__ import annotations

import torch


def m2_descent_reward_active(
    *,
    can_land: torch.Tensor,
    horizontal_error: torch.Tensor,
    horizontal_speed: torch.Tensor,
    body_deck_normal_angle: torch.Tensor,
    upright: torch.Tensor,
    robot_height_above_pad: torch.Tensor,
    align_radius: float,
    align_max_horizontal_speed: float,
    align_body_deck_angle: float,
    align_upright: float,
    align_height_max: float,
) -> torch.Tensor:
    """Gate descent shaping on latched permission plus instantaneous recoverable alignment."""
    recovery_alignment_ok = (
        (horizontal_error < align_radius)
        & (horizontal_speed < align_max_horizontal_speed)
        & (body_deck_normal_angle < align_body_deck_angle)
        & (upright > align_upright)
        & (robot_height_above_pad < align_height_max)
    )
    return can_land & recovery_alignment_ok
