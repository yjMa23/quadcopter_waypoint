import math

import torch

from quadcopter_waypoint.utils.m2_reward_gating import m2_descent_reward_active


DEFAULTS = {
    "align_radius": 0.25,
    "align_max_horizontal_speed": 0.20,
    "align_body_deck_angle": math.radians(10.0),
    "align_upright": 0.95,
    "align_height_max": 1.00,
}


def _gate(
    *,
    can_land: bool = True,
    horizontal_error: float = 0.10,
    horizontal_speed: float = 0.10,
    body_deck_normal_angle: float = math.radians(5.0),
    upright: float = 0.99,
    robot_height_above_pad: float = 0.70,
) -> tuple[bool, torch.Tensor]:
    can_land_tensor = torch.tensor([can_land], dtype=torch.bool)
    active = m2_descent_reward_active(
        can_land=can_land_tensor,
        horizontal_error=torch.tensor([horizontal_error]),
        horizontal_speed=torch.tensor([horizontal_speed]),
        body_deck_normal_angle=torch.tensor([body_deck_normal_angle]),
        upright=torch.tensor([upright]),
        robot_height_above_pad=torch.tensor([robot_height_above_pad]),
        **DEFAULTS,
    )
    return bool(active.item()), can_land_tensor


def test_case_a_never_aligned_keeps_descent_reward_inactive():
    active, can_land = _gate(can_land=False)
    assert active is False
    assert bool(can_land.item()) is False


def test_case_b_latched_and_stable_enables_descent_reward():
    active, can_land = _gate(can_land=True)
    assert active is True
    assert bool(can_land.item()) is True


def test_case_c_latched_then_horizontal_drift_suspends_only_reward_gate():
    active, can_land = _gate(can_land=True, horizontal_error=0.30)
    assert active is False
    assert bool(can_land.item()) is True


def test_case_d_descent_below_align_height_min_stays_reward_eligible():
    # The M2 reward gate intentionally has no align_height_min lower bound.
    active, can_land = _gate(can_land=True, robot_height_above_pad=0.20)
    assert active is True
    assert bool(can_land.item()) is True


def test_case_e_horizontal_speed_recovery_suspends_descent_reward():
    active, can_land = _gate(can_land=True, horizontal_speed=0.25)
    assert active is False
    assert bool(can_land.item()) is True


def test_case_f_body_deck_attitude_recovery_suspends_descent_reward():
    active, can_land = _gate(can_land=True, body_deck_normal_angle=math.radians(12.0))
    assert active is False
    assert bool(can_land.item()) is True
