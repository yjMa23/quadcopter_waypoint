# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rl_games" / "eval_metrics_utils.py"
SPEC = importlib.util.spec_from_file_location("eval_metrics_utils", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
metrics_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics_utils)


class EvalMetricsUtilsTest(unittest.TestCase):
    def test_live_tensor_reference_changes_after_reset(self):
        state = torch.tensor([[1.0, 2.0, 3.0]])
        live_reference = state
        terminal_snapshot = state.clone()

        state.zero_()

        self.assertEqual(live_reference.tolist(), [[0.0, 0.0, 0.0]])
        self.assertEqual(terminal_snapshot.tolist(), [[1.0, 2.0, 3.0]])

    def test_terminal_latch_has_priority(self):
        self.assertEqual(metrics_utils.select_terminal_value(-0.21, 0.0, True), -0.21)
        self.assertEqual(metrics_utils.select_terminal_value(-0.21, 0.0, False), 0.0)

    def test_px4_hierarchical_diagnostics_are_optional(self):
        legacy_task = SimpleNamespace()
        self.assertFalse(metrics_utils.has_px4_hierarchical_diagnostics(legacy_task))

        m2_task = SimpleNamespace(
            **{
                attr_name: object()
                for attr_name in (
                    *metrics_utils.PX4_HIERARCHICAL_SCALAR_LATCHES.values(),
                    *metrics_utils.PX4_HIERARCHICAL_VECTOR_LATCHES.values(),
                )
            }
        )
        self.assertTrue(metrics_utils.has_px4_hierarchical_diagnostics(m2_task))
        delattr(m2_task, "_last_controller_runtime_ms_p95")
        self.assertFalse(metrics_utils.has_px4_hierarchical_diagnostics(m2_task))

    def test_pad_speed_bucket_boundaries(self):
        cases = {
            0.0: "0.00-0.05",
            0.049999: "0.00-0.05",
            0.05: "0.05-0.10",
            0.099999: "0.05-0.10",
            0.10: "0.10-0.15",
            0.149999: "0.10-0.15",
            0.15: ">=0.15",
        }
        for speed, expected in cases.items():
            with self.subTest(speed=speed):
                self.assertEqual(metrics_utils.pad_speed_bucket(speed), expected)

    def test_deck_tilt_bucket_boundaries(self):
        cases = {
            math.radians(0.0): "0-2deg",
            math.radians(1.999): "0-2deg",
            math.radians(2.0): "2-4deg",
            math.radians(3.999): "2-4deg",
            math.radians(4.0): "4-6deg",
            math.radians(5.999): "4-6deg",
            math.radians(6.0): ">=6deg",
        }
        for tilt, expected in cases.items():
            with self.subTest(tilt=tilt):
                self.assertEqual(metrics_utils.deck_tilt_bucket(tilt), expected)

    def test_deck_angular_speed_bucket_boundaries(self):
        cases = {
            0.0: "0.00-0.04",
            0.03999: "0.00-0.04",
            0.04: "0.04-0.08",
            0.07999: "0.04-0.08",
            0.08: "0.08-0.12",
            0.11999: "0.08-0.12",
            0.12: ">=0.12",
        }
        for speed, expected in cases.items():
            with self.subTest(speed=speed):
                self.assertEqual(metrics_utils.deck_angular_speed_bucket(speed), expected)

    def test_touchdown_summary_uses_successful_episodes_only(self):
        episodes = [
            {
                "success": True,
                "align_success": True,
                "crash": False,
                "time_out": False,
                "touchdown_distance": 0.05,
            },
            {
                "success": False,
                "align_success": True,
                "crash": True,
                "time_out": False,
                "touchdown_distance": 99.0,
            },
        ]
        summary = metrics_utils.summarize_ship_landing(episodes)
        self.assertAlmostEqual(summary["touchdown_distance_mean"], 0.05)
        self.assertAlmostEqual(summary["touchdown_distance_p95"], 0.05)

    def test_empty_success_set_returns_nan(self):
        episodes = [
            {
                "success": False,
                "align_success": False,
                "crash": False,
                "time_out": True,
                "touchdown_distance": 0.0,
            }
        ]
        summary = metrics_utils.summarize_ship_landing(episodes)
        self.assertTrue(math.isnan(summary["touchdown_distance_mean"]))
        self.assertTrue(math.isnan(summary["touchdown_distance_p95"]))


if __name__ == "__main__":
    unittest.main()
