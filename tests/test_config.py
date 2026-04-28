import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpeas.config import (
    AdaptiveObjectiveConfig,
    adaptive_lambda,
    clip_importance_weight,
    global_local_lambda_diagnostics,
    lambda_diagnostics_from_history,
    optimization_steps,
    recent_lambda_diagnostics,
)
from tpeas.controllers import (
    PUBLIC_CONTROLLER_MODES,
    TUNABLE_PARAMETERS,
    controller_label,
    resolve_controller_mode,
)


class ConfigTests(unittest.TestCase):
    def test_adaptive_lambda_monotonic_and_bounded(self):
        values = [adaptive_lambda(step, 20) for step in range(0, 25)]
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[20], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))

    def test_adaptive_lambda_matches_formula(self):
        observed = adaptive_lambda(5, 20)
        expected = (1.0 - math.cos((5 / 20) * math.pi)) / 2.0
        self.assertAlmostEqual(observed, expected)

    def test_optimization_steps_are_exactly_one_based_budget_steps(self):
        config = AdaptiveObjectiveConfig(budget=4)
        self.assertEqual(list(optimization_steps(config)), [1, 2, 3, 4])

    def test_clip_importance_weight(self):
        self.assertAlmostEqual(clip_importance_weight(0.1, 0.2), 0.8)
        self.assertAlmostEqual(clip_importance_weight(1.1, 0.2), 1.1)
        self.assertAlmostEqual(clip_importance_weight(9.0, 0.2), 1.2)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            AdaptiveObjectiveConfig(budget=0)
        with self.assertRaises(ValueError):
            AdaptiveObjectiveConfig(budget=10, epsilon=1.0)
        with self.assertRaises(ValueError):
            AdaptiveObjectiveConfig(budget=10, lambda_mode="unknown")
        self.assertEqual(
            AdaptiveObjectiveConfig(budget=10, lambda_mode="global_local").lambda_mode,
            "global_local",
        )

    def test_controller_resolver_prefers_public_controller(self):
        self.assertEqual(PUBLIC_CONTROLLER_MODES, ("budget", "global_local"))
        self.assertTrue(any(parameter.name == "epsilon" for parameter in TUNABLE_PARAMETERS))
        self.assertEqual(resolve_controller_mode("budget", "global_local"), "budget")
        self.assertEqual(resolve_controller_mode("global_local", "budget"), "global_local")
        self.assertEqual(resolve_controller_mode(None, "recent"), "recent")
        with self.assertRaises(ValueError):
            resolve_controller_mode("recent", "budget")
        self.assertEqual(controller_label("budget"), "budget")
        self.assertEqual(controller_label("global_local"), "global_local")
        self.assertEqual(controller_label("recent"), "experimental_recent")

    def test_recent_controller_zero_before_min_history(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="recent",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
        )
        diagnostics = recent_lambda_diagnostics([1.0, 2.0, 3.0], [0.1, 0.1, 0.1], config)
        self.assertEqual(diagnostics.lambda_mode, "recent")
        self.assertAlmostEqual(diagnostics.lambda_t, 0.0)

    def test_recent_high_variance_increases_lambda(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="recent",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
        )
        raw_means = [1.0] * 10
        raw_variances = [0.05] * 5 + [0.40] * 5
        diagnostics = recent_lambda_diagnostics(raw_means, raw_variances, config)
        self.assertGreater(diagnostics.lambda_t, 0.0)
        self.assertGreater(diagnostics.noise_pressure, 0.0)

    def test_recent_mean_improvement_low_variance_keeps_lambda_low(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="recent",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
        )
        raw_means = [0.5] * 5 + [1.2] * 5
        raw_variances = [0.05] * 10
        diagnostics = recent_lambda_diagnostics(raw_means, raw_variances, config)
        self.assertAlmostEqual(diagnostics.lambda_t, 0.0)
        self.assertAlmostEqual(diagnostics.mean_drop_pressure, 0.0)

    def test_recent_mean_deterioration_increases_lambda(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="recent",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
        )
        raw_means = [1.2] * 5 + [0.5] * 5
        raw_variances = [0.05] * 10
        diagnostics = recent_lambda_diagnostics(raw_means, raw_variances, config)
        self.assertGreater(diagnostics.lambda_t, 0.0)
        self.assertGreater(diagnostics.mean_drop_pressure, 0.0)

    def test_budget_mode_dispatch_is_unchanged(self):
        config = AdaptiveObjectiveConfig(budget=20, lambda_mode="budget")
        diagnostics = lambda_diagnostics_from_history(5, [100.0], [999.0], config)
        self.assertEqual(diagnostics.lambda_mode, "budget")
        self.assertAlmostEqual(diagnostics.lambda_t, adaptive_lambda(5, 20))

    def test_global_local_zero_before_min_history(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="global_local",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
            global_window_min_history=10,
        )
        diagnostics = global_local_lambda_diagnostics([1.0] * 5, [0.1] * 5, config)
        self.assertEqual(diagnostics.lambda_mode, "global_local")
        self.assertAlmostEqual(diagnostics.lambda_t, 0.0)

    def test_global_local_high_recent_variance_increases_lambda(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="global_local",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
            global_window_min_history=5,
        )
        raw_means = [1.0] * 20
        raw_variances = [0.05] * 15 + [0.50] * 5
        diagnostics = global_local_lambda_diagnostics(raw_means, raw_variances, config)
        self.assertGreater(diagnostics.lambda_t, 0.0)
        self.assertGreater(diagnostics.global_noise_pressure, 0.0)
        self.assertGreater(diagnostics.local_pressure, 0.0)

    def test_global_local_poor_recent_mean_increases_lambda(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="global_local",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
            global_window_min_history=5,
        )
        raw_means = [1.5] * 15 + [0.2] * 5
        raw_variances = [0.05] * 20
        diagnostics = global_local_lambda_diagnostics(raw_means, raw_variances, config)
        self.assertGreater(diagnostics.lambda_t, 0.0)
        self.assertGreater(diagnostics.global_quality_pressure, 0.0)
        self.assertGreater(diagnostics.mean_drop_pressure, 0.0)

    def test_global_local_strong_recent_mean_normal_variance_stays_low(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="global_local",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
            global_window_min_history=5,
        )
        raw_means = [0.2] * 15 + [1.5] * 5
        raw_variances = [0.05] * 20
        diagnostics = global_local_lambda_diagnostics(raw_means, raw_variances, config)
        self.assertAlmostEqual(diagnostics.lambda_t, 0.0)
        self.assertAlmostEqual(diagnostics.global_pressure, 0.0)
        self.assertAlmostEqual(diagnostics.local_pressure, 0.0)

    def test_global_local_mode_dispatch(self):
        config = AdaptiveObjectiveConfig(
            budget=100,
            lambda_mode="global_local",
            recent_window=5,
            previous_window=5,
            min_recent_history=5,
            global_window_min_history=5,
        )
        diagnostics = lambda_diagnostics_from_history(
            99,
            [1.0] * 15 + [0.5] * 5,
            [0.05] * 15 + [0.40] * 5,
            config,
        )
        self.assertEqual(diagnostics.lambda_mode, "global_local")
        self.assertGreater(diagnostics.lambda_t, 0.0)


if __name__ == "__main__":
    unittest.main()
