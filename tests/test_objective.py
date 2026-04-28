import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpeas.config import AdaptiveObjectiveConfig
from tpeas.objective import adaptive_objective_score


class ObjectiveTests(unittest.TestCase):
    def test_variance_penalty_increases_late_in_budget(self):
        config = AdaptiveObjectiveConfig(budget=10, epsilon=0.2)
        high_variance = adaptive_objective_score([0.0, 4.0], 10, config)
        low_variance = adaptive_objective_score([1.9, 2.1], 10, config)

        self.assertAlmostEqual(high_variance.raw_mean, low_variance.raw_mean)
        self.assertLess(high_variance.objective, low_variance.objective)

    def test_can_disable_variance_penalty_for_raw_baseline(self):
        config = AdaptiveObjectiveConfig(budget=10)
        scored = adaptive_objective_score([0.0, 4.0], 10, config, use_variance_penalty=False)
        self.assertAlmostEqual(scored.objective, 2.0)

    def test_importance_weight_is_clipped_before_variance(self):
        config = AdaptiveObjectiveConfig(budget=10, epsilon=0.1)
        scored = adaptive_objective_score([1.0, 3.0], 10, config, importance_weight=100.0)
        self.assertAlmostEqual(scored.clipped_weight, 1.1)
        self.assertAlmostEqual(scored.raw_variance, 1.0)

    def test_lambda_override_replaces_budget_schedule(self):
        config = AdaptiveObjectiveConfig(budget=10)
        scored = adaptive_objective_score([0.0, 4.0], 10, config, lambda_override=0.0)
        self.assertAlmostEqual(scored.lambda_t, 0.0)
        self.assertAlmostEqual(scored.objective, 2.0)


if __name__ == "__main__":
    unittest.main()
