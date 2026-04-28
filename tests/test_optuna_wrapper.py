import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tpeas.config import AdaptiveObjectiveConfig
from tpeas.evaluators import SyntheticInstabilityBenchmark, synthetic_instability_search_space
from tpeas.optuna_optimizer import OptunaTPEASOptimizer


class OptunaWrapperTests(unittest.TestCase):
    def test_optuna_wrapper_smoke_or_clear_missing_dependency(self):
        optimizer = OptunaTPEASOptimizer(
            synthetic_instability_search_space(),
            SyntheticInstabilityBenchmark(),
            AdaptiveObjectiveConfig(budget=5, startup_trials=2, replicates_per_trial=2),
            seed=3,
        )
        if importlib.util.find_spec("optuna") is None:
            with self.assertRaisesRegex(RuntimeError, "Optuna is required"):
                optimizer.optimize()
        else:
            result = optimizer.optimize()
            self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()

