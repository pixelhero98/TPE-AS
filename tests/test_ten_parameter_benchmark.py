import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.compare_epsilon_tpeas import run_comparison
from experiments.run_ten_parameter_tpeas import run_experiment
from tpeas.evaluators import TenParameterSyntheticBenchmark, ten_parameter_synthetic_search_space
from tpeas.search_space import CategoricalParam, FloatParam, IntParam


STABLE_PARAMS = {
    "trend_strength": 0.55,
    "mean_reversion": 0.25,
    "risk_budget": 0.35,
    "leverage": 1.05,
    "entry_threshold": 0.55,
    "lookback_short": 18,
    "lookback_long": 90,
    "rebalance_days": 7,
    "signal_family": "hybrid",
    "market_regime": "calm",
}

RISKY_PARAMS = {
    "trend_strength": 1.20,
    "mean_reversion": -0.85,
    "risk_budget": 0.82,
    "leverage": 2.35,
    "entry_threshold": 0.20,
    "lookback_short": 8,
    "lookback_long": 34,
    "rebalance_days": 2,
    "signal_family": "trend",
    "market_regime": "volatile",
}


class TenParameterBenchmarkTests(unittest.TestCase):
    def test_search_space_has_exactly_ten_mixed_parameters(self):
        space = ten_parameter_synthetic_search_space()
        self.assertEqual(len(space.params), 10)
        self.assertTrue(any(isinstance(param, FloatParam) for param in space.params))
        self.assertTrue(any(isinstance(param, IntParam) for param in space.params))
        self.assertTrue(any(isinstance(param, CategoricalParam) for param in space.params))
        self.assertEqual(
            set(space.names),
            {
                "trend_strength",
                "mean_reversion",
                "risk_budget",
                "leverage",
                "entry_threshold",
                "lookback_short",
                "lookback_long",
                "rebalance_days",
                "signal_family",
                "market_regime",
            },
        )

    def test_search_space_sample_bounds_are_valid(self):
        space = ten_parameter_synthetic_search_space()
        rng = np.random.default_rng(123)
        sample = space.sample(rng)

        self.assertGreaterEqual(sample["trend_strength"], -1.5)
        self.assertLessEqual(sample["trend_strength"], 1.5)
        self.assertGreaterEqual(sample["mean_reversion"], -1.5)
        self.assertLessEqual(sample["mean_reversion"], 1.5)
        self.assertGreaterEqual(sample["risk_budget"], 0.05)
        self.assertLessEqual(sample["risk_budget"], 1.0)
        self.assertGreaterEqual(sample["leverage"], 0.25)
        self.assertLessEqual(sample["leverage"], 3.0)
        self.assertGreaterEqual(sample["entry_threshold"], 0.05)
        self.assertLessEqual(sample["entry_threshold"], 1.5)
        self.assertGreaterEqual(sample["lookback_short"], 3)
        self.assertLessEqual(sample["lookback_short"], 60)
        self.assertGreaterEqual(sample["lookback_long"], 20)
        self.assertLessEqual(sample["lookback_long"], 180)
        self.assertGreaterEqual(sample["rebalance_days"], 1)
        self.assertLessEqual(sample["rebalance_days"], 30)
        self.assertIn(sample["signal_family"], {"trend", "reversion", "hybrid"})
        self.assertIn(sample["market_regime"], {"calm", "volatile", "crisis"})

    def test_evaluator_is_reproducible_for_same_seed(self):
        evaluator = TenParameterSyntheticBenchmark()
        first = evaluator.evaluate(STABLE_PARAMS, np.random.default_rng(99))
        second = evaluator.evaluate(STABLE_PARAMS, np.random.default_rng(99))
        self.assertAlmostEqual(first, second)
        self.assertEqual(evaluator.basin_label(STABLE_PARAMS), "stable")
        self.assertEqual(evaluator.basin_label(RISKY_PARAMS), "risky")

    def test_risky_basin_has_higher_empirical_variance(self):
        evaluator = TenParameterSyntheticBenchmark()
        stable_rng = np.random.default_rng(12)
        risky_rng = np.random.default_rng(12)
        stable = [evaluator.evaluate(STABLE_PARAMS, stable_rng) for _ in range(400)]
        risky = [evaluator.evaluate(RISKY_PARAMS, risky_rng) for _ in range(400)]
        self.assertGreater(np.var(risky), np.var(stable) * 20.0)

    def test_ten_parameter_runner_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = run_experiment(
                budget=9,
                startup_trials=3,
                replicates_per_trial=2,
                n_candidates=8,
                epsilon=0.035,
                seed=5,
                output_dir=Path(temp_dir),
                run_name="smoke",
                make_plot=False,
            )
            trajectory_path = run_dir / "trajectory.csv"
            summary_path = run_dir / "summary.csv"
            self.assertTrue(trajectory_path.exists())
            self.assertTrue((run_dir / "trajectory.jsonl").exists())
            self.assertTrue(summary_path.exists())

            with trajectory_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            for name in ten_parameter_synthetic_search_space().names:
                self.assertIn(name, rows[0])
            self.assertIn("basin_label", rows[0])

            with summary_path.open("r", encoding="utf-8") as handle:
                metrics = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
            self.assertEqual(metrics["budget"], "9")
            self.assertAlmostEqual(float(metrics["epsilon"]), 0.035)
            self.assertAlmostEqual(float(metrics["clip_lower"]), 0.965)
            self.assertAlmostEqual(float(metrics["clip_upper"]), 1.035)
            self.assertIn("selected_step", metrics)
            self.assertIn("selected_params", metrics)
            self.assertEqual(metrics["selected_step"], metrics["best_objective_step"])
            self.assertEqual(metrics["final_step"], "9")
            self.assertIn("best_raw_params", metrics)
            self.assertIn("late_raw_variance_mean", metrics)

    def test_epsilon_comparison_runner_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_dir = run_comparison(
                epsilons=[0.035, 0.1],
                budget=8,
                startup_trials=3,
                replicates_per_trial=2,
                n_candidates=8,
                seed=4,
                output_dir=Path(temp_dir),
                run_name="comparison_smoke",
                make_plot=False,
            )
            comparison_path = comparison_dir / "epsilon_comparison.csv"
            self.assertTrue(comparison_path.exists())
            self.assertTrue((comparison_dir / "epsilon_0p035" / "trajectory.csv").exists())
            self.assertTrue((comparison_dir / "epsilon_0p1" / "trajectory.csv").exists())

            with comparison_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual([float(row["epsilon"]) for row in rows], [0.035, 0.1])
            self.assertAlmostEqual(float(rows[0]["clip_lower"]), 0.965)
            self.assertAlmostEqual(float(rows[1]["clip_upper"]), 1.1)


if __name__ == "__main__":
    unittest.main()
