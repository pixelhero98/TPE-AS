import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.custom_model_optimization import run_example
from tpeas.config import AdaptiveObjectiveConfig
from tpeas.custom_tpe import CustomTPEASOptimizer
from tpeas.evaluators import SyntheticInstabilityBenchmark, synthetic_instability_search_space
from tpeas.history import OptimizationResult, TrialRecord
from tpeas.search_space import CategoricalParam, FloatParam, IntParam, SearchSpace


class CustomTPETests(unittest.TestCase):
    def test_best_candidate_returns_incumbent_by_objective_not_last_trial(self):
        records = (
            trial_record(step=1, raw_mean=0.0, objective=0.0),
            trial_record(step=2, raw_mean=1.0, objective=2.0),
            trial_record(step=3, raw_mean=3.0, objective=1.0),
        )
        result = OptimizationResult(records=records)

        self.assertEqual(result.best_candidate().step, 2)
        self.assertEqual(result.best_by_objective.step, 2)
        self.assertEqual(result.best_by_raw_mean.step, 3)
        self.assertEqual(result.records[-1].step, 3)

        with self.assertRaises(ValueError):
            OptimizationResult(records=()).best_candidate()

    def test_custom_tpe_smoke(self):
        config = AdaptiveObjectiveConfig(
            budget=18,
            startup_trials=5,
            replicates_per_trial=2,
            n_candidates=12,
        )
        optimizer = CustomTPEASOptimizer(
            synthetic_instability_search_space(),
            SyntheticInstabilityBenchmark(),
            config,
            seed=42,
        )
        result = optimizer.optimize()

        self.assertEqual(len(result), 18)
        self.assertEqual(result.records[0].step, 1)
        self.assertEqual(result.records[-1].step, 18)
        self.assertLessEqual(result.best_candidate().step, 18)
        self.assertIsNotNone(result.best_by_objective.params)
        self.assertTrue(all(record.importance_weight > 0.0 for record in result.records))
        self.assertTrue(all(0.8 <= record.clipped_weight <= 1.2 for record in result.records))

    def test_raw_baseline_uses_raw_mean_as_objective(self):
        config = AdaptiveObjectiveConfig(
            budget=8,
            startup_trials=3,
            replicates_per_trial=2,
            n_candidates=8,
        )
        optimizer = CustomTPEASOptimizer(
            synthetic_instability_search_space(),
            SyntheticInstabilityBenchmark(),
            config,
            seed=1,
            use_variance_penalty=False,
            use_importance=False,
            sampler_name="custom-tpe-raw",
        )
        result = optimizer.optimize()
        for record in result.records:
            self.assertAlmostEqual(record.objective, record.raw_mean)
            self.assertAlmostEqual(record.importance_weight, 1.0)

    def test_custom_tpe_global_local_stops_exactly_at_budget(self):
        config = AdaptiveObjectiveConfig(
            budget=7,
            startup_trials=3,
            replicates_per_trial=2,
            n_candidates=8,
            lambda_mode="global_local",
            recent_window=2,
            previous_window=2,
            min_recent_history=2,
            global_window_min_history=2,
        )
        optimizer = CustomTPEASOptimizer(
            synthetic_instability_search_space(),
            SyntheticInstabilityBenchmark(),
            config,
            seed=7,
        )
        result = optimizer.optimize()

        self.assertEqual(len(result), 7)
        self.assertEqual(result.records[-1].step, 7)
        self.assertEqual(result.records[-1].lambda_mode, "global_local")

    def test_custom_tpe_accepts_more_than_ten_parameters(self):
        space = high_dimensional_search_space()
        config = AdaptiveObjectiveConfig(
            budget=8,
            startup_trials=3,
            replicates_per_trial=2,
            n_candidates=10,
            lambda_mode="global_local",
            recent_window=2,
            previous_window=2,
            min_recent_history=2,
            global_window_min_history=2,
        )
        optimizer = CustomTPEASOptimizer(
            space,
            HighDimensionalBenchmark(),
            config,
            seed=11,
        )
        result = optimizer.optimize()

        self.assertGreater(len(space.params), 10)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(len(record.params) == len(space.params) for record in result.records))
        self.assertEqual(result.records[-1].lambda_mode, "global_local")
        self.assertEqual(set(result.best_candidate().params), set(space.names))

    def test_custom_model_example_runs_with_tiny_budget(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_example(budget=6, seed=2)

        self.assertEqual(len(result), 6)
        self.assertGreater(len(result.best_candidate().params), 10)
        self.assertEqual(result.records[-1].step, 6)


def trial_record(step: int, raw_mean: float, objective: float) -> TrialRecord:
    return TrialRecord(
        step=step,
        params={"x": step},
        raw_samples=(raw_mean,),
        raw_mean=raw_mean,
        raw_variance=0.0,
        lambda_t=0.0,
        importance_weight=1.0,
        clipped_weight=1.0,
        objective=objective,
        elapsed_seconds=0.0,
        sampler="unit-test",
    )


def high_dimensional_search_space() -> SearchSpace:
    return SearchSpace(
        [
            FloatParam("x0", -1.0, 1.0),
            FloatParam("x1", -1.0, 1.0),
            FloatParam("x2", -1.0, 1.0),
            FloatParam("x3", -1.0, 1.0),
            FloatParam("x4", -1.0, 1.0),
            FloatParam("scale", 0.1, 2.0),
            IntParam("i0", 1, 9),
            IntParam("i1", 1, 9),
            IntParam("i2", 1, 9),
            IntParam("i3", 1, 9),
            CategoricalParam("family", ["a", "b", "c"]),
            CategoricalParam("mode", ["fast", "balanced", "careful"]),
        ]
    )


class HighDimensionalBenchmark:
    def evaluate(self, params, rng):
        score = 1.0
        score -= (float(params["x0"]) - 0.2) ** 2
        score -= (float(params["x1"]) + 0.3) ** 2
        score -= 0.5 * (float(params["x2"]) - 0.4) ** 2
        score -= 0.4 * (float(params["x3"]) + 0.1) ** 2
        score -= 0.3 * (float(params["x4"]) - 0.6) ** 2
        score -= 0.2 * (float(params["scale"]) - 1.1) ** 2
        score -= 0.03 * (int(params["i0"]) - 4) ** 2
        score -= 0.02 * (int(params["i1"]) - 6) ** 2
        score -= 0.02 * (int(params["i2"]) - 3) ** 2
        score -= 0.01 * (int(params["i3"]) - 7) ** 2
        score += {"a": 0.0, "b": 0.12, "c": -0.05}[params["family"]]
        score += {"fast": -0.04, "balanced": 0.08, "careful": 0.02}[params["mode"]]
        return float(score + rng.normal(0.0, 0.03))


if __name__ == "__main__":
    unittest.main()
