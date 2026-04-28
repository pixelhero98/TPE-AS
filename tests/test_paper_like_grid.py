import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_paper_like_grid import run_grid
from experiments.compare_lambda_controllers import find_latest_grid_dir
from tpeas.paper_like import (
    MARKET_IDS,
    MARKET_PERIODS,
    MODEL_IDS,
    PaperLikePortfolioBenchmark,
    generate_market_returns,
    paper_like_search_space,
)
from tpeas.search_space import FloatParam, IntParam


EXAMPLE_PARAMS = {
    "entry_threshold": 0.45,
    "exit_threshold": 0.12,
    "risk_budget": 0.35,
    "leverage": 1.2,
    "stop_loss": 0.08,
    "signal_mix": 0.55,
    "lookback_short": 12,
    "lookback_long": 72,
    "rebalance_days": 5,
    "vol_window": 35,
}


class PaperLikeGridTests(unittest.TestCase):
    def test_market_scenarios_are_deterministic_and_expected_length(self):
        for market_id in MARKET_IDS:
            first = generate_market_returns(market_id, seed=11, assets=8)
            second = generate_market_returns(market_id, seed=11, assets=8)
            self.assertEqual(first.shape, (MARKET_PERIODS[market_id], 8))
            self.assertTrue(np.allclose(first, second))
            self.assertGreater(float(np.std(first)), 0.0)

    def test_all_model_ids_evaluate_to_finite_scores(self):
        rng = np.random.default_rng(5)
        for model_id in MODEL_IDS:
            benchmark = PaperLikePortfolioBenchmark(model_id=model_id, market_id="S2", seed=0)
            score = benchmark.evaluate(EXAMPLE_PARAMS, rng)
            deterministic = benchmark.evaluate_deterministic(EXAMPLE_PARAMS)
            self.assertTrue(np.isfinite(score))
            self.assertTrue(np.isfinite(deterministic))

    def test_paper_like_search_space_has_ten_float_and_int_parameters(self):
        space = paper_like_search_space()
        self.assertEqual(len(space.params), 10)
        self.assertEqual(sum(isinstance(param, FloatParam) for param in space.params), 6)
        self.assertEqual(sum(isinstance(param, IntParam) for param in space.params), 4)
        self.assertEqual(
            set(space.names),
            {
                "entry_threshold",
                "exit_threshold",
                "risk_budget",
                "leverage",
                "stop_loss",
                "signal_mix",
                "lookback_short",
                "lookback_long",
                "rebalance_days",
                "vol_window",
            },
        )

    def test_grid_runner_smoke_writes_scenario_and_aggregate_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            grid_dir = run_grid(
                budget=6,
                epsilon=0.1,
                replicates_per_trial=2,
                startup_trials=2,
                n_candidates=6,
                lambda_mode="global_local",
                recent_window=2,
                previous_window=2,
                global_window_min_history=2,
                seed=3,
                models=["M1"],
                markets=["S1"],
                output_dir=Path(temp_dir),
                run_name="paper_smoke",
                make_plot=False,
            )
            scenario_dir = grid_dir / "M1_S1"
            self.assertTrue((scenario_dir / "trajectory.csv").exists())
            self.assertTrue((scenario_dir / "trajectory.jsonl").exists())
            self.assertTrue((scenario_dir / "summary.csv").exists())
            self.assertTrue((grid_dir / "grid_summary.csv").exists())
            self.assertTrue((grid_dir / "scenario_matrix.csv").exists())

            with (scenario_dir / "trajectory.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertIn("importance_weight", rows[0])
            self.assertIn("lambda_mode", rows[0])
            self.assertIn("controller", rows[0])
            self.assertIn("recent_mean", rows[0])
            self.assertIn("noise_pressure", rows[0])
            self.assertIn("global_mean_percentile", rows[0])
            self.assertIn("global_pressure", rows[0])
            self.assertIn("local_pressure", rows[0])
            self.assertIn("entry_threshold", rows[0])

            with (grid_dir / "grid_summary.csv").open("r", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["model_id"], "M1")
            self.assertEqual(summary_rows[0]["market_id"], "S1")
            self.assertEqual(summary_rows[0]["lambda_mode"], "global_local")
            self.assertEqual(summary_rows[0]["controller"], "global_local")
            self.assertIn("global_pressure_mean", summary_rows[0])
            self.assertIn("selected_step", summary_rows[0])
            self.assertIn("selected_params", summary_rows[0])
            self.assertEqual(summary_rows[0]["selected_step"], summary_rows[0]["best_objective_step"])
            self.assertEqual(summary_rows[0]["final_step"], "6")

    def test_controller_precedence_and_legacy_lambda_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            budget_dir = run_grid(
                budget=4,
                epsilon=0.1,
                replicates_per_trial=1,
                startup_trials=1,
                n_candidates=4,
                controller="budget",
                lambda_mode="global_local",
                seed=4,
                models=["M1"],
                markets=["S1"],
                output_dir=Path(temp_dir),
                run_name="controller_precedence",
                make_plot=False,
            )
            with (budget_dir / "grid_summary.csv").open("r", encoding="utf-8") as handle:
                budget_rows = list(csv.DictReader(handle))
            self.assertEqual(budget_rows[0]["lambda_mode"], "budget")
            self.assertEqual(budget_rows[0]["controller"], "budget")

            legacy_dir = run_grid(
                budget=4,
                epsilon=0.1,
                replicates_per_trial=1,
                startup_trials=1,
                n_candidates=4,
                lambda_mode="global_local",
                recent_window=1,
                previous_window=1,
                global_window_min_history=1,
                seed=4,
                models=["M1"],
                markets=["S1"],
                output_dir=Path(temp_dir),
                run_name="legacy_lambda_mode",
                make_plot=False,
            )
            with (legacy_dir / "grid_summary.csv").open("r", encoding="utf-8") as handle:
                legacy_rows = list(csv.DictReader(handle))
            self.assertEqual(legacy_rows[0]["lambda_mode"], "global_local")
            self.assertEqual(legacy_rows[0]["controller"], "global_local")

    def test_comparison_discovers_latest_compatible_grid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            budget_dir = run_grid(
                budget=3,
                epsilon=0.1,
                replicates_per_trial=1,
                startup_trials=1,
                n_candidates=3,
                controller="budget",
                seed=5,
                models=["M1"],
                markets=["S1"],
                output_dir=root,
                run_name="budget_grid",
                make_plot=False,
            )
            recent_dir = run_grid(
                budget=3,
                epsilon=0.1,
                replicates_per_trial=1,
                startup_trials=1,
                n_candidates=3,
                lambda_mode="recent",
                recent_window=1,
                previous_window=1,
                min_recent_history=1,
                seed=5,
                models=["M1"],
                markets=["S1"],
                output_dir=root,
                run_name="recent_grid",
                make_plot=False,
            )
            self.assertEqual(find_latest_grid_dir(root, "budget"), budget_dir)
            self.assertEqual(find_latest_grid_dir(root, "recent"), recent_dir)


if __name__ == "__main__":
    unittest.main()
