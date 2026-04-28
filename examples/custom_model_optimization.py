"""Optimize a user-defined black-box model with a custom mixed search space."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tpeas import (  # noqa: E402
    AdaptiveObjectiveConfig,
    CategoricalParam,
    CustomTPEASOptimizer,
    FloatParam,
    IntParam,
    SearchSpace,
    resolve_controller_mode,
)


class ExampleModelEvaluator:
    """Small stand-in for a user's noisy training, simulation, or backtest loop."""

    def evaluate(self, params: dict[str, Any], rng: np.random.Generator) -> float:
        learning_rate = float(params["learning_rate"])
        dropout = float(params["dropout"])
        l2 = float(params["l2"])
        momentum = float(params["momentum"])
        feature_fraction = float(params["feature_fraction"])
        label_smoothing = float(params["label_smoothing"])
        depth = int(params["depth"])
        width = int(params["width"])
        batch_size = int(params["batch_size"])
        patience = int(params["patience"])
        activation = str(params["activation"])
        optimizer = str(params["optimizer"])
        scheduler = str(params["scheduler"])

        score = 1.5
        score -= _squared_distance(np.log10(learning_rate), np.log10(0.018), 0.45)
        score -= _squared_distance(dropout, 0.16, 0.16)
        score -= _squared_distance(np.log10(l2), np.log10(0.0008), 0.70)
        score -= _squared_distance(momentum, 0.88, 0.10)
        score -= _squared_distance(feature_fraction, 0.72, 0.18)
        score -= _squared_distance(label_smoothing, 0.04, 0.05)
        score -= _squared_distance(depth, 5.0, 2.2)
        score -= _squared_distance(width, 192.0, 96.0)
        score -= _squared_distance(batch_size, 96.0, 60.0)
        score -= _squared_distance(patience, 8.0, 5.0)

        score += {"gelu": 0.10, "relu": 0.04, "tanh": -0.08}[activation]
        score += {"adamw": 0.12, "adam": 0.05, "sgd": -0.10}[optimizer]
        score += {"cosine": 0.10, "plateau": 0.02, "none": -0.06}[scheduler]

        if optimizer == "sgd" and momentum < 0.75:
            score -= 0.25
        if depth >= 8 and dropout < 0.08:
            score -= 0.20

        noise = 0.03 + 0.05 * dropout + 0.04 * (batch_size < 48)
        return float(score + rng.normal(0.0, noise))


def build_search_space() -> SearchSpace:
    """Return a mixed, 13-parameter space to show variable-dimensional use."""

    return SearchSpace(
        [
            FloatParam("learning_rate", 1e-4, 0.1, log_scale=True),
            FloatParam("dropout", 0.0, 0.45),
            FloatParam("l2", 1e-6, 1e-2, log_scale=True),
            FloatParam("momentum", 0.60, 0.98),
            FloatParam("feature_fraction", 0.40, 1.0),
            FloatParam("label_smoothing", 0.0, 0.20),
            IntParam("depth", 2, 10),
            IntParam("width", 32, 512),
            IntParam("batch_size", 16, 256),
            IntParam("patience", 2, 20),
            CategoricalParam("activation", ["relu", "gelu", "tanh"]),
            CategoricalParam("optimizer", ["adam", "adamw", "sgd"]),
            CategoricalParam("scheduler", ["none", "cosine", "plateau"]),
        ]
    )


def run_example(budget: int = 30, seed: int = 0):
    """Run the example and return the full optimization history."""

    controller = "global_local"
    config = AdaptiveObjectiveConfig(
        budget=budget,
        epsilon=0.1,
        quantile=0.15,
        startup_trials=min(8, max(1, budget // 3)),
        replicates_per_trial=3,
        n_candidates=48,
        random_fraction=0.05,
        lambda_mode=resolve_controller_mode(controller),
        recent_window=8,
        previous_window=8,
        min_recent_history=8,
        global_window_min_history=8,
    )
    optimizer = CustomTPEASOptimizer(
        search_space=build_search_space(),
        evaluator=ExampleModelEvaluator(),
        config=config,
        seed=seed,
    )
    result = optimizer.optimize()
    selected = result.best_candidate()

    print(f"evaluations: {len(result)}")
    print(f"selected_step: {selected.step}")
    print(f"selected_objective: {selected.objective:.4f}")
    print(f"selected_raw_mean: {selected.raw_mean:.4f}")
    print(f"selected_params: {selected.params}")
    return result


def _squared_distance(value: float, target: float, scale: float) -> float:
    return ((value - target) / scale) ** 2


if __name__ == "__main__":
    run_example()
