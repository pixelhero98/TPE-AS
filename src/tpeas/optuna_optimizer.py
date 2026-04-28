"""Optuna TPESampler wrapper for the adaptive objective."""

from __future__ import annotations

import time

import numpy as np

from tpeas.config import (
    AdaptiveObjectiveConfig,
    lambda_diagnostics_from_history,
    optimization_steps,
)
from tpeas.evaluators import BlackBoxEvaluator
from tpeas.history import OptimizationResult, TrialRecord
from tpeas.objective import adaptive_objective_score
from tpeas.search_space import SearchSpace


class OptunaTPEASOptimizer:
    """Practical Stage 1 implementation backed by Optuna's TPESampler."""

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: BlackBoxEvaluator,
        config: AdaptiveObjectiveConfig,
        seed: int = 0,
        use_variance_penalty: bool = True,
        sampler_name: str = "optuna-tpe-as",
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_variance_penalty = use_variance_penalty
        self.sampler_name = sampler_name

    def optimize(self) -> OptimizationResult:
        """Run exactly ``config.budget`` Optuna trials and return TPE-AS history."""

        try:
            import optuna
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Optuna is required for OptunaTPEASOptimizer. "
                "Install with: python -m pip install -e .[optuna]"
            ) from exc

        records: list[TrialRecord] = []
        steps = optimization_steps(self.config)
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.config.startup_trials,
            multivariate=True,
            group=True,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            step = len(records) + 1
            start = time.perf_counter()
            params = self.search_space.suggest_optuna(trial)
            samples = [
                self.evaluator.evaluate(params, self.rng)
                for _ in range(self.config.replicates_per_trial)
            ]
            lambda_diagnostics = lambda_diagnostics_from_history(
                step,
                [record.raw_mean for record in records],
                [record.raw_variance for record in records],
                self.config,
            )
            evaluation = adaptive_objective_score(
                samples,
                step,
                self.config,
                importance_weight=1.0,
                use_variance_penalty=self.use_variance_penalty,
                lambda_override=lambda_diagnostics.lambda_t,
            )
            record = TrialRecord(
                step=step,
                params=params,
                raw_samples=evaluation.raw_samples,
                raw_mean=evaluation.raw_mean,
                raw_variance=evaluation.raw_variance,
                lambda_t=evaluation.lambda_t,
                importance_weight=evaluation.importance_weight,
                clipped_weight=evaluation.clipped_weight,
                objective=evaluation.objective,
                elapsed_seconds=time.perf_counter() - start,
                sampler=self.sampler_name,
                lambda_mode=lambda_diagnostics.lambda_mode,
                recent_mean=lambda_diagnostics.recent_mean,
                previous_mean=lambda_diagnostics.previous_mean,
                recent_variance_mean=lambda_diagnostics.recent_variance_mean,
                baseline_variance=lambda_diagnostics.baseline_variance,
                noise_pressure=lambda_diagnostics.noise_pressure,
                mean_drop_pressure=lambda_diagnostics.mean_drop_pressure,
                global_mean_percentile=lambda_diagnostics.global_mean_percentile,
                global_variance_percentile=lambda_diagnostics.global_variance_percentile,
                global_noise_pressure=lambda_diagnostics.global_noise_pressure,
                global_quality_pressure=lambda_diagnostics.global_quality_pressure,
                global_pressure=lambda_diagnostics.global_pressure,
                local_pressure=lambda_diagnostics.local_pressure,
            )
            records.append(record)
            trial.set_user_attr("raw_mean", record.raw_mean)
            trial.set_user_attr("raw_variance", record.raw_variance)
            trial.set_user_attr("lambda_t", record.lambda_t)
            trial.set_user_attr("lambda_mode", record.lambda_mode)
            trial.set_user_attr("raw_samples", record.raw_samples)
            return record.objective

        study.optimize(objective, n_trials=len(steps), show_progress_bar=False)
        return OptimizationResult(records=tuple(records))
