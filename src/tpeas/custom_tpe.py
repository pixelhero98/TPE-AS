"""Custom TPE-AS optimizer with explicit Parzen density models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import exp, log, pi, sqrt
from typing import Any

import numpy as np

from tpeas.config import (
    AdaptiveObjectiveConfig,
    lambda_diagnostics_from_history,
    optimization_steps,
)
from tpeas.evaluators import BlackBoxEvaluator
from tpeas.history import OptimizationResult, TrialRecord
from tpeas.objective import adaptive_objective_score
from tpeas.search_space import CategoricalParam, FloatParam, IntParam, SearchSpace

_LOG_FLOOR = -745.0


class _ParamDensity:
    def sample(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    def log_density(self, value: Any) -> float:
        raise NotImplementedError


@dataclass
class _NumericDensity(_ParamDensity):
    param: FloatParam | IntParam
    centers: np.ndarray
    bandwidth: float
    prior_weight: float = 0.05

    @classmethod
    def fit(cls, param: FloatParam | IntParam, values: list[Any]) -> "_NumericDensity":
        low, high = param.internal_bounds
        if values:
            centers = np.asarray([param.to_internal(value) for value in values], dtype=float)
        else:
            centers = np.asarray([(low + high) / 2.0], dtype=float)

        span = max(high - low, 1e-9)
        if centers.size > 1:
            spread = float(np.std(centers))
        else:
            spread = 0.0
        bandwidth = max(spread, span / (centers.size + 2), span * 0.03, 1e-9)
        return cls(param=param, centers=centers, bandwidth=bandwidth)

    def sample(self, rng: np.random.Generator) -> Any:
        low, high = self.param.internal_bounds
        center = float(rng.choice(self.centers))
        value = float(rng.normal(center, self.bandwidth))
        value = float(np.clip(value, low, high))
        return self.param.from_internal(value)

    def log_density(self, value: Any) -> float:
        low, high = self.param.internal_bounds
        encoded = self.param.to_internal(value)
        z = (encoded - self.centers) / self.bandwidth
        gaussian = np.exp(-0.5 * z * z) / (self.bandwidth * sqrt(2.0 * pi))
        kde_density = float(np.mean(gaussian))
        uniform_density = 1.0 / max(high - low, 1e-9)
        density = (1.0 - self.prior_weight) * kde_density + self.prior_weight * uniform_density
        return log(max(density, 1e-323))


@dataclass
class _CategoricalDensity(_ParamDensity):
    param: CategoricalParam
    probabilities: np.ndarray

    @classmethod
    def fit(cls, param: CategoricalParam, values: list[Any]) -> "_CategoricalDensity":
        counts = np.ones(len(param.choices), dtype=float)
        for value in values:
            counts[param.choices.index(value)] += 1.0
        probabilities = counts / np.sum(counts)
        return cls(param=param, probabilities=probabilities)

    def sample(self, rng: np.random.Generator) -> Any:
        index = int(rng.choice(np.arange(len(self.param.choices)), p=self.probabilities))
        return self.param.choices[index]

    def log_density(self, value: Any) -> float:
        index = self.param.choices.index(value)
        return log(max(float(self.probabilities[index]), 1e-323))


@dataclass
class _ParzenModel:
    densities: dict[str, _ParamDensity]

    @classmethod
    def fit(cls, search_space: SearchSpace, records: list[TrialRecord]) -> "_ParzenModel":
        densities: dict[str, _ParamDensity] = {}
        for param in search_space.params:
            values = [record.params[param.name] for record in records]
            if isinstance(param, CategoricalParam):
                densities[param.name] = _CategoricalDensity.fit(param, values)
            else:
                densities[param.name] = _NumericDensity.fit(param, values)
        return cls(densities=densities)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {name: density.sample(rng) for name, density in self.densities.items()}

    def log_density(self, params: dict[str, Any]) -> float:
        return float(sum(density.log_density(params[name]) for name, density in self.densities.items()))


def _split_records(
    records: list[TrialRecord],
    score_name: str,
    quantile: float,
) -> tuple[list[TrialRecord], list[TrialRecord]]:
    ordered = sorted(records, key=lambda record: getattr(record, score_name), reverse=True)
    elite_count = max(1, int(np.ceil(len(ordered) * quantile)))
    good = ordered[:elite_count]
    bad = ordered[elite_count:] or ordered[-elite_count:]
    return good, bad


def _split_records_by_robust_score(
    records: list[TrialRecord], quantile: float
) -> tuple[list[TrialRecord], list[TrialRecord]]:
    ordered = sorted(
        records,
        key=lambda record: record.raw_mean - record.raw_variance,
        reverse=True,
    )
    elite_count = max(1, int(np.ceil(len(ordered) * quantile)))
    good = ordered[:elite_count]
    bad = ordered[elite_count:] or ordered[-elite_count:]
    return good, bad


class CustomTPEASOptimizer:
    """Generic custom TPE optimizer using the adaptive TPE-AS objective."""

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: BlackBoxEvaluator,
        config: AdaptiveObjectiveConfig,
        seed: int = 0,
        use_variance_penalty: bool = True,
        use_importance: bool = True,
        sampler_name: str = "custom-tpe-as",
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.use_variance_penalty = use_variance_penalty
        self.use_importance = use_importance
        self.sampler_name = sampler_name

    def optimize(self) -> OptimizationResult:
        """Run exactly ``config.budget`` evaluations and return the full history.

        Use ``result.best_candidate()`` to retrieve the selected incumbent. The last
        sampled record is retained as trajectory diagnostics, not as the answer.
        """

        records: list[TrialRecord] = []
        for step in optimization_steps(self.config):
            start = time.perf_counter()
            params, proposal_model, target_model = self._next_params(records)
            importance_weight = self._importance_weight(params, proposal_model, target_model)
            lambda_diagnostics = lambda_diagnostics_from_history(
                step,
                [record.raw_mean for record in records],
                [record.raw_variance for record in records],
                self.config,
            )
            samples = [
                self.evaluator.evaluate(params, self.rng)
                for _ in range(self.config.replicates_per_trial)
            ]
            evaluation = adaptive_objective_score(
                samples,
                step,
                self.config,
                importance_weight=importance_weight,
                use_variance_penalty=self.use_variance_penalty,
                lambda_override=lambda_diagnostics.lambda_t,
            )
            records.append(
                TrialRecord(
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
            )
        return OptimizationResult(records=tuple(records))

    def _next_params(
        self, records: list[TrialRecord]
    ) -> tuple[dict[str, Any], _ParzenModel | None, _ParzenModel | None]:
        if (
            len(records) < max(1, self.config.startup_trials)
            or self.rng.random() < self.config.random_fraction
        ):
            return self.search_space.sample(self.rng), None, None

        good, bad = _split_records(records, "objective", self.config.quantile)
        good_model = _ParzenModel.fit(self.search_space, good)
        bad_model = _ParzenModel.fit(self.search_space, bad)
        robust_good, _ = _split_records_by_robust_score(records, self.config.quantile)
        target_model = _ParzenModel.fit(self.search_space, robust_good)

        best_params: dict[str, Any] | None = None
        best_score = -np.inf
        for _ in range(self.config.n_candidates):
            candidate = good_model.sample(self.rng)
            score = good_model.log_density(candidate) - bad_model.log_density(candidate)
            if score > best_score:
                best_score = score
                best_params = candidate

        if best_params is None:
            best_params = self.search_space.sample(self.rng)
        return best_params, good_model, target_model

    def _importance_weight(
        self,
        params: dict[str, Any],
        proposal_model: _ParzenModel | None,
        target_model: _ParzenModel | None,
    ) -> float:
        if not self.use_importance or proposal_model is None or target_model is None:
            return 1.0
        log_ratio = target_model.log_density(params) - proposal_model.log_density(params)
        log_ratio = max(_LOG_FLOOR, min(50.0, log_ratio))
        return float(exp(log_ratio))
