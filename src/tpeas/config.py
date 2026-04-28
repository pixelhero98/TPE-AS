"""Configuration and scheduling helpers for TPE-AS."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from typing import Sequence

import numpy as np

from tpeas.controllers import LAMBDA_MODES


@dataclass(frozen=True)
class AdaptiveObjectiveConfig:
    """Runtime configuration for adaptive objective evaluation.

    ``budget`` is always the exact evaluation count. Budget-free controllers such
    as ``global_local`` ignore budget progress for lambda, but not for stopping.
    """

    budget: int
    epsilon: float = 0.2
    quantile: float = 0.15
    startup_trials: int = 10
    replicates_per_trial: int = 3
    n_candidates: int = 64
    random_fraction: float = 0.05
    lambda_mode: str = "budget"
    recent_window: int = 30
    previous_window: int = 30
    min_recent_history: int = 30
    recent_variance_weight: float = 0.75
    recent_mean_drop_weight: float = 0.25
    variance_ratio_full_scale: float = 3.0
    global_window_min_history: int = 30
    global_noise_weight: float = 0.5
    global_quality_weight: float = 0.5
    global_controller_weight: float = 1.0
    local_controller_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        if self.startup_trials < 0:
            raise ValueError("startup_trials must be non-negative")
        if self.replicates_per_trial <= 0:
            raise ValueError("replicates_per_trial must be positive")
        if self.n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        if not 0.0 <= self.random_fraction <= 1.0:
            raise ValueError("random_fraction must be in [0, 1]")
        if self.lambda_mode not in LAMBDA_MODES:
            raise ValueError("lambda_mode must be 'budget', 'recent', or 'global_local'")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.previous_window <= 0:
            raise ValueError("previous_window must be positive")
        if self.min_recent_history < 0:
            raise ValueError("min_recent_history must be non-negative")
        if self.recent_variance_weight < 0.0:
            raise ValueError("recent_variance_weight must be non-negative")
        if self.recent_mean_drop_weight < 0.0:
            raise ValueError("recent_mean_drop_weight must be non-negative")
        total_weight = self.recent_variance_weight + self.recent_mean_drop_weight
        if total_weight <= 0.0:
            raise ValueError("recent controller weights must have a positive sum")
        if self.variance_ratio_full_scale <= 1.0:
            raise ValueError("variance_ratio_full_scale must be greater than 1")
        if self.global_window_min_history < 0:
            raise ValueError("global_window_min_history must be non-negative")
        if self.global_noise_weight < 0.0:
            raise ValueError("global_noise_weight must be non-negative")
        if self.global_quality_weight < 0.0:
            raise ValueError("global_quality_weight must be non-negative")
        global_total_weight = self.global_noise_weight + self.global_quality_weight
        if global_total_weight <= 0.0:
            raise ValueError("global controller weights must have a positive sum")
        if self.global_controller_weight < 0.0:
            raise ValueError("global_controller_weight must be non-negative")
        if self.local_controller_weight < 0.0:
            raise ValueError("local_controller_weight must be non-negative")
        controller_weight = self.global_controller_weight + self.local_controller_weight
        if controller_weight <= 0.0:
            raise ValueError("global/local controller weights must have a positive sum")


@dataclass(frozen=True)
class LambdaDiagnostics:
    lambda_t: float
    lambda_mode: str
    recent_mean: float | None = None
    previous_mean: float | None = None
    recent_variance_mean: float | None = None
    baseline_variance: float | None = None
    noise_pressure: float = 0.0
    mean_drop_pressure: float = 0.0
    global_mean_percentile: float | None = None
    global_variance_percentile: float | None = None
    global_noise_pressure: float = 0.0
    global_quality_pressure: float = 0.0
    global_pressure: float = 0.0
    local_pressure: float = 0.0


def adaptive_lambda(step: int, budget: int) -> float:
    """Return the paper's cosine adaptive penalty for a 1-based optimization step."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    if step <= 0:
        return 0.0
    scaled = min((step / budget) * pi, pi)
    return (1.0 - cos(scaled)) / 2.0


def budget_lambda_diagnostics(step: int, config: AdaptiveObjectiveConfig) -> LambdaDiagnostics:
    """Return budget-scheduled lambda diagnostics."""

    return LambdaDiagnostics(lambda_t=adaptive_lambda(step, config.budget), lambda_mode="budget")


def recent_lambda_diagnostics(
    raw_means: Sequence[float],
    raw_variances: Sequence[float],
    config: AdaptiveObjectiveConfig,
) -> LambdaDiagnostics:
    """Return pure recent-trajectory controller diagnostics."""

    if len(raw_means) != len(raw_variances):
        raise ValueError("raw_means and raw_variances must have the same length")
    if len(raw_means) < config.min_recent_history or len(raw_means) < config.recent_window:
        return LambdaDiagnostics(lambda_t=0.0, lambda_mode="recent")

    means = np.asarray(raw_means, dtype=float)
    variances = np.asarray(raw_variances, dtype=float)
    recent_means = means[-config.recent_window :]
    recent_variances = variances[-config.recent_window :]
    previous_start = max(0, len(means) - config.recent_window - config.previous_window)
    previous_end = len(means) - config.recent_window
    previous_means = means[previous_start:previous_end]

    recent_mean = float(np.mean(recent_means))
    previous_mean = float(np.mean(previous_means)) if previous_means.size else recent_mean
    recent_variance_mean = float(np.mean(recent_variances))
    baseline_variance = float(np.median(variances))
    baseline_variance = max(baseline_variance, 1e-12)
    history_std = float(np.std(means))
    history_std = max(history_std, 1e-12)

    variance_ratio = recent_variance_mean / baseline_variance
    noise_pressure = _clip01(
        (variance_ratio - 1.0) / (config.variance_ratio_full_scale - 1.0)
    )
    mean_drop_pressure = _clip01((previous_mean - recent_mean) / history_std)
    total_weight = config.recent_variance_weight + config.recent_mean_drop_weight
    lambda_t = (
        config.recent_variance_weight * noise_pressure
        + config.recent_mean_drop_weight * mean_drop_pressure
    ) / total_weight

    return LambdaDiagnostics(
        lambda_t=float(_clip01(lambda_t)),
        lambda_mode="recent",
        recent_mean=recent_mean,
        previous_mean=previous_mean,
        recent_variance_mean=recent_variance_mean,
        baseline_variance=baseline_variance,
        noise_pressure=float(noise_pressure),
        mean_drop_pressure=float(mean_drop_pressure),
        local_pressure=float(_clip01(lambda_t)),
    )


def global_local_lambda_diagnostics(
    raw_means: Sequence[float],
    raw_variances: Sequence[float],
    config: AdaptiveObjectiveConfig,
) -> LambdaDiagnostics:
    """Return budget-free global-plus-local controller diagnostics."""

    if len(raw_means) != len(raw_variances):
        raise ValueError("raw_means and raw_variances must have the same length")

    min_history = max(
        config.global_window_min_history,
        config.min_recent_history,
        config.recent_window,
    )
    if len(raw_means) < min_history:
        return LambdaDiagnostics(lambda_t=0.0, lambda_mode="global_local")

    local = recent_lambda_diagnostics(raw_means, raw_variances, config)
    if local.recent_mean is None or local.recent_variance_mean is None:
        return LambdaDiagnostics(lambda_t=0.0, lambda_mode="global_local")

    mean_percentile = _percentile_rank(local.recent_mean, raw_means)
    variance_percentile = _percentile_rank(local.recent_variance_mean, raw_variances)
    global_noise_pressure = _clip01(2.0 * (variance_percentile - 0.5))
    global_quality_pressure = _clip01(2.0 * (0.5 - mean_percentile))
    global_weight_total = config.global_noise_weight + config.global_quality_weight
    global_pressure = (
        config.global_noise_weight * global_noise_pressure
        + config.global_quality_weight * global_quality_pressure
    ) / global_weight_total
    local_pressure = local.lambda_t
    lambda_t = _clip01(
        max(
            config.global_controller_weight * global_pressure,
            config.local_controller_weight * local_pressure,
        )
    )

    return LambdaDiagnostics(
        lambda_t=float(lambda_t),
        lambda_mode="global_local",
        recent_mean=local.recent_mean,
        previous_mean=local.previous_mean,
        recent_variance_mean=local.recent_variance_mean,
        baseline_variance=local.baseline_variance,
        noise_pressure=local.noise_pressure,
        mean_drop_pressure=local.mean_drop_pressure,
        global_mean_percentile=float(mean_percentile),
        global_variance_percentile=float(variance_percentile),
        global_noise_pressure=float(global_noise_pressure),
        global_quality_pressure=float(global_quality_pressure),
        global_pressure=float(global_pressure),
        local_pressure=float(local_pressure),
    )


def lambda_diagnostics_from_history(
    step: int,
    raw_means: Sequence[float],
    raw_variances: Sequence[float],
    config: AdaptiveObjectiveConfig,
) -> LambdaDiagnostics:
    """Dispatch to the configured lambda controller."""

    if config.lambda_mode == "global_local":
        return global_local_lambda_diagnostics(raw_means, raw_variances, config)
    if config.lambda_mode == "recent":
        return recent_lambda_diagnostics(raw_means, raw_variances, config)
    return budget_lambda_diagnostics(step, config)


def optimization_steps(config: AdaptiveObjectiveConfig) -> range:
    """Return the exact 1-based evaluation steps for an optimization run."""

    return range(1, config.budget + 1)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _percentile_rank(value: float, samples: Sequence[float]) -> float:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return 0.5
    below = float(np.sum(values < value))
    tied = float(np.sum(np.isclose(values, value, rtol=1e-12, atol=1e-12)))
    return _clip01((below + 0.5 * tied) / values.size)


def clip_importance_weight(weight: float, epsilon: float) -> float:
    """Clip an empirical importance weight to [1 - epsilon, 1 + epsilon]."""

    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    lower = 1.0 - epsilon
    upper = 1.0 + epsilon
    return max(lower, min(upper, float(weight)))
