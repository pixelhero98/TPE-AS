"""Adaptive objective scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tpeas.config import AdaptiveObjectiveConfig, adaptive_lambda, clip_importance_weight


@dataclass(frozen=True)
class AdaptiveEvaluation:
    raw_samples: tuple[float, ...]
    raw_mean: float
    raw_variance: float
    lambda_t: float
    importance_weight: float
    clipped_weight: float
    objective: float


def adaptive_objective_score(
    samples: Iterable[float],
    step: int,
    config: AdaptiveObjectiveConfig,
    importance_weight: float = 1.0,
    use_variance_penalty: bool = True,
    lambda_override: float | None = None,
) -> AdaptiveEvaluation:
    """Score repeated black-box observations with the TPE-AS adaptive objective."""

    raw = np.asarray(tuple(float(value) for value in samples), dtype=float)
    if raw.size == 0:
        raise ValueError("samples must contain at least one observation")

    raw_mean = float(np.mean(raw))
    raw_variance = float(np.var(raw))
    lambda_t = adaptive_lambda(step, config.budget) if lambda_override is None else float(lambda_override)
    clipped_weight = clip_importance_weight(importance_weight, config.epsilon)
    weighted_variance = float(np.var(raw * clipped_weight))
    objective = raw_mean
    if use_variance_penalty:
        objective -= lambda_t * weighted_variance

    return AdaptiveEvaluation(
        raw_samples=tuple(float(value) for value in raw),
        raw_mean=raw_mean,
        raw_variance=raw_variance,
        lambda_t=lambda_t,
        importance_weight=float(importance_weight),
        clipped_weight=clipped_weight,
        objective=float(objective),
    )
