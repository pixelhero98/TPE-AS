"""Controller metadata and public selection helpers for TPE-AS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LAMBDA_MODES = ("budget", "recent", "global_local")
PUBLIC_CONTROLLER_MODES = ("budget", "global_local")
EXPERIMENTAL_CONTROLLER_MODES = ("recent",)

CONTROLLER_DESCRIPTIONS = {
    "budget": "known-budget cosine schedule",
    "global_local": "anytime history-only global/local controller",
}


@dataclass(frozen=True)
class TunableParameter:
    group: str
    name: str
    cli_flag: str
    default: Any
    description: str


TUNABLE_PARAMETERS = (
    TunableParameter("shared", "epsilon", "--epsilon", 0.1, "importance-weight clip half-width"),
    TunableParameter("shared", "quantile", "--quantile", 0.15, "elite split for good/bad TPE densities"),
    TunableParameter(
        "shared",
        "startup_trials",
        "--startup-trials",
        30,
        "random warmup trials before model-guided sampling",
    ),
    TunableParameter(
        "shared",
        "replicates_per_trial",
        "--replicates-per-trial",
        5,
        "repeated black-box samples per candidate",
    ),
    TunableParameter(
        "shared",
        "n_candidates",
        "--n-candidates",
        128,
        "candidate samples scored from the TPE proposal",
    ),
    TunableParameter(
        "shared",
        "random_fraction",
        "--random-fraction",
        0.05,
        "probability of a random exploratory trial after warmup",
    ),
    TunableParameter("budget", "budget", "--budget", 300, "planned evaluation count"),
    TunableParameter(
        "global_local",
        "recent_window",
        "--recent-window",
        30,
        "local window used for recent mean and variance",
    ),
    TunableParameter(
        "global_local",
        "previous_window",
        "--previous-window",
        30,
        "comparison window before the recent window",
    ),
    TunableParameter(
        "global_local",
        "min_recent_history",
        "--min-recent-history",
        30,
        "completed trials required before local pressure activates",
    ),
    TunableParameter(
        "global_local",
        "variance_ratio_full_scale",
        "--variance-ratio-full-scale",
        3.0,
        "recent/baseline variance ratio mapped to full local noise pressure",
    ),
    TunableParameter(
        "global_local",
        "recent_variance_weight",
        "--recent-variance-weight",
        0.75,
        "weight for local noise pressure",
    ),
    TunableParameter(
        "global_local",
        "recent_mean_drop_weight",
        "--recent-mean-drop-weight",
        0.25,
        "weight for local mean-drop pressure",
    ),
    TunableParameter(
        "global_local",
        "global_window_min_history",
        "--global-window-min-history",
        30,
        "completed trials required before global pressure activates",
    ),
    TunableParameter(
        "global_local",
        "global_noise_weight",
        "--global-noise-weight",
        0.5,
        "weight for global high-variance percentile pressure",
    ),
    TunableParameter(
        "global_local",
        "global_quality_weight",
        "--global-quality-weight",
        0.5,
        "weight for global low-mean percentile pressure",
    ),
    TunableParameter(
        "global_local",
        "global_controller_weight",
        "--global-controller-weight",
        1.0,
        "multiplier applied to global pressure",
    ),
    TunableParameter(
        "global_local",
        "local_controller_weight",
        "--local-controller-weight",
        1.0,
        "multiplier applied to local pressure",
    ),
)


def resolve_controller_mode(controller: str | None, lambda_mode: str = "budget") -> str:
    """Resolve the public controller switch with legacy lambda-mode fallback."""

    if controller is not None:
        if controller not in PUBLIC_CONTROLLER_MODES:
            raise ValueError("controller must be 'budget' or 'global_local'")
        return controller
    if lambda_mode not in LAMBDA_MODES:
        raise ValueError("lambda_mode must be 'budget', 'recent', or 'global_local'")
    return lambda_mode


def controller_label(lambda_mode: str) -> str:
    """Return a public metadata label for a lambda controller mode."""

    if lambda_mode in PUBLIC_CONTROLLER_MODES:
        return lambda_mode
    return f"experimental_{lambda_mode}"


def public_controller_help() -> str:
    """Return compact CLI help text for the recommended controller branches."""

    return "; ".join(
        f"{name}: {CONTROLLER_DESCRIPTIONS[name]}" for name in PUBLIC_CONTROLLER_MODES
    )
