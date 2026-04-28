"""TPE-AS replication utilities."""

from tpeas.config import (
    AdaptiveObjectiveConfig,
    LambdaDiagnostics,
    adaptive_lambda,
    clip_importance_weight,
    global_local_lambda_diagnostics,
    lambda_diagnostics_from_history,
    optimization_steps,
    recent_lambda_diagnostics,
)
from tpeas.controllers import (
    CONTROLLER_DESCRIPTIONS,
    EXPERIMENTAL_CONTROLLER_MODES,
    LAMBDA_MODES,
    PUBLIC_CONTROLLER_MODES,
    TUNABLE_PARAMETERS,
    TunableParameter,
    controller_label,
    public_controller_help,
    resolve_controller_mode,
)
from tpeas.custom_tpe import CustomTPEASOptimizer
from tpeas.evaluators import (
    GeneratedReturnsPortfolioBenchmark,
    SyntheticInstabilityBenchmark,
    TenParameterSyntheticBenchmark,
    generated_portfolio_search_space,
    synthetic_instability_search_space,
    ten_parameter_synthetic_search_space,
)
from tpeas.history import OptimizationResult, TrialRecord
from tpeas.objective import AdaptiveEvaluation, adaptive_objective_score
from tpeas.paper_like import (
    MARKET_DESCRIPTIONS,
    MARKET_IDS,
    MARKET_PERIODS,
    MODEL_DESCRIPTIONS,
    MODEL_IDS,
    PaperLikePortfolioBenchmark,
    generate_market_returns,
    paper_like_search_space,
)
from tpeas.random_search import RandomSearchOptimizer
from tpeas.search_space import CategoricalParam, FloatParam, IntParam, SearchSpace

__all__ = [
    "AdaptiveEvaluation",
    "AdaptiveObjectiveConfig",
    "CONTROLLER_DESCRIPTIONS",
    "CategoricalParam",
    "CustomTPEASOptimizer",
    "EXPERIMENTAL_CONTROLLER_MODES",
    "FloatParam",
    "GeneratedReturnsPortfolioBenchmark",
    "IntParam",
    "LambdaDiagnostics",
    "LAMBDA_MODES",
    "MARKET_DESCRIPTIONS",
    "MARKET_IDS",
    "MARKET_PERIODS",
    "MODEL_DESCRIPTIONS",
    "MODEL_IDS",
    "OptimizationResult",
    "PaperLikePortfolioBenchmark",
    "PUBLIC_CONTROLLER_MODES",
    "RandomSearchOptimizer",
    "SearchSpace",
    "SyntheticInstabilityBenchmark",
    "TenParameterSyntheticBenchmark",
    "TrialRecord",
    "TUNABLE_PARAMETERS",
    "TunableParameter",
    "adaptive_lambda",
    "adaptive_objective_score",
    "clip_importance_weight",
    "controller_label",
    "global_local_lambda_diagnostics",
    "lambda_diagnostics_from_history",
    "optimization_steps",
    "public_controller_help",
    "recent_lambda_diagnostics",
    "resolve_controller_mode",
    "generate_market_returns",
    "generated_portfolio_search_space",
    "paper_like_search_space",
    "synthetic_instability_search_space",
    "ten_parameter_synthetic_search_space",
]
