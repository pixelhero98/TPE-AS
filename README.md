# TPE-AS Official Implementation

This project replicates the optimizer mechanism from:

> Improving Bayesian Optimization for Portfolio Management with an Adaptive Scheduling.
> In Proceedings of the 2025 9th International Conference on Advances in Artificial Intelligence (pp. 21-25).

The paper does not disclose any proprietary portfolio models, backtest settings, or
hyperparameters. This repository, therefore, focuses on reusable optimizer mechanics and transparent
synthetic experiments rather than reproducing the paper's exact financial tables.

## What Is Implemented

- Adaptive TPE-AS objective:

```text
J_t = mean(f_samples) - lambda_t * variance(f_samples * clipped_weight)
```

- Stage 1 Optuna wrapper using `TPESampler`.
- Stage 2 custom TPE-AS sampler with explicit good/bad Parzen density models.
- Clipped empirical importance weights.
- Synthetic black-box benchmarks with noisy high-return regions.
- Two public controller branches for variance pressure: `budget` and `global_local`.

## Install

```bash
python -m pip install -e ".[optuna,experiment,dev]"
```

The custom sampler and unit tests only require `numpy`. The Optuna wrapper raises a clear error if
`optuna` is not installed.

## Run Tests

```bash
python -m unittest discover -s tests
```

## Controller Branches

Use `--controller` to switch between the two public controller branches:

```bash
python experiments/run_paper_like_grid.py --controller budget
python experiments/run_paper_like_grid.py --controller global_local
```

| Controller | Use when | Lambda behavior |
|---|---|---|
| `budget` | The optimization budget is known in advance. | Cosine schedule from near `0` to `1`: `(1 - cos(min(t / budget * pi, pi))) / 2`. |
| `global_local` | The run may stop at any time. | Uses only completed trajectory history, combining global percentile pressure and recent-window pressure. |

For both branches, `--budget` is the exact number of evaluations to run. The `global_local`
branch is budget-free only in its lambda calculation; it still stops after `budget` evaluations.
Both branches return the selected incumbent by adaptive objective through
`result.best_candidate()`. Do not treat `result.records[-1]` as the selected answer; it is just the
last sampled point in the trajectory.

The internal `recent` mode remains available through the legacy hidden `--lambda-mode recent` flag
for experiments and comparisons, but it is not a recommended public branch.

## Use With Your Own Model

The optimizer accepts any `SearchSpace` made from `FloatParam`, `IntParam`, and
`CategoricalParam`. There is no 10-parameter limit; use as many parameters as your black-box model
needs. Your evaluator only needs an `evaluate(params, rng)` method that returns one noisy score to
maximize.

```python
import numpy as np

from tpeas import (
    AdaptiveObjectiveConfig,
    CategoricalParam,
    CustomTPEASOptimizer,
    FloatParam,
    IntParam,
    SearchSpace,
    resolve_controller_mode,
)


class MyModelEvaluator:
    def evaluate(self, params, rng: np.random.Generator) -> float:
        # Replace this with training, simulation, backtesting, or another black-box call.
        score = 1.0
        score -= (params["learning_rate"] - 0.02) ** 2 / 0.01
        score -= (params["depth"] - 5) ** 2 / 20.0
        score += {"relu": 0.0, "gelu": 0.1, "tanh": -0.1}[params["activation"]]
        return float(score + rng.normal(0.0, 0.05))


search_space = SearchSpace(
    [
        FloatParam("learning_rate", 1e-4, 0.1, log_scale=True),
        FloatParam("dropout", 0.0, 0.5),
        FloatParam("l2", 1e-6, 1e-2, log_scale=True),
        IntParam("depth", 2, 10),
        IntParam("width", 32, 512),
        CategoricalParam("activation", ["relu", "gelu", "tanh"]),
    ]
)

config = AdaptiveObjectiveConfig(
    budget=80,
    epsilon=0.1,
    startup_trials=12,
    replicates_per_trial=3,
    n_candidates=64,
    lambda_mode=resolve_controller_mode("global_local"),
)

optimizer = CustomTPEASOptimizer(
    search_space=search_space,
    evaluator=MyModelEvaluator(),
    config=config,
    seed=0,
)
result = optimizer.optimize()
selected = result.best_candidate()

print(selected.step)
print(selected.objective)
print(selected.params)
```

A fuller runnable example with 13 mixed parameters is available at
`examples/custom_model_optimization.py`.

## Tunable Parameters

Shared optimizer parameters:

| Parameter | CLI flag | Default | Meaning |
|---|---|---:|---|
| `epsilon` | `--epsilon` | `0.1` | Importance-weight clip half-width; `0.1` gives `[0.9, 1.1]`. |
| `quantile` | `--quantile` | `0.15` | Elite split for good/bad TPE density fitting. |
| `startup_trials` | `--startup-trials` | `30` | Random warmup trials before model-guided sampling. |
| `replicates_per_trial` | `--replicates-per-trial` | `5` | Repeated black-box samples per candidate. |
| `n_candidates` | `--n-candidates` | `128` | Candidate samples scored from the TPE proposal. |
| `random_fraction` | `--random-fraction` | `0.05` | Probability of random exploration after warmup. |

Budget controller parameter:

| Parameter | CLI flag | Default | Meaning |
|---|---|---:|---|
| `budget` | `--budget` | `300` | Exact number of evaluations to run; also used by the budget controller's cosine lambda schedule. |

Global-local controller parameters:

| Parameter | CLI flag | Default | Meaning |
|---|---|---:|---|
| `recent_window` | `--recent-window` | `30` | Local window used for recent mean and variance. |
| `previous_window` | `--previous-window` | `30` | Comparison window before the recent window. |
| `min_recent_history` | `--min-recent-history` | `30` | Completed trials required before local pressure activates. |
| `variance_ratio_full_scale` | `--variance-ratio-full-scale` | `3.0` | Recent/baseline variance ratio mapped to full local noise pressure. |
| `recent_variance_weight` | `--recent-variance-weight` | `0.75` | Weight for local noise pressure. |
| `recent_mean_drop_weight` | `--recent-mean-drop-weight` | `0.25` | Weight for local mean-drop pressure. |
| `global_window_min_history` | `--global-window-min-history` | `30` | Completed trials required before global pressure activates. |
| `global_noise_weight` | `--global-noise-weight` | `0.5` | Weight for global high-variance percentile pressure. |
| `global_quality_weight` | `--global-quality-weight` | `0.5` | Weight for global low-mean percentile pressure. |
| `global_controller_weight` | `--global-controller-weight` | `1.0` | Multiplier applied to global pressure. |
| `local_controller_weight` | `--local-controller-weight` | `1.0` | Multiplier applied to local pressure. |

## Run Experiments

The `ten_parameter_*` and `paper_like_*` components are demo benchmarks. They are useful for
reproducing the repository experiments, but they are not optimizer limitations.

Synthetic comparison:

```bash
python experiments/run_synthetic_comparison.py --budget 80 --seeds 0 1 2
```

10-parameter mixed benchmark:

```bash
python experiments/run_ten_parameter_tpeas.py --budget 120 --seed 0 --controller global_local
```

Paper-like 3-model x 4-market grid:

```bash
python experiments/run_paper_like_grid.py --budget 300 --epsilon 0.1 --seed 0 --controller budget
python experiments/run_paper_like_grid.py --budget 300 --epsilon 0.1 --seed 0 --controller global_local
```

Epsilon sensitivity:

```bash
python experiments/compare_epsilon_tpeas.py --epsilons 0.035 0.05 0.075 0.1 0.2 --budget 120 --seed 0
```

Controller comparison:

```bash
python experiments/compare_lambda_controllers.py --budget-dir results/<budget-run> --recent-dir results/<recent-run>
```

If `--budget-dir` or `--recent-dir` is omitted, the comparison script searches `results/` for the
latest compatible `grid_summary.csv`.

## Outputs

Experiment outputs are written under `results/` by default. Scenario runners write:

- `trajectory.csv`
- `trajectory.jsonl`
- `summary.csv`
- optional `.png` plots when `matplotlib` is installed

Trajectory and summary files include both `lambda_mode` and public `controller` metadata.
Summary files also include `selected_*` fields from `result.best_candidate()`. `final_*` fields are
trajectory diagnostics for the last sampled point.

## Project Boundary

This is a standalone method-replication repo. It does not require any proprietary financial
models or private local projects.
