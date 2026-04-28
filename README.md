# TPE-AS Official Implementation

This project contains the optimizer mechanism from:

> Zinuo You, John Cartlidge, Karen Elliott, Menghan Ge, and Daniel Gold.
> "Improving Bayesian Optimization for Portfolio Management with an Adaptive Scheduling."
> arXiv:2504.13529.

The paper does not disclose its proprietary portfolio models, backtest settings, or
hyperparameters. This project therefore reproduces the method mechanics and qualitative
experiments rather than the paper's exact model-performance-to-optimize tables.

## What Is Implemented

- Adaptive schedule from the paper:

```text
lambda_t = (1 - cos(min(t / budget * pi, pi))) / 2
```

- Adaptive objective:

```text
J_t = mean(f_samples) - lambda_t * variance(f_samples * clipped_weight)
```

- Stage 1 Optuna wrapper using `TPESampler`.
- Stage 2 custom TPE-AS sampler with explicit good/bad Parzen models and clipped empirical
  importance weights.
- Synthetic noisy black-box benchmarks that include high-return but unstable regions.
- Experiment runner comparing TPE-AS, ablations, and random search.

## Install

```powershell
cd D:\projects\tpe_as_replication
python -m pip install -e .[optuna,experiment,dev]
```

The custom sampler and tests only require `numpy`; the Optuna wrapper raises a clear error if
`optuna` is not installed.

## Run Tests

```powershell
cd D:\projects\tpe_as_replication
$env:PYTHONPATH = "D:\projects\tpe_as_replication\src"
python -m unittest discover -s tests
```

## Run A Synthetic Comparison

```powershell
cd D:\projects\tpe_as_replication
$env:PYTHONPATH = "D:\projects\tpe_as_replication\src"
python experiments\run_synthetic_comparison.py --budget 80 --seeds 0 1 2
```

Outputs are written under `results/` as JSONL trial histories and a summary CSV.

## Run The 10-Parameter TPE-AS Trajectory

```powershell
cd D:\projects\tpe_as_replication
$env:PYTHONPATH = "D:\projects\tpe_as_replication\src"
python experiments\run_ten_parameter_tpeas.py --budget 120 --seed 0
```

This runs only the custom TPE-AS sampler on a mixed 10-parameter benchmark. It writes a flat
`trajectory.csv`, `trajectory.jsonl`, `summary.csv`, and `trajectory.png` when `matplotlib` is
installed.

To compare clipped importance-sampling ranges:

```powershell
python experiments\compare_epsilon_tpeas.py --epsilons 0.035 0.05 0.075 0.1 0.2 --budget 120 --seed 0
```

This writes one folder per epsilon plus a combined `epsilon_comparison.csv` and optional
`epsilon_comparison.png`.

## Run The Paper-Like 3x4 Grid

```powershell
python experiments\run_paper_like_grid.py --budget 300 --epsilon 0.1 --seed 0
```

This runs the custom TPE-AS optimizer for 3 lightweight black-box model mimics across 4 generated
market settings. The full default run performs 300 evaluations per model-market scenario and writes
per-scenario trajectories plus `grid_summary.csv`, `scenario_matrix.csv`, and `grid_summary.png`.

Use `--controller` to switch between the two public controller branches:

```powershell
python experiments\run_paper_like_grid.py --budget 300 --epsilon 0.1 --seed 0 --controller budget
python experiments\run_paper_like_grid.py --budget 300 --epsilon 0.1 --seed 0 --controller global_local
```

Controller choices:

- `budget`: paper-style cosine schedule for known-budget optimization.
- `global_local`: anytime history-only controller for runs that may stop early.

Main shared hyperparameters are `epsilon=0.1`, `quantile=0.15`, `startup_trials=30`,
`replicates_per_trial=5`, `n_candidates=128`, and `random_fraction=0.05`.
For `budget`, `budget=300` controls the cosine lambda schedule. For `global_local`, the default
history controls are `recent_window=30`, `previous_window=30`, `min_recent_history=30`,
`variance_ratio_full_scale=3.0`, `recent_variance_weight=0.75`,
`recent_mean_drop_weight=0.25`, `global_window_min_history=30`, `global_noise_weight=0.5`,
`global_quality_weight=0.5`, `global_controller_weight=1.0`, and
`local_controller_weight=1.0`.

To run the global-local controller and compare it against the saved budget and recent baselines:

```powershell
python experiments\compare_lambda_controllers.py --budget 300 --epsilon 0.1 --seed 0
```

## Project Boundary

This is a standalone project. It does not depend on `D:\projects\C3E` and should not read from or
write to that project.
