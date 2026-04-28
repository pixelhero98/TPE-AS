"""Run custom TPE-AS on the 10-parameter synthetic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tpeas.config import AdaptiveObjectiveConfig
from tpeas.controllers import (
    LAMBDA_MODES,
    PUBLIC_CONTROLLER_MODES,
    controller_label,
    public_controller_help,
    resolve_controller_mode,
)
from tpeas.custom_tpe import CustomTPEASOptimizer
from tpeas.evaluators import TenParameterSyntheticBenchmark, ten_parameter_synthetic_search_space
from tpeas.history import OptimizationResult, TrialRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--replicates-per-trial", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=96)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--quantile", type=float, default=0.15)
    parser.add_argument("--random-fraction", type=float, default=0.05)
    parser.add_argument(
        "--controller",
        choices=PUBLIC_CONTROLLER_MODES,
        default=None,
        help=f"Controller branch. {public_controller_help()}",
    )
    parser.add_argument("--lambda-mode", choices=LAMBDA_MODES, default="budget", help=argparse.SUPPRESS)
    parser.add_argument("--recent-window", type=int, default=30)
    parser.add_argument("--previous-window", type=int, default=30)
    parser.add_argument("--min-recent-history", type=int, default=30)
    parser.add_argument("--variance-ratio-full-scale", type=float, default=3.0)
    parser.add_argument("--recent-variance-weight", type=float, default=0.75)
    parser.add_argument("--recent-mean-drop-weight", type=float, default=0.25)
    parser.add_argument("--global-window-min-history", type=int, default=30)
    parser.add_argument("--global-noise-weight", type=float, default=0.5)
    parser.add_argument("--global-quality-weight", type=float, default=0.5)
    parser.add_argument("--global-controller-weight", type=float, default=1.0)
    parser.add_argument("--local-controller-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def run_experiment(
    *,
    budget: int = 120,
    startup_trials: int = 15,
    replicates_per_trial: int = 5,
    n_candidates: int = 96,
    epsilon: float = 0.2,
    quantile: float = 0.15,
    random_fraction: float = 0.05,
    controller: str | None = None,
    lambda_mode: str = "budget",
    recent_window: int = 30,
    previous_window: int = 30,
    min_recent_history: int = 30,
    variance_ratio_full_scale: float = 3.0,
    recent_variance_weight: float = 0.75,
    recent_mean_drop_weight: float = 0.25,
    global_window_min_history: int = 30,
    global_noise_weight: float = 0.5,
    global_quality_weight: float = 0.5,
    global_controller_weight: float = 1.0,
    local_controller_weight: float = 1.0,
    seed: int = 0,
    output_dir: Path = ROOT / "results",
    run_name: str | None = None,
    make_plot: bool = True,
) -> Path:
    run_id = run_name or datetime.now().strftime("ten_parameter_tpeas_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    search_space = ten_parameter_synthetic_search_space()
    evaluator = TenParameterSyntheticBenchmark()
    resolved_lambda_mode = resolve_controller_mode(controller, lambda_mode)
    config = AdaptiveObjectiveConfig(
        budget=budget,
        startup_trials=startup_trials,
        replicates_per_trial=replicates_per_trial,
        n_candidates=n_candidates,
        epsilon=epsilon,
        quantile=quantile,
        random_fraction=random_fraction,
        lambda_mode=resolved_lambda_mode,
        recent_window=recent_window,
        previous_window=previous_window,
        min_recent_history=min_recent_history,
        variance_ratio_full_scale=variance_ratio_full_scale,
        recent_variance_weight=recent_variance_weight,
        recent_mean_drop_weight=recent_mean_drop_weight,
        global_window_min_history=global_window_min_history,
        global_noise_weight=global_noise_weight,
        global_quality_weight=global_quality_weight,
        global_controller_weight=global_controller_weight,
        local_controller_weight=local_controller_weight,
    )
    optimizer = CustomTPEASOptimizer(
        search_space,
        evaluator,
        config,
        seed=seed,
        use_variance_penalty=True,
        use_importance=True,
        sampler_name="custom-tpe-as-10p",
    )
    result = optimizer.optimize()

    param_names = search_space.names
    rows = trajectory_rows(result, evaluator, param_names)
    write_trajectory(run_dir / "trajectory.csv", run_dir / "trajectory.jsonl", rows)
    summary_rows = summary_statistics(
        result,
        evaluator,
        epsilon=epsilon,
        lambda_mode=resolved_lambda_mode,
    )
    write_summary(run_dir / "summary.csv", summary_rows)
    if make_plot:
        plot_trajectory(run_dir / "trajectory.png", rows)

    print_report(run_dir, rows, summary_rows)
    return run_dir


def trajectory_rows(
    result: OptimizationResult,
    evaluator: TenParameterSyntheticBenchmark,
    param_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result.records:
        row = {
            "step": record.step,
            "raw_mean": record.raw_mean,
            "raw_variance": record.raw_variance,
            "objective": record.objective,
            "lambda_t": record.lambda_t,
            "lambda_mode": record.lambda_mode,
            "controller": controller_label(record.lambda_mode),
            "recent_mean": record.recent_mean,
            "previous_mean": record.previous_mean,
            "recent_variance_mean": record.recent_variance_mean,
            "baseline_variance": record.baseline_variance,
            "noise_pressure": record.noise_pressure,
            "mean_drop_pressure": record.mean_drop_pressure,
            "global_mean_percentile": record.global_mean_percentile,
            "global_variance_percentile": record.global_variance_percentile,
            "global_noise_pressure": record.global_noise_pressure,
            "global_quality_pressure": record.global_quality_pressure,
            "global_pressure": record.global_pressure,
            "local_pressure": record.local_pressure,
            "importance_weight": record.importance_weight,
            "clipped_weight": record.clipped_weight,
            "basin_label": evaluator.basin_label(record.params),
            "elapsed_seconds": record.elapsed_seconds,
        }
        for name in param_names:
            row[name] = record.params[name]
        rows.append(row)
    return rows


def write_trajectory(csv_path: Path, jsonl_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("trajectory must contain at least one row")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summary_statistics(
    result: OptimizationResult,
    evaluator: TenParameterSyntheticBenchmark,
    epsilon: float = 0.2,
    lambda_mode: str = "budget",
) -> list[dict[str, str]]:
    records = list(result.records)
    raw_means = np.asarray([record.raw_mean for record in records], dtype=float)
    raw_variances = np.asarray([record.raw_variance for record in records], dtype=float)
    objectives = np.asarray([record.objective for record in records], dtype=float)
    labels = [evaluator.basin_label(record.params) for record in records]
    best_raw = result.best_by_raw_mean
    best_objective = result.best_by_objective
    selected = result.best_candidate()
    final = records[-1]
    thirds = split_thirds(records)

    rows = [
        metric_row("epsilon", epsilon),
        metric_row("lambda_mode", lambda_mode),
        metric_row("controller", controller_label(lambda_mode)),
        metric_row("clip_lower", 1.0 - epsilon),
        metric_row("clip_upper", 1.0 + epsilon),
        metric_row("budget", len(records)),
        metric_row("selected_step", selected.step),
        metric_row("selected_objective", selected.objective),
        metric_row("selected_raw_mean", selected.raw_mean),
        metric_row("selected_raw_variance", selected.raw_variance),
        metric_row("selected_params", selected.params),
        metric_row("best_raw_step", best_raw.step),
        metric_row("best_raw_mean", best_raw.raw_mean),
        metric_row("best_raw_variance", best_raw.raw_variance),
        metric_row("best_raw_params", best_raw.params),
        metric_row("best_raw_basin_label", evaluator.basin_label(best_raw.params)),
        metric_row("best_objective_step", best_objective.step),
        metric_row("best_objective", best_objective.objective),
        metric_row("best_objective_raw_mean", best_objective.raw_mean),
        metric_row("best_objective_params", best_objective.params),
        metric_row("best_objective_basin_label", evaluator.basin_label(best_objective.params)),
        metric_row("final_step", final.step),
        metric_row("final_raw_mean", final.raw_mean),
        metric_row("final_objective", final.objective),
        metric_row("final_params", final.params),
        metric_row("final_basin_label", evaluator.basin_label(final.params)),
        metric_row("raw_mean_mean", float(np.mean(raw_means))),
        metric_row("raw_mean_std", float(np.std(raw_means))),
        metric_row("raw_mean_min", float(np.min(raw_means))),
        metric_row("raw_mean_max", float(np.max(raw_means))),
        metric_row("raw_variance_mean", float(np.mean(raw_variances))),
        metric_row("raw_variance_std", float(np.std(raw_variances))),
        metric_row("raw_variance_min", float(np.min(raw_variances))),
        metric_row("raw_variance_max", float(np.max(raw_variances))),
        metric_row("objective_mean", float(np.mean(objectives))),
        metric_row("objective_std", float(np.std(objectives))),
        metric_row("objective_min", float(np.min(objectives))),
        metric_row("objective_max", float(np.max(objectives))),
        metric_row("trajectory_raw_mean_variance", float(np.var(raw_means))),
        metric_row("early_raw_variance_mean", mean_raw_variance(thirds[0])),
        metric_row("middle_raw_variance_mean", mean_raw_variance(thirds[1])),
        metric_row("late_raw_variance_mean", mean_raw_variance(thirds[2])),
        metric_row("stable_basin_count", labels.count("stable")),
        metric_row("risky_basin_count", labels.count("risky")),
        metric_row("other_basin_count", labels.count("other")),
    ]
    return rows


def split_thirds(records: list[TrialRecord]) -> tuple[list[TrialRecord], list[TrialRecord], list[TrialRecord]]:
    first = max(1, len(records) // 3)
    second = max(first + 1, (2 * len(records)) // 3)
    return records[:first], records[first:second], records[second:]


def mean_raw_variance(records: list[TrialRecord]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([record.raw_variance for record in records]))


def metric_row(metric: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        rendered = json.dumps(value, sort_keys=True)
    else:
        rendered = str(value)
    return {"metric": metric, "value": rendered}


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def plot_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping trajectory.png")
        return

    steps = [row["step"] for row in rows]
    raw_means = [row["raw_mean"] for row in rows]
    raw_variances = [row["raw_variance"] for row in rows]
    objectives = [row["objective"] for row in rows]
    lambdas = [row["lambda_t"] for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, raw_means, label="raw mean f(x)", color="#1f77b4")
    axes[0].plot(steps, objectives, label="adaptive objective J", color="#2ca02c")
    lambda_axis = axes[0].twinx()
    lambda_axis.plot(steps, lambdas, label="lambda_t", color="#7f7f7f", alpha=0.45)
    axes[0].set_ylabel("score")
    lambda_axis.set_ylabel("lambda_t")
    axes[0].legend(loc="upper left")
    lambda_axis.legend(loc="upper right")

    axes[1].plot(steps, raw_variances, label="raw variance", color="#d62728")
    lambda_axis_bottom = axes[1].twinx()
    lambda_axis_bottom.plot(steps, lambdas, label="lambda_t", color="#7f7f7f", alpha=0.45)
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("raw variance")
    lambda_axis_bottom.set_ylabel("lambda_t")
    axes[1].legend(loc="upper left")
    lambda_axis_bottom.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def print_report(run_dir: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, str]]) -> None:
    print(f"Wrote 10-parameter TPE-AS run to {run_dir}")
    print("\nTrajectory preview:")
    if len(rows) <= 10:
        preview_rows = rows
    else:
        preview_rows = rows[:5] + [{"step": "..."}] + rows[-5:]
    for row in preview_rows:
        if row["step"] == "...":
            print("  ...")
            continue
        print(
            "  step={step:>3} raw={raw_mean:>7.3f} var={raw_variance:>7.3f} "
            "J={objective:>7.3f} lambda={lambda_t:>5.3f} basin={basin_label}".format(**row)
        )

    metrics = {row["metric"]: row["value"] for row in summary_rows}
    print("\nTrajectory statistics:")
    for metric in [
        "best_raw_step",
        "epsilon",
        "best_raw_mean",
        "best_raw_variance",
        "best_raw_basin_label",
        "best_objective_step",
        "best_objective",
        "trajectory_raw_mean_variance",
        "early_raw_variance_mean",
        "middle_raw_variance_mean",
        "late_raw_variance_mean",
        "stable_basin_count",
        "risky_basin_count",
        "other_basin_count",
    ]:
        print(f"  {metric}: {metrics[metric]}")


def main() -> None:
    args = parse_args()
    run_experiment(
        budget=args.budget,
        startup_trials=args.startup_trials,
        replicates_per_trial=args.replicates_per_trial,
        n_candidates=args.n_candidates,
        epsilon=args.epsilon,
        quantile=args.quantile,
        random_fraction=args.random_fraction,
        controller=args.controller,
        lambda_mode=args.lambda_mode,
        recent_window=args.recent_window,
        previous_window=args.previous_window,
        min_recent_history=args.min_recent_history,
        variance_ratio_full_scale=args.variance_ratio_full_scale,
        recent_variance_weight=args.recent_variance_weight,
        recent_mean_drop_weight=args.recent_mean_drop_weight,
        global_window_min_history=args.global_window_min_history,
        global_noise_weight=args.global_noise_weight,
        global_quality_weight=args.global_quality_weight,
        global_controller_weight=args.global_controller_weight,
        local_controller_weight=args.local_controller_weight,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
