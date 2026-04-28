"""Run custom TPE-AS over a 3-model x 4-market paper-like grid."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from tpeas.history import OptimizationResult, TrialRecord
from tpeas.paper_like import (
    MARKET_DESCRIPTIONS,
    MARKET_IDS,
    MODEL_DESCRIPTIONS,
    MODEL_IDS,
    PaperLikePortfolioBenchmark,
    paper_like_search_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--quantile", type=float, default=0.15)
    parser.add_argument("--replicates-per-trial", type=int, default=5)
    parser.add_argument("--startup-trials", type=int, default=30)
    parser.add_argument("--n-candidates", type=int, default=128)
    parser.add_argument("--random-fraction", type=float, default=0.05)
    parser.add_argument(
        "--controller",
        choices=PUBLIC_CONTROLLER_MODES,
        default=None,
        help=f"Controller branch. {public_controller_help()}",
    )
    parser.add_argument(
        "--lambda-mode",
        choices=LAMBDA_MODES,
        default="budget",
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--markets", nargs="+", choices=MARKET_IDS, default=list(MARKET_IDS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def run_grid(
    *,
    budget: int = 300,
    epsilon: float = 0.1,
    quantile: float = 0.15,
    replicates_per_trial: int = 5,
    startup_trials: int = 30,
    n_candidates: int = 128,
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
    models: tuple[str, ...] | list[str] = MODEL_IDS,
    markets: tuple[str, ...] | list[str] = MARKET_IDS,
    output_dir: Path = ROOT / "results",
    run_name: str | None = None,
    make_plot: bool = True,
) -> Path:
    run_id = run_name or datetime.now().strftime("paper_like_grid_%Y%m%d_%H%M%S")
    grid_dir = output_dir / run_id
    grid_dir.mkdir(parents=True, exist_ok=True)
    resolved_lambda_mode = resolve_controller_mode(controller, lambda_mode)

    search_space = paper_like_search_space()
    config = AdaptiveObjectiveConfig(
        budget=budget,
        epsilon=epsilon,
        quantile=quantile,
        replicates_per_trial=replicates_per_trial,
        startup_trials=startup_trials,
        n_candidates=n_candidates,
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
    grid_rows: list[dict[str, Any]] = []
    for model_id in models:
        for market_id in markets:
            scenario_seed = seed + 101 * MODEL_IDS.index(model_id) + 17 * MARKET_IDS.index(market_id)
            scenario_dir = grid_dir / f"{model_id}_{market_id}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            benchmark = PaperLikePortfolioBenchmark(model_id=model_id, market_id=market_id, seed=seed)
            optimizer = CustomTPEASOptimizer(
                search_space,
                benchmark,
                config,
                seed=scenario_seed,
                use_variance_penalty=True,
                use_importance=True,
                sampler_name=f"custom-tpe-as-{model_id}-{market_id}",
            )

            scenario_start = time.perf_counter()
            result = optimizer.optimize()
            runtime = time.perf_counter() - scenario_start
            rows = trajectory_rows(result, model_id, market_id, search_space.names)
            write_trajectory(scenario_dir / "trajectory.csv", scenario_dir / "trajectory.jsonl", rows)
            summary = scenario_summary(
                result=result,
                rows=rows,
                model_id=model_id,
                market_id=market_id,
                run_dir=scenario_dir,
                runtime_seconds=runtime,
                epsilon=epsilon,
                lambda_mode=resolved_lambda_mode,
            )
            write_key_value_summary(scenario_dir / "summary.csv", summary)
            if make_plot:
                plot_trajectory(scenario_dir / "trajectory.png", rows, model_id, market_id)
            grid_rows.append(summary)
            print_scenario_report(summary)

    write_grid_summary(grid_dir / "grid_summary.csv", grid_rows)
    write_scenario_matrix(grid_dir / "scenario_matrix.csv", grid_rows)
    if make_plot:
        plot_grid_summary(grid_dir / "grid_summary.png", grid_rows)
    print_grid_report(grid_dir, grid_rows)
    return grid_dir


def trajectory_rows(
    result: OptimizationResult,
    model_id: str,
    market_id: str,
    param_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result.records:
        row = {
            "model_id": model_id,
            "market_id": market_id,
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
            "is_clipped": abs(record.importance_weight - record.clipped_weight) > 1e-12,
            "penalty": record.raw_mean - record.objective,
            "elapsed_seconds": record.elapsed_seconds,
        }
        for name in param_names:
            row[name] = record.params[name]
        rows.append(row)
    return rows


def write_trajectory(csv_path: Path, jsonl_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("trajectory requires at least one row")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def scenario_summary(
    *,
    result: OptimizationResult,
    rows: list[dict[str, Any]],
    model_id: str,
    market_id: str,
    run_dir: Path,
    runtime_seconds: float,
    epsilon: float,
    lambda_mode: str,
) -> dict[str, Any]:
    records = list(result.records)
    best_raw = result.best_by_raw_mean
    best_objective = result.best_by_objective
    selected = result.best_candidate()
    final = records[-1]
    raw_means = np.asarray([record.raw_mean for record in records], dtype=float)
    raw_variances = np.asarray([record.raw_variance for record in records], dtype=float)
    objectives = np.asarray([record.objective for record in records], dtype=float)
    importance = np.asarray([record.importance_weight for record in records], dtype=float)
    clipped = np.asarray([record.clipped_weight for record in records], dtype=float)
    penalties = raw_means - objectives
    clipped_flags = np.asarray([row["is_clipped"] for row in rows], dtype=bool)
    lambdas = np.asarray([record.lambda_t for record in records], dtype=float)
    noise_pressures = np.asarray([record.noise_pressure for record in records], dtype=float)
    mean_drop_pressures = np.asarray([record.mean_drop_pressure for record in records], dtype=float)
    global_noise_pressures = np.asarray([record.global_noise_pressure for record in records], dtype=float)
    global_quality_pressures = np.asarray([record.global_quality_pressure for record in records], dtype=float)
    global_pressures = np.asarray([record.global_pressure for record in records], dtype=float)
    local_pressures = np.asarray([record.local_pressure for record in records], dtype=float)
    global_mean_percentiles = np.asarray(
        [
            record.global_mean_percentile
            for record in records
            if record.global_mean_percentile is not None
        ],
        dtype=float,
    )
    global_variance_percentiles = np.asarray(
        [
            record.global_variance_percentile
            for record in records
            if record.global_variance_percentile is not None
        ],
        dtype=float,
    )
    thirds = split_thirds(records)

    return {
        "model_id": model_id,
        "model_description": MODEL_DESCRIPTIONS[model_id],
        "market_id": market_id,
        "market_description": MARKET_DESCRIPTIONS[market_id],
        "run_dir": str(run_dir),
        "budget": len(records),
        "epsilon": epsilon,
        "lambda_mode": lambda_mode,
        "controller": controller_label(lambda_mode),
        "clip_lower": 1.0 - epsilon,
        "clip_upper": 1.0 + epsilon,
        "selected_step": selected.step,
        "selected_objective": selected.objective,
        "selected_raw_mean": selected.raw_mean,
        "selected_raw_variance": selected.raw_variance,
        "selected_params": json.dumps(selected.params, sort_keys=True),
        "best_raw_step": best_raw.step,
        "best_raw_mean": best_raw.raw_mean,
        "best_raw_variance": best_raw.raw_variance,
        "best_raw_params": json.dumps(best_raw.params, sort_keys=True),
        "best_objective_step": best_objective.step,
        "best_objective": best_objective.objective,
        "best_objective_raw_mean": best_objective.raw_mean,
        "best_objective_params": json.dumps(best_objective.params, sort_keys=True),
        "final_step": final.step,
        "final_raw_mean": final.raw_mean,
        "final_objective": final.objective,
        "trajectory_raw_mean_variance": float(np.var(raw_means)),
        "raw_mean_mean": float(np.mean(raw_means)),
        "raw_mean_std": float(np.std(raw_means)),
        "raw_variance_mean": float(np.mean(raw_variances)),
        "raw_variance_std": float(np.std(raw_variances)),
        "early_raw_variance_mean": mean_raw_variance(thirds[0]),
        "middle_raw_variance_mean": mean_raw_variance(thirds[1]),
        "late_raw_variance_mean": mean_raw_variance(thirds[2]),
        "objective_mean": float(np.mean(objectives)),
        "objective_std": float(np.std(objectives)),
        "lambda_mean": float(np.mean(lambdas)),
        "lambda_max": float(np.max(lambdas)),
        "noise_pressure_mean": float(np.mean(noise_pressures)),
        "mean_drop_pressure_mean": float(np.mean(mean_drop_pressures)),
        "global_mean_percentile_mean": mean_or_nan(global_mean_percentiles),
        "global_variance_percentile_mean": mean_or_nan(global_variance_percentiles),
        "global_noise_pressure_mean": float(np.mean(global_noise_pressures)),
        "global_quality_pressure_mean": float(np.mean(global_quality_pressures)),
        "global_pressure_mean": float(np.mean(global_pressures)),
        "local_pressure_mean": float(np.mean(local_pressures)),
        "is_weight_mean": float(np.mean(importance)),
        "is_weight_std": float(np.std(importance)),
        "is_weight_min": float(np.min(importance)),
        "is_weight_max": float(np.max(importance)),
        "clipped_weight_mean": float(np.mean(clipped)),
        "clipped_weight_std": float(np.std(clipped)),
        "clipped_weight_min": float(np.min(clipped)),
        "clipped_weight_max": float(np.max(clipped)),
        "clipped_trial_count": int(np.sum(clipped_flags)),
        "clipped_trial_percent": float(100.0 * np.mean(clipped_flags)),
        "mean_penalty": float(np.mean(penalties)),
        "max_penalty": float(np.max(penalties)),
        "early_clipped_count": clipped_count(rows[: len(rows) // 3]),
        "middle_clipped_count": clipped_count(rows[len(rows) // 3 : (2 * len(rows)) // 3]),
        "late_clipped_count": clipped_count(rows[(2 * len(rows)) // 3 :]),
        "runtime_seconds": runtime_seconds,
    }


def split_thirds(records: list[TrialRecord]) -> tuple[list[TrialRecord], list[TrialRecord], list[TrialRecord]]:
    first = max(1, len(records) // 3)
    second = max(first + 1, (2 * len(records)) // 3)
    return records[:first], records[first:second], records[second:]


def mean_raw_variance(records: list[TrialRecord]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([record.raw_variance for record in records]))


def mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def clipped_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["is_clipped"])


def write_key_value_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})


def write_grid_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("grid summary requires at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_scenario_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    by_key = {(row["model_id"], row["market_id"]): row for row in rows}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", *MARKET_IDS])
        writer.writeheader()
        for model_id in MODEL_IDS:
            writer.writerow(
                {
                    "model_id": model_id,
                    **{
                        market_id: by_key.get((model_id, market_id), {}).get("best_raw_mean", "")
                        for market_id in MARKET_IDS
                    },
                }
            )


def plot_trajectory(path: Path, rows: list[dict[str, Any]], model_id: str, market_id: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping per-scenario trajectory plots")
        return

    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(steps, [row["raw_mean"] for row in rows], label="raw mean f(x)")
    axes[0].plot(steps, [row["objective"] for row in rows], label="adaptive objective J")
    lambda_axis = axes[0].twinx()
    lambda_axis.plot(steps, [row["lambda_t"] for row in rows], color="gray", alpha=0.45, label="lambda")
    axes[0].set_title(f"{model_id}-{market_id}")
    axes[0].set_ylabel("score")
    axes[0].legend(loc="upper left")
    lambda_axis.legend(loc="upper right")

    axes[1].plot(steps, [row["raw_variance"] for row in rows], color="tab:red", label="raw variance")
    axes[1].set_ylabel("variance")
    axes[1].legend(loc="best")

    axes[2].plot(steps, [row["importance_weight"] for row in rows], label="IS weight")
    axes[2].plot(steps, [row["clipped_weight"] for row in rows], label="clipped weight")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("weight")
    axes[2].legend(loc="best")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_grid_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping grid_summary.png")
        return

    labels = [f"{row['model_id']}-{row['market_id']}" for row in rows]
    best_raw = [row["best_raw_mean"] for row in rows]
    trajectory_var = [row["trajectory_raw_mean_variance"] for row in rows]
    clipped_percent = [row["clipped_trial_percent"] for row in rows]
    lambda_mean = [row["lambda_mean"] for row in rows]

    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].bar(labels, best_raw, color="tab:blue")
    axes[0].set_ylabel("best raw")
    axes[1].bar(labels, trajectory_var, color="tab:orange")
    axes[1].set_ylabel("trajectory var")
    axes[2].bar(labels, clipped_percent, color="tab:green")
    axes[2].set_ylabel("clipped %")
    axes[3].bar(labels, lambda_mean, color="tab:purple")
    axes[3].set_ylabel("mean lambda")
    axes[3].tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def print_scenario_report(summary: dict[str, Any]) -> None:
    print(
        "{model_id}-{market_id}: best_raw={best_raw_mean:.4f} best_J={best_objective:.4f} "
        "traj_var={trajectory_raw_mean_variance:.4f} lambda_mean={lambda_mean:.3f} "
        "clipped={clipped_trial_count}/{budget} "
        "runtime={runtime_seconds:.1f}s".format(**summary)
    )


def print_grid_report(grid_dir: Path, rows: list[dict[str, Any]]) -> None:
    top = max(rows, key=lambda row: row["best_raw_mean"])
    most_stable = min(rows, key=lambda row: row["trajectory_raw_mean_variance"])
    highest_clipping = max(rows, key=lambda row: row["clipped_trial_count"])
    print(f"\nWrote paper-like grid to {grid_dir}")
    print(
        "Top scenario by best raw: "
        f"{top['model_id']}-{top['market_id']} ({top['best_raw_mean']:.4f})"
    )
    print(
        "Most stable trajectory: "
        f"{most_stable['model_id']}-{most_stable['market_id']} "
        f"({most_stable['trajectory_raw_mean_variance']:.4f})"
    )
    print(
        "Highest clipping scenario: "
        f"{highest_clipping['model_id']}-{highest_clipping['market_id']} "
        f"({highest_clipping['clipped_trial_count']}/{highest_clipping['budget']})"
    )
    print("\nPer-model best raw averages:")
    for model_id in MODEL_IDS:
        selected = [row["best_raw_mean"] for row in rows if row["model_id"] == model_id]
        if selected:
            print(f"  {model_id}: {np.mean(selected):.4f}")
    print("\nPer-market best raw averages:")
    for market_id in MARKET_IDS:
        selected = [row["best_raw_mean"] for row in rows if row["market_id"] == market_id]
        if selected:
            print(f"  {market_id}: {np.mean(selected):.4f}")


def main() -> None:
    args = parse_args()
    run_grid(
        budget=args.budget,
        epsilon=args.epsilon,
        quantile=args.quantile,
        replicates_per_trial=args.replicates_per_trial,
        startup_trials=args.startup_trials,
        n_candidates=args.n_candidates,
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
        models=args.models,
        markets=args.markets,
        output_dir=args.output_dir,
        run_name=args.run_name,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
