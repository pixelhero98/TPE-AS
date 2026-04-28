"""Compare epsilon values for custom TPE-AS on the 10-parameter benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_ten_parameter_tpeas import run_experiment
from tpeas.controllers import LAMBDA_MODES, PUBLIC_CONTROLLER_MODES, public_controller_help


DEFAULT_EPSILONS = [0.035, 0.05, 0.075, 0.1, 0.2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--startup-trials", type=int, default=15)
    parser.add_argument("--replicates-per-trial", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=96)
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


def run_comparison(
    *,
    epsilons: list[float] | tuple[float, ...] = tuple(DEFAULT_EPSILONS),
    budget: int = 120,
    startup_trials: int = 15,
    replicates_per_trial: int = 5,
    n_candidates: int = 96,
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
    comparison_id = run_name or datetime.now().strftime("epsilon_comparison_%Y%m%d_%H%M%S")
    comparison_dir = output_dir / comparison_id
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, Any]] = []
    histories: list[tuple[float, list[dict[str, Any]]]] = []
    for epsilon in epsilons:
        run_dir = run_experiment(
            budget=budget,
            startup_trials=startup_trials,
            replicates_per_trial=replicates_per_trial,
            n_candidates=n_candidates,
            epsilon=epsilon,
            quantile=quantile,
            random_fraction=random_fraction,
            controller=controller,
            lambda_mode=lambda_mode,
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
            seed=seed,
            output_dir=comparison_dir,
            run_name=f"epsilon_{epsilon_token(epsilon)}",
            make_plot=make_plot,
        )
        summary = read_summary(run_dir / "summary.csv")
        trajectory = read_trajectory(run_dir / "trajectory.csv")
        histories.append((epsilon, trajectory))
        comparison_rows.append(comparison_metrics(epsilon, run_dir, summary, trajectory))

    write_comparison(comparison_dir / "epsilon_comparison.csv", comparison_rows)
    if make_plot:
        plot_comparison(comparison_dir / "epsilon_comparison.png", histories)
    print_comparison_report(comparison_dir, comparison_rows)
    return comparison_dir


def epsilon_token(epsilon: float) -> str:
    return f"{epsilon:.6g}".replace("-", "m").replace(".", "p")


def read_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_fields = {
        "step",
        "raw_mean",
        "raw_variance",
        "objective",
        "lambda_t",
        "importance_weight",
        "clipped_weight",
        "elapsed_seconds",
    }
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for field in numeric_fields:
                if field == "step":
                    parsed[field] = int(float(parsed[field]))
                else:
                    parsed[field] = float(parsed[field])
            parsed["is_clipped"] = (
                abs(parsed["importance_weight"] - parsed["clipped_weight"]) > 1e-12
            )
            parsed["penalty"] = parsed["raw_mean"] - parsed["objective"]
            rows.append(parsed)
    return rows


def comparison_metrics(
    epsilon: float,
    run_dir: Path,
    summary: dict[str, str],
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    importance = np.asarray([row["importance_weight"] for row in trajectory], dtype=float)
    clipped = np.asarray([row["clipped_weight"] for row in trajectory], dtype=float)
    penalties = np.asarray([row["penalty"] for row in trajectory], dtype=float)
    clipped_flags = np.asarray([row["is_clipped"] for row in trajectory], dtype=bool)
    thirds = split_thirds(trajectory)

    return {
        "epsilon": epsilon,
        "clip_lower": 1.0 - epsilon,
        "clip_upper": 1.0 + epsilon,
        "run_dir": str(run_dir),
        "budget": int(summary["budget"]),
        "lambda_mode": summary.get("lambda_mode", "budget"),
        "controller": summary.get("controller", "budget"),
        "best_raw_step": int(summary["best_raw_step"]),
        "best_raw_mean": float(summary["best_raw_mean"]),
        "best_raw_variance": float(summary["best_raw_variance"]),
        "best_objective_step": int(summary["best_objective_step"]),
        "best_objective": float(summary["best_objective"]),
        "final_raw_mean": float(summary["final_raw_mean"]),
        "final_objective": float(summary["final_objective"]),
        "trajectory_raw_mean_variance": float(summary["trajectory_raw_mean_variance"]),
        "early_raw_variance_mean": float(summary["early_raw_variance_mean"]),
        "middle_raw_variance_mean": float(summary["middle_raw_variance_mean"]),
        "late_raw_variance_mean": float(summary["late_raw_variance_mean"]),
        "stable_basin_count": int(summary["stable_basin_count"]),
        "risky_basin_count": int(summary["risky_basin_count"]),
        "other_basin_count": int(summary["other_basin_count"]),
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
        "early_clipped_count": clipped_count(thirds[0]),
        "middle_clipped_count": clipped_count(thirds[1]),
        "late_clipped_count": clipped_count(thirds[2]),
    }


def split_thirds(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    first = max(1, len(rows) // 3)
    second = max(first + 1, (2 * len(rows)) // 3)
    return rows[:first], rows[first:second], rows[second:]


def clipped_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["is_clipped"])


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("comparison requires at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, histories: list[tuple[float, list[dict[str, Any]]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping epsilon_comparison.png")
        return

    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    panels = [
        ("raw_mean", "raw mean f(x)"),
        ("objective", "adaptive objective J"),
        ("raw_variance", "raw variance"),
        ("importance_weight", "IS weight"),
        ("clipped_weight", "clipped weight"),
        ("is_clipped", "clipping active"),
    ]
    for axis, (field, title) in zip(axes.ravel(), panels, strict=True):
        for epsilon, rows in histories:
            steps = [row["step"] for row in rows]
            if field == "is_clipped":
                values = [1.0 if row[field] else 0.0 for row in rows]
            else:
                values = [row[field] for row in rows]
            axis.plot(steps, values, label=f"eps={epsilon:g}", alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def print_comparison_report(comparison_dir: Path, rows: list[dict[str, Any]]) -> None:
    print(f"Wrote epsilon comparison to {comparison_dir}")
    print("\nEpsilon comparison:")
    for row in rows:
        print(
            "  eps={epsilon:g} clip=[{clip_lower:.3f}, {clip_upper:.3f}] "
            "best_raw={best_raw_mean:.4f} best_J={best_objective:.4f} "
            "stable={stable_basin_count} clipped={clipped_trial_count}/{budget} "
            "IS_max={is_weight_max:.3f} clipped_w_max={clipped_weight_max:.3f}".format(**row)
        )


def main() -> None:
    args = parse_args()
    run_comparison(
        epsilons=args.epsilons,
        budget=args.budget,
        startup_trials=args.startup_trials,
        replicates_per_trial=args.replicates_per_trial,
        n_candidates=args.n_candidates,
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
