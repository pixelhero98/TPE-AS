"""Compare budget, recent, and global-local TPE-AS grid results."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_paper_like_grid import run_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget-dir",
        "--baseline-dir",
        dest="budget_dir",
        type=Path,
        default=None,
        help="Budget-controller grid directory. Defaults to latest compatible run under --output-dir.",
    )
    parser.add_argument(
        "--recent-dir",
        type=Path,
        default=None,
        help="Recent-controller grid directory. Defaults to latest compatible run under --output-dir.",
    )
    parser.add_argument("--global-local-dir", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--quantile", type=float, default=0.15)
    parser.add_argument("--replicates-per-trial", type=int, default=5)
    parser.add_argument("--startup-trials", type=int, default=30)
    parser.add_argument("--n-candidates", type=int, default=128)
    parser.add_argument("--random-fraction", type=float, default=0.05)
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


def run_controller_comparison(
    *,
    budget_dir: Path | None = None,
    recent_dir: Path | None = None,
    global_local_dir: Path | None = None,
    budget: int = 300,
    epsilon: float = 0.1,
    quantile: float = 0.15,
    replicates_per_trial: int = 5,
    startup_trials: int = 30,
    n_candidates: int = 128,
    random_fraction: float = 0.05,
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
    comparison_id = run_name or datetime.now().strftime("lambda_controller_comparison_%Y%m%d_%H%M%S")
    comparison_dir = output_dir / comparison_id
    comparison_dir.mkdir(parents=True, exist_ok=True)
    budget_dir = resolve_grid_dir(budget_dir, output_dir, "budget")
    recent_dir = resolve_grid_dir(recent_dir, output_dir, "recent")
    budget_rows = read_grid_summary(budget_dir / "grid_summary.csv", "budget")
    recent_rows = read_grid_summary(recent_dir / "grid_summary.csv", "recent")

    if global_local_dir is None:
        global_local_dir = run_grid(
            budget=budget,
            epsilon=epsilon,
            quantile=quantile,
            replicates_per_trial=replicates_per_trial,
            startup_trials=startup_trials,
            n_candidates=n_candidates,
            random_fraction=random_fraction,
            controller="global_local",
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
            run_name="global_local_grid",
            make_plot=make_plot,
        )
    global_local_rows = read_grid_summary(
        global_local_dir / "grid_summary.csv",
        "global_local",
    )
    comparison_rows = compare_rows(budget_rows, recent_rows, global_local_rows)
    write_comparison(comparison_dir / "lambda_controller_comparison.csv", comparison_rows)
    write_comparison(comparison_dir / "controller_three_way_comparison.csv", comparison_rows)
    if make_plot:
        plot_comparison(comparison_dir / "lambda_controller_comparison.png", comparison_rows)
        plot_comparison(comparison_dir / "controller_three_way_comparison.png", comparison_rows)
    print_report(comparison_dir, comparison_rows)
    return comparison_dir


def resolve_grid_dir(path: Path | None, results_dir: Path, lambda_mode: str) -> Path:
    if path is not None:
        return path
    return find_latest_grid_dir(results_dir, lambda_mode)


def find_latest_grid_dir(results_dir: Path, lambda_mode: str) -> Path:
    candidates: list[Path] = []
    if not results_dir.exists():
        raise FileNotFoundError(f"results directory does not exist: {results_dir}")
    for summary_path in results_dir.rglob("grid_summary.csv"):
        if grid_summary_matches(summary_path, lambda_mode):
            candidates.append(summary_path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"could not find a {lambda_mode!r} grid_summary.csv under {results_dir}; "
            f"pass the directory explicitly"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def grid_summary_matches(path: Path, lambda_mode: str) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if not rows:
        return False
    modes = {row.get("lambda_mode", "") for row in rows}
    return modes == {lambda_mode}


def read_grid_summary(path: Path, controller: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing grid summary: {path}")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    numeric_fields = {
        "best_raw_mean",
        "best_objective",
        "final_raw_mean",
        "final_objective",
        "trajectory_raw_mean_variance",
        "raw_variance_mean",
        "clipped_trial_count",
        "clipped_trial_percent",
        "lambda_mean",
        "lambda_max",
        "global_pressure_mean",
        "local_pressure_mean",
        "global_noise_pressure_mean",
        "global_quality_pressure_mean",
    }
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for field in numeric_fields:
                if field in parsed:
                    parsed[field] = float(parsed[field])
            parsed["controller"] = controller
            rows[(parsed["model_id"], parsed["market_id"])] = parsed
    return rows


def compare_rows(
    budget: dict[tuple[str, str], dict[str, Any]],
    recent: dict[tuple[str, str], dict[str, Any]],
    global_local: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(budget) & set(recent) & set(global_local)):
        base = budget[key]
        new = recent[key]
        anytime = global_local[key]
        rows.append(
            {
                "model_id": key[0],
                "market_id": key[1],
                "budget_best_raw_mean": base["best_raw_mean"],
                "recent_best_raw_mean": new["best_raw_mean"],
                "global_local_best_raw_mean": anytime["best_raw_mean"],
                "recent_delta_best_raw_vs_budget": new["best_raw_mean"] - base["best_raw_mean"],
                "global_local_delta_best_raw_vs_budget": anytime["best_raw_mean"]
                - base["best_raw_mean"],
                "global_local_delta_best_raw_vs_recent": anytime["best_raw_mean"]
                - new["best_raw_mean"],
                "budget_final_raw_mean": base["final_raw_mean"],
                "recent_final_raw_mean": new["final_raw_mean"],
                "global_local_final_raw_mean": anytime["final_raw_mean"],
                "recent_delta_final_raw_vs_budget": new["final_raw_mean"] - base["final_raw_mean"],
                "global_local_delta_final_raw_vs_budget": anytime["final_raw_mean"]
                - base["final_raw_mean"],
                "global_local_delta_final_raw_vs_recent": anytime["final_raw_mean"]
                - new["final_raw_mean"],
                "budget_trajectory_var": base["trajectory_raw_mean_variance"],
                "recent_trajectory_var": new["trajectory_raw_mean_variance"],
                "global_local_trajectory_var": anytime["trajectory_raw_mean_variance"],
                "recent_delta_trajectory_var_vs_budget": new["trajectory_raw_mean_variance"]
                - base["trajectory_raw_mean_variance"],
                "global_local_delta_trajectory_var_vs_budget": anytime[
                    "trajectory_raw_mean_variance"
                ]
                - base["trajectory_raw_mean_variance"],
                "global_local_delta_trajectory_var_vs_recent": anytime[
                    "trajectory_raw_mean_variance"
                ]
                - new["trajectory_raw_mean_variance"],
                "budget_clipped_count": base["clipped_trial_count"],
                "recent_clipped_count": new["clipped_trial_count"],
                "global_local_clipped_count": anytime["clipped_trial_count"],
                "recent_delta_clipped_count_vs_budget": new["clipped_trial_count"]
                - base["clipped_trial_count"],
                "global_local_delta_clipped_count_vs_budget": anytime["clipped_trial_count"]
                - base["clipped_trial_count"],
                "global_local_delta_clipped_count_vs_recent": anytime["clipped_trial_count"]
                - new["clipped_trial_count"],
                "budget_lambda_mean": base.get("lambda_mean", ""),
                "recent_lambda_mean": new.get("lambda_mean", ""),
                "global_local_lambda_mean": anytime.get("lambda_mean", ""),
                "global_local_global_pressure_mean": anytime.get("global_pressure_mean", ""),
                "global_local_local_pressure_mean": anytime.get("local_pressure_mean", ""),
            }
        )
    return rows


def write_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("controller comparison requires at least one shared scenario")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping lambda_controller_comparison.png")
        return

    labels = [f"{row['model_id']}-{row['market_id']}" for row in rows]
    x = list(range(len(labels)))
    width = 0.38
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].bar(
        [value - width / 2 for value in x],
        [row["recent_delta_best_raw_vs_budget"] for row in rows],
        width=width,
        label="recent",
        color="tab:blue",
    )
    axes[0].bar(
        [value + width / 2 for value in x],
        [row["global_local_delta_best_raw_vs_budget"] for row in rows],
        width=width,
        label="global-local",
        color="tab:purple",
    )
    axes[0].set_ylabel("delta best raw")
    axes[0].legend(loc="best")
    axes[1].bar(
        [value - width / 2 for value in x],
        [row["recent_delta_final_raw_vs_budget"] for row in rows],
        width=width,
        label="recent",
        color="tab:orange",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [row["global_local_delta_final_raw_vs_budget"] for row in rows],
        width=width,
        label="global-local",
        color="tab:red",
    )
    axes[1].set_ylabel("delta final raw")
    axes[1].legend(loc="best")
    axes[2].bar(
        [value - width / 2 for value in x],
        [row["recent_delta_trajectory_var_vs_budget"] for row in rows],
        width=width,
        label="recent",
        color="tab:green",
    )
    axes[2].bar(
        [value + width / 2 for value in x],
        [row["global_local_delta_trajectory_var_vs_budget"] for row in rows],
        width=width,
        label="global-local",
        color="tab:brown",
    )
    axes[2].set_ylabel("delta traj var")
    axes[2].set_xticks(x, labels)
    axes[2].tick_params(axis="x", labelrotation=45)
    axes[2].legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def print_report(comparison_dir: Path, rows: list[dict[str, Any]]) -> None:
    best_gain = max(rows, key=lambda row: row["global_local_delta_best_raw_vs_budget"])
    worst_loss = min(rows, key=lambda row: row["global_local_delta_best_raw_vs_budget"])
    best_vs_recent = max(rows, key=lambda row: row["global_local_delta_best_raw_vs_recent"])
    average_best_delta = sum(row["global_local_delta_best_raw_vs_budget"] for row in rows) / len(rows)
    average_final_delta = sum(row["global_local_delta_final_raw_vs_budget"] for row in rows) / len(rows)
    average_traj_delta = sum(row["global_local_delta_trajectory_var_vs_budget"] for row in rows) / len(rows)
    print(f"Wrote lambda-controller comparison to {comparison_dir}")
    print(
        "Largest global-local best-raw gain vs budget: "
        f"{best_gain['model_id']}-{best_gain['market_id']} "
        f"({best_gain['global_local_delta_best_raw_vs_budget']:.4f})"
    )
    print(
        "Largest global-local best-raw loss vs budget: "
        f"{worst_loss['model_id']}-{worst_loss['market_id']} "
        f"({worst_loss['global_local_delta_best_raw_vs_budget']:.4f})"
    )
    print(
        "Largest global-local best-raw gain vs recent: "
        f"{best_vs_recent['model_id']}-{best_vs_recent['market_id']} "
        f"({best_vs_recent['global_local_delta_best_raw_vs_recent']:.4f})"
    )
    print(
        "Average global-local deltas vs budget: "
        f"best={average_best_delta:.4f}, final={average_final_delta:.4f}, "
        f"trajectory_var={average_traj_delta:.4f}"
    )


def main() -> None:
    args = parse_args()
    run_controller_comparison(
        budget_dir=args.budget_dir,
        recent_dir=args.recent_dir,
        global_local_dir=args.global_local_dir,
        budget=args.budget,
        epsilon=args.epsilon,
        quantile=args.quantile,
        replicates_per_trial=args.replicates_per_trial,
        startup_trials=args.startup_trials,
        n_candidates=args.n_candidates,
        random_fraction=args.random_fraction,
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
