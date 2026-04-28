"""Run a synthetic TPE-AS comparison experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tpeas.config import AdaptiveObjectiveConfig
from tpeas.controllers import (
    LAMBDA_MODES,
    PUBLIC_CONTROLLER_MODES,
    public_controller_help,
    resolve_controller_mode,
)
from tpeas.custom_tpe import CustomTPEASOptimizer
from tpeas.evaluators import SyntheticInstabilityBenchmark, synthetic_instability_search_space
from tpeas.random_search import RandomSearchOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--replicates-per-trial", type=int, default=3)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=64)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("synthetic_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_mode = resolve_controller_mode(args.controller, args.lambda_mode)

    config = AdaptiveObjectiveConfig(
        budget=args.budget,
        epsilon=args.epsilon,
        quantile=args.quantile,
        replicates_per_trial=args.replicates_per_trial,
        startup_trials=args.startup_trials,
        n_candidates=args.n_candidates,
        random_fraction=args.random_fraction,
        lambda_mode=lambda_mode,
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
    )
    search_space = synthetic_instability_search_space()
    evaluator = SyntheticInstabilityBenchmark()

    summaries: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        optimizers = [
            CustomTPEASOptimizer(
                search_space,
                evaluator,
                config,
                seed=seed,
                use_variance_penalty=True,
                use_importance=True,
                sampler_name="custom-tpe-as",
            ),
            CustomTPEASOptimizer(
                search_space,
                evaluator,
                config,
                seed=seed,
                use_variance_penalty=True,
                use_importance=False,
                sampler_name="custom-tpe-as-no-importance",
            ),
            CustomTPEASOptimizer(
                search_space,
                evaluator,
                config,
                seed=seed,
                use_variance_penalty=False,
                use_importance=False,
                sampler_name="custom-tpe-raw",
            ),
            RandomSearchOptimizer(
                search_space,
                evaluator,
                config,
                seed=seed,
                use_variance_penalty=False,
                sampler_name="random-search",
            ),
        ]
        for optimizer in optimizers:
            result = optimizer.optimize()
            sampler = result.records[0].sampler
            selected = result.best_candidate()
            result.to_jsonl(output_dir / f"{sampler}_seed{seed}.jsonl")
            result.to_csv(output_dir / f"{sampler}_seed{seed}.csv")
            summaries.append(
                {
                    "sampler": sampler,
                    "seed": seed,
                    "lambda_mode": lambda_mode,
                    "controller": result.records[0].to_dict()["controller"],
                    "selected_step": selected.step,
                    "selected_objective": selected.objective,
                    "selected_raw_mean": selected.raw_mean,
                    "selected_raw_variance": selected.raw_variance,
                    "selected_params": json.dumps(selected.params, sort_keys=True),
                    "best_objective": result.best_by_objective.objective,
                    "best_raw_mean": result.best_by_raw_mean.raw_mean,
                    "best_raw_variance": result.best_by_raw_mean.raw_variance,
                    "trajectory_raw_mean_variance": result.raw_mean_variance,
                }
            )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    if args.plot:
        plot_histories(output_dir)

    print(f"Wrote results to {output_dir}")


def plot_histories(output_dir: Path) -> None:
    try:
        import json

        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping plots")
        return

    for jsonl_path in output_dir.glob("*.jsonl"):
        steps: list[int] = []
        raw_means: list[float] = []
        objectives: list[float] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                steps.append(row["step"])
                raw_means.append(row["raw_mean"])
                objectives.append(row["objective"])
        plt.figure(figsize=(8, 4))
        plt.plot(steps, raw_means, label="raw mean f(x)")
        plt.plot(steps, objectives, label="adaptive objective J")
        plt.xlabel("optimization step")
        plt.ylabel("score")
        plt.title(jsonl_path.stem)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{jsonl_path.stem}.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
