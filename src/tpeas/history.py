"""Trial-history records and persistence helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tpeas.controllers import controller_label


@dataclass(frozen=True)
class TrialRecord:
    step: int
    params: dict[str, Any]
    raw_samples: tuple[float, ...]
    raw_mean: float
    raw_variance: float
    lambda_t: float
    importance_weight: float
    clipped_weight: float
    objective: float
    elapsed_seconds: float
    sampler: str
    lambda_mode: str = "budget"
    recent_mean: float | None = None
    previous_mean: float | None = None
    recent_variance_mean: float | None = None
    baseline_variance: float | None = None
    noise_pressure: float = 0.0
    mean_drop_pressure: float = 0.0
    global_mean_percentile: float | None = None
    global_variance_percentile: float | None = None
    global_noise_pressure: float = 0.0
    global_quality_pressure: float = 0.0
    global_pressure: float = 0.0
    local_pressure: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "params": self.params,
            "raw_samples": self.raw_samples,
            "raw_mean": self.raw_mean,
            "raw_variance": self.raw_variance,
            "lambda_t": self.lambda_t,
            "importance_weight": self.importance_weight,
            "clipped_weight": self.clipped_weight,
            "objective": self.objective,
            "elapsed_seconds": self.elapsed_seconds,
            "sampler": self.sampler,
            "lambda_mode": self.lambda_mode,
            "controller": controller_label(self.lambda_mode),
            "recent_mean": self.recent_mean,
            "previous_mean": self.previous_mean,
            "recent_variance_mean": self.recent_variance_mean,
            "baseline_variance": self.baseline_variance,
            "noise_pressure": self.noise_pressure,
            "mean_drop_pressure": self.mean_drop_pressure,
            "global_mean_percentile": self.global_mean_percentile,
            "global_variance_percentile": self.global_variance_percentile,
            "global_noise_pressure": self.global_noise_pressure,
            "global_quality_pressure": self.global_quality_pressure,
            "global_pressure": self.global_pressure,
            "local_pressure": self.local_pressure,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        flat = self.to_dict()
        flat["params"] = json.dumps(self.params, sort_keys=True)
        flat["raw_samples"] = json.dumps(self.raw_samples)
        return flat


@dataclass(frozen=True)
class OptimizationResult:
    records: tuple[TrialRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def best_by_objective(self) -> TrialRecord:
        if not self.records:
            raise ValueError("no trial records")
        return max(self.records, key=lambda record: record.objective)

    @property
    def best_by_raw_mean(self) -> TrialRecord:
        if not self.records:
            raise ValueError("no trial records")
        return max(self.records, key=lambda record: record.raw_mean)

    def best_candidate(self) -> TrialRecord:
        """Return the selected incumbent by adaptive objective, not the last trial."""

        return self.best_by_objective

    @property
    def raw_mean_variance(self) -> float:
        if not self.records:
            raise ValueError("no trial records")
        values = [record.raw_mean for record in self.records]
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            path.write_text("", encoding="utf-8")
            return
        rows = [record.to_flat_dict() for record in self.records]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
