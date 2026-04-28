"""Search-space definitions shared by the optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class FloatParam:
    name: str
    low: float
    high: float
    log_scale: bool = False

    kind: str = "float"

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must be greater than low")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"{self.name}: low must be positive for log-scale sampling")

    def sample(self, rng: np.random.Generator) -> float:
        if self.log_scale:
            return float(exp(rng.uniform(log(self.low), log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def to_internal(self, value: Any) -> float:
        value = float(value)
        return float(log(value)) if self.log_scale else value

    def from_internal(self, value: float) -> float:
        decoded = exp(value) if self.log_scale else value
        return float(np.clip(decoded, self.low, self.high))

    @property
    def internal_bounds(self) -> tuple[float, float]:
        if self.log_scale:
            return log(self.low), log(self.high)
        return self.low, self.high

    def suggest_optuna(self, trial: Any) -> float:
        return float(trial.suggest_float(self.name, self.low, self.high, log=self.log_scale))


@dataclass(frozen=True)
class IntParam:
    name: str
    low: int
    high: int
    log_scale: bool = False

    kind: str = "int"

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"{self.name}: high must be greater than or equal to low")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"{self.name}: low must be positive for log-scale sampling")

    def sample(self, rng: np.random.Generator) -> int:
        if self.log_scale:
            value = round(exp(rng.uniform(log(self.low), log(self.high))))
            return int(np.clip(value, self.low, self.high))
        return int(rng.integers(self.low, self.high + 1))

    def to_internal(self, value: Any) -> float:
        value = int(value)
        return float(log(value)) if self.log_scale else float(value)

    def from_internal(self, value: float) -> int:
        decoded = exp(value) if self.log_scale else value
        return int(np.clip(round(decoded), self.low, self.high))

    @property
    def internal_bounds(self) -> tuple[float, float]:
        if self.log_scale:
            return log(self.low), log(self.high)
        return float(self.low), float(self.high)

    def suggest_optuna(self, trial: Any) -> int:
        return int(trial.suggest_int(self.name, self.low, self.high, log=self.log_scale))


@dataclass(frozen=True)
class CategoricalParam:
    name: str
    choices: tuple[Any, ...]

    kind: str = "categorical"

    def __init__(self, name: str, choices: Iterable[Any]):
        choices_tuple = tuple(choices)
        if not choices_tuple:
            raise ValueError(f"{name}: choices must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "choices", choices_tuple)
        object.__setattr__(self, "kind", "categorical")

    def sample(self, rng: np.random.Generator) -> Any:
        index = int(rng.integers(0, len(self.choices)))
        return self.choices[index]

    def to_internal(self, value: Any) -> float:
        return float(self.choices.index(value))

    def from_internal(self, value: float) -> Any:
        index = int(np.clip(round(value), 0, len(self.choices) - 1))
        return self.choices[index]

    @property
    def internal_bounds(self) -> tuple[float, float]:
        return 0.0, float(len(self.choices) - 1)

    def suggest_optuna(self, trial: Any) -> Any:
        return trial.suggest_categorical(self.name, list(self.choices))


Param = FloatParam | IntParam | CategoricalParam


@dataclass(frozen=True)
class SearchSpace:
    """A variable-dimensional mixed search space.

    Optimizers iterate over the provided parameter list, so user spaces can contain
    any positive number of float, integer, and categorical parameters.
    """

    params: tuple[Param, ...]

    def __init__(self, params: Iterable[Param]):
        params_tuple = tuple(params)
        if not params_tuple:
            raise ValueError("SearchSpace requires at least one parameter")
        names = [param.name for param in params_tuple]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        object.__setattr__(self, "params", params_tuple)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {param.name: param.sample(rng) for param in self.params}

    def encode(self, values: dict[str, Any]) -> np.ndarray:
        return np.asarray([param.to_internal(values[param.name]) for param in self.params], dtype=float)

    def decode(self, encoded: Iterable[float]) -> dict[str, Any]:
        return {
            param.name: param.from_internal(value)
            for param, value in zip(self.params, encoded, strict=True)
        }

    def suggest_optuna(self, trial: Any) -> dict[str, Any]:
        return {param.name: param.suggest_optuna(trial) for param in self.params}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(param.name for param in self.params)
