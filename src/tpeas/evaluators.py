"""Transparent black-box evaluators used for replication experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from tpeas.search_space import CategoricalParam, FloatParam, IntParam, SearchSpace


class BlackBoxEvaluator(Protocol):
    def evaluate(self, params: dict[str, Any], rng: np.random.Generator) -> float:
        """Return one noisy observation for a hyperparameter configuration."""


def synthetic_instability_search_space() -> SearchSpace:
    """Mixed search space with float, integer, and categorical parameters."""

    return SearchSpace(
        [
            FloatParam("x", -1.0, 1.0),
            FloatParam("y", -1.0, 1.0),
            IntParam("lookback", 5, 80),
            FloatParam("leverage", 0.25, 2.5),
            CategoricalParam("regime", ["stable", "balanced", "risky"]),
        ]
    )


def ten_parameter_synthetic_search_space() -> SearchSpace:
    """A 10-parameter mixed search space for TPE-AS trajectory demonstrations."""

    return SearchSpace(
        [
            FloatParam("trend_strength", -1.5, 1.5),
            FloatParam("mean_reversion", -1.5, 1.5),
            FloatParam("risk_budget", 0.05, 1.0),
            FloatParam("leverage", 0.25, 3.0),
            FloatParam("entry_threshold", 0.05, 1.5),
            IntParam("lookback_short", 3, 60),
            IntParam("lookback_long", 20, 180),
            IntParam("rebalance_days", 1, 30),
            CategoricalParam("signal_family", ["trend", "reversion", "hybrid"]),
            CategoricalParam("market_regime", ["calm", "volatile", "crisis"]),
        ]
    )


@dataclass(frozen=True)
class SyntheticInstabilityBenchmark:
    """Noisy objective with an attractive but unstable high-return region."""

    def evaluate(self, params: dict[str, Any], rng: np.random.Generator) -> float:
        x = float(params["x"])
        y = float(params["y"])
        lookback = int(params["lookback"])
        leverage = float(params["leverage"])
        regime = str(params["regime"])

        stable_peak = np.exp(-(((x - 0.20) ** 2) / 0.055 + ((y + 0.30) ** 2) / 0.08))
        risky_peak = np.exp(-(((x - 0.72) ** 2) / 0.020 + ((y - 0.58) ** 2) / 0.026))
        lookback_bonus = np.exp(-((lookback - 32) ** 2) / 900.0)

        regime_bonus = {"stable": 0.18, "balanced": 0.05, "risky": -0.02}[regime]
        regime_noise = {"stable": 0.03, "balanced": 0.10, "risky": 0.20}[regime]
        leverage_penalty = 0.08 * max(0.0, leverage - 1.3) ** 2

        deterministic = (
            0.25
            + 1.85 * stable_peak
            + 2.35 * risky_peak
            + 0.22 * lookback_bonus
            + regime_bonus
            - leverage_penalty
        )
        noise_scale = 0.05 + 0.08 * leverage + 0.70 * risky_peak + regime_noise
        return float(deterministic + rng.normal(0.0, noise_scale))


@dataclass(frozen=True)
class TenParameterSyntheticBenchmark:
    """A transparent 10-parameter objective with stable and risky optima."""

    def evaluate(self, params: dict[str, Any], rng: np.random.Generator) -> float:
        stable_peak, risky_peak = self._basin_scores(params)
        signal_family = str(params["signal_family"])
        market_regime = str(params["market_regime"])
        risk_budget = float(params["risk_budget"])
        leverage = float(params["leverage"])
        entry_threshold = float(params["entry_threshold"])
        lookback_short = int(params["lookback_short"])
        lookback_long = int(params["lookback_long"])
        rebalance_days = int(params["rebalance_days"])

        stable_category = self._stable_category_score(signal_family, market_regime)
        risky_category = self._risky_category_score(signal_family, market_regime)
        consistency_bonus = 0.12 * np.exp(-((lookback_long / max(lookback_short, 1) - 4.5) ** 2) / 8.0)
        rebalance_bonus = 0.08 * np.exp(-((rebalance_days - 7) ** 2) / 80.0)

        deterministic = (
            0.25
            + 2.05 * stable_peak * stable_category
            + 2.65 * risky_peak * risky_category
            + consistency_bonus
            + rebalance_bonus
        )
        deterministic -= self._penalty(
            signal_family=signal_family,
            market_regime=market_regime,
            risk_budget=risk_budget,
            leverage=leverage,
            entry_threshold=entry_threshold,
            lookback_short=lookback_short,
            lookback_long=lookback_long,
        )

        noise_scale = (
            0.04
            + 0.08 * risk_budget
            + 0.045 * leverage
            + 0.95 * risky_peak * risky_category
            + {"calm": 0.02, "volatile": 0.16, "crisis": 0.32}[market_regime]
        )
        if signal_family == "trend" and market_regime == "crisis":
            noise_scale += 0.18
        if leverage > 2.0:
            noise_scale += 0.11 * (leverage - 2.0)
        noise_scale = max(0.03, noise_scale - 0.03 * stable_peak * stable_category)

        return float(deterministic + rng.normal(0.0, noise_scale))

    def basin_label(self, params: dict[str, Any]) -> str:
        """Classify a configuration for reporting trajectory region counts."""

        stable_peak, risky_peak = self._basin_scores(params)
        stable_score = stable_peak * self._stable_category_score(
            str(params["signal_family"]), str(params["market_regime"])
        )
        risky_score = risky_peak * self._risky_category_score(
            str(params["signal_family"]), str(params["market_regime"])
        )
        if stable_score >= 0.18 and stable_score >= risky_score:
            return "stable"
        if risky_score >= 0.18 and risky_score > stable_score:
            return "risky"
        return "other"

    def _basin_scores(self, params: dict[str, Any]) -> tuple[float, float]:
        stable_distance = (
            self._scaled_sq(float(params["trend_strength"]), 0.55, 0.45)
            + self._scaled_sq(float(params["mean_reversion"]), 0.25, 0.40)
            + self._scaled_sq(float(params["risk_budget"]), 0.35, 0.18)
            + self._scaled_sq(float(params["leverage"]), 1.05, 0.35)
            + self._scaled_sq(float(params["entry_threshold"]), 0.55, 0.30)
            + self._scaled_sq(int(params["lookback_short"]), 18.0, 11.0)
            + self._scaled_sq(int(params["lookback_long"]), 90.0, 30.0)
            + self._scaled_sq(int(params["rebalance_days"]), 7.0, 5.0)
        )
        risky_distance = (
            self._scaled_sq(float(params["trend_strength"]), 1.20, 0.28)
            + self._scaled_sq(float(params["mean_reversion"]), -0.85, 0.34)
            + self._scaled_sq(float(params["risk_budget"]), 0.82, 0.13)
            + self._scaled_sq(float(params["leverage"]), 2.35, 0.32)
            + self._scaled_sq(float(params["entry_threshold"]), 0.20, 0.18)
            + self._scaled_sq(int(params["lookback_short"]), 8.0, 6.0)
            + self._scaled_sq(int(params["lookback_long"]), 34.0, 12.0)
            + self._scaled_sq(int(params["rebalance_days"]), 2.0, 2.5)
        )
        dimension_count = 8.0
        return (
            float(np.exp(-0.5 * stable_distance / dimension_count)),
            float(np.exp(-0.5 * risky_distance / dimension_count)),
        )

    @staticmethod
    def _scaled_sq(value: float, target: float, scale: float) -> float:
        return ((value - target) / scale) ** 2

    @staticmethod
    def _stable_category_score(signal_family: str, market_regime: str) -> float:
        signal_score = {"hybrid": 1.0, "reversion": 0.70, "trend": 0.55}[signal_family]
        regime_score = {"calm": 1.0, "volatile": 0.62, "crisis": 0.20}[market_regime]
        return signal_score * regime_score

    @staticmethod
    def _risky_category_score(signal_family: str, market_regime: str) -> float:
        signal_score = {"trend": 1.0, "hybrid": 0.75, "reversion": 0.35}[signal_family]
        regime_score = {"volatile": 1.0, "crisis": 0.82, "calm": 0.25}[market_regime]
        return signal_score * regime_score

    @staticmethod
    def _penalty(
        signal_family: str,
        market_regime: str,
        risk_budget: float,
        leverage: float,
        entry_threshold: float,
        lookback_short: int,
        lookback_long: int,
    ) -> float:
        penalty = 0.0
        if lookback_short >= lookback_long:
            penalty += 0.8 + 0.025 * (lookback_short - lookback_long + 1)
        if leverage > 2.2:
            penalty += 0.20 * (leverage - 2.2) ** 2
        if risk_budget * leverage > 1.65:
            penalty += 0.45 * (risk_budget * leverage - 1.65) ** 2
        if entry_threshold < 0.18 and leverage > 1.8:
            penalty += 0.22 * (1.8 - entry_threshold)
        if signal_family == "reversion" and market_regime == "crisis":
            penalty += 0.28
        if signal_family == "trend" and market_regime == "calm":
            penalty += 0.12
        return float(penalty)


def generated_portfolio_search_space() -> SearchSpace:
    """Finance-like generated-data search space; no proprietary data required."""

    return SearchSpace(
        [
            IntParam("lookback", 5, 90),
            FloatParam("entry_threshold", 0.05, 1.5),
            FloatParam("max_weight", 0.05, 0.40),
            FloatParam("stop_loss", 0.02, 0.25),
            IntParam("rebalance_every", 1, 21),
            CategoricalParam("style", ["trend", "mean_reversion", "hybrid"]),
        ]
    )


@dataclass
class GeneratedReturnsPortfolioBenchmark:
    """Small generated-return portfolio benchmark returning annualized Sharpe."""

    periods: int = 756
    assets: int = 8
    seed: int = 123

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        market = rng.normal(0.00035, 0.009, size=self.periods)
        asset_loadings = rng.uniform(0.65, 1.25, size=self.assets)
        idiosyncratic = rng.normal(0.0, 0.012, size=(self.periods, self.assets))
        drift = rng.normal(0.00005, 0.00018, size=self.assets)
        self.returns = market[:, None] * asset_loadings[None, :] + idiosyncratic + drift[None, :]

    def evaluate(self, params: dict[str, Any], rng: np.random.Generator) -> float:
        lookback = int(params["lookback"])
        entry_threshold = float(params["entry_threshold"])
        max_weight = float(params["max_weight"])
        stop_loss = float(params["stop_loss"])
        rebalance_every = int(params["rebalance_every"])
        style = str(params["style"])

        rolling_signal = self._rolling_mean(lookback)
        signal_scale = np.std(rolling_signal, axis=1, keepdims=True) + 1e-8
        z_signal = rolling_signal / signal_scale

        if style == "trend":
            raw_signal = z_signal
        elif style == "mean_reversion":
            raw_signal = -z_signal
        else:
            raw_signal = np.where(np.abs(z_signal) > entry_threshold, z_signal, -0.35 * z_signal)

        positions = np.where(np.abs(raw_signal) >= entry_threshold, np.sign(raw_signal), 0.0)
        positions *= max_weight
        positions[::rebalance_every] = positions[::rebalance_every]
        for t in range(1, len(positions)):
            if t % rebalance_every != 0:
                positions[t] = positions[t - 1]

        strategy_returns = np.sum(positions[:-1] * self.returns[1:], axis=1)
        strategy_returns = np.where(strategy_returns < -stop_loss, -stop_loss, strategy_returns)
        transaction_cost = 0.0003 * np.sum(np.abs(np.diff(positions, axis=0)), axis=1)
        strategy_returns = strategy_returns - transaction_cost

        # Add tiny evaluator noise to mimic stochastic backtest/model variation.
        strategy_returns = strategy_returns + rng.normal(0.0, 0.0004, size=strategy_returns.shape)
        std = float(np.std(strategy_returns))
        if std <= 1e-12:
            return 0.0
        return float(np.sqrt(252.0) * np.mean(strategy_returns) / std)

    def _rolling_mean(self, lookback: int) -> np.ndarray:
        cumulative = np.cumsum(self.returns, axis=0)
        signal = np.zeros_like(self.returns)
        signal[lookback:] = (cumulative[lookback:] - cumulative[:-lookback]) / lookback
        return signal
