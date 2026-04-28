"""Lightweight paper-like portfolio benchmark for TPE-AS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tpeas.search_space import FloatParam, IntParam, SearchSpace

MODEL_IDS = ("M1", "M2", "M3")
MARKET_IDS = ("S1", "S2", "S3", "S4")

MODEL_DESCRIPTIONS = {
    "M1": "Trend-following momentum mimic",
    "M2": "Mean-reversion deviation mimic",
    "M3": "Threshold-based hybrid mimic",
}

MARKET_DESCRIPTIONS = {
    "S1": "High-volatility one-year market with regime switches",
    "S2": "Stable three-year bull market",
    "S3": "Five-year range-bound sideways market",
    "S4": "Four-year choppy range-bound sideways market",
}

MARKET_PERIODS = {
    "S1": 252,
    "S2": 756,
    "S3": 1260,
    "S4": 1008,
}


def paper_like_search_space() -> SearchSpace:
    """Shared 10-parameter mixed search space for all model-market scenarios."""

    return SearchSpace(
        [
            FloatParam("entry_threshold", 0.05, 1.5),
            FloatParam("exit_threshold", 0.01, 1.0),
            FloatParam("risk_budget", 0.05, 1.0),
            FloatParam("leverage", 0.25, 3.0),
            FloatParam("stop_loss", 0.01, 0.25),
            FloatParam("signal_mix", 0.0, 1.0),
            IntParam("lookback_short", 3, 60),
            IntParam("lookback_long", 20, 180),
            IntParam("rebalance_days", 1, 30),
            IntParam("vol_window", 10, 120),
        ]
    )


def generate_market_returns(market_id: str, seed: int = 0, assets: int = 8) -> np.ndarray:
    """Generate deterministic daily asset returns for a paper-like market setting."""

    if market_id not in MARKET_IDS:
        raise ValueError(f"unknown market_id: {market_id}")
    rng = np.random.default_rng(seed + 1009 * (MARKET_IDS.index(market_id) + 1))
    periods = MARKET_PERIODS[market_id]

    if market_id == "S1":
        factor = _high_vol_market(rng, periods)
        idio_scale = 0.018
        asset_drift_scale = 0.00018
    elif market_id == "S2":
        factor = _stable_bull_market(rng, periods)
        idio_scale = 0.007
        asset_drift_scale = 0.00010
    elif market_id == "S3":
        factor = _range_bound_market(rng, periods, shock_scale=0.008, reversion=-0.32)
        idio_scale = 0.010
        asset_drift_scale = 0.00008
    else:
        factor = _range_bound_market(rng, periods, shock_scale=0.012, reversion=-0.48)
        idio_scale = 0.013
        asset_drift_scale = 0.00012

    loadings = rng.uniform(0.55, 1.35, size=assets)
    idiosyncratic = rng.normal(0.0, idio_scale, size=(periods, assets))
    drift = rng.normal(0.0, asset_drift_scale, size=assets)
    returns = factor[:, None] * loadings[None, :] + idiosyncratic + drift[None, :]
    return returns.astype(float)


def _high_vol_market(rng: np.random.Generator, periods: int) -> np.ndarray:
    factor = np.empty(periods, dtype=float)
    drift_choices = np.asarray([-0.0009, -0.00025, 0.0002, 0.0008])
    for start in range(0, periods, 21):
        end = min(periods, start + 21)
        drift = float(rng.choice(drift_choices))
        vol = float(rng.uniform(0.013, 0.026))
        factor[start:end] = rng.normal(drift, vol, size=end - start)
    for start in range(30, periods, 63):
        end = min(periods, start + 5)
        factor[start:end] += rng.normal(0.0, 0.035, size=end - start)
    return factor


def _stable_bull_market(rng: np.random.Generator, periods: int) -> np.ndarray:
    slow_cycle = 0.00012 * np.sin(np.linspace(0.0, 5.0 * np.pi, periods))
    return rng.normal(0.00055 + slow_cycle, 0.0058, size=periods)


def _range_bound_market(
    rng: np.random.Generator,
    periods: int,
    shock_scale: float,
    reversion: float,
) -> np.ndarray:
    factor = np.empty(periods, dtype=float)
    previous = 0.0
    seasonal = 0.0007 * np.sin(np.linspace(0.0, 12.0 * np.pi, periods))
    for idx in range(periods):
        shock = rng.normal(0.0, shock_scale)
        factor[idx] = reversion * previous + seasonal[idx] + shock
        previous = factor[idx]
    return factor


@dataclass(frozen=True)
class PaperLikePortfolioBenchmark:
    """Black-box model mimic evaluated on one generated market setting."""

    model_id: str
    market_id: str
    seed: int = 0
    assets: int = 8

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise ValueError(f"unknown model_id: {self.model_id}")
        if self.market_id not in MARKET_IDS:
            raise ValueError(f"unknown market_id: {self.market_id}")
        returns = generate_market_returns(self.market_id, seed=self.seed, assets=self.assets)
        object.__setattr__(self, "returns", returns)
        object.__setattr__(self, "_score_cache", {})

    def evaluate(self, params: dict[str, Any], rng: np.random.Generator) -> float:
        """Return a noisy annualized Sharpe-like score for one configuration."""

        clean_score = self._cached_deterministic(params)
        evaluation_noise = self._evaluation_noise(params)
        return float(clean_score + rng.normal(0.0, evaluation_noise))

    def evaluate_deterministic(self, params: dict[str, Any]) -> float:
        positions = self._positions(params)
        if len(positions) < 2:
            return 0.0
        raw_strategy_returns = np.sum(positions[:-1] * self.returns[1:], axis=1)
        strategy_returns = self._apply_risk_controls(raw_strategy_returns, positions, params)
        std = float(np.std(strategy_returns))
        if std <= 1e-12:
            return 0.0
        sharpe = float(np.sqrt(252.0) * np.mean(strategy_returns) / std)
        return sharpe - self._configuration_penalty(params)

    def _cached_deterministic(self, params: dict[str, Any]) -> float:
        key = tuple((name, params[name]) for name in paper_like_search_space().names)
        cache = self._score_cache
        if key not in cache:
            cache[key] = self.evaluate_deterministic(params)
        return float(cache[key])

    def _positions(self, params: dict[str, Any]) -> np.ndarray:
        lookback_short = int(params["lookback_short"])
        lookback_long = int(params["lookback_long"])
        rebalance_days = int(params["rebalance_days"])
        vol_window = int(params["vol_window"])
        entry = float(params["entry_threshold"])
        exit_threshold = float(params["exit_threshold"])
        signal_mix = float(params["signal_mix"])
        leverage = float(params["leverage"])
        risk_budget = float(params["risk_budget"])

        short_momentum = _rolling_mean(self.returns, lookback_short)
        long_momentum = _rolling_mean(self.returns, lookback_long)
        volatility = _rolling_std(self.returns, vol_window)
        z_price = _rolling_zscore(self.returns, lookback_long)
        momentum_signal = (short_momentum - long_momentum) / (volatility + 1e-8)
        reversion_signal = -z_price

        if self.model_id == "M1":
            raw_signal = momentum_signal
        elif self.model_id == "M2":
            raw_signal = reversion_signal
        else:
            raw_signal = signal_mix * momentum_signal + (1.0 - signal_mix) * reversion_signal
            raw_signal = np.where(np.abs(momentum_signal) >= exit_threshold, raw_signal, 0.0)

        active = np.abs(raw_signal) >= entry
        positions = np.where(active, np.sign(raw_signal), 0.0)
        positions = positions * leverage * risk_budget
        scale = np.sum(np.abs(positions), axis=1, keepdims=True)
        positions = np.divide(positions, np.maximum(scale, 1.0), out=np.zeros_like(positions), where=scale > 0)
        positions = positions * min(leverage, 3.0)

        for idx in range(1, len(positions)):
            if idx % rebalance_days != 0:
                positions[idx] = positions[idx - 1]
        return positions

    def _apply_risk_controls(
        self,
        raw_strategy_returns: np.ndarray,
        positions: np.ndarray,
        params: dict[str, Any],
    ) -> np.ndarray:
        stop_loss = float(params["stop_loss"])
        leverage = float(params["leverage"])
        turnover = np.sum(np.abs(np.diff(positions, axis=0)), axis=1)
        transaction_cost = 0.00035 * turnover
        capped = np.maximum(raw_strategy_returns, -stop_loss * max(1.0, leverage))
        return capped - transaction_cost

    def _configuration_penalty(self, params: dict[str, Any]) -> float:
        lookback_short = int(params["lookback_short"])
        lookback_long = int(params["lookback_long"])
        entry = float(params["entry_threshold"])
        exit_threshold = float(params["exit_threshold"])
        leverage = float(params["leverage"])
        risk_budget = float(params["risk_budget"])
        signal_mix = float(params["signal_mix"])

        penalty = 0.0
        if lookback_short >= lookback_long:
            penalty += 0.25 + 0.004 * (lookback_short - lookback_long + 1)
        if exit_threshold > entry:
            penalty += 0.12 * (exit_threshold - entry)
        if leverage * risk_budget > 2.0:
            penalty += 0.08 * (leverage * risk_budget - 2.0) ** 2
        if self.model_id == "M1" and signal_mix < 0.25:
            penalty += 0.04
        if self.model_id == "M2" and signal_mix > 0.75:
            penalty += 0.04
        if self.market_id == "S1" and leverage > 2.2:
            penalty += 0.08 * (leverage - 2.2)
        return float(penalty)

    def _evaluation_noise(self, params: dict[str, Any]) -> float:
        leverage = float(params["leverage"])
        risk_budget = float(params["risk_budget"])
        market_noise = {"S1": 0.20, "S2": 0.07, "S3": 0.10, "S4": 0.14}[self.market_id]
        model_noise = {"M1": 0.06, "M2": 0.08, "M3": 0.11}[self.model_id]
        return float(market_noise + model_noise + 0.04 * leverage + 0.04 * risk_budget)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    cumulative = np.cumsum(values, axis=0)
    result = np.zeros_like(values)
    if window < len(values):
        result[window:] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(values, window)
    second_moment = _rolling_mean(values * values, window)
    variance = np.maximum(second_moment - mean * mean, 1e-10)
    return np.sqrt(variance)


def _rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(values, window)
    std = _rolling_std(values, window)
    return (values - mean) / (std + 1e-8)
