from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from quant_stack.module5 import Module5Config, run_backtest


@dataclass(frozen=True)
class MonteCarloStressConfig:
    n_simulations: int = 10_000
    confidence_level: float = 0.99
    random_state: int = 42


@dataclass(frozen=True)
class MonteCarloStressResult:
    config: MonteCarloStressConfig
    simulations: int
    periods_per_path: int
    var_99_terminal_return: float
    cvar_99_terminal_return: float
    var_99_max_drawdown: float
    median_max_drawdown: float
    drawdown_distribution: np.ndarray
    terminal_return_distribution: np.ndarray


@dataclass(frozen=True)
class GapScenarioResult:
    gap_size: float
    label_distribution: dict[int, int]
    avg_return: float
    worst_return: float
    best_return: float


@dataclass(frozen=True)
class FeeSensitivityResult:
    summary: pl.DataFrame
    breakeven_fee_bps: float | None


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    clean = np.asarray(returns, dtype=np.float64)
    if clean.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.nan_to_num(clean, nan=0.0))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(np.nanmin(drawdown))


def simulate_monte_carlo_max_drawdown(
    returns: np.ndarray | pd.Series | pl.Series,
    config: MonteCarloStressConfig = MonteCarloStressConfig(),
) -> MonteCarloStressResult:
    if isinstance(returns, pl.Series):
        values = returns.to_numpy()
    elif isinstance(returns, pd.Series):
        values = returns.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(returns, dtype=np.float64)

    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite returns are required for Monte Carlo stress testing.")

    rng = np.random.default_rng(config.random_state)
    sampled_indices = rng.integers(0, values.size, size=(config.n_simulations, values.size))
    simulated_returns = values[sampled_indices]

    equity = np.cumprod(1.0 + simulated_returns, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity / running_max - 1.0
    max_drawdowns = np.min(drawdowns, axis=1)
    terminal_returns = equity[:, -1] - 1.0

    tail_quantile = max(1.0 - float(config.confidence_level), 1e-6)
    var_99_terminal = float(np.quantile(terminal_returns, tail_quantile))
    var_99_drawdown = float(np.quantile(max_drawdowns, tail_quantile))
    tail_mask = terminal_returns <= var_99_terminal
    if np.any(tail_mask):
        cvar_99_terminal = float(np.mean(terminal_returns[tail_mask]))
    else:
        cvar_99_terminal = var_99_terminal

    return MonteCarloStressResult(
        config=config,
        simulations=int(config.n_simulations),
        periods_per_path=int(values.size),
        var_99_terminal_return=var_99_terminal,
        cvar_99_terminal_return=cvar_99_terminal,
        var_99_max_drawdown=var_99_drawdown,
        median_max_drawdown=float(np.median(max_drawdowns)),
        drawdown_distribution=max_drawdowns,
        terminal_return_distribution=terminal_returns,
    )


def simulate_gap_scenarios(
    label_frame: pl.DataFrame,
    gap_size: float = 0.05,
) -> GapScenarioResult:
    if label_frame.is_empty():
        return GapScenarioResult(
            gap_size=float(gap_size),
            label_distribution={-1: 0, 0: 0, 1: 0},
            avg_return=0.0,
            worst_return=0.0,
            best_return=0.0,
        )

    rets = label_frame["ret"].to_numpy()
    trgt = np.maximum(label_frame["trgt"].to_numpy(), 1e-12)
    shifted_returns = rets + float(gap_size) * np.sign(np.where(np.abs(rets) > 0, rets, 1.0))
    labels = np.where(
        shifted_returns >= trgt,
        1,
        np.where(shifted_returns <= -trgt, -1, 0),
    )
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {-1: 0, 0: 0, 1: 0}
    for label, count in zip(unique, counts, strict=True):
        distribution[int(label)] = int(count)
    return GapScenarioResult(
        gap_size=float(gap_size),
        label_distribution=distribution,
        avg_return=float(np.mean(shifted_returns)),
        worst_return=float(np.min(shifted_returns)),
        best_return=float(np.max(shifted_returns)),
    )


def run_fee_sensitivity(
    position_sizing: pl.DataFrame | pd.DataFrame | str | Path,
    stationary_bars: pl.DataFrame | pd.DataFrame | str | Path,
    fee_bps_values: Sequence[float] = (2.0, 4.0, 6.0, 8.0, 10.0),
    backtest_config: Module5Config = Module5Config(),
) -> FeeSensitivityResult:
    rows: list[dict[str, float]] = []
    for fee_bps in fee_bps_values:
        artifacts = run_backtest(
            position_sizing,
            stationary_bars,
            Module5Config(
                benchmark_symbol=backtest_config.benchmark_symbol,
                execution_delay_bars=backtest_config.execution_delay_bars,
                fee_bps=float(fee_bps),
                slippage_vol_multiplier=backtest_config.slippage_vol_multiplier,
                meta_prob_win_threshold=backtest_config.meta_prob_win_threshold,
                output_root=backtest_config.output_root,
                save_plots=False,
            ),
        )
        metrics = artifacts.tearsheet["strategies"]["meta_strategy"]
        rows.append(
            {
                "fee_bps": float(fee_bps),
                "meta_sharpe": float(metrics["sharpe"]),
                "meta_max_drawdown": float(metrics["max_drawdown"]),
                "meta_profit_factor": float(metrics["profit_factor"]),
                "meta_annual_return": float(metrics["annual_return"]),
            }
        )

    summary = pl.DataFrame(rows).sort("fee_bps")
    positive = summary.filter(pl.col("meta_annual_return") > 0.0)
    breakeven = float(positive["fee_bps"].max()) if positive.height > 0 else None
    return FeeSensitivityResult(summary=summary, breakeven_fee_bps=breakeven)


def regime_shift_adaptation_latency(
    detected_regimes: Sequence[str],
    forced_regime: str = "bull",
) -> int | None:
    if not detected_regimes:
        return None
    for idx, regime in enumerate(detected_regimes):
        if str(regime).lower() != forced_regime.lower():
            return idx
    return None
