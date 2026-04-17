from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True)
class Module5Config:
    benchmark_symbol: str = "SPY"
    execution_delay_bars: int = 1
    fee_bps: float = 20.0
    slippage_vol_multiplier: float = 0.1
    meta_prob_win_threshold: float = 0.55
    output_root: Path = Path("artifacts/module5")
    save_plots: bool = True


@dataclass(frozen=True)
class BacktestArtifacts:
    config: Module5Config
    tearsheet: dict[str, Any]
    portfolio_frame: pl.DataFrame
    instrument_frame: pl.DataFrame
    figure_paths: dict[str, Path]


def _load_frame(frame_or_path: pl.DataFrame | pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(frame_or_path, pl.DataFrame):
        return frame_or_path.to_pandas()
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path.copy()
    path = Path(frame_or_path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input type or file extension: {frame_or_path}")


def _periods_per_year(index: pd.Series) -> float:
    timestamps = pd.to_datetime(index, utc=True).sort_values()
    if timestamps.shape[0] < 2:
        return 1.0
    span_seconds = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    if span_seconds <= 0.0:
        return float(timestamps.shape[0])
    years = span_seconds / (365.25 * 24.0 * 60.0 * 60.0)
    return max(float(timestamps.shape[0] / years), 1.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _strategy_metrics(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    clean = returns.fillna(0.0).astype(float)
    equity = (1.0 + clean).cumprod()
    drawdown = _drawdown(equity)
    mean_return = float(clean.mean())
    std_return = float(clean.std(ddof=1))
    downside = clean.where(clean < 0.0, 0.0)
    downside_std = float(downside.std(ddof=1))
    sharpe = 0.0 if std_return == 0.0 or not np.isfinite(std_return) else mean_return / std_return * np.sqrt(periods_per_year)
    sortino = 0.0 if downside_std == 0.0 or not np.isfinite(downside_std) else mean_return / downside_std * np.sqrt(periods_per_year)
    mdd = float(drawdown.min()) if drawdown.shape[0] else 0.0
    n_periods = max(clean.shape[0], 1)
    annual_return = float((equity.iloc[-1] ** (periods_per_year / n_periods)) - 1.0) if equity.shape[0] else 0.0
    calmar = 0.0 if mdd == 0.0 else annual_return / abs(mdd)
    gross_gain = float(clean[clean > 0.0].sum())
    gross_loss = float(abs(clean[clean < 0.0].sum()))
    profit_factor = float("inf") if gross_loss == 0.0 and gross_gain > 0.0 else (gross_gain / gross_loss if gross_loss > 0.0 else 0.0)
    return {
        "annual_return": annual_return,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "calmar": float(calmar),
        "profit_factor": float(profit_factor),
    }


def _regime_attribution(
    instrument_frame: pd.DataFrame,
    periods_per_year: float,
) -> dict[str, dict[str, float]]:
    if "regime_label" not in instrument_frame.columns:
        return {}

    attribution: dict[str, dict[str, float]] = {}
    for regime_label, regime_frame in instrument_frame.groupby("regime_label", dropna=False, sort=True):
        label = str(regime_label) if pd.notna(regime_label) else "unknown"
        metrics = _strategy_metrics(regime_frame["meta_strategy_return_net"], periods_per_year)
        metrics["alpha_raw_mean_return"] = float(regime_frame["alpha_raw_return_net"].mean())
        metrics["meta_mean_return"] = float(regime_frame["meta_strategy_return_net"].mean())
        metrics["observations"] = float(regime_frame.shape[0])
        metrics["avg_regime_penalty"] = float(
            regime_frame["regime_penalty"].mean() if "regime_penalty" in regime_frame.columns else 1.0
        )
        attribution[label] = metrics
    return attribution


def _plot_equity_curves(portfolio: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio["timestamp"], portfolio["passive_equity"], label="Passive", linewidth=2.0)
    plt.plot(portfolio["timestamp"], portfolio["alpha_raw_equity"], label="Alpha-Raw", linewidth=1.5)
    plt.plot(portfolio["timestamp"], portfolio["meta_strategy_equity"], label="Meta-Strategy", linewidth=1.5)
    plt.title("Comparative Equity Curves")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_underwater_chart(portfolio: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(portfolio["timestamp"], portfolio["passive_drawdown"], label="Passive", linewidth=1.5)
    plt.plot(portfolio["timestamp"], portfolio["alpha_raw_drawdown"], label="Alpha-Raw", linewidth=1.5)
    plt.plot(portfolio["timestamp"], portfolio["meta_strategy_drawdown"], label="Meta-Strategy", linewidth=1.5)
    plt.fill_between(portfolio["timestamp"], portfolio["meta_strategy_drawdown"], 0.0, alpha=0.15)
    plt.title("Underwater Drawdown Chart")
    plt.xlabel("Timestamp")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _build_signal_frame(
    bars: pd.DataFrame,
    positions: pd.DataFrame,
    config: Module5Config,
) -> pd.DataFrame:
    positions = positions.copy()
    if "pred_prob_win" not in positions.columns:
        positions["pred_prob_win"] = 1.0
    if "label" not in positions.columns:
        positions["label"] = np.nan
    signal_columns = [
        "instrument_id",
        "event_time",
        "pred_side",
        "bet_size",
        "pred_prob_win",
        "label",
        "regime_state",
        "regime_label",
        "regime_penalty",
    ]
    available_columns = [column for column in signal_columns if column in positions.columns]
    signals = positions.loc[:, available_columns].copy()
    signals["event_time"] = pd.to_datetime(signals["event_time"], utc=True)
    signals = signals.sort_values(["instrument_id", "event_time"]).drop_duplicates(
        subset=["instrument_id", "event_time"], keep="last"
    )

    merged = bars.merge(
        signals,
        left_on=["instrument_id", "timestamp"],
        right_on=["instrument_id", "event_time"],
        how="left",
    )
    merged["alpha_event_target"] = merged["pred_side"].astype(float)
    merged["meta_event_target"] = np.where(
        merged["pred_prob_win"].fillna(0.0) >= config.meta_prob_win_threshold,
        merged["pred_side"].fillna(0.0).astype(float) * merged["bet_size"].fillna(0.0).astype(float),
        0.0,
    )

    grouped = merged.groupby("instrument_id", sort=False)
    merged["alpha_position"] = grouped["alpha_event_target"].transform(
        lambda series: series.shift(config.execution_delay_bars).ffill().fillna(0.0)
    )
    merged["meta_position"] = grouped["meta_event_target"].transform(
        lambda series: series.shift(config.execution_delay_bars).ffill().fillna(0.0)
    )
    if "regime_state" in merged.columns:
        merged["regime_state"] = grouped["regime_state"].transform(lambda series: series.ffill().fillna(-1)).astype(int)
    if "regime_label" in merged.columns:
        # Cast to pandas string dtype before filling so fillna does not trigger
        # object downcast warnings on newer pandas versions.
        merged["regime_label"] = grouped["regime_label"].transform(
            lambda series: series.astype("string").ffill().fillna("unknown")
        )
    if "regime_penalty" in merged.columns:
        merged["regime_penalty"] = grouped["regime_penalty"].transform(lambda series: series.ffill().fillna(1.0))

    merged["open_to_open_return"] = grouped["open"].transform(lambda series: series.shift(-1) / series - 1.0).fillna(0.0)
    merged["slippage_rate"] = merged["sigma_t"].fillna(0.0).clip(lower=0.0) * config.slippage_vol_multiplier
    merged["transaction_cost_rate"] = (config.fee_bps * 1e-4) + merged["slippage_rate"]
    merged["alpha_turnover"] = grouped["alpha_position"].transform(lambda series: series.diff().fillna(series))
    merged["meta_turnover"] = grouped["meta_position"].transform(lambda series: series.diff().fillna(series))

    merged["alpha_raw_return_gross"] = merged["alpha_position"] * merged["open_to_open_return"]
    merged["meta_strategy_return_gross"] = merged["meta_position"] * merged["open_to_open_return"]
    merged["alpha_raw_cost"] = merged["alpha_turnover"].abs() * merged["transaction_cost_rate"]
    merged["meta_strategy_cost"] = merged["meta_turnover"].abs() * merged["transaction_cost_rate"]
    merged["alpha_raw_return_net"] = merged["alpha_raw_return_gross"] - merged["alpha_raw_cost"]
    merged["meta_strategy_return_net"] = merged["meta_strategy_return_gross"] - merged["meta_strategy_cost"]
    return merged


def run_backtest(
    position_sizing_artifact: pl.DataFrame | pd.DataFrame | str | Path,
    bar_data_artifact: pl.DataFrame | pd.DataFrame | str | Path,
    config: Module5Config = Module5Config(),
) -> BacktestArtifacts:
    positions = _load_frame(position_sizing_artifact)
    bars = _load_frame(bar_data_artifact)
    positions["event_time"] = pd.to_datetime(positions["event_time"], utc=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)

    required_bar_columns = {"instrument_id", "symbol", "timestamp", "open", "sigma_t"}
    missing_bar = required_bar_columns - set(bars.columns)
    if missing_bar:
        raise ValueError(f"Bar data is missing required columns: {sorted(missing_bar)}")

    required_position_columns = {"instrument_id", "event_time", "pred_side", "bet_size"}
    missing_signal = required_position_columns - set(positions.columns)
    if missing_signal:
        raise ValueError(f"Position sizing artifact is missing required columns: {sorted(missing_signal)}")

    bars = bars.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    instrument_frame = _build_signal_frame(bars, positions, config)
    n_instruments = max(int(instrument_frame["instrument_id"].nunique()), 1)

    benchmark_rows = instrument_frame.loc[
        instrument_frame["symbol"].astype(str).str.upper() == config.benchmark_symbol.upper()
    ].copy()
    if benchmark_rows.empty:
        raise ValueError(f"Benchmark symbol {config.benchmark_symbol} was not found in the bar data.")

    portfolio = (
        instrument_frame.groupby("timestamp", sort=True)
        .agg(
            alpha_raw_return=("alpha_raw_return_net", lambda series: float(series.sum()) / n_instruments),
            meta_strategy_return=("meta_strategy_return_net", lambda series: float(series.sum()) / n_instruments),
            alpha_raw_gross_return=("alpha_raw_return_gross", lambda series: float(series.sum()) / n_instruments),
            meta_strategy_gross_return=("meta_strategy_return_gross", lambda series: float(series.sum()) / n_instruments),
            alpha_raw_cost=("alpha_raw_cost", lambda series: float(series.sum()) / n_instruments),
            meta_strategy_cost=("meta_strategy_cost", lambda series: float(series.sum()) / n_instruments),
            alpha_raw_exposure=("alpha_position", lambda series: float(np.mean(np.abs(series)))),
            meta_strategy_exposure=("meta_position", lambda series: float(np.mean(np.abs(series)))),
        )
        .reset_index()
    )
    benchmark_series = (
        benchmark_rows.loc[:, ["timestamp", "open_to_open_return"]]
        .rename(columns={"open_to_open_return": "passive_return"})
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
    )
    portfolio = portfolio.merge(benchmark_series, on="timestamp", how="left")
    portfolio["passive_return"] = portfolio["passive_return"].fillna(0.0)

    for prefix in ("passive", "alpha_raw", "meta_strategy"):
        return_col = f"{prefix}_return"
        equity_col = f"{prefix}_equity"
        portfolio[equity_col] = (1.0 + portfolio[return_col]).cumprod()
        portfolio[f"{prefix}_drawdown"] = _drawdown(portfolio[equity_col])

    periods_per_year = _periods_per_year(portfolio["timestamp"])
    passive_metrics = _strategy_metrics(portfolio["passive_return"], periods_per_year)
    alpha_metrics = _strategy_metrics(portfolio["alpha_raw_return"], periods_per_year)
    meta_metrics = _strategy_metrics(portfolio["meta_strategy_return"], periods_per_year)

    alpha_mdd = alpha_metrics["max_drawdown"]
    meta_mdd = meta_metrics["max_drawdown"]
    if alpha_mdd == 0.0:
        drawdown_reduction_pct = 0.0
    else:
        drawdown_reduction_pct = 100.0 * (abs(alpha_mdd) - abs(meta_mdd)) / abs(alpha_mdd)

    tearsheet = {
        "periods_per_year": periods_per_year,
        "execution": {
            "execution_delay_bars": config.execution_delay_bars,
            "fee_bps": config.fee_bps,
            "slippage_vol_multiplier": config.slippage_vol_multiplier,
            "meta_prob_win_threshold": config.meta_prob_win_threshold,
            "instrument_count": n_instruments,
        },
        "strategies": {
            "passive": passive_metrics,
            "alpha_raw": alpha_metrics,
            "meta_strategy": meta_metrics,
        },
        "alpha_meta_attribution": {
            "drawdown_reduction_pct": drawdown_reduction_pct,
            "alpha_raw_mdd": alpha_mdd,
            "meta_strategy_mdd": meta_mdd,
        },
        "regime_attribution": _regime_attribution(instrument_frame, periods_per_year),
    }

    figure_paths: dict[str, Path] = {}
    if config.save_plots:
        output_root = Path(config.output_root)
        equity_path = _plot_equity_curves(portfolio, output_root / "equity_curves.png")
        underwater_path = _plot_underwater_chart(portfolio, output_root / "underwater_chart.png")
        figure_paths = {
            "equity_curves": equity_path,
            "underwater_chart": underwater_path,
        }

    return BacktestArtifacts(
        config=config,
        tearsheet=tearsheet,
        portfolio_frame=pl.from_pandas(portfolio),
        instrument_frame=pl.from_pandas(instrument_frame),
        figure_paths=figure_paths,
    )
