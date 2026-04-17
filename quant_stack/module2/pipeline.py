from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from quant_stack.module6.regime_detector import RegimeDetectorConfig, build_regime_feature_frame


@dataclass(frozen=True)
class Module2Config:
    momentum_windows: tuple[int, ...] = (3, 8, 21)
    realized_vol_windows: tuple[int, ...] = (5, 13, 34)
    volume_windows: tuple[int, ...] = (5, 20)
    illiquidity_windows: tuple[int, ...] = (5, 20)
    benchmark_symbol: str = "SPY"
    use_regime_detector: bool = True
    regime: RegimeDetectorConfig = field(default_factory=RegimeDetectorConfig)


def _group_sorted(frame: pd.DataFrame, column: str) -> pd.core.groupby.generic.SeriesGroupBy:
    return frame.groupby("instrument_id", sort=False)[column]


def _rolling_sum(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return _group_sorted(frame, column).transform(lambda series: series.rolling(window).sum())


def _rolling_mean(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return _group_sorted(frame, column).transform(lambda series: series.rolling(window).mean())


def _rolling_std(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return _group_sorted(frame, column).transform(lambda series: series.rolling(window).std())


def build_module2_features(stationary_df: pl.DataFrame | pd.DataFrame, config: Module2Config) -> pd.DataFrame:
    if isinstance(stationary_df, pl.DataFrame):
        frame = stationary_df.sort(["instrument_id", "timestamp"]).to_pandas()
    else:
        frame = stationary_df.sort_values(["instrument_id", "timestamp"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    epsilon = 1e-12

    if config.use_regime_detector:
        regime_features = build_regime_feature_frame(frame, config.regime)
        frame = frame.merge(
            regime_features,
            on=["instrument_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )

    frame["m2_log_ret"] = frame["raw_log_return"].astype(float)
    frame["m2_log_hl"] = np.log(frame["high"].clip(lower=epsilon) / frame["low"].clip(lower=epsilon))
    frame["m2_log_co"] = np.log(frame["close"].clip(lower=epsilon) / frame["open"].clip(lower=epsilon))
    frame["m2_dollar_volume"] = frame["close"] * frame["volume"]
    frame["m2_abs_return"] = np.abs(frame["m2_log_ret"])
    frame["m2_signed_pressure"] = frame["feat_body_pct"] if "feat_body_pct" in frame else (
        (frame["close"] - frame["open"]) / frame["open"].clip(lower=epsilon)
    )
    frame["m2_amihud_bar"] = frame["m2_abs_return"] / frame["m2_dollar_volume"].clip(lower=epsilon)

    for window in config.momentum_windows:
        frame[f"m2_momentum_{window}"] = _rolling_sum(frame, "m2_log_ret", window)

    benchmark_mask = frame["symbol"].astype(str).str.upper() == config.benchmark_symbol.upper()
    benchmark_frame = frame.loc[benchmark_mask, ["timestamp"] + [f"m2_momentum_{w}" for w in config.momentum_windows]]
    benchmark_frame = benchmark_frame.rename(
        columns={f"m2_momentum_{w}": f"benchmark_momentum_{w}" for w in config.momentum_windows}
    )
    frame = frame.merge(benchmark_frame, on="timestamp", how="left")
    for window in config.momentum_windows:
        benchmark_column = f"benchmark_momentum_{window}"
        frame[f"m2_rel_strength_{window}"] = frame[f"m2_momentum_{window}"] - frame[benchmark_column].fillna(0.0)

    parkinson_variance = (frame["m2_log_hl"] ** 2) / (4.0 * np.log(2.0))
    gk_variance = 0.5 * (frame["m2_log_hl"] ** 2) - (2.0 * np.log(2.0) - 1.0) * (frame["m2_log_co"] ** 2)
    frame["m2_parkinson_var_bar"] = parkinson_variance.clip(lower=0.0)
    frame["m2_gk_var_bar"] = gk_variance.clip(lower=0.0)

    for window in config.realized_vol_windows:
        frame[f"m2_realized_vol_{window}"] = _rolling_std(frame, "m2_log_ret", window) * np.sqrt(window)
        frame[f"m2_parkinson_vol_{window}"] = np.sqrt(_rolling_mean(frame, "m2_parkinson_var_bar", window))
        frame[f"m2_gk_vol_{window}"] = np.sqrt(_rolling_mean(frame, "m2_gk_var_bar", window))
        net_move = np.abs(_rolling_sum(frame, "m2_log_ret", window))
        total_move = _group_sorted(frame, "m2_abs_return").transform(lambda series: series.rolling(window).sum())
        frame[f"m2_efficiency_ratio_{window}"] = net_move / total_move.clip(lower=epsilon)

    for window in config.volume_windows:
        volume_mean = _rolling_mean(frame, "volume", window)
        volume_std = _rolling_std(frame, "volume", window)
        dollar_mean = _rolling_mean(frame, "m2_dollar_volume", window)
        frame[f"m2_relative_volume_{window}"] = frame["volume"] / volume_mean.clip(lower=epsilon)
        frame[f"m2_volume_z_{window}"] = (frame["volume"] - volume_mean) / volume_std.clip(lower=epsilon)
        frame[f"m2_relative_dollar_volume_{window}"] = frame["m2_dollar_volume"] / dollar_mean.clip(lower=epsilon)

    for window in config.illiquidity_windows:
        frame[f"m2_amihud_{window}"] = _rolling_mean(frame, "m2_amihud_bar", window)

    frame["m2_trade_intensity"] = frame["input_bar_count"].astype(float) / frame["bar_minutes"].clip(lower=1.0)

    cross_sectional_columns = [
        *(f"m2_momentum_{window}" for window in config.momentum_windows),
        *(f"m2_rel_strength_{window}" for window in config.momentum_windows),
        *(f"m2_realized_vol_{window}" for window in config.realized_vol_windows),
        *(f"m2_relative_volume_{window}" for window in config.volume_windows),
        "m2_trade_intensity",
    ]
    for column in cross_sectional_columns:
        frame[f"{column}_rank"] = frame.groupby("timestamp")[column].rank(pct=True, method="average")

    feature_columns = [
        "instrument_id",
        "timestamp",
        *[column for column in frame.columns if column.startswith("m2_")],
        *[column for column in frame.columns if column.startswith("feat_regime_")],
    ]
    result = frame.loc[:, feature_columns].copy()
    result = result.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    return result
