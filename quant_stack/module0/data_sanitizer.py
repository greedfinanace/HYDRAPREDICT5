from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class DataSanitizerConfig:
    bar_every: str = "1m"
    max_one_minute_move: float = 0.10
    min_volume_allowed: float = 0.0


def detect_timestamp_gaps(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            {
                "instrument_id": [],
                "missing_from": [],
                "missing_to": [],
                "missing_minutes": [],
            }
        )
    rows: list[dict[str, Any]] = []
    for part in frame.sort(["instrument_id", "timestamp"]).partition_by("instrument_id", maintain_order=True):
        timestamps = part["timestamp"].to_list()
        for prev, cur in zip(timestamps[:-1], timestamps[1:], strict=True):
            diff_minutes = int((cur - prev).total_seconds() // 60)
            if diff_minutes > 1:
                rows.append(
                    {
                        "instrument_id": part["instrument_id"][0],
                        "missing_from": prev,
                        "missing_to": cur,
                        "missing_minutes": diff_minutes - 1,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame({"instrument_id": [], "missing_from": [], "missing_to": [], "missing_minutes": []})


def upsample_and_fill(frame: pl.DataFrame, config: DataSanitizerConfig = DataSanitizerConfig()) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    cleaned_parts: list[pl.DataFrame] = []
    for part in frame.sort(["instrument_id", "timestamp"]).partition_by("instrument_id", maintain_order=True):
        base = part.sort("timestamp")
        upsampled = base.upsample(time_column="timestamp", every=config.bar_every)
        if "is_imputed" in upsampled.columns:
            imputed_marker = (
                pl.when(pl.col("open").is_null())
                .then(pl.lit(True))
                .otherwise(pl.col("is_imputed").cast(pl.Boolean).fill_null(False))
                .alias("is_imputed")
            )
        else:
            imputed_marker = pl.col("open").is_null().alias("is_imputed")
        upsampled = upsampled.with_columns(
            [
                imputed_marker,
                pl.col("instrument_id").fill_null(strategy="forward"),
                pl.col("symbol").fill_null(strategy="forward"),
                pl.col("venue").fill_null(strategy="forward"),
                pl.col("asset_group").fill_null(strategy="forward"),
                pl.col("source_bucket").fill_null(strategy="forward"),
                pl.col("source_bar_minutes").fill_null(strategy="forward"),
                pl.col("open").fill_null(strategy="forward"),
                pl.col("high").fill_null(strategy="forward"),
                pl.col("low").fill_null(strategy="forward"),
                pl.col("close").fill_null(strategy="forward"),
                pl.col("open_interest").fill_null(strategy="forward"),
                pl.col("volume").fill_null(0.0),
            ]
        )
        cleaned_parts.append(upsampled)
    return pl.concat(cleaned_parts).sort(["instrument_id", "timestamp"])


def spike_filter(frame: pl.DataFrame, config: DataSanitizerConfig = DataSanitizerConfig()) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.sort(["instrument_id", "timestamp"])
        .with_columns(
            (
                (pl.col("close") / pl.col("close").shift(1).over("instrument_id")) - 1.0
            ).alias("_ret_1m")
        )
        .filter(pl.col("_ret_1m").is_null() | (pl.col("_ret_1m").abs() <= config.max_one_minute_move))
        .drop("_ret_1m")
    )


def clean_market_data(frame: pl.DataFrame, config: DataSanitizerConfig = DataSanitizerConfig()) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    filled = upsample_and_fill(frame, config)
    filtered = spike_filter(filled, config)
    return filtered.filter(pl.col("volume") >= config.min_volume_allowed).sort(["instrument_id", "timestamp"])


def check_data_integrity(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "ok": False,
            "reason": "empty_frame",
            "gap_count": 0,
            "zero_volume_bars": 0,
            "raw_zero_volume_bars": 0,
            "imputed_zero_volume_bars": 0,
        }
    gaps = detect_timestamp_gaps(frame)
    imputed_expr = (
        pl.col("is_imputed").cast(pl.Boolean).fill_null(False)
        if "is_imputed" in frame.columns
        else pl.lit(False)
    )
    raw_zero_volume = int(frame.filter((pl.col("volume") <= 0.0) & (~imputed_expr)).height)
    imputed_zero_volume = int(frame.filter((pl.col("volume") <= 0.0) & imputed_expr).height)
    total_zero_volume = raw_zero_volume + imputed_zero_volume
    has_gaps = not gaps.is_empty()
    ok = (not has_gaps) and raw_zero_volume == 0
    reason = "ok"
    if has_gaps:
        reason = "timestamp_gaps_detected"
    elif raw_zero_volume > 0:
        reason = "raw_zero_volume_detected"
    return {
        "ok": ok,
        "reason": reason,
        "gap_count": int(gaps.height),
        "zero_volume_bars": int(total_zero_volume),
        "raw_zero_volume_bars": int(raw_zero_volume),
        "imputed_zero_volume_bars": int(imputed_zero_volume),
    }
