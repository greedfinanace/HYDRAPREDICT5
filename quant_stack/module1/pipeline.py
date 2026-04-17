from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import adfuller


DEFAULT_SOURCE_ROOT = Path("data/data/hourly/us")
DEFAULT_OUTPUT_ROOT = Path("artifacts/module1")
SUPPORTED_SOURCE_FORMATS = {"auto", "txt", "parquet"}


def _default_d_grid() -> tuple[float, ...]:
    values: list[float] = []
    current = 0.10
    while current <= 0.9000001:
        values.append(round(current, 2))
        current += 0.05
    return tuple(values)


@dataclass(frozen=True)
class Module1Config:
    source_root: Path = DEFAULT_SOURCE_ROOT
    source_format: str = "auto"
    symbols: Sequence[str] | None = None
    start: str | datetime | pd.Timestamp | None = None
    end: str | datetime | pd.Timestamp | None = None
    bar_minutes: int = 60
    sampling_mode: str = "time"
    volume_bar_threshold: float | None = None
    volume_threshold_window: int = 1000
    volume_threshold_multiplier: float = 50.0
    source_timezone: str = "UTC"
    session_timezone: str = "America/New_York"
    fracdiff_d_grid: tuple[float, ...] = field(default_factory=_default_d_grid)
    adf_alpha: float = 0.05
    weight_eps: float = 1e-5
    adaptive_weight_eps: bool = True
    min_weight_eps: float = 1e-7
    cusum_sigma_mult: float = 1.0
    vol_lookback_bars: int = 63
    pt_mult: float = 1.5
    sl_mult: float = 1.5
    vertical_barrier_bars: int = 21
    vertical_barrier_days: float | None = 3.0
    vertical_barrier_session_window: int = 20
    output_root: Path = DEFAULT_OUTPUT_ROOT

    def normalized_symbols(self) -> tuple[str, ...] | None:
        if self.symbols is None:
            return None
        return tuple(sorted({str(symbol).upper() for symbol in self.symbols}))

    def normalized_source_format(self) -> str:
        return str(self.source_format).strip().lower()


@dataclass(frozen=True)
class Module1Artifacts:
    config: Module1Config
    bars: pl.DataFrame
    stationary: pl.DataFrame
    fracdiff_params: pl.DataFrame
    events: pl.DataFrame
    labels: pl.DataFrame
    bars_path: Path
    stationary_path: Path
    fracdiff_params_path: Path
    events_path: Path
    labels_path: Path


@dataclass(frozen=True)
class _FracDiffCandidate:
    d: float
    series: np.ndarray
    pvalue: float
    retained_correlation: float
    effective_window: int
    valid_start_index: int
    passed_adf: bool


def _coerce_timestamp_bound(
    value: str | datetime | pd.Timestamp | None,
    timezone_name: str,
) -> datetime | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone_name)
    else:
        ts = ts.tz_convert(timezone_name)
    return ts.to_pydatetime()


def _infer_source_format(source_root: Path, config: Module1Config) -> str:
    source_format = config.normalized_source_format()
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(
            f"Unsupported source_format: {config.source_format}. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_FORMATS)}"
        )

    if source_format != "auto":
        return source_format

    if source_root.is_file():
        suffix = source_root.suffix.lower()
        if suffix == ".parquet":
            return "parquet"
        if suffix == ".txt":
            return "txt"
        raise ValueError(f"Unsupported source file extension: {source_root.suffix}")

    has_parquet = _has_source_files(source_root, ".parquet")
    if has_parquet:
        return "parquet"

    has_text = _has_source_files(source_root, ".txt")
    if has_text:
        return "txt"

    raise FileNotFoundError(
        f"No supported source files (*.parquet or *.txt) were found under {source_root}"
    )


def _iter_candidate_files(source_root: Path, source_format: str) -> list[Path]:
    if source_format == "parquet":
        patterns = ("*.parquet", "*/*/*/*.parquet")
    else:
        patterns = ("*.txt", "*/*.txt", "*/*/*.txt", "*/*/*/*.txt")

    discovered: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(source_root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            discovered.append(path)

    if discovered:
        return discovered

    # Fallback for uncommon legacy layouts.
    return [path for path in sorted(source_root.rglob(f"*.{source_format}")) if path.is_file()]


def _recover_parquet_candidates_for_symbols(
    source_root: Path,
    missing_symbols: set[str],
    existing: set[Path],
) -> list[Path]:
    if not missing_symbols:
        return []

    recovered: list[Path] = []
    unresolved = set(missing_symbols)
    for path in sorted(source_root.rglob("*.parquet")):
        if not path.is_file() or path in existing:
            continue
        parent_symbol = path.parent.name.upper()
        stem_symbol = path.stem.upper()
        if parent_symbol in unresolved or stem_symbol in unresolved:
            recovered.append(path)
            unresolved.discard(parent_symbol)
            unresolved.discard(stem_symbol)
            if not unresolved:
                break
    return recovered


def _has_source_files(source_root: Path, suffix: str) -> bool:
    suffix = suffix.lower()
    if suffix == ".parquet":
        patterns = ("*.parquet", "*/*/*/*.parquet")
    elif suffix == ".txt":
        patterns = ("*.txt", "*/*.txt", "*/*/*.txt", "*/*/*/*.txt")
    else:
        patterns = (f"*{suffix}",)

    for pattern in patterns:
        if next(source_root.glob(pattern), None) is not None:
            return True
    return next(source_root.rglob(f"*{suffix}"), None) is not None


def _discover_symbol_files(config: Module1Config) -> list[Path]:
    source_root = Path(config.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    source_format = _infer_source_format(source_root, config)
    if source_root.is_file():
        return [source_root]

    desired_symbols = set(config.normalized_symbols() or [])
    discovered: list[Path] = []
    found_symbols: set[str] = set()

    candidates = _iter_candidate_files(source_root, source_format)
    if source_format == "parquet":
        parquet_candidates = candidates
    else:
        parquet_candidates = []

    for path in candidates:
        symbol = path.name.split(".")[0].upper()
        if source_format == "txt" and desired_symbols and symbol not in desired_symbols:
            continue
        discovered.append(path)
        found_symbols.add(symbol)

    if source_format == "parquet":
        if desired_symbols:
            prefiltered = [
                path
                for path in parquet_candidates
                if path.parent.name.upper() in desired_symbols
            ]
            discovered = prefiltered
            covered_symbols = {path.parent.name.upper() for path in discovered}
            missing_symbols = set(desired_symbols - covered_symbols)
            if missing_symbols:
                recovered = _recover_parquet_candidates_for_symbols(
                    source_root=source_root,
                    missing_symbols=missing_symbols,
                    existing=set(parquet_candidates),
                )
                discovered.extend(recovered)
            if not discovered:
                discovered = parquet_candidates
        else:
            discovered = parquet_candidates

    if source_format == "txt" and desired_symbols:
        missing = sorted(desired_symbols - found_symbols)
        if missing:
            raise FileNotFoundError(
                "Requested symbols were not found under "
                f"{source_root}: {', '.join(missing)}"
            )

    if not discovered:
        raise FileNotFoundError(
            f"No {source_format} source files found under {source_root}"
        )
    return discovered


def _parse_path_metadata(path: Path, source_root: Path) -> tuple[str, str, str | None]:
    if source_root.is_file():
        source_root = source_root.parent
    rel_parts = path.relative_to(source_root).parts
    top_level = rel_parts[0] if rel_parts else "unknown unknown"
    tokens = top_level.split()
    venue = tokens[0].lower() if tokens else "unknown"
    asset_group = "_".join(tokens[1:]).lower() if len(tokens) > 1 else "unknown"
    bucket = rel_parts[1] if len(rel_parts) > 2 else None
    return venue, asset_group, bucket


def _normalize_symbol_expression(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8)
        .str.replace(r"\.US$", "", literal=False)
        .str.to_uppercase()
    )


def _first_present(schema_names: set[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in schema_names:
            return candidate
    return None


def _scan_text_file(path: Path, config: Module1Config) -> pl.LazyFrame:
    venue, asset_group, bucket = _parse_path_metadata(path, Path(config.source_root))
    source_bucket = "" if bucket is None else str(bucket)

    lf = pl.scan_csv(
        str(path),
        has_header=True,
        schema_overrides={
            "<TICKER>": pl.Utf8,
            "<PER>": pl.Int64,
            "<DATE>": pl.Int64,
            "<TIME>": pl.Int64,
            "<OPEN>": pl.Float64,
            "<HIGH>": pl.Float64,
            "<LOW>": pl.Float64,
            "<CLOSE>": pl.Float64,
            "<VOL>": pl.Float64,
            "<OPENINT>": pl.Int64,
        },
    )

    timestamp_expr = (
        pl.concat_str(
            [
                pl.col("<DATE>").cast(pl.Utf8),
                pl.col("<TIME>").cast(pl.Utf8).str.zfill(6),
            ]
        )
        .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S", strict=True)
        .dt.replace_time_zone(config.source_timezone)
    )

    lf = (
        lf.filter(pl.col("<PER>") == config.bar_minutes)
        .with_columns(
            [
                timestamp_expr.alias("timestamp"),
                _normalize_symbol_expression("<TICKER>").alias("symbol"),
                pl.lit(venue).alias("venue"),
                pl.lit(asset_group).alias("asset_group"),
                pl.lit(source_bucket).cast(pl.Utf8).alias("source_bucket"),
            ]
        )
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("symbol"),
                        pl.lit("@"),
                        pl.col("venue"),
                        pl.lit("_"),
                        pl.col("asset_group"),
                    ]
                ).alias("instrument_id")
            ]
        )
        .rename(
            {
                "<PER>": "source_bar_minutes",
                "<OPEN>": "open",
                "<HIGH>": "high",
                "<LOW>": "low",
                "<CLOSE>": "close",
                "<VOL>": "volume",
                "<OPENINT>": "open_interest",
            }
        )
        .with_columns(
            [
                pl.col("source_bar_minutes").alias("bar_minutes"),
                pl.lit("time").alias("sampling_mode"),
                pl.lit(1).alias("input_bar_count"),
                pl.lit(False).alias("is_partial_bar"),
                pl.lit(None, dtype=pl.Float64).alias("volume_threshold_used"),
            ]
        )
        .select(
            [
                "instrument_id",
                "symbol",
                "venue",
                "asset_group",
                "source_bucket",
                "timestamp",
                "sampling_mode",
                "source_bar_minutes",
                "bar_minutes",
                "input_bar_count",
                "is_partial_bar",
                "volume_threshold_used",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )
    )

    start_bound = _coerce_timestamp_bound(config.start, config.source_timezone)
    end_bound = _coerce_timestamp_bound(config.end, config.source_timezone)
    if start_bound is not None:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start_bound))
    if end_bound is not None:
        lf = lf.filter(pl.col("timestamp") <= pl.lit(end_bound))
    return lf


def _timestamp_expr_from_parquet(
    lf: pl.LazyFrame,
    config: Module1Config,
) -> pl.Expr:
    schema = lf.collect_schema()
    schema_names = set(schema.names())

    if "<DATE>" in schema_names and "<TIME>" in schema_names:
        return (
            pl.concat_str(
                [
                    pl.col("<DATE>").cast(pl.Utf8),
                    pl.col("<TIME>").cast(pl.Utf8).str.zfill(6),
                ]
            )
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S", strict=True)
            .dt.replace_time_zone(config.source_timezone)
        )

    timestamp_column = _first_present(
        schema_names,
        ("timestamp", "Timestamp", "datetime", "Datetime", "date_time", "DateTime", "ts", "TS"),
    )
    if timestamp_column is None:
        raise ValueError("Parquet source must contain a timestamp/datetime column or <DATE>/<TIME> columns.")

    return (
        pl.col(timestamp_column)
        .cast(pl.Utf8)
        .str.to_datetime(strict=False, time_zone=config.source_timezone)
    )


def _scan_parquet_file(path: Path, config: Module1Config) -> pl.LazyFrame:
    venue, asset_group, bucket = _parse_path_metadata(path, Path(config.source_root))
    lf = pl.scan_parquet(str(path))
    schema = lf.collect_schema()
    schema_names = set(schema.names())

    symbol_column = _first_present(
        schema_names,
        ("symbol", "Symbol", "ticker", "Ticker", "<TICKER>"),
    )
    source_bar_minutes_column = _first_present(
        schema_names,
        ("source_bar_minutes", "bar_minutes", "BarMinutes", "period", "Period", "<PER>"),
    )
    open_column = _first_present(schema_names, ("open", "Open", "<OPEN>"))
    high_column = _first_present(schema_names, ("high", "High", "<HIGH>"))
    low_column = _first_present(schema_names, ("low", "Low", "<LOW>"))
    close_column = _first_present(schema_names, ("close", "Close", "<CLOSE>"))
    volume_column = _first_present(schema_names, ("volume", "Volume", "<VOL>"))
    open_interest_column = _first_present(
        schema_names,
        ("open_interest", "OpenInterest", "openinterest", "<OPENINT>"),
    )
    venue_column = _first_present(schema_names, ("venue", "Venue"))
    asset_group_column = _first_present(schema_names, ("asset_group", "AssetGroup"))
    source_bucket_column = _first_present(schema_names, ("source_bucket", "SourceBucket"))

    required_columns = {
        "open": open_column,
        "high": high_column,
        "low": low_column,
        "close": close_column,
        "volume": volume_column,
    }
    missing_required = [name for name, column in required_columns.items() if column is None]
    if missing_required:
        raise ValueError(
            f"Parquet source {path} is missing required OHLCV columns: {', '.join(missing_required)}"
        )

    if symbol_column is None:
        inferred_symbol = path.name.split(".")[0].upper()
        symbol_expr = pl.lit(inferred_symbol)
    else:
        symbol_expr = _normalize_symbol_expression(symbol_column)

    timestamp_expr = _timestamp_expr_from_parquet(lf, config)
    desired_symbols = list(config.normalized_symbols() or [])

    lf = (
        lf.with_columns(
            [
                timestamp_expr.alias("timestamp"),
                symbol_expr.alias("symbol"),
                (
                    pl.col(source_bar_minutes_column).cast(pl.Int64)
                    if source_bar_minutes_column is not None
                    else pl.lit(config.bar_minutes).cast(pl.Int64)
                ).alias("source_bar_minutes"),
                (
                    pl.col(venue_column).cast(pl.Utf8)
                    if venue_column is not None
                    else pl.lit(venue)
                ).alias("venue"),
                (
                    pl.col(asset_group_column).cast(pl.Utf8)
                    if asset_group_column is not None
                    else pl.lit(asset_group)
                ).alias("asset_group"),
                (
                    pl.col(source_bucket_column).cast(pl.Utf8)
                    if source_bucket_column is not None
                    else pl.lit("" if bucket is None else str(bucket)).cast(pl.Utf8)
                ).alias("source_bucket"),
            ]
        )
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("symbol"),
                        pl.lit("@"),
                        pl.col("venue"),
                        pl.lit("_"),
                        pl.col("asset_group"),
                    ]
                ).alias("instrument_id"),
                pl.col(open_column).cast(pl.Float64).alias("open"),
                pl.col(high_column).cast(pl.Float64).alias("high"),
                pl.col(low_column).cast(pl.Float64).alias("low"),
                pl.col(close_column).cast(pl.Float64).alias("close"),
                pl.col(volume_column).cast(pl.Float64).alias("volume"),
                (
                    pl.col(open_interest_column).fill_null(0).cast(pl.Int64)
                    if open_interest_column is not None
                    else pl.lit(0).cast(pl.Int64)
                ).alias("open_interest"),
            ]
        )
        .with_columns(
            [
                pl.col("source_bar_minutes").alias("bar_minutes"),
                pl.lit("time").alias("sampling_mode"),
                pl.lit(1).alias("input_bar_count"),
                pl.lit(False).alias("is_partial_bar"),
                pl.lit(None, dtype=pl.Float64).alias("volume_threshold_used"),
            ]
        )
    )

    if source_bar_minutes_column is not None:
        lf = lf.filter(pl.col("source_bar_minutes") == config.bar_minutes)
    if desired_symbols:
        lf = lf.filter(pl.col("symbol").is_in(desired_symbols))

    start_bound = _coerce_timestamp_bound(config.start, config.source_timezone)
    end_bound = _coerce_timestamp_bound(config.end, config.source_timezone)
    if start_bound is not None:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start_bound))
    if end_bound is not None:
        lf = lf.filter(pl.col("timestamp") <= pl.lit(end_bound))

    return lf.select(
        [
            "instrument_id",
            "symbol",
            "venue",
            "asset_group",
            "source_bucket",
            "timestamp",
            "sampling_mode",
            "source_bar_minutes",
            "bar_minutes",
            "input_bar_count",
            "is_partial_bar",
            "volume_threshold_used",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ]
    )


def _attach_bar_indices(bars: pl.DataFrame, config: Module1Config) -> pl.DataFrame:
    bars = bars.sort(["instrument_id", "timestamp"])
    bars = bars.with_columns(
        [
            pl.col("timestamp")
            .dt.convert_time_zone(config.session_timezone)
            .dt.date()
            .alias("session_date")
        ]
    )
    return bars.with_columns(
        [
            (pl.col("timestamp").cum_count().over("instrument_id") - 1).alias("bar_index"),
            (
                pl.col("timestamp").cum_count().over(["instrument_id", "session_date"]) - 1
            ).alias("session_bar_index"),
        ]
    )


def _load_source_bars(config: Module1Config) -> pl.DataFrame:
    files = _discover_symbol_files(config)
    bars = pl.concat(
        [
            _scan_parquet_file(path, config)
            if path.suffix.lower() == ".parquet"
            else _scan_text_file(path, config)
            for path in files
        ]
    ).collect()
    desired_symbols = set(config.normalized_symbols() or [])
    if desired_symbols:
        found_symbols = set(bars["symbol"].unique().to_list()) if not bars.is_empty() else set()
        missing_symbols = sorted(desired_symbols - found_symbols)
        if missing_symbols:
            raise FileNotFoundError(
                "Requested symbols were not found after scanning source files under "
                f"{config.source_root}: {', '.join(missing_symbols)}"
            )
    bars = bars.unique(subset=list(bars.columns), maintain_order=True)
    return _attach_bar_indices(bars, config).select(
        [
            "instrument_id",
            "symbol",
            "venue",
            "asset_group",
            "source_bucket",
            "timestamp",
            "sampling_mode",
            "session_date",
            "bar_index",
            "session_bar_index",
            "source_bar_minutes",
            "bar_minutes",
            "input_bar_count",
            "is_partial_bar",
            "volume_threshold_used",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ]
    )


def _rolling_volume_thresholds(volumes: np.ndarray, config: Module1Config) -> np.ndarray:
    if config.volume_bar_threshold is not None:
        threshold = np.full(volumes.shape[0], float(config.volume_bar_threshold), dtype=np.float64)
    else:
        rolling = (
            pd.Series(volumes, dtype="float64")
            .rolling(window=max(int(config.volume_threshold_window), 1), min_periods=1)
            .mean()
            .shift(1)
            .bfill()
        )
        threshold = rolling.to_numpy(dtype=np.float64) * float(config.volume_threshold_multiplier)

    valid = np.isfinite(threshold) & (threshold > 0.0)
    if not np.any(valid):
        fallback = float(np.nanmean(volumes[np.isfinite(volumes) & (volumes > 0.0)]))
        fallback = fallback if np.isfinite(fallback) and fallback > 0.0 else 1.0
        threshold = np.full(volumes.shape[0], fallback, dtype=np.float64)
    else:
        fallback = float(np.nanmean(threshold[valid]))
        threshold = np.where(valid, threshold, fallback)
    return np.clip(threshold, 1e-12, None)


def _resolve_weight_eps(bars: pl.DataFrame, config: Module1Config) -> float:
    if not config.adaptive_weight_eps or bars.is_empty():
        return float(config.weight_eps)

    source_minutes = bars["source_bar_minutes"].to_numpy()
    finite_minutes = source_minutes[np.isfinite(source_minutes)]
    base_minutes = float(np.nanmedian(finite_minutes)) if finite_minutes.size else float(config.bar_minutes)
    if not np.isfinite(base_minutes) or base_minutes <= 0.0 or base_minutes >= 60.0:
        return float(config.weight_eps)

    density_ratio = max(60.0 / base_minutes, 1.0)
    adjusted = float(config.weight_eps) / np.sqrt(density_ratio)
    return float(max(adjusted, config.min_weight_eps))


def _session_vertical_barriers(raw_frame: pl.DataFrame, config: Module1Config) -> dict[object, int]:
    session_dates = raw_frame["session_date"].unique(maintain_order=True).to_list()
    fixed_barrier = max(int(config.vertical_barrier_bars), 1)
    if config.sampling_mode != "volume" or config.vertical_barrier_days is None:
        return {session_date: fixed_barrier for session_date in session_dates}

    session_counts = (
        raw_frame.group_by("session_date", maintain_order=True)
        .agg(pl.len().alias("bars_in_session"))
        .sort("session_date")
    )
    bars_per_session = session_counts["bars_in_session"].to_numpy().astype(np.float64)
    if bars_per_session.size == 0:
        return {}

    lookback = max(int(config.vertical_barrier_session_window), 1)
    history = (
        pd.Series(bars_per_session, dtype="float64")
        .rolling(window=lookback, min_periods=1)
        .mean()
        .shift(1)
    )
    fallback = float(np.nanmedian(bars_per_session))
    history = history.fillna(fallback)
    resolved = np.clip(
        np.rint(history.to_numpy(dtype=np.float64) * float(config.vertical_barrier_days)),
        1,
        None,
    ).astype(np.int64)
    return dict(zip(session_counts["session_date"].to_list(), resolved.tolist(), strict=True))


def build_volume_bars(source_bars: pl.DataFrame, config: Module1Config) -> pl.DataFrame:
    volume_frames: list[pl.DataFrame] = []

    for frame in source_bars.partition_by("instrument_id", maintain_order=True):
        timestamps = frame["timestamp"].to_list()
        session_dates = frame["session_date"].to_list()
        opens = frame["open"].to_numpy()
        highs = frame["high"].to_numpy()
        lows = frame["low"].to_numpy()
        closes = frame["close"].to_numpy()
        volumes = frame["volume"].to_numpy()
        open_interest = frame["open_interest"].to_numpy()
        thresholds = _rolling_volume_thresholds(volumes, config)

        rows: list[dict[str, object]] = []
        current_row: dict[str, object] | None = None
        current_session_date = None

        for idx in range(frame.height):
            row_session = session_dates[idx]
            if current_row is not None and row_session != current_session_date:
                current_row["bar_minutes"] = int(current_row["input_bar_count"] * config.bar_minutes)
                current_row["is_partial_bar"] = True
                rows.append(current_row)
                current_row = None

            if current_row is None:
                current_session_date = row_session
                current_row = {
                    "instrument_id": frame["instrument_id"][0],
                    "symbol": frame["symbol"][0],
                    "venue": frame["venue"][0],
                    "asset_group": frame["asset_group"][0],
                    "source_bucket": frame["source_bucket"][0],
                    "timestamp": timestamps[idx],
                    "sampling_mode": "volume",
                    "session_date": row_session,
                    "source_bar_minutes": int(frame["source_bar_minutes"][idx]),
                    "bar_minutes": 0,
                    "input_bar_count": 0,
                    "is_partial_bar": False,
                    "volume_threshold_used": float(thresholds[idx]),
                    "open": float(opens[idx]),
                    "high": float(highs[idx]),
                    "low": float(lows[idx]),
                    "close": float(closes[idx]),
                    "volume": 0.0,
                    "open_interest": int(open_interest[idx]),
                }

            current_row["timestamp"] = timestamps[idx]
            current_row["high"] = max(float(current_row["high"]), float(highs[idx]))
            current_row["low"] = min(float(current_row["low"]), float(lows[idx]))
            current_row["close"] = float(closes[idx])
            current_row["volume"] = float(current_row["volume"]) + float(volumes[idx])
            current_row["open_interest"] = int(open_interest[idx])
            current_row["input_bar_count"] = int(current_row["input_bar_count"]) + 1

            if float(current_row["volume"]) >= float(current_row["volume_threshold_used"]):
                current_row["bar_minutes"] = int(current_row["input_bar_count"] * config.bar_minutes)
                rows.append(current_row)
                current_row = None

        if current_row is not None:
            current_row["bar_minutes"] = int(current_row["input_bar_count"] * config.bar_minutes)
            current_row["is_partial_bar"] = True
            rows.append(current_row)

        volume_frames.append(pl.DataFrame(rows))

    bars = pl.concat(volume_frames)
    bars = _attach_bar_indices(bars, config)
    return bars.select(
        [
            "instrument_id",
            "symbol",
            "venue",
            "asset_group",
            "source_bucket",
            "timestamp",
            "sampling_mode",
            "session_date",
            "bar_index",
            "session_bar_index",
            "source_bar_minutes",
            "bar_minutes",
            "input_bar_count",
            "is_partial_bar",
            "volume_threshold_used",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ]
    )


def load_bars(config: Module1Config) -> pl.DataFrame:
    source_bars = _load_source_bars(config)
    if config.sampling_mode == "time":
        return source_bars
    if config.sampling_mode == "volume":
        return build_volume_bars(source_bars, config)
    raise ValueError(f"Unsupported sampling_mode: {config.sampling_mode}")


def _fracdiff_weights(d: float, max_size: int, weight_eps: float) -> np.ndarray:
    weights = [1.0]
    for k in range(1, max_size):
        next_weight = -weights[-1] * (d - k + 1.0) / k
        if abs(next_weight) < weight_eps:
            break
        weights.append(next_weight)
    return np.asarray(weights[::-1], dtype=np.float64)


def _apply_fracdiff(log_close: np.ndarray, weights: np.ndarray) -> np.ndarray:
    width = len(weights)
    series = np.full(log_close.shape[0], np.nan, dtype=np.float64)
    if width == 0 or width > log_close.shape[0]:
        return series
    series[width - 1 :] = np.convolve(log_close, weights, mode="valid")
    return series


def _safe_correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    if mask.sum() < 3:
        return float("nan")
    lhs_valid = lhs[mask]
    rhs_valid = rhs[mask]
    if np.std(lhs_valid) == 0.0 or np.std(rhs_valid) == 0.0:
        return float("nan")
    return float(np.corrcoef(lhs_valid, rhs_valid)[0, 1])


def _adf_pvalue(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.shape[0] < 25:
        return float("inf")
    try:
        return float(adfuller(clean, regression="c", autolag="AIC")[1])
    except Exception:
        return float("inf")


def _evaluate_fracdiff_candidates(
    log_close: np.ndarray,
    config: Module1Config,
    weight_eps: float,
) -> tuple[_FracDiffCandidate, list[_FracDiffCandidate]]:
    candidates: list[_FracDiffCandidate] = []
    for d in sorted(config.fracdiff_d_grid):
        weights = _fracdiff_weights(d=d, max_size=log_close.shape[0], weight_eps=weight_eps)
        transformed = _apply_fracdiff(log_close, weights)
        pvalue = _adf_pvalue(transformed)
        corr = _safe_correlation(log_close, transformed)
        candidate = _FracDiffCandidate(
            d=d,
            series=transformed,
            pvalue=pvalue,
            retained_correlation=corr,
            effective_window=len(weights),
            valid_start_index=max(len(weights) - 1, 0),
            passed_adf=bool(np.isfinite(pvalue) and pvalue <= config.adf_alpha),
        )
        candidates.append(candidate)
        if candidate.passed_adf:
            return candidate, candidates

    fallback = min(
        candidates,
        key=lambda candidate: (
            candidate.pvalue if np.isfinite(candidate.pvalue) else float("inf"),
            -(candidate.retained_correlation if np.isfinite(candidate.retained_correlation) else -np.inf),
        ),
    )
    return fallback, candidates


def _ewm_volatility(raw_log_return: np.ndarray, span: int) -> np.ndarray:
    return (
        pd.Series(raw_log_return)
        .ewm(span=span, adjust=False, min_periods=span)
        .std(bias=False)
        .to_numpy(dtype=np.float64)
    )


def fit_fractional_diff(
    bars: pl.DataFrame,
    config: Module1Config,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    stationary_frames: list[pl.DataFrame] = []
    params_rows: list[dict[str, object]] = []
    resolved_weight_eps = _resolve_weight_eps(bars, config)

    for frame in bars.partition_by("instrument_id", maintain_order=True):
        close = frame.get_column("close").to_numpy()
        log_close = np.log(close)
        raw_log_return = np.concatenate(([np.nan], np.diff(log_close)))
        selected, all_candidates = _evaluate_fracdiff_candidates(
            log_close,
            config,
            weight_eps=resolved_weight_eps,
        )
        fracdiff_return = np.concatenate(([np.nan], np.diff(selected.series)))
        sigma_t = _ewm_volatility(raw_log_return, span=config.vol_lookback_bars)

        stationary_frames.append(
            frame.with_columns(
                [
                    pl.Series("log_close", log_close),
                    pl.Series("raw_log_return", raw_log_return),
                    pl.Series("fracdiff_log_close", selected.series),
                    pl.Series("fracdiff_return", fracdiff_return),
                    pl.Series("sigma_t", sigma_t),
                    pl.lit(selected.d).alias("selected_d"),
                    pl.lit(selected.effective_window).alias("fracdiff_window"),
                    pl.lit(selected.valid_start_index).alias("valid_start_index"),
                ]
            )
        )

        params_rows.append(
            {
                "instrument_id": frame["instrument_id"][0],
                "symbol": frame["symbol"][0],
                "venue": frame["venue"][0],
                "asset_group": frame["asset_group"][0],
                "selected_d": selected.d,
                "adf_pvalue": selected.pvalue,
                "retained_correlation": selected.retained_correlation,
                "effective_window": selected.effective_window,
                "valid_start_index": selected.valid_start_index,
                "adf_passed": selected.passed_adf,
                "n_observations": frame.height,
                "raw_log_return_std": float(np.nanstd(raw_log_return)),
                "weight_eps_used": float(resolved_weight_eps),
                "candidate_grid": ",".join(f"{candidate.d:.2f}" for candidate in all_candidates),
            }
        )

    stationary = pl.concat(stationary_frames).sort(["instrument_id", "timestamp"])
    params = pl.DataFrame(params_rows).sort(["instrument_id"])
    return stationary, params


def sample_events(stationary_df: pl.DataFrame, config: Module1Config) -> pl.DataFrame:
    event_frames: list[pl.DataFrame] = []

    for frame in stationary_df.partition_by("instrument_id", maintain_order=True):
        fd_returns = frame.get_column("fracdiff_return").to_numpy()
        sigmas = frame.get_column("sigma_t").to_numpy()
        timestamps = frame.get_column("timestamp").to_list()
        bar_indices = frame.get_column("bar_index").to_numpy()

        positive_cumsum = 0.0
        negative_cumsum = 0.0
        events: list[dict[str, object]] = []

        for idx, (fd_ret, sigma_t) in enumerate(zip(fd_returns, sigmas, strict=True)):
            if not np.isfinite(fd_ret) or not np.isfinite(sigma_t) or sigma_t <= 0.0:
                continue

            threshold = config.cusum_sigma_mult * sigma_t
            positive_cumsum = max(0.0, positive_cumsum + fd_ret)
            negative_cumsum = min(0.0, negative_cumsum + fd_ret)

            if positive_cumsum > threshold or negative_cumsum < -threshold:
                events.append(
                    {
                        "instrument_id": frame["instrument_id"][0],
                        "symbol": frame["symbol"][0],
                        "venue": frame["venue"][0],
                        "asset_group": frame["asset_group"][0],
                        "event_time": timestamps[idx],
                        "event_bar_index": int(bar_indices[idx]),
                        "cusum_side": 1 if positive_cumsum > threshold else -1,
                        "trgt": float(sigma_t),
                        "fd_return": float(fd_ret),
                    }
                )
                positive_cumsum = 0.0
                negative_cumsum = 0.0

        if not events:
            continue

        event_frame = pl.DataFrame(events).sort("event_time")
        event_frame = event_frame.with_columns(
            (pl.col("event_time").cum_count().over("instrument_id") - 1).alias("event_seq")
        )
        event_frames.append(event_frame)

    if not event_frames:
        return pl.DataFrame(
            schema={
                "instrument_id": pl.Utf8,
                "symbol": pl.Utf8,
                "venue": pl.Utf8,
                "asset_group": pl.Utf8,
                "event_time": pl.Datetime(time_zone=config.source_timezone),
                "event_bar_index": pl.Int64,
                "cusum_side": pl.Int64,
                "trgt": pl.Float64,
                "fd_return": pl.Float64,
                "event_seq": pl.Int64,
            }
        )

    return pl.concat(event_frames).sort(["instrument_id", "event_time"])


def _compute_overlap_metadata(starts: np.ndarray, ends: np.ndarray, n_bars: int) -> list[dict[str, float]]:
    delta = np.zeros(n_bars + 1, dtype=np.int64)
    for start, end in zip(starts, ends, strict=True):
        delta[start] += 1
        if end + 1 < delta.shape[0]:
            delta[end + 1] -= 1

    concurrency = np.cumsum(delta[:-1])
    # Count interval intersections in O(n log n):
    # overlaps_i = count(starts <= end_i) - count(ends < start_i) - 1(self).
    sorted_starts = np.sort(starts)
    sorted_ends = np.sort(ends)
    overlap_counts = (
        np.searchsorted(sorted_starts, ends, side="right")
        - np.searchsorted(sorted_ends, starts, side="left")
        - 1
    )
    overlaps = []
    for idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        interval_concurrency = concurrency[start : end + 1]
        overlap_count = int(max(overlap_counts[idx], 0))
        overlaps.append(
            {
                "concurrency_at_event": int(concurrency[start]),
                "max_concurrency": int(interval_concurrency.max()),
                "mean_concurrency": float(interval_concurrency.mean()),
                "avg_uniqueness": float(np.mean(1.0 / interval_concurrency)),
                "overlap_count": overlap_count,
            }
        )
    return overlaps


def label_events(events: pl.DataFrame, raw_bars: pl.DataFrame, config: Module1Config) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame(
            schema={
                "instrument_id": pl.Utf8,
                "symbol": pl.Utf8,
                "venue": pl.Utf8,
                "asset_group": pl.Utf8,
                "event_time": pl.Datetime(time_zone=config.source_timezone),
                "t1": pl.Datetime(time_zone=config.source_timezone),
                "vertical_barrier_time": pl.Datetime(time_zone=config.source_timezone),
                "trgt": pl.Float64,
                "label": pl.Int64,
                "hit_type": pl.Utf8,
                "ret": pl.Float64,
                "event_bar_index": pl.Int64,
                "t1_bar_index": pl.Int64,
                "vertical_barrier_bar_index": pl.Int64,
                "vertical_barrier_bars_used": pl.Int64,
                "truncated_vertical_barrier": pl.Boolean,
                "concurrency_at_event": pl.Int64,
                "max_concurrency": pl.Int64,
                "mean_concurrency": pl.Float64,
                "avg_uniqueness": pl.Float64,
                "overlap_count": pl.Int64,
            }
        )

    labeled_frames: list[pl.DataFrame] = []
    raw_by_instrument = {
        frame["instrument_id"][0]: frame.sort("bar_index")
        for frame in raw_bars.partition_by("instrument_id", maintain_order=True)
    }

    for event_frame in events.partition_by("instrument_id", maintain_order=True):
        instrument_id = event_frame["instrument_id"][0]
        raw_frame = raw_by_instrument[instrument_id]

        closes = raw_frame.get_column("close").to_numpy()
        log_close = np.log(closes)
        timestamps = raw_frame.get_column("timestamp").to_list()
        session_dates = raw_frame.get_column("session_date").to_list()
        last_bar_index = int(raw_frame["bar_index"][-1])
        session_vertical_barriers = _session_vertical_barriers(raw_frame, config)

        rows: list[dict[str, object]] = []
        starts: list[int] = []
        ends: list[int] = []

        for event in event_frame.iter_rows(named=True):
            entry_index = int(event["event_bar_index"])
            event_session_date = session_dates[entry_index]
            vertical_barrier_bars = int(
                session_vertical_barriers.get(event_session_date, max(int(config.vertical_barrier_bars), 1))
            )
            vertical_barrier_index = min(entry_index + vertical_barrier_bars, last_bar_index)
            truncated = vertical_barrier_index < entry_index + vertical_barrier_bars

            exit_index = vertical_barrier_index
            hit_type = "vertical"
            label = 0
            ret = 0.0

            if vertical_barrier_index > entry_index:
                future_slice = log_close[entry_index + 1 : vertical_barrier_index + 1] - log_close[entry_index]
                pt_level = config.pt_mult * float(event["trgt"])
                sl_level = -config.sl_mult * float(event["trgt"])

                pt_hits = np.flatnonzero(future_slice >= pt_level)
                sl_hits = np.flatnonzero(future_slice <= sl_level)
                first_pt = int(pt_hits[0]) if pt_hits.size else None
                first_sl = int(sl_hits[0]) if sl_hits.size else None

                if first_pt is not None and (first_sl is None or first_pt <= first_sl):
                    exit_index = entry_index + 1 + first_pt
                    hit_type = "pt"
                    label = 1
                    ret = float(future_slice[first_pt])
                elif first_sl is not None:
                    exit_index = entry_index + 1 + first_sl
                    hit_type = "sl"
                    label = -1
                    ret = float(future_slice[first_sl])
                elif future_slice.size:
                    ret = float(future_slice[-1])

            starts.append(entry_index)
            ends.append(exit_index)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "symbol": event["symbol"],
                    "venue": event["venue"],
                    "asset_group": event["asset_group"],
                    "event_time": event["event_time"],
                    "t1": timestamps[exit_index],
                    "vertical_barrier_time": timestamps[vertical_barrier_index],
                    "trgt": float(event["trgt"]),
                    "label": int(label),
                    "hit_type": hit_type,
                    "ret": ret,
                    "event_bar_index": entry_index,
                    "t1_bar_index": exit_index,
                    "vertical_barrier_bar_index": vertical_barrier_index,
                    "vertical_barrier_bars_used": vertical_barrier_bars,
                    "truncated_vertical_barrier": bool(truncated),
                }
            )

        overlap_metadata = _compute_overlap_metadata(
            starts=np.asarray(starts, dtype=np.int64),
            ends=np.asarray(ends, dtype=np.int64),
            n_bars=raw_frame.height,
        )

        for row, overlap in zip(rows, overlap_metadata, strict=True):
            row.update(overlap)

        labeled_frames.append(pl.DataFrame(rows))

    return pl.concat(labeled_frames).sort(["instrument_id", "event_time"])


def _write_artifact(frame: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def build_module1_dataset(config: Module1Config) -> Module1Artifacts:
    bars = load_bars(config)
    if bars.is_empty():
        raise ValueError(
            "Module 1 ingestion produced zero bars. Verify the source_root, source_format, "
            "symbol filter, and bar_minutes settings for the dense-data transition."
        )
    stationary, fracdiff_params = fit_fractional_diff(bars, config)
    events = sample_events(stationary, config)
    labels = label_events(events, bars, config)

    output_root = Path(config.output_root)
    bars_path = _write_artifact(bars, output_root / "normalized_bars.parquet")
    stationary_path = _write_artifact(stationary, output_root / "stationary_series.parquet")
    fracdiff_params_path = _write_artifact(fracdiff_params, output_root / "fracdiff_params.parquet")
    events_path = _write_artifact(events, output_root / "sampled_events.parquet")
    labels_path = _write_artifact(labels, output_root / "triple_barrier_labels.parquet")

    return Module1Artifacts(
        config=config,
        bars=bars,
        stationary=stationary,
        fracdiff_params=fracdiff_params,
        events=events,
        labels=labels,
        bars_path=bars_path,
        stationary_path=stationary_path,
        fracdiff_params_path=fracdiff_params_path,
        events_path=events_path,
        labels_path=labels_path,
    )
