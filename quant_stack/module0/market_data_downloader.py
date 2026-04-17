from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import polars as pl

from .data_sanitizer import DataSanitizerConfig, clean_market_data


@dataclass(frozen=True)
class DownloaderConfig:
    provider: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT",)
    timeframe: str = "1m"
    lookback_days: int = 7
    output_root: Path = Path("market_data")
    venue: str = "auto"
    asset_group: str = "auto"
    source_bucket: str = "0"
    bar_minutes: int = 1
    sanitizer: DataSanitizerConfig = DataSanitizerConfig()


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def canonical_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol)


def _timeframe_to_ms(timeframe: str) -> int:
    tf = str(timeframe).strip().lower()
    match = re.fullmatch(r"(\d+)([mhdw])", tf)
    if not match:
        raise ValueError(f"Unsupported timeframe for pagination: {timeframe}")
    size = int(match.group(1))
    unit = match.group(2)
    unit_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }[unit]
    return size * unit_ms


def _normalize_equity_timeframe(timeframe: str) -> tuple[str, int]:
    normalized = str(timeframe).strip().lower()
    mapping = {
        "1m": ("1m", 1),
        "60m": ("60m", 60),
        "1h": ("60m", 60),
        "1d": ("1d", 1440),
        "d": ("1d", 1440),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported equity timeframe: {timeframe}")
    return mapping[normalized]


def _to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_frame(
    symbol: str,
    timestamps_ms: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    *,
    venue: str,
    asset_group: str,
    source_bucket: str,
    source_bar_minutes: int,
) -> pl.DataFrame:
    ts = [datetime.fromtimestamp(t / 1000.0, tz=timezone.utc) for t in timestamps_ms]
    symbol_original = str(symbol).strip().upper()
    symbol_code = _normalize_symbol(symbol_original)
    return pl.DataFrame(
        {
            "instrument_id": [f"{symbol_code}@{venue}_{asset_group}"] * len(ts),
            "symbol": [symbol_code] * len(ts),
            "source_symbol": [symbol_original] * len(ts),
            "venue": [venue] * len(ts),
            "asset_group": [asset_group] * len(ts),
            "source_bucket": [source_bucket] * len(ts),
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "open_interest": [0] * len(ts),
            "source_bar_minutes": [source_bar_minutes] * len(ts),
        }
    )


def _fetch_binance(symbol: str, lookback_days: int, timeframe: str) -> pl.DataFrame:
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    step_ms = _timeframe_to_ms(timeframe)
    since = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
    all_rows: list[list[float]] = []
    seen_timestamps: set[int] = set()
    cursor = since

    while cursor < now_ms:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1500)
        if not rows:
            break
        appended = 0
        for row in rows:
            ts_ms = int(row[0])
            if ts_ms in seen_timestamps:
                continue
            seen_timestamps.add(ts_ms)
            all_rows.append(row)
            appended += 1
        last_ts = int(rows[-1][0])
        next_cursor = last_ts + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if appended == 0:
            break

    if not all_rows:
        return pl.DataFrame()
    all_rows.sort(key=lambda value: int(value[0]))
    timestamps_ms = [int(r[0]) for r in all_rows]
    return _to_frame(
        symbol,
        timestamps_ms,
        [float(r[1]) for r in all_rows],
        [float(r[2]) for r in all_rows],
        [float(r[3]) for r in all_rows],
        [float(r[4]) for r in all_rows],
        [float(r[5]) for r in all_rows],
        venue="binance",
        asset_group="crypto",
        source_bucket="0",
        source_bar_minutes=1,
    )


def _fetch_alpaca_or_yfinance(symbol: str, lookback_days: int, timeframe: str) -> pl.DataFrame:
    provider_timeframe, source_bar_minutes = _normalize_equity_timeframe(timeframe)
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if api_key and api_secret:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = StockHistoricalDataClient(api_key, api_secret)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        timeframe_value = {
            "1m": TimeFrame.Minute,
            "60m": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }[provider_timeframe]
        req = StockBarsRequest(symbol_or_symbols=[symbol], timeframe=timeframe_value, start=start, end=end)
        bars = client.get_stock_bars(req).df.reset_index()
        if bars.empty:
            return pl.DataFrame()
        bars = bars.rename(
            columns={"timestamp": "ts", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        )
        ts_col = "ts" if "ts" in bars.columns else (bars.columns[0] if bars.columns.size else "ts")
        timestamps_ms = [int(_to_utc_timestamp(v).timestamp() * 1000) for v in bars[ts_col].tolist()]
        return _to_frame(
            symbol,
            timestamps_ms,
            bars["open"].astype(float).tolist(),
            bars["high"].astype(float).tolist(),
            bars["low"].astype(float).tolist(),
            bars["close"].astype(float).tolist(),
            bars["volume"].astype(float).tolist(),
            venue="alpaca",
            asset_group="stocks",
            source_bucket="0",
            source_bar_minutes=source_bar_minutes,
        )

    import yfinance as yf

    ticker = yf.Ticker(symbol)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(int(lookback_days), 1))
    hist = ticker.history(start=start, end=end, interval=provider_timeframe, auto_adjust=False)
    if hist.empty:
        return pl.DataFrame()
    hist = hist.reset_index()
    if "Datetime" in hist.columns:
        ts_col = "Datetime"
    elif "Date" in hist.columns:
        ts_col = "Date"
    else:
        ts_col = hist.columns[0]
    timestamps_ms = [int(_to_utc_timestamp(v).timestamp() * 1000) for v in hist[ts_col].tolist()]
    return _to_frame(
        symbol,
        timestamps_ms,
        hist["Open"].astype(float).tolist(),
        hist["High"].astype(float).tolist(),
        hist["Low"].astype(float).tolist(),
        hist["Close"].astype(float).tolist(),
        hist["Volume"].astype(float).tolist(),
        venue="yfinance",
        asset_group="stocks",
        source_bucket="0",
        source_bar_minutes=source_bar_minutes,
    )


def save_incremental_parquet(frame: pl.DataFrame, output_root: Path) -> list[Path]:
    if frame.is_empty():
        return []

    def _merge_and_write(path: Path, incoming: pl.DataFrame) -> None:
        if path.exists():
            existing = pl.read_parquet(path).sort("timestamp")
            merged = (
                pl.concat([existing, incoming], how="vertical_relaxed")
                .sort("timestamp")
                .unique(subset=["timestamp"], keep="last")
                .sort("timestamp")
            )
            if not merged.equals(existing):
                merged.write_parquet(path)
            return
        incoming.write_parquet(path)

    def _migrate_legacy_symbol_file(legacy_path: Path, symbol_dir: Path) -> None:
        if not legacy_path.exists():
            return
        legacy = pl.read_parquet(legacy_path)
        if legacy.is_empty():
            legacy_path.unlink(missing_ok=True)
            return
        legacy = (
            legacy.sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
            .with_columns(pl.col("timestamp").dt.strftime("%Y%m%d").alias("_partition_day"))
        )
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for day_slice in legacy.partition_by("_partition_day", maintain_order=True):
            partition_day = str(day_slice["_partition_day"][0])
            _merge_and_write(symbol_dir / f"{partition_day}.parquet", day_slice.drop("_partition_day"))
        try:
            legacy_path.unlink()
        except OSError:
            legacy_path.rename(legacy_path.with_suffix(".legacy.bak"))

    written: list[Path] = []
    for part in frame.partition_by(["venue", "asset_group", "source_bucket", "symbol"], maintain_order=True):
        symbol = str(part["symbol"][0]).lower()
        venue = str(part["venue"][0]).lower()
        asset_group = str(part["asset_group"][0]).lower()
        bucket = str(part["source_bucket"][0])
        source_bar_minutes = int(part["source_bar_minutes"][0]) if "source_bar_minutes" in part.columns else 1
        out_dir = output_root / f"{venue} {asset_group}" / bucket
        if source_bar_minutes >= 1440:
            out_dir.mkdir(parents=True, exist_ok=True)
            symbol_path = out_dir / f"{symbol}.parquet"
            incoming = part.sort("timestamp").unique(subset=["timestamp"], keep="last")
            _merge_and_write(symbol_path, incoming)
            written.append(symbol_path)
            continue

        symbol_dir = out_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        legacy_path = out_dir / f"{symbol}.parquet"
        _migrate_legacy_symbol_file(legacy_path, symbol_dir)

        by_day = (
            part.sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
            .with_columns(pl.col("timestamp").dt.strftime("%Y%m%d").alias("_partition_day"))
        )
        for day_slice in by_day.partition_by("_partition_day", maintain_order=True):
            partition_day = str(day_slice["_partition_day"][0])
            partition_path = symbol_dir / f"{partition_day}.parquet"
            incoming = day_slice.drop("_partition_day")
            _merge_and_write(partition_path, incoming)
            written.append(partition_path)
    return written


class MarketDataDownloader:
    def __init__(self, config: DownloaderConfig = DownloaderConfig()) -> None:
        self.config = config

    def fetch_raw(self, lookback_days: int | None = None) -> pl.DataFrame:
        days = int(lookback_days if lookback_days is not None else self.config.lookback_days)
        frames: list[pl.DataFrame] = []
        for symbol in self.config.symbols:
            provider = self.config.provider.lower()
            if provider == "binance":
                fetched = _fetch_binance(symbol, days, self.config.timeframe)
            elif provider in {"alpaca", "yfinance"}:
                fetched = _fetch_alpaca_or_yfinance(symbol, days, self.config.timeframe)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            if not fetched.is_empty():
                frames.append(fetched)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical_relaxed").sort(["instrument_id", "timestamp"])

    def sanitize(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        return clean_market_data(frame, self.config.sanitizer)

    def fetch(self, lookback_days: int | None = None) -> pl.DataFrame:
        raw = self.fetch_raw(lookback_days=lookback_days)
        return self.sanitize(raw)

    def fetch_and_store(self, lookback_days: int | None = None) -> tuple[pl.DataFrame, list[Path]]:
        frame = self.fetch(lookback_days=lookback_days)
        written = save_incremental_parquet(frame, self.config.output_root)
        return frame, written
