from __future__ import annotations

import json
import logging
import smtplib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from quant_stack.json_records import append_json_record
from quant_stack.module6.regime_detector import RegimeDetector, RegimeDetectorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveIngestorConfig:
    symbol: str
    venue: str = "binance"
    source: str = "ccxt"
    interval: str = "1m"
    source_timezone: str = "UTC"
    volume_threshold_window: int = 1000
    volume_threshold_multiplier: float = 50.0
    fracdiff_d: float = 0.3
    weight_eps: float = 1e-5
    vol_lookback_bars: int = 63
    max_minute_buffer: int = 6000
    max_volume_buffer: int = 4000
    regime_refit_interval: int = 20
    regime_min_observations: int = 120
    regime_fit_window: int = 1000
    heartbeat_timeout_minutes: int = 5
    heartbeat_alert_path: Path = Path("artifacts/module7/feed_alerts.json")
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    alert_from_email: str | None = None
    alert_to_emails: tuple[str, ...] = ()
    poll_seconds: float = 10.0


@dataclass(frozen=True)
class MinuteBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class LiveVolumeBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    input_bar_count: int
    volume_threshold_used: float
    raw_log_return: float
    fracdiff_log_close: float
    fracdiff_return: float
    sigma_t: float
    regime_state: int
    regime_label: str
    regime_penalty: float


class MinuteBarSource(Protocol):
    def fetch_new_bars(self, since: datetime | None = None) -> list[MinuteBar]:
        ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(timestamp: datetime | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _fracdiff_weights(d: float, max_size: int, weight_eps: float) -> np.ndarray:
    weights = [1.0]
    for k in range(1, max_size):
        next_weight = -weights[-1] * (d - k + 1.0) / k
        if abs(next_weight) < weight_eps:
            break
        weights.append(next_weight)
    return np.asarray(weights[::-1], dtype=np.float64)


def _timeframe_to_ms(timeframe: str) -> int:
    token = str(timeframe).strip().lower()
    if token.endswith("m"):
        return int(token[:-1]) * 60_000
    if token.endswith("h"):
        return int(token[:-1]) * 3_600_000
    if token.endswith("d"):
        return int(token[:-1]) * 86_400_000
    raise ValueError(f"Unsupported timeframe for recovery paging: {timeframe}")


class RollingLiveBuffer:
    def __init__(
        self,
        config: LiveIngestorConfig,
        regime_config: RegimeDetectorConfig | None = None,
    ) -> None:
        self.config = config
        self._minute_bars: deque[MinuteBar] = deque(maxlen=config.max_minute_buffer)
        self._volume_bars: deque[LiveVolumeBar] = deque(maxlen=config.max_volume_buffer)
        self._current_bucket: dict[str, float | datetime | int] | None = None
        self._recent_minute_volumes: deque[float] = deque(maxlen=max(int(config.volume_threshold_window), 1))
        self._recent_minute_volume_sum = 0.0
        self._log_close_buffer: deque[float] = deque(maxlen=config.max_volume_buffer)
        sigma_window = max(int(config.vol_lookback_bars) * 4, int(config.vol_lookback_bars) + 2)
        self._raw_return_buffer: deque[float] = deque(maxlen=max(sigma_window, 2))
        self._fracdiff_value_buffer: deque[float] = deque(maxlen=config.max_volume_buffer)
        self._fracdiff_weights = _fracdiff_weights(
            d=config.fracdiff_d,
            max_size=config.max_volume_buffer,
            weight_eps=config.weight_eps,
        )
        self._last_regime = {
            "state": -1,
            "label": "unknown",
            "penalty": 1.0,
        }
        self._regime_detector = RegimeDetector(
            regime_config
            or RegimeDetectorConfig(
                benchmark_symbol=config.symbol,
                min_observations=max(config.regime_min_observations, 30),
            )
        )
        self._volume_bars_since_refit = 0

    @property
    def minute_bars(self) -> tuple[MinuteBar, ...]:
        return tuple(self._minute_bars)

    @property
    def volume_bars(self) -> tuple[LiveVolumeBar, ...]:
        return tuple(self._volume_bars)

    def _push_minute_volume(self, volume: float) -> None:
        if len(self._recent_minute_volumes) == self._recent_minute_volumes.maxlen:
            self._recent_minute_volume_sum -= float(self._recent_minute_volumes[0])
        self._recent_minute_volumes.append(float(volume))
        self._recent_minute_volume_sum += float(volume)

    def _current_threshold(self, fallback_volume: float) -> float:
        if not self._recent_minute_volumes:
            return max(float(fallback_volume) * self.config.volume_threshold_multiplier, 1e-12)
        mean_volume = float(self._recent_minute_volume_sum / len(self._recent_minute_volumes))
        if not np.isfinite(mean_volume) or mean_volume <= 0.0:
            mean_volume = float(fallback_volume)
        return max(mean_volume * self.config.volume_threshold_multiplier, 1e-12)

    def _build_regime_frame(self) -> pd.DataFrame:
        if not self._volume_bars:
            return pd.DataFrame()
        window = max(int(self.config.regime_fit_window), int(self.config.regime_min_observations))
        tail = list(self._volume_bars)[-window:]
        return pd.DataFrame(
            {
                "instrument_id": [f"{self.config.symbol}@{self.config.venue}"] * len(tail),
                "symbol": [self.config.symbol] * len(tail),
                "timestamp": [bar.timestamp for bar in tail],
                "raw_log_return": [bar.raw_log_return for bar in tail],
                "sigma_t": [bar.sigma_t for bar in tail],
            }
        )

    def _maybe_update_regime(self) -> None:
        if len(self._volume_bars) < self.config.regime_min_observations:
            return
        if self._volume_bars_since_refit < self.config.regime_refit_interval:
            return
        frame = self._build_regime_frame()
        if frame.empty:
            return
        try:
            regime_features = self._regime_detector.fit_transform(frame).feature_frame
            latest = regime_features.iloc[-1]
            self._last_regime = {
                "state": int(latest["m2_regime_state"]),
                "label": str(latest["m2_regime_label"]),
                "penalty": float(latest["m2_regime_penalty"]),
            }
            self._volume_bars_since_refit = 0
        except Exception:
            # Keep the last known regime when realtime fit fails.
            logger.warning("Realtime regime refit failed for %s@%s", self.config.symbol, self.config.venue, exc_info=True)
            self._volume_bars_since_refit = 0

    def _latest_metrics(self, close_price: float) -> tuple[float, float, float, float]:
        log_close = float(np.log(max(float(close_price), 1e-12)))
        previous_log_close = self._log_close_buffer[-1] if self._log_close_buffer else float("nan")
        raw_return = float(log_close - previous_log_close) if np.isfinite(previous_log_close) else float("nan")

        self._log_close_buffer.append(log_close)
        self._raw_return_buffer.append(raw_return)

        fracdiff_log_close = float("nan")
        width = len(self._fracdiff_weights)
        if len(self._log_close_buffer) >= width:
            log_tail = np.asarray(list(self._log_close_buffer)[-width:], dtype=np.float64)
            fracdiff_log_close = float(np.dot(log_tail, self._fracdiff_weights))

        previous_fracdiff = self._fracdiff_value_buffer[-1] if self._fracdiff_value_buffer else float("nan")
        fracdiff_return = (
            float(fracdiff_log_close - previous_fracdiff)
            if np.isfinite(fracdiff_log_close) and np.isfinite(previous_fracdiff)
            else float("nan")
        )
        self._fracdiff_value_buffer.append(fracdiff_log_close)

        finite_returns = np.asarray(
            [value for value in self._raw_return_buffer if np.isfinite(value)],
            dtype=np.float64,
        )
        sigma_t = float("nan")
        if finite_returns.size >= 2:
            min_periods = min(self.config.vol_lookback_bars, max(2, finite_returns.size))
            sigma_series = (
                pd.Series(finite_returns)
                .ewm(
                    span=self.config.vol_lookback_bars,
                    adjust=False,
                    min_periods=min_periods,
                )
                .std(bias=False)
            )
            if not sigma_series.empty:
                sigma_t = float(sigma_series.iloc[-1])
        return raw_return, fracdiff_log_close, fracdiff_return, sigma_t

    def on_minute_bar(self, bar: MinuteBar) -> LiveVolumeBar | None:
        normalized = MinuteBar(
            timestamp=_as_utc(bar.timestamp),
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )
        threshold = self._current_threshold(fallback_volume=normalized.volume)
        self._minute_bars.append(normalized)
        self._push_minute_volume(normalized.volume)

        if self._current_bucket is None:
            self._current_bucket = {
                "timestamp": normalized.timestamp,
                "open": normalized.open,
                "high": normalized.high,
                "low": normalized.low,
                "close": normalized.close,
                "volume": 0.0,
                "input_bar_count": 0,
                "volume_threshold_used": threshold,
            }

        bucket = self._current_bucket
        bucket["timestamp"] = normalized.timestamp
        bucket["high"] = max(float(bucket["high"]), normalized.high)
        bucket["low"] = min(float(bucket["low"]), normalized.low)
        bucket["close"] = normalized.close
        bucket["volume"] = float(bucket["volume"]) + normalized.volume
        bucket["input_bar_count"] = int(bucket["input_bar_count"]) + 1

        if float(bucket["volume"]) < float(bucket["volume_threshold_used"]):
            return None

        provisional = LiveVolumeBar(
            timestamp=bucket["timestamp"],  # type: ignore[arg-type]
            symbol=self.config.symbol,
            open=float(bucket["open"]),
            high=float(bucket["high"]),
            low=float(bucket["low"]),
            close=float(bucket["close"]),
            volume=float(bucket["volume"]),
            input_bar_count=int(bucket["input_bar_count"]),
            volume_threshold_used=float(bucket["volume_threshold_used"]),
            raw_log_return=float("nan"),
            fracdiff_log_close=float("nan"),
            fracdiff_return=float("nan"),
            sigma_t=float("nan"),
            regime_state=int(self._last_regime["state"]),
            regime_label=str(self._last_regime["label"]),
            regime_penalty=float(self._last_regime["penalty"]),
        )
        self._volume_bars.append(provisional)
        self._volume_bars_since_refit += 1

        raw_ret, fracdiff_log_close, fracdiff_return, sigma_t = self._latest_metrics(provisional.close)
        self._maybe_update_regime()
        finalized = LiveVolumeBar(
            timestamp=provisional.timestamp,
            symbol=provisional.symbol,
            open=provisional.open,
            high=provisional.high,
            low=provisional.low,
            close=provisional.close,
            volume=provisional.volume,
            input_bar_count=provisional.input_bar_count,
            volume_threshold_used=provisional.volume_threshold_used,
            raw_log_return=raw_ret,
            fracdiff_log_close=fracdiff_log_close,
            fracdiff_return=fracdiff_return,
            sigma_t=sigma_t,
            regime_state=int(self._last_regime["state"]),
            regime_label=str(self._last_regime["label"]),
            regime_penalty=float(self._last_regime["penalty"]),
        )
        self._volume_bars[-1] = finalized
        self._current_bucket = None
        return finalized


@dataclass(frozen=True)
class FeedHeartbeat:
    timeout_minutes: int = 5
    alert_path: Path = Path("artifacts/module7/feed_alerts.json")
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    from_email: str | None = None
    to_emails: tuple[str, ...] = ()
    _last_seen: datetime | None = field(default=None, init=False, repr=False, compare=False)
    _last_alert_at: datetime | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_last_seen", None)
        object.__setattr__(self, "_last_alert_at", None)

    def mark_data(self, timestamp: datetime) -> None:
        object.__setattr__(self, "_last_seen", _as_utc(timestamp))

    def _send_email_alert(self, subject: str, body: str) -> None:
        if not self.smtp_host or not self.to_emails or not self.from_email:
            return
        try:
            message = EmailMessage()
            message["From"] = self.from_email
            message["To"] = ",".join(self.to_emails)
            message["Subject"] = subject
            message.set_content(body)
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)
        except Exception:
            logger.warning("Failed to send feed heartbeat email alert.", exc_info=True)
            return

    def _write_alert(self, payload: dict[str, object]) -> None:
        append_json_record(self.alert_path, payload)

    def check(self, now: datetime | None = None) -> bool:
        current_time = _as_utc(now or _utc_now())
        if self._last_seen is None:
            return True
        timeout = timedelta(minutes=max(int(self.timeout_minutes), 1))
        if current_time - self._last_seen <= timeout:
            return True

        if self._last_alert_at is not None and current_time - self._last_alert_at <= timedelta(minutes=1):
            return False

        payload = {
            "timestamp": current_time.isoformat(),
            "status": "stale_feed",
            "last_seen": self._last_seen.isoformat(),
            "silence_seconds": int((current_time - self._last_seen).total_seconds()),
        }
        self._write_alert(payload)
        self._send_email_alert(
            subject="Quant Feed Heartbeat Alert",
            body=json.dumps(payload, indent=2),
        )
        object.__setattr__(self, "_last_alert_at", current_time)
        return False


class YFinanceMinuteSource:
    def __init__(self, symbol: str, interval: str = "1m", backfill_days: int = 30) -> None:
        self.symbol = symbol
        self.interval = interval
        self.backfill_days = max(int(backfill_days), 1)

    def fetch_new_bars(self, since: datetime | None = None) -> list[MinuteBar]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is required for YFinanceMinuteSource.") from exc

        end = pd.Timestamp(_utc_now())
        if since is None:
            start = end - pd.Timedelta(days=self.backfill_days)
        else:
            start = pd.Timestamp(_as_utc(since))
        data = yf.download(
            tickers=self.symbol,
            start=start,
            end=end,
            interval=self.interval,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            return []
        data = data.reset_index()
        if "Datetime" in data.columns:
            ts_column = "Datetime"
        elif "Date" in data.columns:
            ts_column = "Date"
        else:
            ts_column = data.columns[0]
        data[ts_column] = pd.to_datetime(data[ts_column], utc=True)
        if since is not None:
            data = data.loc[data[ts_column] > pd.Timestamp(_as_utc(since))]
        bars: list[MinuteBar] = []
        for row in data.itertuples(index=False):
            bars.append(
                MinuteBar(
                    timestamp=_as_utc(getattr(row, ts_column)),
                    open=float(getattr(row, "Open")),
                    high=float(getattr(row, "High")),
                    low=float(getattr(row, "Low")),
                    close=float(getattr(row, "Close")),
                    volume=float(getattr(row, "Volume")),
                )
            )
        return bars


class CCXTMinuteSource:
    def __init__(self, symbol: str, exchange_name: str = "binance", timeframe: str = "1m", backfill_days: int = 30) -> None:
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.timeframe = timeframe
        self.backfill_days = max(int(backfill_days), 1)
        try:
            import ccxt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("ccxt is required for CCXTMinuteSource.") from exc
        exchange_class = getattr(ccxt, exchange_name, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported ccxt exchange: {exchange_name}")
        self.exchange = exchange_class()

    def fetch_new_bars(self, since: datetime | None = None) -> list[MinuteBar]:
        if since is None:
            since_ms = int((_utc_now() - timedelta(days=self.backfill_days)).timestamp() * 1000)
        else:
            since_ms = int(_as_utc(since).timestamp() * 1000)
        step_ms = _timeframe_to_ms(self.timeframe)
        rows: list[list[float]] = []
        seen: set[int] = set()
        cursor = since_ms
        limit = 1000
        while True:
            batch = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=cursor, limit=limit)
            if not batch:
                break
            appended = 0
            for row in batch:
                ts_ms = int(row[0])
                if ts_ms in seen:
                    continue
                seen.add(ts_ms)
                rows.append(row)
                appended += 1
            last_ts = int(batch[-1][0])
            next_cursor = last_ts + step_ms
            if next_cursor <= int(cursor):
                break
            cursor = next_cursor
            if appended == 0:
                break

        rows.sort(key=lambda row: int(row[0]))
        bars: list[MinuteBar] = []
        for ts_ms, open_price, high_price, low_price, close_price, volume in rows:
            bars.append(
                MinuteBar(
                    timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                    open=float(open_price),
                    high=float(high_price),
                    low=float(low_price),
                    close=float(close_price),
                    volume=float(volume),
                )
            )
        if since is None:
            return bars
        since_utc = _as_utc(since)
        return [bar for bar in bars if bar.timestamp > since_utc]


class LiveIngestor:
    def __init__(
        self,
        config: LiveIngestorConfig,
        source: MinuteBarSource,
        buffer: RollingLiveBuffer | None = None,
        heartbeat: FeedHeartbeat | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.buffer = buffer or RollingLiveBuffer(config)
        self.heartbeat = heartbeat or FeedHeartbeat(
            timeout_minutes=config.heartbeat_timeout_minutes,
            alert_path=config.heartbeat_alert_path,
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            smtp_username=config.smtp_username,
            smtp_password=config.smtp_password,
            from_email=config.alert_from_email,
            to_emails=config.alert_to_emails,
        )
        self._last_minute_timestamp: datetime | None = None

    def poll_once(self) -> list[LiveVolumeBar]:
        minute_bars = self.source.fetch_new_bars(self._last_minute_timestamp)
        closed: list[LiveVolumeBar] = []
        for minute_bar in minute_bars:
            self._last_minute_timestamp = minute_bar.timestamp
            self.heartbeat.mark_data(minute_bar.timestamp)
            volume_bar = self.buffer.on_minute_bar(minute_bar)
            if volume_bar is not None:
                closed.append(volume_bar)
        self.heartbeat.check()
        return closed

    def run_forever(self, on_volume_bar: Callable[[LiveVolumeBar], None]) -> None:
        while True:
            for volume_bar in self.poll_once():
                on_volume_bar(volume_bar)
            time.sleep(max(float(self.config.poll_seconds), 0.1))
