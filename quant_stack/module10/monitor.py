from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_stack.json_records import append_json_record

logger = logging.getLogger(__name__)

from .notifier import AlertNotifier, NotificationConfig


@dataclass(frozen=True)
class MonitorConfig:
    heartbeat_path: Path = Path("artifacts/module7/heartbeat.json")
    alerts_path: Path = Path("artifacts/module10/monitor_alerts.json")
    check_interval_seconds: int = 60
    stale_data_minutes_threshold: float = 5.0
    disk_usage_threshold: float = 0.80
    memory_rss_mb_threshold: float = 1500.0
    parquet_root: Path = Path("artifacts")
    pid_to_watch: int | None = None
    notifier: NotificationConfig = NotificationConfig()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _append_alert(path: Path, payload: dict[str, Any]) -> None:
    append_json_record(path, payload)


def _get_process_rss_mb(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(pid)
        rss_bytes = float(process.memory_info().rss)
        return rss_bytes / (1024.0 * 1024.0)
    except Exception:
        logger.warning("Failed to read process RSS for pid=%s.", pid, exc_info=True)
        return None


def _read_last_heartbeat(path: Path) -> datetime | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to parse heartbeat payload at %s.", path, exc_info=True)
        return None
    timestamp = payload.get("timestamp") if isinstance(payload, dict) else None
    if not timestamp:
        return None
    ts = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def run_health_check(config: MonitorConfig = MonitorConfig()) -> dict[str, Any]:
    now = _utc_now()
    alerts: list[str] = []
    notifier = AlertNotifier(config.notifier)

    heartbeat_ts = _read_last_heartbeat(config.heartbeat_path)
    stale_data_minutes = None
    if heartbeat_ts is None:
        alerts.append("heartbeat_missing")
    else:
        stale_data_minutes = (now - heartbeat_ts).total_seconds() / 60.0
        if stale_data_minutes > float(config.stale_data_minutes_threshold):
            alerts.append("stale_data")

    disk = shutil.disk_usage(config.parquet_root)
    disk_used_ratio = float(disk.used / max(disk.total, 1))
    if disk_used_ratio > float(config.disk_usage_threshold):
        alerts.append("disk_usage_high")

    watched_pid = config.pid_to_watch if config.pid_to_watch is not None else os.getpid()
    memory_rss_mb = _get_process_rss_mb(int(watched_pid))
    if memory_rss_mb is not None and memory_rss_mb > float(config.memory_rss_mb_threshold):
        alerts.append("memory_usage_high")

    result = {
        "timestamp": now.isoformat(),
        "alerts": alerts,
        "stale_data_minutes": stale_data_minutes,
        "disk_used_ratio": disk_used_ratio,
        "memory_rss_mb": memory_rss_mb,
        "watched_pid": watched_pid,
    }
    _append_alert(config.alerts_path, result)
    if alerts:
        notifier.send_summary(
            daily_pnl=0.0,
            alerts=alerts,
            latency_ms=None,
            slippage_bps=None,
            stale_data_minutes=stale_data_minutes,
        )
    return result
