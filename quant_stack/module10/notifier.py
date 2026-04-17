from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_stack.json_records import append_json_record

@dataclass(frozen=True)
class NotificationConfig:
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    slack_webhook_url: str | None = None
    notification_log_path: Path = Path("artifacts/module10/notifications_log.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_json(path: Path, payload: dict[str, Any]) -> None:
    append_json_record(path, payload)


class AlertNotifier:
    def __init__(self, config: NotificationConfig = NotificationConfig()) -> None:
        self.config = config

    def _post_json(self, url: str, body: dict[str, Any]) -> tuple[bool, str]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status_code = int(response.status)
                ok = 200 <= status_code < 300
                return ok, f"http_{status_code}"
        except urllib.error.URLError as exc:
            return False, str(exc)

    def send_telegram(self, message: str) -> bool:
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        if not token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        body = {"chat_id": chat_id, "text": message}
        ok, detail = self._post_json(url, body)
        _append_json(
            self.config.notification_log_path,
            {
                "timestamp": _utc_now_iso(),
                "channel": "telegram",
                "ok": ok,
                "detail": detail,
                "message": message,
            },
        )
        return ok

    def send_slack(self, message: str) -> bool:
        webhook = self.config.slack_webhook_url
        if not webhook:
            return False
        ok, detail = self._post_json(webhook, {"text": message})
        _append_json(
            self.config.notification_log_path,
            {
                "timestamp": _utc_now_iso(),
                "channel": "slack",
                "ok": ok,
                "detail": detail,
                "message": message,
            },
        )
        return ok

    def send_summary(
        self,
        *,
        daily_pnl: float,
        alerts: list[str],
        latency_ms: float | None = None,
        slippage_bps: float | None = None,
        stale_data_minutes: float | None = None,
    ) -> dict[str, bool]:
        parts = [
            f"Daily PnL: {daily_pnl:.4f}",
            f"Critical Alerts: {len(alerts)}",
        ]
        if latency_ms is not None:
            parts.append(f"Latency(ms): {latency_ms:.1f}")
        if slippage_bps is not None:
            parts.append(f"Slippage(bps): {slippage_bps:.2f}")
        if stale_data_minutes is not None:
            parts.append(f"Stale Data(min): {stale_data_minutes:.1f}")
        if alerts:
            parts.append("Alerts: " + " | ".join(alerts[:5]))
        message = " | ".join(parts)
        return {
            "telegram": self.send_telegram(message),
            "slack": self.send_slack(message),
        }

    def send_backtest_summary(
        self,
        *,
        run_name: str,
        sharpe: float,
        max_drawdown: float,
        profit_factor: float | None = None,
    ) -> dict[str, bool]:
        parts = [
            f"Run: {run_name}",
            f"Sharpe: {sharpe:.4f}",
            f"MaxDD: {max_drawdown:.4f}",
        ]
        if profit_factor is not None:
            parts.append(f"PF: {profit_factor:.3f}")
        return {
            "telegram": self.send_telegram(" | ".join(parts)),
            "slack": self.send_slack(" | ".join(parts)),
        }
