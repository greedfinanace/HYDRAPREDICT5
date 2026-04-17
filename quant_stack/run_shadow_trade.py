from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_stack.json_records import (
    append_json_record as _append_record_fast,
    read_json_records as _read_records_fast,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return _read_records_fast(path, limit=limit)


def _append_json_record(path: Path, record: dict[str, Any]) -> None:
    _append_record_fast(path, record)


@dataclass(frozen=True)
class ShadowTradeConfig:
    signal_log_path: Path = Path("artifacts/module7/live_signals_log.json")
    shadow_log_path: Path = Path("artifacts/module7/shadow_trades.json")
    slippage_audit_path: Path = Path("artifacts/module7/slippage_audit.json")
    alpha_decay_alert_path: Path = Path("artifacts/module7/alpha_decay_alerts.json")
    expected_return_window: int = 250
    alpha_decay_z_threshold: float = 2.0
    mock_fill_spread_bps: float = 1.0


@dataclass(frozen=True)
class ShadowTradeDecision:
    timestamp: str
    symbol: str
    action: str
    bet_size: float
    pred_prob_win: float
    expected_return: float
    realized_return: float
    mid_price: float
    fill_price: float
    realized_slippage_bps: float
    alpha_decay_zscore: float
    alpha_decay_flag: bool


class ShadowTradeLogger:
    def __init__(self, config: ShadowTradeConfig = ShadowTradeConfig()) -> None:
        self.config = config

    def _mock_fill_price(self, action: str, mid_price: float) -> float:
        spread = float(self.config.mock_fill_spread_bps) * 1e-4
        side = action.lower()
        if side == "buy":
            return mid_price * (1.0 + spread)
        if side == "sell":
            return mid_price * (1.0 - spread)
        return mid_price

    def _alpha_decay_check(self, expected_series: np.ndarray, realized_series: np.ndarray) -> tuple[float, bool]:
        if expected_series.size < 10 or realized_series.size < 10:
            return 0.0, False
        diff = realized_series - expected_series
        mu = float(np.mean(diff))
        sigma = float(np.std(diff, ddof=1))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            return 0.0, False
        z = mu / sigma
        return float(z), bool(abs(z) > float(self.config.alpha_decay_z_threshold))

    def process_latest_signal(self) -> ShadowTradeDecision | None:
        signals = _read_json_records(self.config.signal_log_path, limit=1)
        if not signals:
            return None

        latest = signals[-1]
        action = str(latest.get("final_action", "hold")).lower()
        symbol = str(latest.get("symbol", "UNKNOWN"))
        bet_size = float(latest.get("bet_size", 0.0))
        pred_prob_win = float(latest.get("pred_prob_win", 0.0))
        expected_return = float(latest.get("expected_return", latest.get("meta_expected_return", 0.0)))
        realized_return = float(latest.get("realized_return", latest.get("label_return", 0.0)))
        mid_price = float(latest.get("mid_price", latest.get("close", 0.0)))
        fill_price = self._mock_fill_price(action=action, mid_price=mid_price) if mid_price > 0.0 else 0.0

        if mid_price > 0.0 and action in {"buy", "sell"}:
            if action == "buy":
                slip = (fill_price / mid_price - 1.0) * 10_000.0
            else:
                slip = (mid_price / max(fill_price, 1e-12) - 1.0) * 10_000.0
        else:
            slip = 0.0

        history = _read_json_records(
            self.config.shadow_log_path,
            limit=max(int(self.config.expected_return_window), 1),
        )
        expected_series = np.array(
            [float(r.get("expected_return", 0.0)) for r in history[-self.config.expected_return_window :]] + [expected_return],
            dtype=np.float64,
        )
        realized_series = np.array(
            [float(r.get("realized_return", 0.0)) for r in history[-self.config.expected_return_window :]] + [realized_return],
            dtype=np.float64,
        )
        zscore, alpha_decay_flag = self._alpha_decay_check(expected_series, realized_series)

        decision = ShadowTradeDecision(
            timestamp=_utc_now_iso(),
            symbol=symbol,
            action=action,
            bet_size=bet_size,
            pred_prob_win=pred_prob_win,
            expected_return=expected_return,
            realized_return=realized_return,
            mid_price=mid_price,
            fill_price=fill_price,
            realized_slippage_bps=float(slip),
            alpha_decay_zscore=float(zscore),
            alpha_decay_flag=bool(alpha_decay_flag),
        )

        record = {
            "timestamp": decision.timestamp,
            "symbol": decision.symbol,
            "action": decision.action,
            "bet_size": decision.bet_size,
            "pred_prob_win": decision.pred_prob_win,
            "expected_return": decision.expected_return,
            "realized_return": decision.realized_return,
            "mid_price": decision.mid_price,
            "fill_price": decision.fill_price,
            "realized_slippage_bps": decision.realized_slippage_bps,
            "alpha_decay_zscore": decision.alpha_decay_zscore,
            "alpha_decay_flag": decision.alpha_decay_flag,
        }
        _append_json_record(self.config.shadow_log_path, record)
        _append_json_record(
            self.config.slippage_audit_path,
            {
                "timestamp": decision.timestamp,
                "symbol": decision.symbol,
                "action": decision.action,
                "mid_price": decision.mid_price,
                "fill_price": decision.fill_price,
                "realized_slippage_bps": decision.realized_slippage_bps,
            },
        )
        if alpha_decay_flag:
            _append_json_record(
                self.config.alpha_decay_alert_path,
                {
                    "timestamp": decision.timestamp,
                    "symbol": decision.symbol,
                    "zscore": decision.alpha_decay_zscore,
                    "threshold": float(self.config.alpha_decay_z_threshold),
                    "status": "alpha_decay_flagged",
                },
            )
        return decision


def run_shadow_trade(
    config: ShadowTradeConfig = ShadowTradeConfig(),
) -> pd.DataFrame:
    logger = ShadowTradeLogger(config)
    decision = logger.process_latest_signal()
    if decision is None:
        return pd.DataFrame()
    return pd.DataFrame([decision.__dict__])
