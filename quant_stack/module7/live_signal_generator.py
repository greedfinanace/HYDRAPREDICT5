from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_stack.json_records import append_json_record
from quant_stack.module7.live_ingestor import LiveVolumeBar


@dataclass(frozen=True)
class OMSConfig:
    min_free_balance: float = 25.0
    max_notional_fraction: float = 0.20
    min_order_notional: float = 10.0
    wash_trade_cooldown_seconds: int = 60
    signals_log_path: Path = Path("artifacts/module7/live_signals_log.json")


@dataclass(frozen=True)
class AccountSnapshot:
    timestamp: datetime
    total_balance: float
    free_balance: float
    position_qty: float
    mark_price: float


@dataclass(frozen=True)
class LiveSignalConfig:
    best_params_path: Path = Path("artifacts/module6_hpo/best_params.json")
    alpha_model_path: Path = Path("artifacts/module7/alpha_model.pkl")
    meta_model_path: Path = Path("artifacts/module7/meta_model.pkl")
    meta_probability_floor: float = 0.50
    max_bet_size: float = 1.0
    alpha_class_order: tuple[int, ...] = (-1, 0, 1)
    symbol: str = "BTC/USDT"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_timestamp(value: datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload
    except Exception:
        return default


def _average_model_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "trained_models"):
        predictions = []
        for fold_model in model.trained_models:
            estimator = getattr(fold_model, "estimator", fold_model)
            probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
            if probabilities.ndim == 1:
                probabilities = np.column_stack([1.0 - probabilities, probabilities])
            predictions.append(probabilities)
        if not predictions:
            raise ValueError("No trained models found in loaded artifact.")
        return np.mean(np.stack(predictions, axis=0), axis=0)

    if isinstance(model, Sequence) and not isinstance(model, (str, bytes, bytearray)):
        predictions = []
        for estimator in model:
            probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
            if probabilities.ndim == 1:
                probabilities = np.column_stack([1.0 - probabilities, probabilities])
            predictions.append(probabilities)
        if predictions:
            return np.mean(np.stack(predictions, axis=0), axis=0)

    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    return probabilities


def _infer_classes(model: Any, fallback: Sequence[int]) -> list[int]:
    if hasattr(model, "classes_"):
        return [int(value) for value in np.asarray(model.classes_).tolist()]
    if hasattr(model, "trained_models") and model.trained_models:
        first = model.trained_models[0]
        available = getattr(first, "available_classes", None)
        if available:
            return [int(value) for value in available]
    return [int(value) for value in fallback]


class OrderManagementSafetyLayer:
    def __init__(self, config: OMSConfig = OMSConfig()) -> None:
        self.config = config
        self._last_order_by_symbol: dict[str, dict[str, object]] = {}

    def _log(self, payload: dict[str, object]) -> None:
        append_json_record(self.config.signals_log_path, payload)

    def validate(
        self,
        *,
        symbol: str,
        action: str,
        bet_size: float,
        account: AccountSnapshot,
    ) -> tuple[bool, str, float]:
        if action == "hold" or bet_size <= 0.0:
            return False, "no_action", 0.0

        if account.free_balance < self.config.min_free_balance:
            return False, "insufficient_free_balance", 0.0

        target_notional = account.free_balance * self.config.max_notional_fraction * float(np.clip(bet_size, 0.0, 1.0))
        if target_notional < self.config.min_order_notional:
            return False, "below_min_notional", target_notional

        previous = self._last_order_by_symbol.get(symbol)
        if previous is not None:
            previous_action = str(previous["action"])
            previous_time = _as_utc_timestamp(str(previous["timestamp"]))
            now = _as_utc_timestamp(account.timestamp)
            if previous_action != action and now - previous_time <= pd.Timedelta(seconds=self.config.wash_trade_cooldown_seconds):
                return False, "wash_trade_cooldown", target_notional

        return True, "approved", target_notional

    def register_order(self, symbol: str, action: str, timestamp: datetime, notional: float) -> None:
        self._last_order_by_symbol[symbol] = {
            "action": action,
            "timestamp": _as_utc_timestamp(timestamp).isoformat(),
            "notional": float(notional),
        }

    def log_decision(self, payload: dict[str, object]) -> None:
        self._log(payload)


class LiveSignalGenerator:
    def __init__(
        self,
        config: LiveSignalConfig = LiveSignalConfig(),
        oms: OrderManagementSafetyLayer | None = None,
    ) -> None:
        self.config = config
        self.oms = oms or OrderManagementSafetyLayer()
        self.best_params = _safe_json_load(config.best_params_path, default={})
        if not config.alpha_model_path.exists():
            raise FileNotFoundError(f"Alpha model file not found: {config.alpha_model_path}")
        if not config.meta_model_path.exists():
            raise FileNotFoundError(f"Meta model file not found: {config.meta_model_path}")
        with config.alpha_model_path.open("rb") as handle:
            self.alpha_model = pickle.load(handle)
        with config.meta_model_path.open("rb") as handle:
            self.meta_model = pickle.load(handle)
        self.alpha_classes = _infer_classes(self.alpha_model, self.config.alpha_class_order)

    def _bet_size(self, probability_win: float) -> float:
        floor = float(np.clip(self.config.meta_probability_floor, 0.0, 0.999999))
        if probability_win <= floor:
            return 0.0
        scaled = (probability_win - floor) / (1.0 - floor)
        return float(np.clip(scaled, 0.0, self.config.max_bet_size))

    def build_feature_frame(
        self,
        volume_bar: LiveVolumeBar,
        extra_features: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        row = {
            "feat_fracdiff_return": volume_bar.fracdiff_return,
            "feat_raw_log_return": volume_bar.raw_log_return,
            "feat_sigma_t": volume_bar.sigma_t,
            "m2_regime_state": float(volume_bar.regime_state),
            "m2_regime_penalty": float(volume_bar.regime_penalty),
            "feat_regime_bull": 1.0 if volume_bar.regime_label == "bull" else 0.0,
            "feat_regime_bear": 1.0 if volume_bar.regime_label == "bear" else 0.0,
            "feat_regime_sideways": 1.0 if volume_bar.regime_label == "sideways" else 0.0,
            "trgt": max(float(volume_bar.sigma_t), 0.0),
        }
        if extra_features:
            for key, value in extra_features.items():
                row[key] = float(value)
        return pd.DataFrame([row])

    def generate_signal(
        self,
        volume_bar: LiveVolumeBar,
        account: AccountSnapshot,
        extra_features: dict[str, float] | None = None,
    ) -> dict[str, object]:
        feature_frame = self.build_feature_frame(volume_bar, extra_features=extra_features)

        alpha_probabilities = _average_model_probability(self.alpha_model, feature_frame)
        if alpha_probabilities.shape[1] != len(self.alpha_classes):
            # Fallback: align to class order length.
            alpha_probabilities = alpha_probabilities[:, : len(self.alpha_classes)]
        alpha_row = alpha_probabilities[0]
        alpha_side = int(self.alpha_classes[int(np.argmax(alpha_row))])

        meta_probabilities = _average_model_probability(self.meta_model, feature_frame)
        meta_row = meta_probabilities[0]
        if meta_row.shape[0] == 1:
            pred_prob_win = float(meta_row[0])
        else:
            pred_prob_win = float(meta_row[-1])
        pred_prob_win = float(np.clip(pred_prob_win, 0.0, 1.0))

        raw_bet_size = self._bet_size(pred_prob_win)
        bet_size = raw_bet_size * float(np.clip(volume_bar.regime_penalty, 0.0, 1.0))
        if alpha_side == 0 or bet_size <= 0.0:
            action = "hold"
            bet_size = 0.0
        else:
            action = "buy" if alpha_side > 0 else "sell"

        approved, reason, notional = self.oms.validate(
            symbol=self.config.symbol,
            action=action,
            bet_size=bet_size,
            account=account,
        )
        if not approved:
            action = "hold"
            bet_size = 0.0
        else:
            self.oms.register_order(
                symbol=self.config.symbol,
                action=action,
                timestamp=account.timestamp,
                notional=notional,
            )

        decision = {
            "timestamp": _utc_now().isoformat(),
            "symbol": self.config.symbol,
            "bar_timestamp": volume_bar.timestamp.isoformat(),
            "final_action": action,
            "pred_side": alpha_side,
            "bet_size": float(np.clip(bet_size, 0.0, 1.0)),
            "pred_prob_win": pred_prob_win,
            "pred_prob_buy": float(alpha_row[self.alpha_classes.index(1)]) if 1 in self.alpha_classes else 0.0,
            "pred_prob_hold": float(alpha_row[self.alpha_classes.index(0)]) if 0 in self.alpha_classes else 0.0,
            "pred_prob_sell": float(alpha_row[self.alpha_classes.index(-1)]) if -1 in self.alpha_classes else 0.0,
            "regime_state": int(volume_bar.regime_state),
            "regime_label": volume_bar.regime_label,
            "regime_penalty": float(volume_bar.regime_penalty),
            "free_balance": float(account.free_balance),
            "mark_price": float(account.mark_price),
            "order_notional": float(notional),
            "safety_reason": reason,
        }
        self.oms.log_decision(decision)
        return decision


def emergency_stop(oms_config: OMSConfig, reason: str = "manual_kill_switch") -> Path:
    kill_path = oms_config.signals_log_path.parent / "kill_switch.flag"
    payload = {
        "timestamp": _utc_now().isoformat(),
        "status": "stopped",
        "reason": reason,
    }
    kill_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return kill_path
