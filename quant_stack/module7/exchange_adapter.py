from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quant_stack.json_records import append_json_record

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExchangeAdapterConfig:
    provider: str = "ccxtpro"
    venue: str = "binance"
    symbol: str = "BTC/USDT"
    env_path: Path | None = Path(".env")
    dotenv_override: bool = False
    quote_currency: str = "USDT"
    tick_size: float = 0.01
    offset_ticks: int = 2
    price_precision: int | None = None
    min_free_balance: float = 25.0
    max_notional_fraction: float = 0.20
    manual_position_tolerance: float = 1e-8
    latency_alert_ms: float = 500.0
    dry_run: bool = True
    execution_log_path: Path = Path("artifacts/module7/order_execution_log.json")
    network_alert_log_path: Path = Path("artifacts/module7/network_alerts.json")


@dataclass(frozen=True)
class ExchangeCredentials:
    api_key: str
    api_secret: str
    passphrase: str | None = None
    paper: bool = True


class ExchangeClient(Protocol):
    def fetch_reference_price(self, symbol: str) -> float:
        ...

    def fetch_balance(self, quote_currency: str) -> dict[str, float]:
        ...

    def fetch_position_qty(self, symbol: str) -> float:
        ...

    def place_limit_order(self, symbol: str, side: str, quantity: float, limit_price: float) -> dict[str, Any]:
        ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_timestamp(value: datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _safe_append_json(path: Path, payload: dict[str, object]) -> None:
    append_json_record(path, payload)


def load_exchange_credentials(config: ExchangeAdapterConfig) -> ExchangeCredentials:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError("python-dotenv is required for secure credential loading.") from exc

    load_dotenv(dotenv_path=config.env_path, override=config.dotenv_override)
    import os

    provider = config.provider.lower()
    if provider.startswith("alpaca"):
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        paper_flag = os.getenv("ALPACA_PAPER", "true").strip().lower() in {"1", "true", "yes", "on"}
        if not api_key or not api_secret:
            raise ValueError("Missing Alpaca credentials. Expected ALPACA_API_KEY and ALPACA_API_SECRET.")
        return ExchangeCredentials(api_key=api_key, api_secret=api_secret, paper=paper_flag)

    api_key = os.getenv("EXCHANGE_API_KEY")
    api_secret = os.getenv("EXCHANGE_API_SECRET")
    passphrase = os.getenv("EXCHANGE_API_PASSPHRASE")
    if not api_key or not api_secret:
        raise ValueError("Missing exchange credentials. Expected EXCHANGE_API_KEY and EXCHANGE_API_SECRET.")
    return ExchangeCredentials(api_key=api_key, api_secret=api_secret, passphrase=passphrase)


class CCXTProClient:
    def __init__(self, config: ExchangeAdapterConfig, credentials: ExchangeCredentials) -> None:
        provider = config.provider.lower()
        if provider == "ccxtpro":
            module_name = "ccxt.pro"
        else:
            module_name = "ccxt"
        try:
            ccxt_module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"{module_name} is required for provider={config.provider}. "
                "Install ccxt.pro for websocket mode or ccxt for REST mode."
            ) from exc

        exchange_class = getattr(ccxt_module, config.venue, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange venue for {module_name}: {config.venue}")

        params: dict[str, Any] = {
            "apiKey": credentials.api_key,
            "secret": credentials.api_secret,
            "enableRateLimit": True,
        }
        if credentials.passphrase:
            params["password"] = credentials.passphrase
        self.exchange = exchange_class(params)

    def fetch_reference_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        last = ticker.get("last")
        if bid is not None and ask is not None and np.isfinite(bid) and np.isfinite(ask):
            return float((float(bid) + float(ask)) / 2.0)
        if last is None:
            raise ValueError("Exchange ticker did not provide bid/ask/last.")
        return float(last)

    def fetch_balance(self, quote_currency: str) -> dict[str, float]:
        payload = self.exchange.fetch_balance()
        free_map = payload.get("free", {}) if isinstance(payload, dict) else {}
        total_map = payload.get("total", {}) if isinstance(payload, dict) else {}
        free_balance = float(free_map.get(quote_currency, 0.0))
        total_balance = float(total_map.get(quote_currency, free_balance))
        return {
            "free_balance": free_balance,
            "total_balance": total_balance,
        }

    def fetch_position_qty(self, symbol: str) -> float:
        if hasattr(self.exchange, "fetch_positions"):
            try:
                positions = self.exchange.fetch_positions([symbol])
            except Exception:
                logger.warning("fetch_positions failed for symbol=%s on CCXT client.", symbol, exc_info=True)
                positions = []
            for position in positions or []:
                if str(position.get("symbol")) == symbol:
                    contracts = position.get("contracts")
                    if contracts is None:
                        contracts = position.get("positionAmt", 0.0)
                    return float(contracts)
        return 0.0

    def place_limit_order(self, symbol: str, side: str, quantity: float, limit_price: float) -> dict[str, Any]:
        return self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=float(quantity),
            price=float(limit_price),
            params={"timeInForce": "GTC"},
        )


class AlpacaRESTClient:
    def __init__(self, credentials: ExchangeCredentials) -> None:
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import LimitOrderRequest
        except ImportError as exc:
            raise ImportError("alpaca-py is required for provider='alpaca'.") from exc

        self._StockLatestQuoteRequest = StockLatestQuoteRequest
        self._LimitOrderRequest = LimitOrderRequest
        self.trading_client = TradingClient(credentials.api_key, credentials.api_secret, paper=credentials.paper)
        self.data_client = StockHistoricalDataClient(credentials.api_key, credentials.api_secret)

    def fetch_reference_price(self, symbol: str) -> float:
        request = self._StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = self.data_client.get_stock_latest_quote(request)
        quote = quotes[symbol]
        bid = float(getattr(quote, "bid_price", 0.0))
        ask = float(getattr(quote, "ask_price", 0.0))
        if bid > 0.0 and ask > 0.0:
            return (bid + ask) / 2.0
        return max(bid, ask)

    def fetch_balance(self, quote_currency: str) -> dict[str, float]:
        account = self.trading_client.get_account()
        return {
            "free_balance": float(account.cash),
            "total_balance": float(account.equity),
        }

    def fetch_position_qty(self, symbol: str) -> float:
        try:
            position = self.trading_client.get_open_position(symbol)
            return float(position.qty)
        except Exception:
            logger.warning("Alpaca get_open_position failed for symbol=%s; assuming flat.", symbol, exc_info=True)
            return 0.0

    def place_limit_order(self, symbol: str, side: str, quantity: float, limit_price: float) -> dict[str, Any]:
        order_request = self._LimitOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=side,
            limit_price=float(limit_price),
            time_in_force="gtc",
        )
        order = self.trading_client.submit_order(order_data=order_request)
        return {"id": str(order.id), "status": str(order.status)}


class ExchangeAdapter:
    def __init__(
        self,
        config: ExchangeAdapterConfig = ExchangeAdapterConfig(),
        client: ExchangeClient | None = None,
    ) -> None:
        self.config = config
        self._cached_position_qty: float | None = None
        if client is not None:
            self.client = client
        else:
            credentials = load_exchange_credentials(config)
            if config.provider.lower().startswith("alpaca"):
                self.client = AlpacaRESTClient(credentials)
            else:
                self.client = CCXTProClient(config, credentials)

    def _round_to_tick(self, price: float, side: str) -> float:
        tick = max(float(self.config.tick_size), 1e-12)
        steps = price / tick
        if side.lower() == "buy":
            rounded = np.ceil(steps) * tick
        else:
            rounded = np.floor(steps) * tick
        rounded = max(float(rounded), tick)
        if self.config.price_precision is not None:
            rounded = round(rounded, int(self.config.price_precision))
        return rounded

    def compute_limit_price(
        self,
        *,
        side: str,
        reference_price: float,
        tick_size: float | None = None,
        offset_ticks: int | None = None,
    ) -> float:
        tick = max(float(tick_size if tick_size is not None else self.config.tick_size), 1e-12)
        offset = abs(int(offset_ticks if offset_ticks is not None else self.config.offset_ticks))
        signed_offset = tick * offset
        side_lower = side.lower()
        if side_lower == "buy":
            raw = float(reference_price) + signed_offset
        elif side_lower == "sell":
            raw = float(reference_price) - signed_offset
        else:
            raise ValueError(f"Unsupported side for limit pricing: {side}")
        return self._round_to_tick(raw, side_lower)

    def sync_positions(
        self,
        *,
        requested_bet_size: float,
        risk_notional: float,
    ) -> dict[str, float | bool]:
        balance = self.client.fetch_balance(self.config.quote_currency)
        free_balance = float(balance.get("free_balance", 0.0))
        total_balance = float(balance.get("total_balance", free_balance))
        live_position_qty = float(self.client.fetch_position_qty(self.config.symbol))

        manual_override_detected = False
        if self._cached_position_qty is not None:
            manual_override_detected = abs(live_position_qty - self._cached_position_qty) > float(
                self.config.manual_position_tolerance
            )
        self._cached_position_qty = live_position_qty

        capacity = max(free_balance - float(self.config.min_free_balance), 0.0) * float(self.config.max_notional_fraction)
        requested_notional = max(float(risk_notional) * float(requested_bet_size), 0.0)
        if risk_notional <= 0.0:
            adjusted_bet_size = 0.0
        elif requested_notional <= 0.0:
            adjusted_bet_size = 0.0
        else:
            adjusted_bet_size = min(requested_notional, capacity) / float(risk_notional)

        return {
            "free_balance": free_balance,
            "total_balance": total_balance,
            "live_position_qty": live_position_qty,
            "manual_override_detected": manual_override_detected,
            "adjusted_bet_size": float(np.clip(adjusted_bet_size, 0.0, 1.0)),
            "capacity_notional": capacity,
            "requested_notional": requested_notional,
        }

    def _log_latency(
        self,
        *,
        signal_generated_at: datetime,
        order_acknowledged_at: datetime,
        context: dict[str, object],
    ) -> float:
        latency_ms = float(
            (_as_utc_timestamp(order_acknowledged_at) - _as_utc_timestamp(signal_generated_at)).total_seconds() * 1000.0
        )
        if latency_ms > float(self.config.latency_alert_ms):
            payload = {
                "timestamp": _utc_now().isoformat(),
                "status": "high_latency",
                "latency_ms": latency_ms,
                "threshold_ms": float(self.config.latency_alert_ms),
                **context,
            }
            _safe_append_json(self.config.network_alert_log_path, payload)
        return latency_ms

    def place_order(
        self,
        *,
        signal_generated_at: datetime,
        action: str,
        bet_size: float,
        risk_notional: float,
        reference_price: float | None = None,
        quantity: float | None = None,
    ) -> dict[str, object]:
        action_lower = action.lower()
        if action_lower not in {"buy", "sell", "hold"}:
            raise ValueError(f"Unsupported action: {action}")
        if action_lower == "hold" or bet_size <= 0.0:
            result = {
                "timestamp": _utc_now().isoformat(),
                "symbol": self.config.symbol,
                "action": "hold",
                "status": "skipped",
                "reason": "no_action",
                "bet_size": 0.0,
            }
            _safe_append_json(self.config.execution_log_path, result)
            return result

        sync = self.sync_positions(
            requested_bet_size=float(bet_size),
            risk_notional=float(risk_notional),
        )
        adjusted_bet_size = float(sync["adjusted_bet_size"])
        if adjusted_bet_size <= 0.0:
            result = {
                "timestamp": _utc_now().isoformat(),
                "symbol": self.config.symbol,
                "action": action_lower,
                "status": "skipped",
                "reason": "insufficient_capacity",
                "bet_size": adjusted_bet_size,
                "manual_override_detected": bool(sync["manual_override_detected"]),
            }
            _safe_append_json(self.config.execution_log_path, result)
            return result

        reference = float(reference_price) if reference_price is not None else float(
            self.client.fetch_reference_price(self.config.symbol)
        )
        limit_price = self.compute_limit_price(side=action_lower, reference_price=reference)
        if quantity is None:
            target_notional = float(risk_notional) * adjusted_bet_size
            order_quantity = max(target_notional / max(reference, 1e-12), 0.0)
        else:
            order_quantity = float(quantity)

        submitted_at = _utc_now()
        if self.config.dry_run:
            order_response = {
                "id": f"dry_run_{submitted_at.timestamp()}",
                "status": "accepted",
                "type": "limit",
            }
        else:
            order_response = self.client.place_limit_order(
                symbol=self.config.symbol,
                side=action_lower,
                quantity=order_quantity,
                limit_price=limit_price,
            )
        acknowledged_at = _utc_now()
        latency_ms = self._log_latency(
            signal_generated_at=signal_generated_at,
            order_acknowledged_at=acknowledged_at,
            context={
                "symbol": self.config.symbol,
                "action": action_lower,
                "dry_run": self.config.dry_run,
            },
        )

        result = {
            "timestamp": acknowledged_at.isoformat(),
            "symbol": self.config.symbol,
            "action": action_lower,
            "status": str(order_response.get("status", "submitted")),
            "order_id": str(order_response.get("id", "")),
            "type": "limit",
            "reference_price": reference,
            "limit_price": limit_price,
            "tick_size": float(self.config.tick_size),
            "offset_ticks": int(self.config.offset_ticks),
            "requested_bet_size": float(bet_size),
            "adjusted_bet_size": adjusted_bet_size,
            "quantity": order_quantity,
            "risk_notional": float(risk_notional),
            "manual_override_detected": bool(sync["manual_override_detected"]),
            "signal_generated_at": _as_utc_timestamp(signal_generated_at).isoformat(),
            "order_acknowledged_at": acknowledged_at.isoformat(),
            "latency_ms": latency_ms,
            "dry_run": self.config.dry_run,
        }
        _safe_append_json(self.config.execution_log_path, result)
        return result
