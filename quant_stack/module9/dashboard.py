from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_stack.json_records import read_json_records


DEFAULT_LOG_PATH = Path("artifacts/module7/live_signals_log.json")
DEFAULT_KILL_SWITCH_PATH = Path("artifacts/module9/kill_switch.flag")
DEFAULT_SQLITE_PATH = Path("artifacts/module9/live_signals.db")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "timestamp" in normalized:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    if "bar_timestamp" in normalized:
        normalized["bar_timestamp"] = pd.to_datetime(normalized["bar_timestamp"], utc=True)
    return normalized


def load_live_signals_log(log_path: str | Path = DEFAULT_LOG_PATH, limit: int | None = None) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    records = read_json_records(path, limit=limit)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame([_normalize_record(record) for record in records])
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def migrate_signals_json_to_sqlite(
    json_path: str | Path = DEFAULT_LOG_PATH,
    sqlite_path: str | Path = DEFAULT_SQLITE_PATH,
) -> int:
    frame = load_live_signals_log(json_path, limit=None)
    if frame.empty:
        return 0
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    frame_to_store = frame.copy()
    for column in ("timestamp", "bar_timestamp"):
        if column in frame_to_store.columns:
            frame_to_store[column] = pd.to_datetime(frame_to_store[column], utc=True).astype(str)
    def _dedupe_key(record: dict[str, Any]) -> str:
        canonical = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS live_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dedupe_key TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL
            )
            """
        )
        existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(live_signals)").fetchall()]
        if "dedupe_key" not in existing_columns:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_signals_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                )
                """
            )
            legacy_rows = conn.execute("SELECT payload FROM live_signals").fetchall()
            legacy_payloads: list[tuple[str, str]] = []
            for row in legacy_rows:
                try:
                    payload_obj = json.loads(str(row[0]))
                except Exception:
                    continue
                if not isinstance(payload_obj, dict):
                    continue
                legacy_payloads.append((_dedupe_key(payload_obj), json.dumps(payload_obj, default=str)))
            if legacy_payloads:
                conn.executemany(
                    "INSERT OR IGNORE INTO live_signals_v2(dedupe_key, payload) VALUES (?, ?)",
                    legacy_payloads,
                )
            conn.execute("DROP TABLE live_signals")
            conn.execute("ALTER TABLE live_signals_v2 RENAME TO live_signals")
        payloads = [
            (_dedupe_key(record), json.dumps(record, default=str))
            for record in frame_to_store.to_dict(orient="records")
        ]
        before_changes = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO live_signals(dedupe_key, payload) VALUES (?, ?)", payloads)
        inserted = int(conn.total_changes - before_changes)
    return inserted


def load_live_signals_sqlite(
    sqlite_path: str | Path = DEFAULT_SQLITE_PATH,
    limit: int = 5000,
) -> pd.DataFrame:
    db_path = Path(sqlite_path)
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT payload FROM live_signals ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    records = [json.loads(row[0]) for row in reversed(rows)]
    frame = pd.DataFrame([_normalize_record(record) for record in records])
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _regime_color(regime: str) -> str:
    mapping = {
        "bull": "#e6f7ea",
        "bear": "#fde8e8",
        "sideways": "#e8eefc",
    }
    return mapping.get(str(regime).lower(), "#f5f5f5")


def _parse_feature_importance(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame(columns=["feature", "importance"])
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                items = list(parsed.items())
            else:
                items = []
        except json.JSONDecodeError:
            items = []
    else:
        items = []
    frame = pd.DataFrame(items, columns=["feature", "importance"])
    if frame.empty:
        return frame
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0.0)
    return frame.sort_values("importance", ascending=False).head(10)


def _build_equity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "realized_return" in out.columns:
        out["realized_return"] = pd.to_numeric(out["realized_return"], errors="coerce").fillna(0.0)
        out["live_equity"] = (1.0 + out["realized_return"]).cumprod()
    elif "live_equity" in out.columns:
        out["live_equity"] = pd.to_numeric(out["live_equity"], errors="coerce").fillna(method="ffill")
    else:
        out["live_equity"] = np.nan

    if "benchmark_return" in out.columns:
        out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce").fillna(0.0)
        out["benchmark_equity"] = (1.0 + out["benchmark_return"]).cumprod()
    elif "benchmark_equity" in out.columns:
        out["benchmark_equity"] = pd.to_numeric(out["benchmark_equity"], errors="coerce").fillna(method="ffill")
    else:
        out["benchmark_equity"] = np.nan
    return out


def render_dashboard(
    log_path: str | Path = DEFAULT_LOG_PATH,
    kill_switch_path: str | Path = DEFAULT_KILL_SWITCH_PATH,
    sqlite_path: str | Path = DEFAULT_SQLITE_PATH,
) -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st
        from streamlit_autorefresh import st_autorefresh
        import streamlit_authenticator as stauth
    except ImportError as exc:
        raise ImportError(
            "Dashboard dependencies missing. Install streamlit, plotly, streamlit-autorefresh, and streamlit-authenticator."
        ) from exc

    st.set_page_config(page_title="Quant Live Command Center", layout="wide")
    st_autorefresh(interval=10_000, key="quant-dashboard-refresh")

    auth_usernames = st.secrets.get("auth_usernames", [])
    auth_names = st.secrets.get("auth_names", auth_usernames)
    auth_password_hashes = st.secrets.get("auth_password_hashes", [])
    if auth_usernames and auth_password_hashes:
        authenticator = stauth.Authenticate(
            auth_names,
            auth_usernames,
            auth_password_hashes,
            cookie_name="quant_dashboard_cookie",
            cookie_key=st.secrets.get("auth_cookie_key", "quant_dashboard_cookie_key"),
            cookie_expiry_days=float(st.secrets.get("auth_cookie_expiry_days", 1.0)),
        )
        authenticator.login(location="main", fields={"Form name": "Login"})
        status = st.session_state.get("authentication_status")
        if status is False:
            st.error("Invalid username or password.")
            return
        if status is None:
            st.warning("Please log in to access the dashboard.")
            return
        authenticator.logout("Logout", "sidebar")

    st.title("Quant Live Command Center")

    source_mode = st.radio("Signal Source", options=["sqlite", "json"], index=0, horizontal=True)
    if source_mode == "sqlite":
        sqlite_value = st.text_input("SQLite Path", value=str(Path(sqlite_path)))
        if st.button("Migrate JSON -> SQLite"):
            migrated = migrate_signals_json_to_sqlite(json_path=log_path, sqlite_path=sqlite_value)
            st.success(f"Migrated {migrated} rows.")
        log_frame = load_live_signals_sqlite(sqlite_value)
    else:
        path_value = st.text_input("Signals Log Path", value=str(Path(log_path)))
        log_frame = load_live_signals_log(path_value, limit=5000)
    if log_frame.empty:
        st.warning("No signal records found.")
        return

    latest = log_frame.iloc[-1]
    regime_label = str(latest.get("regime_label", "unknown")).lower()
    st.markdown(
        f"""
        <div style="padding:10px;border-radius:6px;background:{_regime_color(regime_label)};">
        Current regime: <strong>{regime_label.upper()}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Final Action", str(latest.get("final_action", "hold")).upper())
    c2.metric("Bet Size", f"{float(latest.get('bet_size', 0.0)):.3f}")
    c3.metric("Pred Prob Win", f"{float(latest.get('pred_prob_win', 0.0)):.3f}")

    pie_frame = pd.DataFrame(
        {
            "side": ["sell", "hold", "buy"],
            "probability": [
                float(latest.get("pred_prob_sell", 0.0)),
                float(latest.get("pred_prob_hold", 0.0)),
                float(latest.get("pred_prob_buy", 0.0)),
            ],
        }
    )
    pie_chart = px.pie(pie_frame, names="side", values="probability", title="Alpha Side Probabilities")
    st.plotly_chart(pie_chart, use_container_width=True)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(latest.get("bet_size", 0.0)),
            title={"text": "Meta Bet Size"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#d94d4d"},
                "steps": [
                    {"range": [0, 0.4], "color": "#f8d7da"},
                    {"range": [0.4, 0.7], "color": "#fce5cd"},
                    {"range": [0.7, 1.0], "color": "#d9ead3"},
                ],
            },
        )
    )
    st.plotly_chart(gauge, use_container_width=True)

    equity_frame = _build_equity_frame(log_frame)
    if "timestamp" in equity_frame.columns:
        line = go.Figure()
        line.add_trace(
            go.Scatter(
                x=equity_frame["timestamp"],
                y=equity_frame["live_equity"],
                name="Live Equity",
                mode="lines",
            )
        )
        line.add_trace(
            go.Scatter(
                x=equity_frame["timestamp"],
                y=equity_frame["benchmark_equity"],
                name="Benchmark",
                mode="lines",
            )
        )
        line.update_layout(title="Live Equity vs Benchmark")
        st.plotly_chart(line, use_container_width=True)

    importance_frame = _parse_feature_importance(latest.get("feature_importance"))
    st.subheader("Top Feature Drivers")
    if importance_frame.empty:
        st.info("No feature importance snapshot in current signal log.")
    else:
        st.dataframe(importance_frame, use_container_width=True)

    kill_path = Path(kill_switch_path)
    if st.button("STOP ALL LIVE ORDERS", type="primary"):
        kill_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": _utc_now_iso(),
            "status": "stopped",
            "reason": "dashboard_kill_switch",
        }
        kill_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        st.error(f"Kill switch enabled: {kill_path}")

    preview_columns = [col for col in ["timestamp", "symbol", "final_action", "bet_size", "pred_prob_win", "regime_label"] if col in log_frame.columns]
    st.subheader("Recent Decisions")
    st.dataframe(log_frame[preview_columns].tail(50), use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
