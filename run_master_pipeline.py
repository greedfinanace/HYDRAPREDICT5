from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl

from quant_stack.module0 import (
    DownloaderConfig,
    MarketDataDownloader,
    canonical_symbol,
    check_data_integrity,
    save_incremental_parquet,
)
from quant_stack.module1 import Module1Config, build_module1_dataset
from quant_stack.module2 import Module2Config, build_module2_features
from quant_stack.module3 import CPCVConfig, Module3Config, run_alpha_module
from quant_stack.module4 import Module4Config, run_meta_module
from quant_stack.module5 import Module5Config, run_backtest
from quant_stack.module6 import RegimeDetectorConfig, build_regime_feature_frame
from quant_stack.module10 import AlertNotifier, NotificationConfig

PRODUCT_NAME = "HydraPredict 5"

STEP_ORDER = (
    "module0.fetch_data",
    "module07.clean_data",
    "module1.build_volume_bars",
    "module2.generate_features",
    "module65.detect_regimes",
    "module34.train_models",
    "module5.run_backtest",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _artifact_row_count(artifact: object, *names: str) -> int:
    for name in names:
        value = getattr(artifact, name, None)
        if value is None:
            continue
        if hasattr(value, "height"):
            return int(value.height)
        if hasattr(value, "shape"):
            return int(value.shape[0])
    return 0


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_backtest_pdf(report_dir: Path, title: str, tearsheet: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / f"backtest_report_{_utc_stamp()}.pdf"
    meta_metrics = tearsheet["strategies"]["meta_strategy"]
    alpha_metrics = tearsheet["strategies"]["alpha_raw"]

    fig = plt.figure(figsize=(8.5, 11))
    lines = [
        title,
        "",
        f"Generated UTC: {_utc_now().isoformat()}",
        "",
        "Primary Engine",
        f"Sharpe: {meta_metrics['sharpe']:.4f}",
        f"Sortino: {meta_metrics['sortino']:.4f}",
        f"Max Drawdown: {meta_metrics['max_drawdown']:.4f}",
        f"Profit Factor: {meta_metrics['profit_factor']:.4f}",
        "",
        "Secondary Engine",
        f"Sharpe: {alpha_metrics['sharpe']:.4f}",
        f"Max Drawdown: {alpha_metrics['max_drawdown']:.4f}",
    ]
    fig.text(0.08, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    return pdf_path


def _written_file_footprint(paths: list[Path]) -> list[dict[str, Any]]:
    footprint: list[dict[str, Any]] = []
    for path in sorted((Path(path) for path in paths), key=lambda value: str(value).lower()):
        if path.exists():
            stat = path.stat()
            footprint.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        else:
            footprint.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": -1,
                    "mtime_ns": -1,
                }
            )
    return footprint


def _build_data_signature(
    *,
    provider: str,
    symbols: tuple[str, ...],
    fetched: pl.DataFrame,
    written_paths: list[Path],
) -> dict[str, Any]:
    latest_ts: str | None = None
    if "timestamp" in fetched.columns and fetched.height > 0:
        value = fetched["timestamp"].max()
        if value is not None:
            latest_ts = value.isoformat()

    payload: dict[str, Any] = {
        "provider": provider.lower(),
        "symbols": sorted(str(symbol).upper() for symbol in symbols),
        "row_count": int(fetched.height),
        "latest_timestamp": latest_ts,
        "written_files": _written_file_footprint(written_paths),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "hash": digest,
        "payload": payload,
    }


def _flow_mermaid(run_timestamp: str, step_status: dict[str, str]) -> str:
    lines = [
        "flowchart TD",
        f'    run["run_timestamp\\n{run_timestamp}"]',
    ]
    node_ids: list[str] = []
    for idx, step in enumerate(STEP_ORDER):
        node_id = f"s{idx}"
        node_ids.append(node_id)
        status = step_status.get(step, "pending")
        lines.append(f'    {node_id}["{step}\\nstatus: {status}"]')

    if node_ids:
        lines.append(f"    run --> {node_ids[0]}")
        for left, right in zip(node_ids[:-1], node_ids[1:], strict=True):
            lines.append(f"    {left} --> {right}")
    return "\n".join(lines) + "\n"


def _status_color(status: str) -> str:
    if status == "success":
        return "#d9ead3"
    if status == "failed":
        return "#f4cccc"
    if status.startswith("skipped"):
        return "#d9d9d9"
    return "#fff2cc"


def _write_flow_png(path: Path, run_timestamp: str, step_status: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.5,
        0.97,
        f"Master Pipeline Flow\n{run_timestamp}",
        ha="center",
        va="top",
        fontsize=12,
        family="monospace",
    )
    top = 0.88
    step_gap = 0.11
    for idx, step in enumerate(STEP_ORDER):
        y = top - idx * step_gap
        status = step_status.get(step, "pending")
        ax.text(
            0.5,
            y,
            f"{step}\nstatus: {status}",
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": _status_color(status), "edgecolor": "black"},
        )
        if idx < len(STEP_ORDER) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - 0.045),
                xytext=(0.5, y - 0.07),
                arrowprops={"arrowstyle": "->", "linewidth": 1.2},
            )
    fig.savefig(path, format="png", dpi=160)
    plt.close(fig)
    return path


def _write_pipeline_flow_artifacts(
    *,
    artifacts_root: Path,
    run_label: str,
    run_timestamp: str,
    step_status: dict[str, str],
) -> dict[str, Path]:
    flow_root = artifacts_root / "flows"
    flow_root.mkdir(parents=True, exist_ok=True)
    mmd_path = flow_root / f"pipeline_flow_{run_label}.mmd"
    png_path = flow_root / f"pipeline_flow_{run_label}.png"
    mmd_path.write_text(_flow_mermaid(run_timestamp, step_status), encoding="utf-8")
    _write_flow_png(png_path, run_timestamp, step_status)
    return {"mermaid": mmd_path, "png": png_path}


def run_master_pipeline(
    *,
    provider: str = "binance",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    lookback_days: int = 7,
    source_root: Path | None = None,
    source_format: str | None = None,
    market_data_root: Path = Path("market_data"),
    artifacts_root: Path = Path("artifacts/master_pipeline"),
    state_path: Path = Path("artifacts/master_pipeline/pipeline_state.json"),
    best_params_path: Path | None = None,
    notifier_config: NotificationConfig | None = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    input_bar_minutes = 60 if source_root is not None and str(source_format or "").strip().lower() == "txt" else 1
    notifier = AlertNotifier(notifier_config or NotificationConfig())
    state = _load_state(state_path)
    run_label = _utc_stamp()
    run_timestamp = _utc_now().isoformat()
    step_status = {step: "pending" for step in STEP_ORDER}
    try:
        if source_root is None:
            downloader = MarketDataDownloader(
                DownloaderConfig(
                    provider=provider,
                    symbols=symbols,
                    lookback_days=lookback_days,
                    output_root=market_data_root,
                )
            )
            raw_fetched = downloader.fetch_raw(lookback_days=lookback_days)
            step_status["module0.fetch_data"] = "success"
            if raw_fetched.is_empty():
                step_status["module0.fetch_data"] = "failed"
                raise RuntimeError("module0.fetch_data returned empty payload.")

            raw_integrity = check_data_integrity(raw_fetched)
            if not raw_integrity["ok"]:
                step_status["module07.clean_data"] = "failed"
                raise RuntimeError(f"module0.raw_integrity failure: {raw_integrity}")

            fetched = downloader.sanitize(raw_fetched)
            written = save_incremental_parquet(fetched, market_data_root)
            if fetched.is_empty() or not written:
                step_status["module07.clean_data"] = "failed"
                raise RuntimeError("module07.clean_data produced no persisted bars.")

            integrity = check_data_integrity(fetched)
            step_status["module07.clean_data"] = "success" if integrity["ok"] else "failed"
            if not integrity["ok"]:
                raise RuntimeError(f"module07.clean_data integrity failure: {integrity}")
        else:
            raw_fetched = pl.DataFrame()
            fetched = pl.DataFrame()
            raw_integrity = {"ok": True, "reason": "local_source_mode", "gap_count": 0, "zero_volume_bars": 0, "raw_zero_volume_bars": 0, "imputed_zero_volume_bars": 0}
            integrity = raw_integrity
            written = []
            step_status["module0.fetch_data"] = "skipped_local_source"
            step_status["module07.clean_data"] = "skipped_local_source"

        data_signature = _build_data_signature(
            provider=provider,
            symbols=symbols,
            fetched=fetched,
            written_paths=[Path(path) for path in written],
        )
        previous_signature = state.get("data_signature", {})
        previous_hash = str(previous_signature.get("hash", "")) if isinstance(previous_signature, dict) else ""
        if source_root is None and not force_retrain and previous_hash == data_signature["hash"]:
            for step in STEP_ORDER[2:]:
                step_status[step] = "skipped_no_new_data"
            flow_paths = _write_pipeline_flow_artifacts(
                artifacts_root=artifacts_root,
                run_label=run_label,
                run_timestamp=run_timestamp,
                step_status=step_status,
            )
            return {
                "product_name": PRODUCT_NAME,
                "timestamp": run_timestamp,
                "status": "skipped_no_new_data",
                "run_mode": "local_source" if source_root is not None else "live_fetch",
                "provider": provider,
                "symbols": list(symbols),
                "raw_rows_downloaded": int(raw_fetched.height),
                "rows_downloaded": int(fetched.height),
                "module1_source_rows": _artifact_row_count(module1 if "module1" in locals() else fetched, "bars", "stationary"),
                "module1_stationary_rows": _artifact_row_count(module1 if "module1" in locals() else fetched, "stationary", "bars"),
                "module1_event_rows": _artifact_row_count(module1 if "module1" in locals() else fetched, "events"),
                "module1_label_rows": _artifact_row_count(module1 if "module1" in locals() else fetched, "labels"),
                "integrity_raw": raw_integrity,
                "integrity_cleaned": integrity,
                "data_signature": data_signature,
                "state_path": str(state_path),
                "pipeline_flow_mermaid": str(flow_paths["mermaid"]),
                "pipeline_flow_png": str(flow_paths["png"]),
                "last_meta_sharpe": state.get("last_meta_sharpe"),
                "last_train_accuracy": state.get("last_train_accuracy"),
            }

        module1 = build_module1_dataset(
            Module1Config(
                source_root=Path(source_root) if source_root is not None else market_data_root,
                source_format=source_format or "parquet",
                symbols=tuple(canonical_symbol(s) for s in symbols),
                bar_minutes=input_bar_minutes,
                sampling_mode="volume",
                output_root=artifacts_root / "module1",
                vertical_barrier_days=3.0,
            )
        )
        step_status["module1.build_volume_bars"] = "success"

        stationary_pd = module1.stationary.to_pandas()
        module2_features = build_module2_features(stationary_pd, Module2Config())
        module2_path = artifacts_root / "module2_features.parquet"
        module2_path.parent.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(module2_features).write_parquet(module2_path)
        step_status["module2.generate_features"] = "success"

        regime_columns = [
            "instrument_id",
            "timestamp",
            "m2_regime_state",
            "m2_regime_label",
            "m2_regime_penalty",
            "feat_regime_bull",
            "feat_regime_bear",
            "feat_regime_sideways",
        ]
        if all(column in module2_features.columns for column in regime_columns):
            regimes = module2_features.loc[:, regime_columns].copy()
        else:
            regimes = build_regime_feature_frame(stationary_pd, RegimeDetectorConfig())
        regimes_path = artifacts_root / "module65_regimes.parquet"
        pl.from_pandas(regimes).write_parquet(regimes_path)
        step_status["module65.detect_regimes"] = "success"

        module3_config = Module3Config(optimized_params_path=best_params_path)
        if source_root is not None:
            module3_config = Module3Config(
                optimized_params_path=best_params_path,
                cpcv=CPCVConfig(n_groups=3, test_groups=1, min_train_samples=20, min_test_samples=10),
            )
        module3 = run_alpha_module(module1, module3_config)
        module4_config = Module4Config(optimized_params_path=best_params_path)
        if source_root is not None:
            module4_config = Module4Config(
                optimized_params_path=best_params_path,
                cpcv=CPCVConfig(n_groups=3, test_groups=1, min_train_samples=20, min_test_samples=10),
            )
        module4 = run_meta_module(module3, module4_config)
        step_status["module34.train_models"] = "success"

        module5_config = Module5Config(output_root=artifacts_root / "module5")
        if source_root is not None and symbols:
            module5_config = Module5Config(
                benchmark_symbol=canonical_symbol(symbols[0]),
                output_root=artifacts_root / "module5",
            )
        module5 = run_backtest(
            module4.position_sizing,
            module1.stationary,
            module5_config,
        )
        step_status["module5.run_backtest"] = "success"

        meta_metrics = module5.tearsheet["strategies"]["meta_strategy"]
        report_pdf = _write_backtest_pdf(
            artifacts_root / "reports",
            f"{PRODUCT_NAME} Automated Loop Report",
            module5.tearsheet,
        )
        flow_paths = _write_pipeline_flow_artifacts(
            artifacts_root=artifacts_root,
            run_label=run_label,
            run_timestamp=run_timestamp,
            step_status=step_status,
        )
        out = {
            "product_name": PRODUCT_NAME,
            "timestamp": run_timestamp,
            "status": "completed",
            "run_mode": "local_source" if source_root is not None else "live_fetch",
            "provider": provider,
            "symbols": list(symbols),
            "raw_rows_downloaded": int(raw_fetched.height),
            "rows_downloaded": int(fetched.height),
            "module1_source_rows": _artifact_row_count(module1, "bars", "stationary"),
            "module1_stationary_rows": _artifact_row_count(module1, "stationary", "bars"),
            "module1_event_rows": _artifact_row_count(module1, "events"),
            "module1_label_rows": _artifact_row_count(module1, "labels"),
            "integrity_raw": raw_integrity,
            "integrity_cleaned": integrity,
            "meta_sharpe": float(meta_metrics["sharpe"]),
            "meta_max_drawdown": float(meta_metrics["max_drawdown"]),
            "meta_profit_factor": float(meta_metrics["profit_factor"]),
            "report_pdf": str(report_pdf),
            "pipeline_flow_mermaid": str(flow_paths["mermaid"]),
            "pipeline_flow_png": str(flow_paths["png"]),
            "data_signature": data_signature,
        }

        state.update(
            {
                "last_fetch_time": out["timestamp"],
                "last_train_accuracy": float(module3.overall_metrics.get("accuracy", 0.0)),
                "last_meta_sharpe": out["meta_sharpe"],
                "last_rows_downloaded": out["rows_downloaded"],
                "data_signature": data_signature,
            }
        )
        _save_state(state_path, state)

        notifier.send_backtest_summary(
            run_name="master_pipeline",
            sharpe=out["meta_sharpe"],
            max_drawdown=out["meta_max_drawdown"],
            profit_factor=out["meta_profit_factor"],
        )
        return out
    except Exception as exc:
        if all(step_status[step] != "failed" for step in STEP_ORDER):
            for step in STEP_ORDER:
                if step_status[step] == "pending":
                    step_status[step] = "failed"
                    break
        flow_error: Exception | None = None
        try:
            _write_pipeline_flow_artifacts(
                artifacts_root=artifacts_root,
                run_label=run_label,
                run_timestamp=run_timestamp,
                step_status=step_status,
            )
        except Exception as flow_exc:
            flow_error = flow_exc
        tb = traceback.format_exc()
        alerts = [f"pipeline_failure: {type(exc).__name__}", tb[-400:]]
        if flow_error is not None:
            alerts.append(f"pipeline_flow_artifact_failure: {type(flow_error).__name__}: {flow_error}")
        notifier.send_summary(
            daily_pnl=0.0,
            alerts=alerts,
        )
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {PRODUCT_NAME} master pipeline.")
    parser.add_argument("--provider", default="binance", choices=["binance", "alpaca", "yfinance"])
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"])
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-format", type=str, default=None)
    parser.add_argument("--market-data-root", type=Path, default=Path("market_data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/master_pipeline"))
    parser.add_argument("--state-path", type=Path, default=Path("artifacts/master_pipeline/pipeline_state.json"))
    parser.add_argument("--best-params-path", type=Path, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        out = run_master_pipeline(
            provider=args.provider,
            symbols=tuple(args.symbols),
            lookback_days=args.lookback_days,
            source_root=args.source_root,
            source_format=args.source_format,
            market_data_root=args.market_data_root,
            artifacts_root=args.artifacts_root,
            state_path=args.state_path,
            best_params_path=args.best_params_path,
            force_retrain=bool(args.force_retrain),
        )
        print(json.dumps(out, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
