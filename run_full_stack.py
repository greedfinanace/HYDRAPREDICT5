from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

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
from quant_stack.module3 import CPCVConfig, Module3Config, run_alpha_module
from quant_stack.module4 import Module4Config, run_meta_module
from quant_stack.module5 import Module5Config, run_backtest
from quant_stack.module10 import AlertNotifier, NotificationConfig

PRODUCT_NAME = "HydraPredict 5"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _write_backtest_pdf(report_dir: Path, title: str, tearsheet: dict[str, object]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / f"backtest_report_{_utc_stamp()}.pdf"
    strategies = tearsheet["strategies"] if isinstance(tearsheet, dict) else {}
    meta_metrics = strategies.get("meta_strategy", {}) if isinstance(strategies, dict) else {}
    alpha_metrics = strategies.get("alpha_raw", {}) if isinstance(strategies, dict) else {}

    fig = plt.figure(figsize=(8.5, 11))
    lines = [
        title,
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Primary Engine",
        f"Sharpe: {float(meta_metrics.get('sharpe', 0.0)):.4f}",
        f"Sortino: {float(meta_metrics.get('sortino', 0.0)):.4f}",
        f"Max Drawdown: {float(meta_metrics.get('max_drawdown', 0.0)):.4f}",
        f"Profit Factor: {float(meta_metrics.get('profit_factor', 0.0)):.4f}",
        "",
        "Secondary Engine",
        f"Sharpe: {float(alpha_metrics.get('sharpe', 0.0)):.4f}",
        f"Max Drawdown: {float(alpha_metrics.get('max_drawdown', 0.0)):.4f}",
    ]
    fig.text(0.08, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    return pdf_path


def run_full_stack(
    *,
    provider: str,
    symbols: tuple[str, ...],
    lookback_days: int,
    source_root: Path | None = None,
    source_format: str | None = None,
    market_data_root: Path,
    artifacts_root: Path,
    best_params_path: Path | None = None,
    notifier_config: NotificationConfig | None = None,
) -> dict[str, object]:
    input_bar_minutes = 60 if source_root is not None and str(source_format or "").strip().lower() == "txt" else 1
    if source_root is None:
        downloader = MarketDataDownloader(
            DownloaderConfig(
                provider=provider,
                symbols=symbols,
                lookback_days=lookback_days,
                output_root=market_data_root,
            )
        )
        raw_downloaded = downloader.fetch_raw(lookback_days=lookback_days)
        if raw_downloaded.is_empty():
            raise RuntimeError("Downloader returned no data; aborting train/backtest.")

        raw_integrity = check_data_integrity(raw_downloaded)
        if not bool(raw_integrity.get("ok", False)):
            raise RuntimeError(f"Raw data integrity check failed: {raw_integrity}")

        downloaded = downloader.sanitize(raw_downloaded)
        written = save_incremental_parquet(downloaded, market_data_root)
        if downloaded.is_empty() or not written:
            raise RuntimeError("Sanitized dataset produced no persisted bars; aborting train/backtest.")

        cleaned_integrity = check_data_integrity(downloaded)
        if not bool(cleaned_integrity.get("ok", False)):
            raise RuntimeError(f"Post-clean integrity check failed: {cleaned_integrity}")
    else:
        raw_downloaded = pl.DataFrame()
        downloaded = pl.DataFrame()
        raw_integrity = {"ok": True, "reason": "local_source_mode", "gap_count": 0, "zero_volume_bars": 0, "raw_zero_volume_bars": 0, "imputed_zero_volume_bars": 0}
        cleaned_integrity = raw_integrity
        written = []

    module1_config = Module1Config(
        source_root=Path(source_root) if source_root is not None else market_data_root,
        source_format=source_format or "parquet",
        symbols=tuple(canonical_symbol(s) for s in symbols),
        bar_minutes=input_bar_minutes,
        sampling_mode="volume",
        output_root=artifacts_root / "module1",
        vertical_barrier_days=3.0,
    )
    module1 = build_module1_dataset(module1_config)

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
    module4 = run_meta_module(
        module3,
        module4_config,
    )
    module5_config = Module5Config(output_root=artifacts_root / "module5")
    if source_root is not None and symbols:
        module5_config = Module5Config(
            benchmark_symbol=canonical_symbol(symbols[0]),
            output_root=artifacts_root / "module5",
        )
    backtest = run_backtest(
        module4.position_sizing,
        module1.stationary,
        module5_config,
    )
    metrics = backtest.tearsheet["strategies"]["meta_strategy"]
    out = {
        "product_name": PRODUCT_NAME,
        "timestamp": _utc_stamp(),
        "run_mode": "local_source" if source_root is not None else "live_fetch",
        "provider": provider,
        "symbols": list(symbols),
        "lookback_days": lookback_days,
        "raw_rows_downloaded": int(raw_downloaded.height),
        "rows_downloaded": int(downloaded.height),
        "module1_source_rows": _artifact_row_count(module1, "bars", "stationary"),
        "module1_stationary_rows": _artifact_row_count(module1, "stationary", "bars"),
        "module1_event_rows": _artifact_row_count(module1, "events"),
        "module1_label_rows": _artifact_row_count(module1, "labels"),
        "integrity_raw": raw_integrity,
        "integrity_cleaned": cleaned_integrity,
        "meta_sharpe": float(metrics["sharpe"]),
        "meta_max_drawdown": float(metrics["max_drawdown"]),
        "meta_profit_factor": float(metrics["profit_factor"]),
        "artifacts_root": str(artifacts_root),
    }
    report_pdf_path = _write_backtest_pdf(
        artifacts_root / "reports",
        f"{PRODUCT_NAME} Full Stack Backtest Report",
        backtest.tearsheet,
    )
    out["report_pdf"] = str(report_pdf_path)

    report_json_path = artifacts_root / f"run_full_stack_report_{_utc_stamp()}.json"
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["report_path"] = str(report_json_path)
    out["report_json"] = str(report_json_path)

    if notifier_config is not None:
        AlertNotifier(notifier_config).send_backtest_summary(
            run_name="run_full_stack",
            sharpe=float(metrics["sharpe"]),
            max_drawdown=float(metrics["max_drawdown"]),
            profit_factor=float(metrics["profit_factor"]),
        )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run automated {PRODUCT_NAME} fetch -> train -> backtest stack.")
    parser.add_argument("--provider", default="binance", choices=["binance", "alpaca", "yfinance"])
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"])
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-format", type=str, default=None)
    parser.add_argument("--market-data-root", type=Path, default=Path("market_data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/auto_full_stack"))
    parser.add_argument("--best-params-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = run_full_stack(
            provider=args.provider,
            symbols=tuple(args.symbols),
            lookback_days=args.lookback_days,
            source_root=args.source_root,
            source_format=args.source_format,
            market_data_root=args.market_data_root,
            artifacts_root=args.artifacts_root,
            best_params_path=args.best_params_path,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
