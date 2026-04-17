from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRODUCT_NAME = "HydraPredict 5"
TEMPLATE_ROOT = Path("web/hydrapredict5")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Build a hostable {PRODUCT_NAME} backtest website.")
    parser.add_argument("--report-json", type=Path, required=True, help="Path to walkforward_report.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/hydrapredict5_site"),
        help="Output folder for hostable site bundle",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(report_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    from_report_parent = (report_path.parent / candidate).resolve()
    if from_report_parent.exists():
        return from_report_parent
    from_cwd = (Path.cwd() / candidate).resolve()
    if from_cwd.exists():
        return from_cwd
    return None


def _copy_optional(src: Path | None, dest: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest.relative_to(dest.parents[1]).as_posix())


def _public_engine_label(strategy_mode: str | None) -> str:
    normalized = str(strategy_mode or "").strip().lower()
    if normalized == "alpha_raw_long_only":
        return f"{PRODUCT_NAME} Core Engine"
    if normalized == "meta":
        return f"{PRODUCT_NAME} Adaptive Engine"
    return f"{PRODUCT_NAME} Engine"


def _scrub_payload(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, raw in value.items():
            if key == "strategy_mode":
                output[key] = "hydrapredict5_engine"
                continue
            output[key] = _scrub_payload(raw)
        return output
    if isinstance(value, list):
        return [_scrub_payload(item) for item in value]
    if isinstance(value, str):
        return value.replace("alpha_raw_long_only", "hydrapredict5_engine").replace("meta", "engine")
    return value


def _build_site_payload(report: dict[str, Any], assets: dict[str, str | None]) -> dict[str, Any]:
    return {
        "product_name": PRODUCT_NAME,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "timeframe": report.get("timeframe"),
        "engine_label": _public_engine_label(str(report.get("strategy_mode", ""))),
        "universe_mode": report.get("universe_mode"),
        "universe": report.get("universe", []),
        "benchmark_symbol": report.get("benchmark_symbol"),
        "train_period": report.get("train_period", {}),
        "test_period": report.get("test_period", {}),
        "is_metrics": report.get("is_metrics", {}),
        "oos_metrics": report.get("oos_metrics", {}),
        "benchmark_metrics": report.get("benchmark_metrics", {}),
        "verdict": report.get("verdict", {}),
        "module1": report.get("module1", {}),
        "oos_trade_count": report.get("oos_trade_count"),
        "assets": assets,
        "payload_sanitized": _scrub_payload(report),
    }


def _build_diff_text(report: dict[str, Any]) -> str:
    universe = ", ".join(report.get("universe", []))
    is_metrics = report.get("is_metrics", {})
    oos_metrics = report.get("oos_metrics", {})
    benchmark_oos = report.get("benchmark_metrics", {}).get("out_of_sample", {})
    verdict = report.get("verdict", {})
    lines = [
        f"--- {PRODUCT_NAME} walkforward_report.json",
        f"+++ {PRODUCT_NAME} hosted_site_bundle",
        "@@ report summary @@",
        f"+ product_name: {PRODUCT_NAME}",
        f"+ engine_label: {_public_engine_label(str(report.get('strategy_mode', '')))}",
        f"+ timeframe: {report.get('timeframe')}",
        f"+ universe_mode: {report.get('universe_mode')}",
        f"+ benchmark_symbol: {report.get('benchmark_symbol')}",
        f"+ universe: {universe}",
        f"+ train_period: {report.get('train_period', {}).get('start')} -> {report.get('train_period', {}).get('end')}",
        f"+ test_period: {report.get('test_period', {}).get('start')} -> {report.get('test_period', {}).get('end')}",
        "@@ phase 1 @@",
        f"+ IS ending_equity: {is_metrics.get('ending_equity')}",
        f"+ IS return: {is_metrics.get('total_return')}",
        f"+ IS annual_return: {is_metrics.get('annualized_return')}",
        f"+ IS volatility: {is_metrics.get('annualized_volatility')}",
        f"+ IS sharpe: {is_metrics.get('sharpe')}",
        f"+ IS max_drawdown: {is_metrics.get('max_drawdown')}",
        "@@ phase 2 @@",
        f"+ OOS ending_equity: {oos_metrics.get('ending_equity')}",
        f"+ OOS return: {oos_metrics.get('total_return')}",
        f"+ OOS annual_return: {oos_metrics.get('annualized_return')}",
        f"+ OOS volatility: {oos_metrics.get('annualized_volatility')}",
        f"+ OOS sharpe: {oos_metrics.get('sharpe')}",
        f"+ OOS max_drawdown: {oos_metrics.get('max_drawdown')}",
        f"+ OOS profit_factor: {oos_metrics.get('profit_factor')}",
        "@@ benchmark @@",
        f"+ benchmark_oos_sharpe: {benchmark_oos.get('sharpe')}",
        f"+ benchmark_oos_max_drawdown: {benchmark_oos.get('max_drawdown')}",
        "@@ verdict @@",
        f"+ status: {verdict.get('status')}",
        f"+ summary: {verdict.get('summary')}",
        "@@ graphs @@",
        f"+ curves_png: assets/walkforward_equity_curves.png",
        f"+ report_pdf: assets/walkforward_report.pdf",
        f"+ report_txt: assets/walkforward_report.txt",
    ]
    return "\n".join(lines) + "\n"


def _copy_templates(output_root: Path) -> None:
    for rel_path in ("index.html", "styles.css", "app.js"):
        src = TEMPLATE_ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Missing site template file: {src}")
        dest = output_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def build_site(report_json_path: Path, output_root: Path) -> Path:
    report_json_path = report_json_path.resolve()
    if not report_json_path.exists():
        raise FileNotFoundError(f"Report JSON does not exist: {report_json_path}")

    report = _read_json(report_json_path)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_templates(output_root)

    assets_root = output_root / "assets"
    data_root = output_root / "data"
    assets_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    report_txt = _resolve_artifact_path(report_json_path, report.get("report_txt"))
    report_pdf = _resolve_artifact_path(report_json_path, report.get("report_pdf"))
    curves_png = _resolve_artifact_path(report_json_path, report.get("curves_png"))

    asset_refs = {
        "report_txt": _copy_optional(report_txt, assets_root / "walkforward_report.txt"),
        "report_pdf": _copy_optional(report_pdf, assets_root / "walkforward_report.pdf"),
        "curves_png": _copy_optional(curves_png, assets_root / "walkforward_equity_curves.png"),
    }
    diff_path = assets_root / "backtest_report.diff"
    diff_path.write_text(_build_diff_text(report), encoding="utf-8")
    asset_refs["report_diff"] = str(diff_path.relative_to(output_root).as_posix())

    site_payload = _build_site_payload(report, asset_refs)
    (data_root / "report.json").write_text(json.dumps(site_payload, indent=2), encoding="utf-8")
    return output_root


def main() -> int:
    args = _parse_args()
    out = build_site(args.report_json, args.output_root)
    print(json.dumps({"product_name": PRODUCT_NAME, "site_root": str(out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
