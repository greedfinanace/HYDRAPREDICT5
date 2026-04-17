from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from quant_stack.module1 import Module1Artifacts, Module1Config, build_module1_dataset
from quant_stack.module3 import AlphaModelArtifacts, Module3Config, run_alpha_module
from quant_stack.module4 import MetaModelArtifacts, Module4Config, run_meta_module
from quant_stack.module5 import BacktestArtifacts, Module5Config, run_backtest
from quant_stack.module6 import apply_hpo_overrides


@dataclass(frozen=True)
class ComparisonRunArtifacts:
    name: str
    module1: Module1Artifacts
    alpha: AlphaModelArtifacts
    meta: MetaModelArtifacts
    backtest: BacktestArtifacts


def _execute_variant(
    name: str,
    module1_config: Module1Config,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    run_root: Path,
) -> ComparisonRunArtifacts:
    variant_root = run_root / name
    module1_artifacts = build_module1_dataset(replace(module1_config, output_root=variant_root / "module1"))
    alpha_artifacts = run_alpha_module(module1_artifacts, module3_config)
    meta_artifacts = run_meta_module(alpha_artifacts, module4_config)
    backtest_artifacts = run_backtest(
        meta_artifacts.position_sizing,
        module1_artifacts.stationary,
        replace(module5_config, output_root=variant_root / "module5"),
    )
    return ComparisonRunArtifacts(
        name=name,
        module1=module1_artifacts,
        alpha=alpha_artifacts,
        meta=meta_artifacts,
        backtest=backtest_artifacts,
    )


def run_comparison(
    hourly_module1_config: Module1Config,
    dense_module1_config: Module1Config,
    module3_config: Module3Config = Module3Config(),
    module4_config: Module4Config = Module4Config(),
    module5_config: Module5Config = Module5Config(),
    params_path: str | Path | None = None,
    results_root: Path = Path("results"),
) -> tuple[pl.DataFrame, dict[str, ComparisonRunArtifacts]]:
    timestamp_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(results_root) / timestamp_label
    run_root.mkdir(parents=True, exist_ok=True)

    if params_path is not None and Path(params_path).exists():
        hourly_module1_config, module3_config, module4_config = apply_hpo_overrides(
            hourly_module1_config,
            module3_config,
            module4_config,
            params_path,
        )
        dense_module1_config, _, _ = apply_hpo_overrides(
            dense_module1_config,
            module3_config,
            module4_config,
            params_path,
        )

    runs = {
        "hourly_baseline": _execute_variant(
            "hourly_baseline",
            hourly_module1_config,
            module3_config,
            module4_config,
            module5_config,
            run_root,
        ),
        "dense_volume_bar": _execute_variant(
            "dense_volume_bar",
            dense_module1_config,
            module3_config,
            module4_config,
            module5_config,
            run_root,
        ),
    }

    rows: list[dict[str, Any]] = []
    for name, artifacts in runs.items():
        meta_metrics = artifacts.backtest.tearsheet["strategies"]["meta_strategy"]
        alpha_metrics = artifacts.backtest.tearsheet["strategies"]["alpha_raw"]
        rows.append(
            {
                "run_name": name,
                "bars": artifacts.module1.bars.height,
                "events": artifacts.module1.events.height,
                "labels": artifacts.module1.labels.height,
                "meta_sharpe": meta_metrics["sharpe"],
                "meta_max_drawdown": meta_metrics["max_drawdown"],
                "meta_profit_factor": meta_metrics["profit_factor"],
                "alpha_sharpe": alpha_metrics["sharpe"],
                "alpha_max_drawdown": alpha_metrics["max_drawdown"],
                "alpha_profit_factor": alpha_metrics["profit_factor"],
                "drawdown_reduction_pct": artifacts.backtest.tearsheet["alpha_meta_attribution"]["drawdown_reduction_pct"],
            }
        )

    comparison = pl.DataFrame(rows).sort("run_name")
    comparison.write_csv(run_root / "comparison_tearsheet.csv")
    (run_root / "comparison_tearsheet.json").write_text(
        json.dumps(comparison.to_dicts(), indent=2),
        encoding="utf-8",
    )
    return comparison, runs
