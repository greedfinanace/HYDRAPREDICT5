from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from quant_stack.module0 import (
    LeveragedSectorUniverseSelectionConfig,
    LiquidUniverseResearchArtifacts,
    TightUniverseSelectionConfig,
    UniverseSelectionConfig,
    canonical_symbol,
    prepare_liquid_etf_research_artifacts,
    select_leveraged_sector_universe,
    select_liquid_etf_universe,
    select_tight_liquid_etf_universe,
)
from quant_stack.module0.research_universe import _greedy_multi_start_basket
from quant_stack.module1 import Module1Artifacts, Module1Config, build_module1_dataset
from quant_stack.module3 import (
    AlphaDataset,
    CPCVConfig,
    Module3Config,
    build_alpha_inference_dataset,
    predict_alpha_probabilities,
    run_alpha_module,
)
from quant_stack.module4 import (
    MetaDataset,
    Module4Config,
    build_meta_dataset_from_predictions,
    predict_meta_bet_sizes,
    run_meta_module,
)
from quant_stack.module5 import BacktestArtifacts, Module5Config, run_backtest


DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_OUTPUT_ROOT = Path("artifacts/walkforward_backtest")
PRODUCT_NAME = "HydraPredict 5"


@dataclass(frozen=True)
class WalkforwardConfig:
    source_root: Path
    source_format: str = "auto"
    timeframe: str = "auto"
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None
    benchmark_symbol: str = "SPY"
    universe_mode: str = "liquid_etfs"
    universe_file: Path | None = None
    symbols: tuple[str, ...] | None = None
    target_universe_size: int = 50
    broad_liquid_candidate_size: int = 20
    broad_liquid_target_size: int = 6
    broad_liquid_etf_only: bool = True
    tight_universe_target_size: int = 10
    tight_universe_etf_only: bool = True
    leveraged_sector_target_size: int = 8
    leveraged_sector_etf_only: bool = True
    min_median_dollar_volume: float = 1_000_000.0
    min_coverage_ratio: float = 0.98
    max_zero_volume_fraction: float = 0.0
    strategy_mode: str = "meta"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    hpo_trials: int = 0
    hpo_sampler_seed: int = 42


def _normalize_date(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def _date_range_bounds(start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_value = pd.Timestamp(start_date)
    end_value = pd.Timestamp(end_date)
    start_ts = start_value.tz_localize("UTC") if start_value.tzinfo is None else start_value.tz_convert("UTC")
    end_shifted = end_value + pd.Timedelta(days=1)
    end_exclusive = end_shifted.tz_localize("UTC") if end_shifted.tzinfo is None else end_shifted.tz_convert("UTC")
    return start_ts, end_exclusive


def _module1_date_end(end_date: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _infer_bar_minutes(timeframe: str) -> int:
    normalized = str(timeframe).strip().lower()
    if normalized in {"60m", "1h"}:
        return 60
    if normalized in {"1d", "d"}:
        return 1440
    raise ValueError(f"Unsupported walkforward timeframe: {timeframe}")


def _candidate_timeframes(value: str) -> tuple[str, ...]:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return ("60m", "1d")
    if normalized in {"60m", "1h"}:
        return ("60m",)
    if normalized in {"1d", "d"}:
        return ("1d",)
    raise ValueError(f"Unsupported walkforward timeframe: {value}")


def _phase_defaults(
    benchmark_start: pd.Timestamp,
    benchmark_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    test_end = benchmark_end.normalize()
    test_start = (test_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).normalize()
    train_end = (test_start - pd.Timedelta(days=1)).normalize()
    train_start = (train_end - pd.DateOffset(years=6) + pd.Timedelta(days=1)).normalize()
    if benchmark_start > train_start:
        raise ValueError(
            "Insufficient history for the default 6y IS + 1y OOS policy. "
            f"Available benchmark range is {benchmark_start.date()} to {benchmark_end.date()}."
        )
    return train_start, train_end, test_start, test_end


def _resolve_timeframe_and_period(
    config: WalkforwardConfig,
) -> tuple[str, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train_start = _normalize_date(config.train_start)
    train_end = _normalize_date(config.train_end)
    test_start = _normalize_date(config.test_start)
    test_end = _normalize_date(config.test_end)

    failure_messages: list[str] = []
    for timeframe in _candidate_timeframes(config.timeframe):
        bar_minutes = _infer_bar_minutes(timeframe)
        try:
            benchmark_artifacts = select_liquid_etf_universe(
                UniverseSelectionConfig(
                    source_root=config.source_root,
                    source_format=config.source_format,
                    start=None,
                    end=None,
                    bar_minutes=bar_minutes,
                    target_size=1,
                    benchmark_symbol=config.benchmark_symbol,
                    min_coverage_ratio=0.0,
                    min_median_dollar_volume=0.0,
                    max_zero_volume_fraction=1.0,
                    etf_only=False,
                )
            )
            summary = benchmark_artifacts.summary.to_pandas()
            benchmark_row = summary.loc[summary["symbol"].astype(str).str.upper() == config.benchmark_symbol.upper()]
            if benchmark_row.empty:
                raise ValueError(f"Benchmark symbol {config.benchmark_symbol} is unavailable at timeframe {timeframe}.")
            benchmark_start = pd.Timestamp(benchmark_row.iloc[0]["start_date"]).normalize()
            benchmark_end = pd.Timestamp(benchmark_row.iloc[0]["end_date"]).normalize()

            if any(value is None for value in (train_start, train_end, test_start, test_end)):
                train_start_resolved, train_end_resolved, test_start_resolved, test_end_resolved = _phase_defaults(
                    benchmark_start,
                    benchmark_end,
                )
            else:
                train_start_resolved = train_start
                train_end_resolved = train_end
                test_start_resolved = test_start
                test_end_resolved = test_end

            if benchmark_start > train_start_resolved or benchmark_end < test_end_resolved:
                raise ValueError(
                    f"Benchmark range {benchmark_start.date()} to {benchmark_end.date()} does not cover "
                    f"requested window {train_start_resolved.date()} to {test_end_resolved.date()}."
                )
            return (
                timeframe,
                bar_minutes,
                train_start_resolved,
                train_end_resolved,
                test_start_resolved,
                test_end_resolved,
            )
        except Exception as exc:
            failure_messages.append(f"{timeframe}: {exc}")
    raise ValueError("Unable to resolve a valid timeframe/history window. " + " | ".join(failure_messages))


def _resolve_symbols(
    config: WalkforwardConfig,
    *,
    bar_minutes: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[str, ...]:
    if config.symbols:
        symbols = tuple(canonical_symbol(symbol) for symbol in config.symbols)
        if canonical_symbol(config.benchmark_symbol) not in symbols:
            symbols = (canonical_symbol(config.benchmark_symbol),) + symbols
        return tuple(dict.fromkeys(symbols))
    if config.universe_file is not None:
        lines = [line.strip() for line in config.universe_file.read_text(encoding="utf-8").splitlines()]
        symbols = tuple(canonical_symbol(line) for line in lines if line)
        if canonical_symbol(config.benchmark_symbol) not in symbols:
            symbols = (canonical_symbol(config.benchmark_symbol),) + symbols
        return tuple(dict.fromkeys(symbols))
    if config.universe_mode == "broad_liquid_etfs":
        artifacts = select_liquid_etf_universe(
            UniverseSelectionConfig(
                source_root=config.source_root,
                source_format=config.source_format,
                start=train_start,
                end=train_end,
                bar_minutes=bar_minutes,
                target_size=max(int(config.broad_liquid_candidate_size), int(config.broad_liquid_target_size)),
                benchmark_symbol=config.benchmark_symbol,
                min_coverage_ratio=config.min_coverage_ratio,
                min_median_dollar_volume=config.min_median_dollar_volume,
                max_zero_volume_fraction=config.max_zero_volume_fraction,
                etf_only=config.broad_liquid_etf_only,
            )
        )
        return artifacts.selected_symbols
    if config.universe_mode == "tight_liquid_etfs":
        artifacts = select_tight_liquid_etf_universe(
            TightUniverseSelectionConfig(
                source_root=config.source_root,
                source_format=config.source_format,
                train_start=train_start,
                train_end=train_end,
                bar_minutes=bar_minutes,
                target_size=config.tight_universe_target_size,
                candidate_pool_size=max(config.tight_universe_target_size * 4, 24),
                benchmark_symbol=config.benchmark_symbol,
                min_coverage_ratio=config.min_coverage_ratio,
                min_median_dollar_volume=config.min_median_dollar_volume,
                max_zero_volume_fraction=config.max_zero_volume_fraction,
                etf_only=config.tight_universe_etf_only,
            )
        )
        return artifacts.selected_symbols
    if config.universe_mode == "leveraged_sector_etfs":
        artifacts = select_leveraged_sector_universe(
            LeveragedSectorUniverseSelectionConfig(
                source_root=config.source_root,
                source_format=config.source_format,
                train_start=train_start,
                train_end=train_end,
                bar_minutes=bar_minutes,
                target_size=config.leveraged_sector_target_size,
                candidate_pool_size=max(config.leveraged_sector_target_size * 6, 24),
                benchmark_symbol=config.benchmark_symbol,
                min_coverage_ratio=config.min_coverage_ratio,
                min_median_dollar_volume=config.min_median_dollar_volume,
                max_zero_volume_fraction=config.max_zero_volume_fraction,
                etf_only=config.leveraged_sector_etf_only,
            )
        )
        return artifacts.selected_symbols
    if config.universe_mode != "liquid_etfs":
        raise ValueError(f"Unsupported universe_mode: {config.universe_mode}")
    artifacts = select_liquid_etf_universe(
        UniverseSelectionConfig(
            source_root=config.source_root,
            source_format=config.source_format,
            start=train_start,
            end=train_end,
            bar_minutes=bar_minutes,
            target_size=config.target_universe_size,
            benchmark_symbol=config.benchmark_symbol,
            min_coverage_ratio=config.min_coverage_ratio,
            min_median_dollar_volume=config.min_median_dollar_volume,
            max_zero_volume_fraction=config.max_zero_volume_fraction,
            etf_only=True,
        )
    )
    return artifacts.selected_symbols


def _search_broad_liquid_universe(
    config: WalkforwardConfig,
    *,
    bar_minutes: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    strategy_mode: str,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    candidate_selection = prepare_liquid_etf_research_artifacts(
        UniverseSelectionConfig(
            source_root=config.source_root,
            source_format=config.source_format,
            start=train_start,
            end=train_end,
            bar_minutes=bar_minutes,
            target_size=max(int(config.broad_liquid_candidate_size), int(config.broad_liquid_target_size)),
            benchmark_symbol=config.benchmark_symbol,
            min_coverage_ratio=config.min_coverage_ratio,
            min_median_dollar_volume=config.min_median_dollar_volume,
            max_zero_volume_fraction=config.max_zero_volume_fraction,
            etf_only=config.broad_liquid_etf_only,
        )
    )
    benchmark_symbol = canonical_symbol(config.benchmark_symbol)
    candidate_pool = list(dict.fromkeys(candidate_selection.selected_symbols))
    if benchmark_symbol not in candidate_pool:
        candidate_pool.insert(0, benchmark_symbol)
    if len(candidate_pool) < 4:
        raise ValueError("Broad liquid ETF search requires at least four candidate symbols.")

    best_symbols: tuple[str, ...] | None = None
    best_score = float("-inf")
    best_details: dict[str, Any] = {}
    evaluations: list[dict[str, Any]] = []
    max_target = min(max(int(config.broad_liquid_target_size), 4), len(candidate_pool))
    seed_pool = candidate_pool[: max(6, min(len(candidate_pool), max_target * 2))]
    for basket_size in range(4, max_target + 1):
        proxy_symbols, proxy_score, proxy_metrics = _greedy_multi_start_basket(
            benchmark_symbol=benchmark_symbol,
            benchmark_sessions=set(candidate_selection.benchmark_sessions),
            returns_by_symbol=candidate_selection.returns_by_symbol,
            candidate_symbols=[symbol for symbol in candidate_pool if symbol != benchmark_symbol],
            target_size=basket_size,
            score_drawdown_weight=3.0,
            score_correlation_weight=0.25,
            score_return_weight=0.5,
            max_pairwise_correlation=0.92,
            min_selected_symbols=min(4, basket_size),
            seed_symbols=seed_pool,
        )
        score, details, _ = _score_training_universe(
            tuple(proxy_symbols),
            walkforward_config=config,
            bar_minutes=bar_minutes,
            train_start=train_start,
            train_end=train_end,
            module3_config=module3_config,
            module4_config=module4_config,
            module5_config=module5_config,
            strategy_mode=strategy_mode,
            output_root=config.output_root,
        )
        result = {
            "target_size": int(len(proxy_symbols)),
            "symbols": list(proxy_symbols),
            "score": float(score),
            "proxy_score": float(proxy_score),
            "proxy_train_sharpe": float(proxy_metrics.get("sharpe", 0.0)),
            "proxy_train_max_drawdown": float(proxy_metrics.get("max_drawdown", 0.0)),
            "validation_median_sharpe": float(details.get("median_validation_sharpe", 0.0)),
            "validation_non_positive_ratio": float(details.get("validation_non_positive_ratio", 1.0)),
            "objective_used": str(details.get("objective_used", "unknown")),
        }
        evaluations.append(result)
        if score > best_score:
            best_symbols = tuple(proxy_symbols)
            best_score = float(score)
            best_details = dict(details)

    if best_symbols is None:
        raise ValueError("Broad liquid ETF search did not yield any valid basket.")

    evaluations.sort(key=lambda row: float(row["score"]), reverse=True)
    diagnostics = {
        "objective": "training_window_backtest",
        "strategy_mode": strategy_mode,
        "candidate_pool": candidate_pool,
        "candidate_pool_size": int(len(candidate_pool)),
        "target_size_max": int(max_target),
        "evaluation_count": int(len(evaluations)),
        "top_evaluations": evaluations[:25],
        "selected_score": float(best_score),
        "selected_train_metrics": best_details,
        "selected_target_size": int(len(best_symbols)),
    }
    return best_symbols, diagnostics


def _score_training_universe(
    symbols: tuple[str, ...],
    *,
    walkforward_config: WalkforwardConfig,
    bar_minutes: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    strategy_mode: str,
    output_root: Path,
) -> tuple[float, dict[str, Any], Module1Artifacts]:
    digest = hashlib.sha1("|".join(symbols).encode("utf-8")).hexdigest()[:12]
    module1_config = Module1Config(
        source_root=walkforward_config.source_root,
        source_format=walkforward_config.source_format,
        symbols=symbols,
        start=train_start.to_pydatetime(),
        end=_module1_date_end(train_end).to_pydatetime(),
        bar_minutes=bar_minutes,
        sampling_mode="time",
        vertical_barrier_days=20.0 if bar_minutes >= 1440 else 3.0,
        output_root=output_root / "universe_search" / digest,
    )
    artifacts = build_module1_dataset(module1_config)
    train_artifacts = _phase_module1_artifacts(artifacts, phase_start=train_start, phase_end=train_end)
    _ensure_phase_has_data(train_artifacts, "universe_search_train")
    try:
        score, details = _validation_score(train_artifacts, module3_config, module4_config, module5_config)
        validated_details = dict(details)
        validated_details["objective_used"] = "nested_validation"
        return float(score), validated_details, artifacts
    except ValueError as exc:
        backtest, _ = _phase_backtest(
            "in_sample",
            None,
            train_artifacts,
            module3_config,
            module4_config,
            module5_config,
            strategy_mode=strategy_mode,
        )
        strategy_key, position_prefix, return_prefix = _strategy_spec(strategy_mode)
        metrics = _phase_metrics(
            backtest,
            starting_capital=walkforward_config.initial_capital,
            strategy_key=strategy_key,
            position_prefix=position_prefix,
            return_prefix=return_prefix,
        )
        score = (
            float(metrics["sharpe"])
            - 1.5 * abs(float(metrics["max_drawdown"]))
            - 0.05 * float(metrics["avg_turnover"])
        )
        fallback_details: dict[str, Any] = {
            "objective_used": "full_train_backtest",
            "fallback_reason": str(exc),
            "median_validation_sharpe": float(metrics["sharpe"]),
            "validation_non_positive_ratio": 0.0 if float(metrics["sharpe"]) > 0.0 else 1.0,
            "validation_sharpes": [float(metrics["sharpe"])],
            "validation_mdds": [float(metrics["max_drawdown"])],
            "validation_turnovers": [float(metrics["avg_turnover"])],
        }
        return float(score), fallback_details, artifacts


def _search_leveraged_sector_universe(
    config: WalkforwardConfig,
    *,
    bar_minutes: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    strategy_mode: str,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    min_target = min(4, max(1, int(config.leveraged_sector_target_size)))
    max_target = max(int(config.leveraged_sector_target_size), min_target)
    candidate_sizes = list(range(min_target, max_target + 1))

    evaluations: list[dict[str, Any]] = []
    seen: dict[tuple[str, ...], dict[str, Any]] = {}
    best_symbols: tuple[str, ...] | None = None
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    sweep_baskets: list[tuple[str, ...]] = []

    for target_size in candidate_sizes:
        selection = select_leveraged_sector_universe(
            LeveragedSectorUniverseSelectionConfig(
                source_root=config.source_root,
                source_format=config.source_format,
                train_start=train_start,
                train_end=train_end,
                bar_minutes=bar_minutes,
                target_size=target_size,
                candidate_pool_size=max(int(target_size) * 6, 24),
                benchmark_symbol=config.benchmark_symbol,
                min_coverage_ratio=config.min_coverage_ratio,
                min_median_dollar_volume=config.min_median_dollar_volume,
                max_zero_volume_fraction=config.max_zero_volume_fraction,
                etf_only=config.leveraged_sector_etf_only,
            )
        )
        symbols = tuple(selection.selected_symbols)
        sweep_baskets.append(symbols)
        if symbols in seen:
            continue
        score, metrics, _ = _score_training_universe(
            symbols,
            walkforward_config=config,
            bar_minutes=bar_minutes,
            train_start=train_start,
            train_end=train_end,
            module3_config=module3_config,
            module4_config=module4_config,
            module5_config=module5_config,
            strategy_mode=strategy_mode,
            output_root=config.output_root,
        )
        result = {
            "target_size": int(target_size),
            "symbols": list(symbols),
            "score": float(score),
            "validation_median_sharpe": float(metrics["median_validation_sharpe"]),
            "validation_non_positive_ratio": float(metrics["validation_non_positive_ratio"]),
            "validation_sharpes": [float(value) for value in metrics["validation_sharpes"]],
            "validation_mdds": [float(value) for value in metrics["validation_mdds"]],
            "validation_turnovers": [float(value) for value in metrics["validation_turnovers"]],
        }
        evaluations.append(result)
        seen[symbols] = result
        if score > best_score:
            best_symbols = symbols
            best_score = float(score)
            best_metrics = dict(metrics)

    candidate_pool = sorted({symbol for basket in sweep_baskets for symbol in basket})
    if canonical_symbol(config.benchmark_symbol) not in candidate_pool:
        candidate_pool = [canonical_symbol(config.benchmark_symbol)] + candidate_pool

    benchmark_symbol = canonical_symbol(config.benchmark_symbol)
    exhaustive_evaluations: list[dict[str, Any]] = []
    max_subset_size = min(max_target, len(candidate_pool))
    if len(candidate_pool) <= 10 and max_subset_size >= min_target:
        other_symbols = [symbol for symbol in candidate_pool if symbol != benchmark_symbol]
        for subset_size in range(min_target, max_subset_size + 1):
            if subset_size == 1:
                candidate_subsets = [(benchmark_symbol,)]
            else:
                candidate_subsets = (
                    (benchmark_symbol,) + combo for combo in itertools.combinations(other_symbols, subset_size - 1)
                )
            for basket in candidate_subsets:
                if basket in seen:
                    continue
                score, metrics, _ = _score_training_universe(
                    basket,
                    walkforward_config=config,
                    bar_minutes=bar_minutes,
                    train_start=train_start,
                    train_end=train_end,
                    module3_config=module3_config,
                    module4_config=module4_config,
                    module5_config=module5_config,
                    strategy_mode=strategy_mode,
                    output_root=config.output_root,
                )
                result = {
                    "target_size": int(len(basket)),
                    "symbols": list(basket),
                    "score": float(score),
                    "validation_median_sharpe": float(metrics["median_validation_sharpe"]),
                    "validation_non_positive_ratio": float(metrics["validation_non_positive_ratio"]),
                    "validation_sharpes": [float(value) for value in metrics["validation_sharpes"]],
                    "validation_mdds": [float(value) for value in metrics["validation_mdds"]],
                    "validation_turnovers": [float(value) for value in metrics["validation_turnovers"]],
                }
                exhaustive_evaluations.append(result)
                seen[basket] = result
                if score > best_score:
                    best_symbols = basket
                    best_score = float(score)
                    best_metrics = dict(metrics)

    all_evaluations = evaluations + exhaustive_evaluations

    if best_symbols is None:
        raise ValueError("Leveraged sector universe search did not yield any valid basket.")

    diagnostics = {
        "objective": "training_window_backtest",
        "strategy_mode": strategy_mode,
        "candidate_sizes": candidate_sizes,
        "evaluations": all_evaluations,
        "selected_score": float(best_score),
        "selected_train_metrics": best_metrics,
        "selected_target_size": int(len(best_symbols)),
        "candidate_pool": candidate_pool,
    }
    return best_symbols, diagnostics


def _phase_module1_artifacts(
    artifacts: Module1Artifacts,
    *,
    phase_start: pd.Timestamp,
    phase_end: pd.Timestamp,
) -> Module1Artifacts:
    start_ts, end_exclusive = _date_range_bounds(phase_start, phase_end)
    stationary = artifacts.stationary.filter(
        (pl.col("timestamp") >= pl.lit(start_ts)) & (pl.col("timestamp") < pl.lit(end_exclusive))
    )
    bars = artifacts.bars.filter(
        (pl.col("timestamp") >= pl.lit(start_ts)) & (pl.col("timestamp") < pl.lit(end_exclusive))
    )
    labels = artifacts.labels.filter(
        (pl.col("event_time") >= pl.lit(start_ts))
        & (pl.col("event_time") < pl.lit(end_exclusive))
        & (pl.col("t1") < pl.lit(end_exclusive))
    )
    if "event_time" in artifacts.events.columns:
        events = artifacts.events.filter(
            (pl.col("event_time") >= pl.lit(start_ts)) & (pl.col("event_time") < pl.lit(end_exclusive))
        )
    else:
        events = labels.select([column for column in artifacts.events.columns if column in labels.columns])
    return Module1Artifacts(
        config=artifacts.config,
        bars=bars,
        stationary=stationary,
        fracdiff_params=artifacts.fracdiff_params,
        events=events,
        labels=labels,
        bars_path=artifacts.bars_path,
        stationary_path=artifacts.stationary_path,
        fracdiff_params_path=artifacts.fracdiff_params_path,
        events_path=artifacts.events_path,
        labels_path=artifacts.labels_path,
    )


def _ensure_phase_has_data(artifacts: Module1Artifacts, phase_name: str) -> None:
    if artifacts.stationary.is_empty():
        raise ValueError(f"{phase_name} phase contains no stationary bars.")
    if artifacts.labels.is_empty():
        raise ValueError(f"{phase_name} phase contains no fully-realized labels/events.")


def _strategy_spec(strategy_mode: str) -> tuple[str, str, str]:
    normalized = str(strategy_mode).strip().lower()
    if normalized == "meta":
        return "meta_strategy", "meta", "meta_strategy"
    if normalized == "alpha_raw_long_only":
        return "alpha_raw", "alpha", "alpha_raw"
    raise ValueError(f"Unsupported strategy_mode: {strategy_mode}")


def _oos_positions(
    train_artifacts: Module1Artifacts,
    test_artifacts: Module1Artifacts,
    module3_config: Module3Config,
    module4_config: Module4Config,
) -> tuple[AlphaDataset, MetaDataset, pl.DataFrame]:
    alpha_train = run_alpha_module(train_artifacts, module3_config)
    meta_train = run_meta_module(alpha_train, module4_config)
    oos_events = test_artifacts.labels.drop("label") if "label" in test_artifacts.labels.columns else test_artifacts.labels
    alpha_dataset = build_alpha_inference_dataset(test_artifacts.stationary, oos_events, module3_config)
    alpha_predictions = predict_alpha_probabilities(alpha_dataset, alpha_train)
    meta_dataset = build_meta_dataset_from_predictions(alpha_dataset, alpha_predictions, module4_config)
    positions = predict_meta_bet_sizes(meta_dataset, meta_train)
    return alpha_dataset, meta_dataset, positions


def _phase_backtest(
    phase_name: str,
    train_artifacts: Module1Artifacts | None,
    phase_artifacts: Module1Artifacts,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    *,
    strategy_mode: str = "meta",
) -> tuple[BacktestArtifacts, pl.DataFrame]:
    normalized_mode = str(strategy_mode).strip().lower()
    if normalized_mode == "meta":
        if phase_name == "in_sample":
            alpha_artifacts = run_alpha_module(phase_artifacts, module3_config)
            meta_artifacts = run_meta_module(alpha_artifacts, module4_config)
            positions = meta_artifacts.position_sizing
        else:
            if train_artifacts is None:
                raise ValueError("OOS backtest requires training artifacts.")
            _, _, positions = _oos_positions(train_artifacts, phase_artifacts, module3_config, module4_config)
    elif normalized_mode == "alpha_raw_long_only":
        if phase_name == "in_sample":
            alpha_artifacts = run_alpha_module(phase_artifacts, module3_config)
            positions = alpha_artifacts.oof_predictions.with_columns(
                [
                    pl.lit(1.0).alias("pred_side"),
                    pl.lit(1.0).alias("bet_size"),
                ]
            )
        else:
            if train_artifacts is None:
                raise ValueError("OOS backtest requires training artifacts.")
            alpha_train = run_alpha_module(train_artifacts, module3_config)
            alpha_dataset = build_alpha_inference_dataset(phase_artifacts.stationary, phase_artifacts.labels, module3_config)
            alpha_predictions = predict_alpha_probabilities(alpha_dataset, alpha_train)
            positions = alpha_predictions.with_columns(
                [
                    pl.lit(1.0).alias("pred_side"),
                    pl.lit(1.0).alias("bet_size"),
                ]
            )
    else:
        raise ValueError(f"Unsupported strategy_mode: {strategy_mode}")
    backtest = run_backtest(
        positions,
        phase_artifacts.stationary,
        replace(module5_config, save_plots=False),
    )
    return backtest, positions


def _validation_windows(labels: pl.DataFrame, min_windows: int = 2) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    frame = labels.to_pandas()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    unique_times = sorted(pd.Timestamp(value) for value in frame["event_time"].drop_duplicates().tolist())
    if len(unique_times) < 12:
        return []
    grouped = [chunk.tolist() for chunk in np.array_split(np.asarray(unique_times, dtype=object), 4) if len(chunk) > 0]
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for chunk in grouped[1:]:
        if not chunk:
            continue
        start = pd.Timestamp(chunk[0])
        end = pd.Timestamp(chunk[-1])
        if start.tzinfo is not None:
            start = start.tz_convert(None)
        if end.tzinfo is not None:
            end = end.tz_convert(None)
        windows.append((start.normalize(), end.normalize()))
    return windows if len(windows) >= min_windows else []


def _validation_score(
    module1_artifacts: Module1Artifacts,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
) -> tuple[float, dict[str, Any]]:
    windows = _validation_windows(module1_artifacts.labels)
    if not windows:
        raise ValueError("Training window does not contain enough events for nested validation.")

    event_times = pd.to_datetime(module1_artifacts.labels["event_time"].to_list(), utc=True)
    full_start = pd.Timestamp(event_times.min()).tz_convert(None).normalize()
    sharpes: list[float] = []
    mdds: list[float] = []
    turnovers: list[float] = []

    for window_start, window_end in windows:
        train_end = window_start - pd.Timedelta(days=1)
        train_slice = _phase_module1_artifacts(
            module1_artifacts,
            phase_start=full_start,
            phase_end=train_end,
        )
        valid_slice = _phase_module1_artifacts(module1_artifacts, phase_start=window_start, phase_end=window_end)
        if train_slice.labels.height < 10 or valid_slice.labels.height < 5:
            continue
        _ensure_phase_has_data(train_slice, "validation_train")
        _ensure_phase_has_data(valid_slice, "validation_test")
        backtest, _ = _phase_backtest(
            "out_of_sample",
            train_slice,
            valid_slice,
            module3_config,
            module4_config,
            module5_config,
        )
        metrics = backtest.tearsheet["strategies"]["meta_strategy"]
        sharpes.append(float(metrics["sharpe"]))
        mdds.append(float(metrics["max_drawdown"]))
        instrument_frame = backtest.instrument_frame.to_pandas()
        turnovers.append(float(np.mean(np.abs(instrument_frame["meta_turnover"].to_numpy(dtype=np.float64)))))

    if not sharpes:
        raise ValueError("Nested validation produced no usable windows.")

    median_sharpe = float(np.median(sharpes))
    score = median_sharpe - (0.25 * float(np.std(sharpes)))
    score -= 2.0 * max(0.0, abs(float(np.median(mdds))) - 0.12)
    score -= 0.1 * float(np.median(turnovers))
    non_positive_ratio = float(sum(value <= 0.0 for value in sharpes)) / float(len(sharpes))
    score -= 0.5 * non_positive_ratio
    return score, {
        "validation_sharpes": sharpes,
        "validation_mdds": mdds,
        "validation_turnovers": turnovers,
        "validation_non_positive_ratio": non_positive_ratio,
        "median_validation_sharpe": median_sharpe,
    }


def _trial_payload(trial: optuna.Trial) -> dict[str, dict[str, float | int]]:
    return {
        "module1": {
            "fracdiff_d": float(trial.suggest_float("fracdiff_d", 0.1, 0.9, step=0.05)),
            "weight_eps": float(trial.suggest_float("weight_eps", 1e-6, 1e-4, log=True)),
            "cusum_sigma_mult": float(trial.suggest_float("cusum_sigma_mult", 0.25, 2.0)),
            "pt_mult": float(trial.suggest_float("pt_mult", 1.0, 3.0)),
            "sl_mult": float(trial.suggest_float("sl_mult", 1.0, 3.0)),
            "volume_threshold_multiplier": float(trial.suggest_float("volume_threshold_multiplier", 10.0, 80.0)),
            "vertical_barrier_days": float(trial.suggest_float("vertical_barrier_days", 5.0, 60.0, step=5.0)),
        },
        "module3": {
            "num_leaves": int(trial.suggest_int("alpha_num_leaves", 20, 150)),
            "learning_rate": float(trial.suggest_float("alpha_learning_rate", 0.001, 0.1, log=True)),
            "feature_fraction": float(trial.suggest_float("alpha_feature_fraction", 0.5, 0.9)),
            "min_child_samples": int(trial.suggest_int("alpha_min_child_samples", 5, 60)),
            "subsample": float(trial.suggest_float("alpha_subsample", 0.5, 1.0)),
        },
        "module4": {
            "num_leaves": int(trial.suggest_int("meta_num_leaves", 20, 150)),
            "learning_rate": float(trial.suggest_float("meta_learning_rate", 0.001, 0.1, log=True)),
            "feature_fraction": float(trial.suggest_float("meta_feature_fraction", 0.5, 0.9)),
            "probability_floor": float(trial.suggest_float("meta_probability_floor", 0.0, 0.55)),
            "bet_sizing_power": float(trial.suggest_float("meta_bet_sizing_power", 0.25, 2.5)),
            "min_child_samples": int(trial.suggest_int("meta_min_child_samples", 5, 60)),
        },
        "module5": {
            "meta_prob_win_threshold": float(trial.suggest_float("meta_prob_win_threshold", 0.0, 0.60)),
        },
    }


def _apply_trial_payload(
    module1_config: Module1Config,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    payload: dict[str, dict[str, float | int]],
) -> tuple[Module1Config, Module3Config, Module4Config, Module5Config]:
    resolved_module1 = replace(
        module1_config,
        fracdiff_d_grid=(float(payload["module1"]["fracdiff_d"]),),
        weight_eps=float(payload["module1"]["weight_eps"]),
        adaptive_weight_eps=False,
        cusum_sigma_mult=float(payload["module1"]["cusum_sigma_mult"]),
        pt_mult=float(payload["module1"]["pt_mult"]),
        sl_mult=float(payload["module1"]["sl_mult"]),
        volume_threshold_multiplier=float(payload["module1"]["volume_threshold_multiplier"]),
        vertical_barrier_days=float(payload["module1"]["vertical_barrier_days"]),
    )
    resolved_module3 = replace(
        module3_config,
        num_leaves=int(payload["module3"]["num_leaves"]),
        learning_rate=float(payload["module3"]["learning_rate"]),
        feature_fraction=float(payload["module3"]["feature_fraction"]),
        min_child_samples=int(payload["module3"]["min_child_samples"]),
        subsample=float(payload["module3"]["subsample"]),
        optimized_params_path=None,
    )
    resolved_module4 = replace(
        module4_config,
        num_leaves=int(payload["module4"]["num_leaves"]),
        learning_rate=float(payload["module4"]["learning_rate"]),
        feature_fraction=float(payload["module4"]["feature_fraction"]),
        probability_floor=float(payload["module4"]["probability_floor"]),
        bet_sizing_power=float(payload["module4"]["bet_sizing_power"]),
        min_child_samples=int(payload["module4"]["min_child_samples"]),
        optimized_params_path=None,
    )
    resolved_module5 = replace(
        module5_config,
        meta_prob_win_threshold=float(payload["module5"]["meta_prob_win_threshold"]),
        save_plots=False,
    )
    return resolved_module1, resolved_module3, resolved_module4, resolved_module5


def _run_training_hpo(
    module1_config: Module1Config,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    *,
    output_root: Path,
    n_trials: int,
    sampler_seed: int,
) -> tuple[Module1Config, Module3Config, Module4Config, Module5Config, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    sampler = TPESampler(seed=sampler_seed)
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        payload = _trial_payload(trial)
        trial_root = output_root / f"trial_{trial.number:04d}"
        trial_module1, trial_module3, trial_module4, trial_module5 = _apply_trial_payload(
            replace(module1_config, output_root=trial_root / "module1"),
            module3_config,
            module4_config,
            replace(module5_config, output_root=trial_root / "module5", save_plots=False),
            payload,
        )
        try:
            artifacts = build_module1_dataset(trial_module1)
            score, details = _validation_score(artifacts, trial_module3, trial_module4, trial_module5)
            trial.set_user_attr("validation", details)
            trial.report(float(details["median_validation_sharpe"]), step=0)
            if trial.should_prune():
                raise optuna.TrialPruned("Median validation Sharpe was pruned.")
            return score
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            raise optuna.TrialPruned(str(exc)) from exc

    study.optimize(objective, n_trials=n_trials)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise ValueError("Training HPO finished without a completed trial.")

    best_trial = study.best_trial
    payload = _trial_payload(best_trial)
    best_params = {
        "objective": {
            "name": "robust_is_validation_score",
            "best_value": float(best_trial.value),
            "trial_number": int(best_trial.number),
        },
        "module1": payload["module1"],
        "module3": payload["module3"],
        "module4": payload["module4"],
        "module5": payload["module5"],
        "trial_user_attributes": best_trial.user_attrs,
    }
    best_params_path = output_root / "best_params.json"
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(output_root / "study_trials.csv", index=False)
    resolved = _apply_trial_payload(module1_config, module3_config, module4_config, module5_config, payload)
    return (*resolved, best_params_path)


def _periods_per_year(index: pd.Series) -> float:
    timestamps = pd.to_datetime(index, utc=True).sort_values()
    if timestamps.shape[0] < 2:
        return 1.0
    span_seconds = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    if span_seconds <= 0.0:
        return float(timestamps.shape[0])
    years = span_seconds / (365.25 * 24.0 * 60.0 * 60.0)
    return max(float(timestamps.shape[0] / years), 1.0)


def _phase_metrics(
    backtest: BacktestArtifacts,
    *,
    starting_capital: float,
    strategy_key: str,
    position_prefix: str,
    return_prefix: str,
) -> dict[str, float]:
    portfolio = backtest.portfolio_frame.to_pandas().sort_values("timestamp").reset_index(drop=True)
    returns = portfolio[f"{return_prefix}_return"].fillna(0.0).astype(float)
    periods_per_year = _periods_per_year(portfolio["timestamp"])
    equity = (1.0 + returns).cumprod()
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns.index) > 1 else 0.0
    instrument_frame = backtest.instrument_frame.to_pandas()
    strategy = backtest.tearsheet["strategies"][strategy_key]
    return {
        "starting_equity": float(starting_capital),
        "ending_equity": float(starting_capital * equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(strategy["annual_return"]),
        "annualized_volatility": annualized_volatility,
        "sharpe": float(strategy["sharpe"]),
        "sortino": float(strategy["sortino"]),
        "max_drawdown": float(strategy["max_drawdown"]),
        "profit_factor": float(strategy["profit_factor"]),
        "trade_count": float((instrument_frame[f"{position_prefix}_turnover"].abs() > 0.0).sum()),
        "avg_exposure": float(np.mean(np.abs(instrument_frame[f"{position_prefix}_position"].to_numpy(dtype=np.float64)))),
        "avg_turnover": float(np.mean(np.abs(instrument_frame[f"{position_prefix}_turnover"].to_numpy(dtype=np.float64)))),
    }


def _passive_metrics(backtest: BacktestArtifacts, *, starting_capital: float) -> dict[str, float]:
    portfolio = backtest.portfolio_frame.to_pandas().sort_values("timestamp").reset_index(drop=True)
    returns = portfolio["passive_return"].fillna(0.0).astype(float)
    periods_per_year = _periods_per_year(portfolio["timestamp"])
    equity = (1.0 + returns).cumprod()
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns.index) > 1 else 0.0
    passive = backtest.tearsheet["strategies"]["passive"]
    return {
        "starting_equity": float(starting_capital),
        "ending_equity": float(starting_capital * equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(passive["annual_return"]),
        "annualized_volatility": annualized_volatility,
        "sharpe": float(passive["sharpe"]),
        "max_drawdown": float(passive["max_drawdown"]),
        "profit_factor": float(passive["profit_factor"]),
    }


def _verdict(is_metrics: dict[str, float], oos_metrics: dict[str, float]) -> dict[str, str]:
    is_sharpe = float(is_metrics["sharpe"])
    oos_sharpe = float(oos_metrics["sharpe"])
    oos_mdd = abs(float(oos_metrics["max_drawdown"]))
    oos_pf = float(oos_metrics["profit_factor"])
    sharpe_ratio = 0.0 if is_sharpe <= 0.0 else oos_sharpe / is_sharpe

    if oos_sharpe >= 1.7 and oos_mdd <= 0.10 and sharpe_ratio >= 0.75:
        return {
            "status": "CLEARED",
            "summary": "OOS Sharpe holds up against IS under strict costs with limited drawdown degradation.",
        }
    if oos_sharpe >= 1.25 and oos_pf > 1.2 and oos_mdd <= 0.12:
        return {
            "status": "CANDIDATE",
            "summary": "OOS performance clears the minimum deployment gate but does not yet meet the stretch target.",
        }
    return {
        "status": "REJECTED",
        "summary": "OOS performance does not satisfy the minimum robustness gate under strict execution frictions.",
    }


def _public_strategy_label(strategy_mode: str) -> str:
    normalized = str(strategy_mode).strip().lower()
    if normalized == "meta":
        return f"{PRODUCT_NAME} Adaptive Engine"
    if normalized == "alpha_raw_long_only":
        return f"{PRODUCT_NAME} Core Engine"
    return f"{PRODUCT_NAME} Engine"


def _phase_block(title: str, metrics: dict[str, float]) -> list[str]:
    return [
        f"{title}",
        "=" * 60,
        f"Starting Equity         : ${metrics['starting_equity']:,.2f}",
        f"Ending Equity           : ${metrics['ending_equity']:,.2f}",
        f"Total Return            : {metrics['total_return']:+.2%}",
        f"Annualized Return       : {metrics['annualized_return']:+.2%}",
        f"Annualized Volatility   : {metrics['annualized_volatility']:+.2%}",
        f"Sharpe Ratio            : {metrics['sharpe']:.2f}",
        f"Max Drawdown            : {metrics['max_drawdown']:+.2%}",
        "",
    ]


def _report_text(
    *,
    timeframe: str,
    universe_mode: str,
    strategy_mode: str,
    train_period: tuple[pd.Timestamp, pd.Timestamp],
    test_period: tuple[pd.Timestamp, pd.Timestamp],
    universe: tuple[str, ...],
    is_metrics: dict[str, float],
    oos_metrics: dict[str, float],
    verdict: dict[str, str],
) -> str:
    lines = [
        f"{PRODUCT_NAME} WALKFORWARD BACKTEST ({timeframe})",
        "",
        f"Universe Mode          : {universe_mode}",
        f"Primary Engine         : {_public_strategy_label(strategy_mode)}",
        f"Universe               : {', '.join(universe)}",
        "",
    ]
    lines.extend(_phase_block(f"PHASE 1: IN-SAMPLE ({train_period[0].date()} to {train_period[1].date()})", is_metrics))
    lines.extend(_phase_block(f"PHASE 2: OUT-OF-SAMPLE ({test_period[0].date()} to {test_period[1].date()})", oos_metrics))
    lines.extend(
        [
            "[VERDICT]",
            f"Status                 : {verdict['status']}",
            verdict["summary"],
            "",
        ]
    )
    return "\n".join(lines)


def _write_curves_png(
    output_path: Path,
    is_backtest: BacktestArtifacts,
    oos_backtest: BacktestArtifacts,
    initial_capital: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_port = is_backtest.portfolio_frame.to_pandas().sort_values("timestamp").reset_index(drop=True)
    oos_port = oos_backtest.portfolio_frame.to_pandas().sort_values("timestamp").reset_index(drop=True)

    is_port["scaled_meta_equity"] = initial_capital * is_port["meta_strategy_equity"]
    is_port["scaled_alpha_equity"] = initial_capital * is_port["alpha_raw_equity"]
    is_port["scaled_passive_equity"] = initial_capital * is_port["passive_equity"]

    oos_start_capital = float(is_port["scaled_meta_equity"].iloc[-1])
    oos_port["scaled_meta_equity"] = oos_start_capital * oos_port["meta_strategy_equity"]
    oos_port["scaled_alpha_equity"] = oos_start_capital * oos_port["alpha_raw_equity"]
    oos_port["scaled_passive_equity"] = oos_start_capital * oos_port["passive_equity"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    for axis, frame, title in [
        (axes[0], is_port, "In-Sample Equity Curves"),
        (axes[1], oos_port, "Out-of-Sample Equity Curves"),
    ]:
        axis.plot(frame["timestamp"], frame["scaled_passive_equity"], label="Benchmark", linewidth=2.0)
        axis.plot(frame["timestamp"], frame["scaled_alpha_equity"], label="Engine Stream A", linewidth=1.4)
        axis.plot(frame["timestamp"], frame["scaled_meta_equity"], label="Engine Stream B", linewidth=1.6)
        axis.set_title(title)
        axis.set_ylabel("Equity")
        axis.legend()
    axes[1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _write_report_pdf(output_path: Path, report_text: str, curves_png: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = plt.imread(curves_png)
    with PdfPages(output_path) as pdf:
        text_fig = plt.figure(figsize=(8.5, 11))
        text_fig.text(0.05, 0.98, report_text, va="top", family="monospace", fontsize=10)
        pdf.savefig(text_fig, bbox_inches="tight")
        plt.close(text_fig)

        plot_fig = plt.figure(figsize=(11, 8.5))
        plt.imshow(image)
        plt.axis("off")
        pdf.savefig(plot_fig, bbox_inches="tight")
        plt.close(plot_fig)
    return output_path


def run_walkforward_backtest(config: WalkforwardConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    timeframe, bar_minutes, train_start, train_end, test_start, test_end = _resolve_timeframe_and_period(config)
    hpo_cpcv = CPCVConfig(n_groups=5, test_groups=1, min_train_samples=80, min_test_samples=20)
    eval_cpcv = CPCVConfig(n_groups=5, test_groups=1, min_train_samples=1, min_test_samples=1)
    hpo_module3 = Module3Config(cpcv=hpo_cpcv, n_jobs=1)
    hpo_module4 = Module4Config(cpcv=hpo_cpcv, n_jobs=1)
    eval_module3 = Module3Config(cpcv=eval_cpcv, n_jobs=1)
    eval_module4 = Module4Config(cpcv=eval_cpcv, n_jobs=1)
    base_module5 = Module5Config(
        benchmark_symbol=canonical_symbol(config.benchmark_symbol),
        output_root=config.output_root / "module5",
        save_plots=False,
    )

    universe_search: dict[str, Any] = {"objective": "direct_symbol_resolution"}
    if config.universe_mode == "broad_liquid_etfs":
        symbols, universe_search = _search_broad_liquid_universe(
            config,
            bar_minutes=bar_minutes,
            train_start=train_start,
            train_end=train_end,
            module3_config=eval_module3,
            module4_config=eval_module4,
            module5_config=base_module5,
            strategy_mode=config.strategy_mode,
        )
    elif config.universe_mode == "leveraged_sector_etfs":
        symbols, universe_search = _search_leveraged_sector_universe(
            config,
            bar_minutes=bar_minutes,
            train_start=train_start,
            train_end=train_end,
            module3_config=eval_module3,
            module4_config=eval_module4,
            module5_config=base_module5,
            strategy_mode=config.strategy_mode,
        )
    else:
        symbols = _resolve_symbols(config, bar_minutes=bar_minutes, train_start=train_start, train_end=train_end)
    strategy_key, position_prefix, return_prefix = _strategy_spec(config.strategy_mode)

    base_module1 = Module1Config(
        source_root=config.source_root,
        source_format=config.source_format,
        symbols=symbols,
        start=train_start.to_pydatetime(),
        end=_module1_date_end(test_end).to_pydatetime(),
        bar_minutes=bar_minutes,
        sampling_mode="time",
        vertical_barrier_days=20.0 if bar_minutes >= 1440 else 3.0,
        output_root=config.output_root / "module1",
    )

    training_module1 = replace(
        base_module1,
        start=train_start.to_pydatetime(),
        end=_module1_date_end(train_end).to_pydatetime(),
        output_root=config.output_root / "module1_train",
    )
    final_module1 = base_module1
    final_module3 = replace(eval_module3, optimized_params_path=None)
    final_module4 = replace(eval_module4, optimized_params_path=None)
    final_module5 = base_module5
    best_params_path: Path | None = None
    if config.hpo_trials > 0:
        final_module1, _, _, final_module5, best_params_path = _run_training_hpo(
            training_module1,
            hpo_module3,
            hpo_module4,
            base_module5,
            output_root=config.output_root / "hpo",
            n_trials=config.hpo_trials,
            sampler_seed=config.hpo_sampler_seed,
        )
        final_module3 = replace(eval_module3, optimized_params_path=best_params_path)
        final_module4 = replace(eval_module4, optimized_params_path=best_params_path)
        final_module1 = replace(
            final_module1,
            start=train_start.to_pydatetime(),
            end=_module1_date_end(test_end).to_pydatetime(),
            output_root=config.output_root / "module1",
        )

    full_artifacts = build_module1_dataset(final_module1)
    train_artifacts = _phase_module1_artifacts(full_artifacts, phase_start=train_start, phase_end=train_end)
    test_artifacts = _phase_module1_artifacts(full_artifacts, phase_start=test_start, phase_end=test_end)
    _ensure_phase_has_data(train_artifacts, "in_sample")
    _ensure_phase_has_data(test_artifacts, "out_of_sample")

    is_backtest, _ = _phase_backtest(
        "in_sample",
        None,
        train_artifacts,
        final_module3,
        final_module4,
        final_module5,
        strategy_mode=config.strategy_mode,
    )
    oos_backtest, oos_positions = _phase_backtest(
        "out_of_sample",
        train_artifacts,
        test_artifacts,
        final_module3,
        final_module4,
        final_module5,
        strategy_mode=config.strategy_mode,
    )

    is_metrics = _phase_metrics(
        is_backtest,
        starting_capital=config.initial_capital,
        strategy_key=strategy_key,
        position_prefix=position_prefix,
        return_prefix=return_prefix,
    )
    oos_metrics = _phase_metrics(
        oos_backtest,
        starting_capital=is_metrics["ending_equity"],
        strategy_key=strategy_key,
        position_prefix=position_prefix,
        return_prefix=return_prefix,
    )
    benchmark_metrics = {
        "in_sample": _passive_metrics(is_backtest, starting_capital=config.initial_capital),
        "out_of_sample": _passive_metrics(oos_backtest, starting_capital=is_metrics["ending_equity"]),
    }
    verdict = _verdict(is_metrics, oos_metrics)

    report_text = _report_text(
        timeframe=timeframe,
        universe_mode=config.universe_mode,
        strategy_mode=config.strategy_mode,
        train_period=(train_start, train_end),
        test_period=(test_start, test_end),
        universe=symbols,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        verdict=verdict,
    )
    report_txt = config.output_root / "walkforward_report.txt"
    report_txt.write_text(report_text, encoding="utf-8")
    curves_png = _write_curves_png(config.output_root / "walkforward_equity_curves.png", is_backtest, oos_backtest, config.initial_capital)
    report_pdf = _write_report_pdf(config.output_root / "walkforward_report.pdf", report_text, curves_png)

    payload = {
        "product_name": PRODUCT_NAME,
        "run_mode": "walkforward_local_source",
        "timeframe": timeframe,
        "universe_mode": config.universe_mode,
        "strategy_mode": config.strategy_mode,
        "strategy_label": _public_strategy_label(config.strategy_mode),
        "train_period": {"start": str(train_start.date()), "end": str(train_end.date())},
        "test_period": {"start": str(test_start.date()), "end": str(test_end.date())},
        "universe": list(symbols),
        "benchmark_symbol": canonical_symbol(config.benchmark_symbol),
        "universe_search": universe_search,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "benchmark_metrics": benchmark_metrics,
        "verdict": verdict,
        "module1": {
            "bars": int(full_artifacts.bars.height),
            "stationary": int(full_artifacts.stationary.height),
            "events": int(full_artifacts.events.height),
            "labels": int(full_artifacts.labels.height),
        },
        "oos_trade_count": int((oos_positions["bet_size"] > 0.0).sum()),
        "report_txt": str(report_txt),
        "report_pdf": str(report_pdf),
        "curves_png": str(curves_png),
    }
    if best_params_path is not None:
        payload["best_params_path"] = str(best_params_path)

    report_json = config.output_root / "walkforward_report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_json"] = str(report_json)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run a historical {PRODUCT_NAME} walkforward backtest on local data.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-format", type=str, default="auto")
    parser.add_argument("--timeframe", type=str, default="auto", choices=["auto", "60m", "1h", "1d", "d"])
    parser.add_argument("--train-start", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--benchmark-symbol", type=str, default="SPY")
    parser.add_argument(
        "--universe-mode",
        type=str,
        default="liquid_etfs",
        choices=["liquid_etfs", "broad_liquid_etfs", "tight_liquid_etfs", "leveraged_sector_etfs"],
    )
    parser.add_argument("--universe-file", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--target-universe-size", type=int, default=50)
    parser.add_argument("--broad-liquid-candidate-size", type=int, default=20)
    parser.add_argument("--broad-liquid-target-size", type=int, default=6)
    parser.add_argument("--broad-liquid-etf-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tight-universe-size", type=int, default=10)
    parser.add_argument("--tight-universe-etf-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leveraged-sector-size", type=int, default=8)
    parser.add_argument("--leveraged-sector-etf-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-median-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--min-coverage-ratio", type=float, default=0.98)
    parser.add_argument("--max-zero-volume-fraction", type=float, default=0.0)
    parser.add_argument("--strategy-mode", type=str, default="meta", choices=["meta", "alpha_raw_long_only"])
    parser.add_argument("--hpo-trials", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = run_walkforward_backtest(
            WalkforwardConfig(
                source_root=args.source_root,
                source_format=args.source_format,
                timeframe=args.timeframe,
                train_start=args.train_start,
                train_end=args.train_end,
                test_start=args.test_start,
                test_end=args.test_end,
                benchmark_symbol=args.benchmark_symbol,
                universe_mode=args.universe_mode,
                universe_file=args.universe_file,
                symbols=tuple(args.symbols) if args.symbols else None,
                target_universe_size=args.target_universe_size,
                broad_liquid_candidate_size=args.broad_liquid_candidate_size,
                broad_liquid_target_size=args.broad_liquid_target_size,
                broad_liquid_etf_only=bool(args.broad_liquid_etf_only),
                tight_universe_target_size=args.tight_universe_size,
                tight_universe_etf_only=bool(args.tight_universe_etf_only),
                leveraged_sector_target_size=args.leveraged_sector_size,
                leveraged_sector_etf_only=bool(args.leveraged_sector_etf_only),
                min_median_dollar_volume=args.min_median_dollar_volume,
                min_coverage_ratio=args.min_coverage_ratio,
                max_zero_volume_fraction=args.max_zero_volume_fraction,
                strategy_mode=args.strategy_mode,
                output_root=args.output_root,
                hpo_trials=args.hpo_trials,
            )
        )
        print(json.dumps(payload, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
