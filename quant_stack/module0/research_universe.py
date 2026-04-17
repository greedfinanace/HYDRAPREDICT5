from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


@dataclass(frozen=True)
class UniverseSelectionConfig:
    source_root: Path
    source_format: str = "auto"
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    bar_minutes: int = 60
    target_size: int = 50
    benchmark_symbol: str = "SPY"
    min_coverage_ratio: float = 0.98
    min_median_dollar_volume: float = 1_000_000.0
    max_zero_volume_fraction: float = 0.0
    etf_only: bool = True


@dataclass(frozen=True)
class UniverseSelectionArtifacts:
    config: UniverseSelectionConfig
    selected_symbols: tuple[str, ...]
    summary: pl.DataFrame


@dataclass(frozen=True)
class LiquidUniverseResearchArtifacts:
    config: UniverseSelectionConfig
    selected_symbols: tuple[str, ...]
    summary: pl.DataFrame
    returns_by_symbol: dict[str, pd.Series]
    benchmark_sessions: frozenset[pd.Timestamp]


@dataclass(frozen=True)
class TightUniverseSelectionConfig:
    source_root: Path
    source_format: str = "auto"
    train_start: str | pd.Timestamp | None = None
    train_end: str | pd.Timestamp | None = None
    bar_minutes: int = 1440
    target_size: int = 10
    candidate_pool_size: int = 40
    benchmark_symbol: str = "SPY"
    min_coverage_ratio: float = 0.98
    min_median_dollar_volume: float = 1_000_000.0
    max_zero_volume_fraction: float = 0.0
    max_pairwise_correlation: float = 0.92
    score_drawdown_weight: float = 3.0
    score_correlation_weight: float = 0.25
    score_return_weight: float = 0.5
    etf_only: bool = True


@dataclass(frozen=True)
class LeveragedSectorUniverseSelectionConfig:
    source_root: Path
    source_format: str = "auto"
    train_start: str | pd.Timestamp | None = None
    train_end: str | pd.Timestamp | None = None
    bar_minutes: int = 1440
    target_size: int = 8
    candidate_pool_size: int = 48
    benchmark_symbol: str = "SPY"
    allowed_families: tuple[str, ...] = ("sector", "leveraged")
    min_coverage_ratio: float = 0.98
    min_median_dollar_volume: float = 1_000_000.0
    max_zero_volume_fraction: float = 0.0
    max_pairwise_correlation: float = 0.90
    score_drawdown_weight: float = 1.8
    score_correlation_weight: float = 0.20
    score_return_weight: float = 0.75
    max_leveraged_symbols: int = 3
    min_seed_pool_size: int = 6
    min_selected_symbols: int = 4
    etf_only: bool = True


SECTOR_SYMBOLS = {
    "XBI",
    "XHB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
    "KBE",
    "KRE",
    "SMH",
    "SOXX",
    "XRT",
    "IYR",
    "IYT",
    "ITA",
    "IBB",
    "OIH",
    "PPA",
}

LEVERAGED_SYMBOLS = {
    "QLD",
    "SDS",
    "SH",
    "SOXL",
    "SOXS",
    "SPXU",
    "SPXL",
    "SSO",
    "TNA",
    "TQQQ",
    "TWM",
    "UPRO",
    "UVXY",
    "VXX",
    "FAZ",
    "FAS",
    "LABD",
    "LABU",
    "NAIL",
    "NUGT",
    "DUST",
    "JDST",
    "DUG",
    "DIG",
    "TBT",
    "TMF",
    "TECL",
    "TZA",
    "UDOW",
    "UWM",
    "UYG",
    "QLD",
    "RWM",
    "SPXS",
}


def _coerce_date(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value).normalize()


def _infer_source_format(source_root: Path, source_format: str) -> str:
    normalized = str(source_format).strip().lower()
    if normalized in {"txt", "parquet"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"Unsupported source format: {source_format}")
    if next(source_root.rglob("*.parquet"), None) is not None:
        return "parquet"
    if next(source_root.rglob("*.txt"), None) is not None:
        return "txt"
    raise FileNotFoundError(f"No supported source files were found under {source_root}")


def _symbol_from_path(path: Path, source_format: str) -> str:
    if source_format == "txt":
        return path.name.split(".")[0].upper()
    if path.stem.isdigit():
        return path.parent.name.upper()
    return path.stem.split(".")[0].upper()


def _path_is_etf(path: Path, source_root: Path) -> bool:
    rel = [part.lower() for part in path.relative_to(source_root).parts]
    return any("etf" in part for part in rel)


def _symbol_family(symbol: str) -> str:
    normalized = str(symbol).upper()
    if normalized in LEVERAGED_SYMBOLS:
        return "leveraged"
    if normalized in SECTOR_SYMBOLS:
        return "sector"
    return "other"


def _iter_candidate_paths(source_root: Path, source_format: str, etf_only: bool) -> list[Path]:
    pattern = "*.txt" if source_format == "txt" else "*.parquet"
    candidates = [path for path in sorted(source_root.rglob(pattern)) if path.is_file()]
    if not etf_only:
        return candidates
    return [path for path in candidates if _path_is_etf(path, source_root)]


def _group_candidate_paths(source_root: Path, source_format: str, etf_only: bool) -> dict[str, list[Path]]:
    by_symbol: dict[str, list[Path]] = {}
    for path in _iter_candidate_paths(source_root, source_format, etf_only):
        by_symbol.setdefault(_symbol_from_path(path, source_format), []).append(path)
    return by_symbol


def _collect_symbol_frames(
    by_symbol: dict[str, list[Path]],
    *,
    source_format: str,
    bar_minutes: int,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, paths in sorted(by_symbol.items()):
        if source_format == "txt":
            if len(paths) != 1:
                raise ValueError(f"Expected one txt source file per symbol, found {len(paths)} for {symbol}")
            frame = _read_txt_summary(paths[0], bar_minutes, start, end)
        else:
            frame = _read_parquet_summary(paths, bar_minutes, start, end)
        if frame.empty:
            continue
        frame = frame.sort_values("session_date").drop_duplicates(subset=["session_date"], keep="last").reset_index(
            drop=True
        )
        frames[symbol] = frame
    return frames


def _periods_per_year(index: pd.Index | pd.Series) -> float:
    timestamps = pd.to_datetime(index, utc=True).sort_values()
    if timestamps.shape[0] < 2:
        return 1.0
    span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    if span_seconds <= 0.0:
        return float(timestamps.shape[0])
    years = span_seconds / (365.25 * 24.0 * 60.0 * 60.0)
    return max(float(timestamps.shape[0] / years), 1.0)


def _portfolio_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.fillna(0.0).astype(float)
    if clean.empty:
        return {
            "annual_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
        }

    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    periods_per_year = _periods_per_year(clean.index)
    mean_return = float(clean.mean())
    std_return = float(clean.std(ddof=1))
    downside = clean.where(clean < 0.0, 0.0)
    downside_std = float(downside.std(ddof=1))
    annual_return = float((equity.iloc[-1] ** (periods_per_year / max(clean.shape[0], 1))) - 1.0)
    annualized_volatility = float(std_return * np.sqrt(periods_per_year)) if np.isfinite(std_return) else 0.0
    sharpe = 0.0 if std_return == 0.0 or not np.isfinite(std_return) else mean_return / std_return * np.sqrt(periods_per_year)
    sortino = 0.0 if downside_std == 0.0 or not np.isfinite(downside_std) else mean_return / downside_std * np.sqrt(periods_per_year)
    mdd = float(drawdown.min())
    calmar = 0.0 if mdd == 0.0 else annual_return / abs(mdd)
    gross_gain = float(clean[clean > 0.0].sum())
    gross_loss = float(abs(clean[clean < 0.0].sum()))
    profit_factor = float("inf") if gross_loss == 0.0 and gross_gain > 0.0 else (gross_gain / gross_loss if gross_loss > 0.0 else 0.0)
    return {
        "annual_return": annual_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "calmar": float(calmar),
        "profit_factor": float(profit_factor),
    }


def _average_pairwise_correlation(returns_frame: pd.DataFrame) -> float:
    if returns_frame.shape[1] <= 1:
        return 0.0
    correlation = returns_frame.corr().abs()
    if correlation.empty:
        return 0.0
    values = correlation.to_numpy(dtype=np.float64)
    mask = ~np.eye(values.shape[0], dtype=bool)
    finite_values = values[mask]
    finite_values = finite_values[np.isfinite(finite_values)]
    return float(finite_values.mean()) if finite_values.size else 0.0


def _basket_score(
    returns_frame: pd.DataFrame,
    *,
    score_drawdown_weight: float,
    score_correlation_weight: float,
    score_return_weight: float,
) -> tuple[float, dict[str, float]]:
    aligned = returns_frame.dropna(axis=0, how="any")
    if aligned.empty:
        return float("-inf"), _portfolio_metrics(pd.Series(dtype=float))
    basket_returns = aligned.mean(axis=1)
    metrics = _portfolio_metrics(basket_returns)
    metrics["avg_pairwise_correlation"] = _average_pairwise_correlation(aligned)
    score = (
        float(metrics["sharpe"])
        - float(score_drawdown_weight) * abs(float(metrics["max_drawdown"]))
        - float(score_correlation_weight) * float(metrics["avg_pairwise_correlation"])
        + float(score_return_weight) * float(metrics["annual_return"])
    )
    return score, metrics


def _score_candidate_basket(
    returns_frame: pd.DataFrame,
    symbols: list[str],
    *,
    score_drawdown_weight: float,
    score_correlation_weight: float,
    score_return_weight: float,
) -> tuple[float, dict[str, float]]:
    frame = returns_frame.loc[:, symbols].dropna(axis=0, how="any")
    if frame.empty:
        return float("-inf"), _portfolio_metrics(pd.Series(dtype=float))
    return _basket_score(
        frame,
        score_drawdown_weight=score_drawdown_weight,
        score_correlation_weight=score_correlation_weight,
        score_return_weight=score_return_weight,
    )


def _greedy_multi_start_basket(
    *,
    benchmark_symbol: str,
    benchmark_sessions: set[pd.Timestamp],
    returns_by_symbol: dict[str, pd.Series],
    candidate_symbols: list[str],
    target_size: int,
    score_drawdown_weight: float,
    score_correlation_weight: float,
    score_return_weight: float,
    max_pairwise_correlation: float,
    min_selected_symbols: int,
    max_leveraged_symbols: int | None = None,
    seed_symbols: list[str] | None = None,
) -> tuple[list[str], float, dict[str, float]]:
    all_symbols = [benchmark_symbol] + [symbol for symbol in candidate_symbols if symbol != benchmark_symbol]
    returns_frame = pd.DataFrame(
        {symbol: returns_by_symbol.get(symbol, pd.Series(dtype=float)).reindex(sorted(benchmark_sessions)) for symbol in all_symbols}
    )
    returns_frame.index = pd.Index(sorted(benchmark_sessions), name="session_date")

    if seed_symbols is None:
        seed_pool = candidate_symbols[: max(min(len(candidate_symbols), max(target_size * 2, min_selected_symbols)), 1)]
    else:
        seed_pool = [symbol for symbol in seed_symbols if symbol in candidate_symbols and symbol != benchmark_symbol]
    if not seed_pool:
        seed_pool = [None]  # type: ignore[list-item]

    best_selected = [benchmark_symbol]
    best_score = float("-inf")
    best_metrics: dict[str, float] = _portfolio_metrics(pd.Series(dtype=float))

    def _evaluate(selected: list[str]) -> tuple[float, dict[str, float]]:
        return _score_candidate_basket(
            returns_frame,
            selected,
            score_drawdown_weight=score_drawdown_weight,
            score_correlation_weight=score_correlation_weight,
            score_return_weight=score_return_weight,
        )

    candidate_families = {symbol: _symbol_family(symbol) for symbol in candidate_symbols}

    for seed in seed_pool:
        selected = [benchmark_symbol]
        if seed is not None:
            selected.append(seed)

        current_score, current_metrics = _evaluate(selected)
        if not np.isfinite(current_score):
            continue

        remaining = [symbol for symbol in candidate_symbols if symbol not in selected]
        while remaining and len(selected) < max(int(target_size), 1):
            best_choice: str | None = None
            best_choice_score = float("-inf")
            best_choice_metrics: dict[str, float] | None = None
            for symbol in remaining:
                family = candidate_families.get(symbol, "other")
                if family == "leveraged" and max_leveraged_symbols is not None:
                    leveraged_count = sum(candidate_families.get(item, "other") == "leveraged" for item in selected)
                    if leveraged_count >= int(max_leveraged_symbols):
                        continue
                trial = selected + [symbol]
                score, metrics = _evaluate(trial)
                if not np.isfinite(score):
                    continue
                if len(trial) > 2 and metrics.get("avg_pairwise_correlation", 0.0) > float(max_pairwise_correlation):
                    continue
                if score > best_choice_score:
                    best_choice = symbol
                    best_choice_score = score
                    best_choice_metrics = metrics
            if best_choice is None:
                break
            if len(selected) >= int(min_selected_symbols) and best_choice_score <= current_score:
                break
            selected.append(best_choice)
            remaining.remove(best_choice)
            current_score = best_choice_score
            current_metrics = best_choice_metrics or current_metrics

        # one-pass local refinement: try swapping each non-benchmark symbol once
        improved = True
        while improved:
            improved = False
            current_selected = list(selected)
            current_score, current_metrics = _evaluate(current_selected)
            remaining = [symbol for symbol in candidate_symbols if symbol not in current_selected]
            for idx in range(1, len(current_selected)):
                for symbol in remaining:
                    family = candidate_families.get(symbol, "other")
                    if family == "leveraged" and max_leveraged_symbols is not None:
                        leveraged_count = sum(candidate_families.get(item, "other") == "leveraged" for item in current_selected if item != current_selected[idx])
                        if leveraged_count >= int(max_leveraged_symbols):
                            continue
                    trial = current_selected[:idx] + [symbol] + current_selected[idx + 1 :]
                    score, metrics = _evaluate(trial)
                    if not np.isfinite(score):
                        continue
                    if len(trial) > 2 and metrics.get("avg_pairwise_correlation", 0.0) > float(max_pairwise_correlation):
                        continue
                    if score > current_score + 1e-12:
                        selected = trial
                        current_score = score
                        current_metrics = metrics
                        improved = True
                        break
                if improved:
                    break

        if current_score > best_score:
            best_selected = list(selected)
            best_score = float(current_score)
            best_metrics = dict(current_metrics)

    if not np.isfinite(best_score):
        raise ValueError("Unable to construct a viable basket from the provided symbols.")
    return best_selected, float(best_score), best_metrics


def _finalize_summary_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {
                "symbol": [],
                "observations": [],
                "session_count": [],
                "coverage_ratio": [],
                "median_dollar_volume": [],
                "zero_volume_fraction": [],
                "start_date": [],
                "end_date": [],
            }
        )
    return pl.DataFrame(rows).sort(["median_dollar_volume", "symbol"], descending=[True, False])


def _read_txt_summary(path: Path, bar_minutes: int, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, usecols=["<DATE>", "<PER>", "<CLOSE>", "<VOL>"])
    except (ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["session_date", "close", "volume", "dollar_volume"])
    frame = frame.loc[frame["<PER>"].astype(int) == int(bar_minutes)].copy()
    if frame.empty:
        return frame
    frame["session_date"] = pd.to_datetime(frame["<DATE>"].astype(str), format="%Y%m%d", utc=False).dt.normalize()
    if start is not None:
        frame = frame.loc[frame["session_date"] >= start]
    if end is not None:
        frame = frame.loc[frame["session_date"] <= end]
    if frame.empty:
        return frame
    frame = frame.rename(columns={"<CLOSE>": "close", "<VOL>": "volume"})
    frame["dollar_volume"] = frame["close"].astype(float) * frame["volume"].astype(float)
    return frame.loc[:, ["session_date", "close", "volume", "dollar_volume"]]


def _read_parquet_summary(paths: list[Path], bar_minutes: int, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        if "source_bar_minutes" in frame.columns:
            frame = frame.loc[frame["source_bar_minutes"].astype(int) == int(bar_minutes)].copy()
        elif "bar_minutes" in frame.columns:
            frame = frame.loc[frame["bar_minutes"].astype(int) == int(bar_minutes)].copy()
        if frame.empty:
            continue
        if "timestamp" in frame.columns:
            session_dates = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
        elif "<DATE>" in frame.columns:
            session_dates = pd.to_datetime(frame["<DATE>"].astype(str), format="%Y%m%d", utc=False).dt.normalize()
        else:
            continue
        frame = frame.copy()
        frame["session_date"] = session_dates
        if start is not None:
            frame = frame.loc[frame["session_date"] >= start]
        if end is not None:
            frame = frame.loc[frame["session_date"] <= end]
        if frame.empty:
            continue
        close_column = "close" if "close" in frame.columns else "<CLOSE>"
        volume_column = "volume" if "volume" in frame.columns else "<VOL>"
        frame["dollar_volume"] = frame[close_column].astype(float) * frame[volume_column].astype(float)
        frame = frame.rename(columns={close_column: "close", volume_column: "volume"})
        frames.append(frame.loc[:, ["session_date", "close", "volume", "dollar_volume"]])
    if not frames:
        return pd.DataFrame(columns=["session_date", "close", "volume", "dollar_volume"])
    return pd.concat(frames, ignore_index=True).sort_values("session_date")


def select_liquid_etf_universe(config: UniverseSelectionConfig) -> UniverseSelectionArtifacts:
    source_root = Path(config.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Universe source root does not exist: {source_root}")

    source_format = _infer_source_format(source_root, config.source_format)
    start = _coerce_date(config.start)
    end = _coerce_date(config.end)

    by_symbol = _group_candidate_paths(source_root, source_format, config.etf_only)
    if not by_symbol:
        raise FileNotFoundError(f"No candidate symbols were found under {source_root}")

    benchmark_symbol = str(config.benchmark_symbol).upper()
    if benchmark_symbol not in by_symbol:
        raise ValueError(f"Benchmark symbol {benchmark_symbol} was not found in the candidate universe.")

    frames = _collect_symbol_frames(
        by_symbol,
        source_format=source_format,
        bar_minutes=config.bar_minutes,
        start=start,
        end=end,
    )

    summaries: dict[str, dict[str, Any]] = {}
    session_sets: dict[str, set[pd.Timestamp]] = {}
    for symbol, frame in sorted(frames.items()):
        sessions = {pd.Timestamp(value).normalize() for value in frame["session_date"].drop_duplicates().tolist()}
        session_sets[symbol] = sessions
        zero_volume_fraction = float((frame["volume"].astype(float) <= 0.0).mean()) if len(frame.index) else 1.0
        summaries[symbol] = {
            "symbol": symbol,
            "observations": int(len(frame.index)),
            "session_count": int(len(sessions)),
            "median_dollar_volume": float(frame["dollar_volume"].median()),
            "zero_volume_fraction": zero_volume_fraction,
            "start_date": str(frame["session_date"].min().date()),
            "end_date": str(frame["session_date"].max().date()),
        }

    if benchmark_symbol not in session_sets:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} does not contain bar_minutes={config.bar_minutes} data in the requested range."
        )

    benchmark_sessions = session_sets[benchmark_symbol]
    benchmark_summary = summaries[benchmark_symbol]
    if start is not None and pd.Timestamp(benchmark_summary["start_date"]) > start:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} starts after the requested range "
            f"for bar_minutes={config.bar_minutes}: {benchmark_summary['start_date']} > {start.date()}"
        )
    if end is not None and pd.Timestamp(benchmark_summary["end_date"]) < end:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} ends before the requested range "
            f"for bar_minutes={config.bar_minutes}: {benchmark_summary['end_date']} < {end.date()}"
        )

    ranked_rows: list[dict[str, Any]] = []
    for symbol, summary in summaries.items():
        coverage_ratio = (
            float(len(session_sets[symbol] & benchmark_sessions) / len(benchmark_sessions))
            if benchmark_sessions
            else 0.0
        )
        row = dict(summary)
        row["coverage_ratio"] = coverage_ratio
        ranked_rows.append(row)

    eligible = [
        row
        for row in ranked_rows
        if row["coverage_ratio"] >= float(config.min_coverage_ratio)
        and row["median_dollar_volume"] >= float(config.min_median_dollar_volume)
        and row["zero_volume_fraction"] <= float(config.max_zero_volume_fraction)
    ]
    eligible.sort(key=lambda row: (-float(row["median_dollar_volume"]), str(row["symbol"])))

    selected: list[str] = []
    for row in eligible:
        symbol = str(row["symbol"])
        if symbol == benchmark_symbol:
            continue
        selected.append(symbol)
        if len(selected) >= max(int(config.target_size) - 1, 0):
            break

    if benchmark_symbol in summaries and benchmark_symbol not in selected:
        selected.insert(0, benchmark_symbol)
    if not selected:
        raise ValueError("Universe selection produced no eligible symbols under the configured filters.")

    selected_rows = [row for row in ranked_rows if str(row["symbol"]) in set(selected)]
    if benchmark_symbol not in {str(row["symbol"]) for row in selected_rows}:
        selected_rows.append(dict(summaries[benchmark_symbol], coverage_ratio=1.0))
    selected_rows.sort(key=lambda row: (str(row["symbol"]) != benchmark_symbol, -float(row["median_dollar_volume"]), str(row["symbol"])))

    return UniverseSelectionArtifacts(
        config=config,
        selected_symbols=tuple(row["symbol"] for row in selected_rows),
        summary=_finalize_summary_frame(selected_rows),
    )


def prepare_liquid_etf_research_artifacts(config: UniverseSelectionConfig) -> LiquidUniverseResearchArtifacts:
    selection = select_liquid_etf_universe(config)
    source_root = Path(config.source_root)
    source_format = _infer_source_format(source_root, config.source_format)
    start = _coerce_date(config.start)
    end = _coerce_date(config.end)
    by_symbol = _group_candidate_paths(source_root, source_format, config.etf_only)
    selected_candidates = {symbol: by_symbol[symbol] for symbol in selection.selected_symbols if symbol in by_symbol}
    frames = _collect_symbol_frames(
        selected_candidates,
        source_format=source_format,
        bar_minutes=config.bar_minutes,
        start=start,
        end=end,
    )
    benchmark_symbol = str(config.benchmark_symbol).upper()
    if benchmark_symbol not in frames:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} does not contain bar_minutes={config.bar_minutes} data in the requested range."
        )

    benchmark_frame = frames[benchmark_symbol]
    benchmark_sessions = frozenset(
        pd.Timestamp(value).normalize() for value in benchmark_frame["session_date"].drop_duplicates().tolist()
    )

    returns_by_symbol: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        metrics_source = frame.loc[:, ["session_date", "close"]].copy()
        metrics_source["return"] = metrics_source["close"].astype(float).pct_change()
        returns = metrics_source.dropna(subset=["return"]).set_index("session_date")["return"].astype(float)
        returns_by_symbol[symbol] = returns

    return LiquidUniverseResearchArtifacts(
        config=config,
        selected_symbols=selection.selected_symbols,
        summary=selection.summary,
        returns_by_symbol=returns_by_symbol,
        benchmark_sessions=benchmark_sessions,
    )


def select_tight_liquid_etf_universe(config: TightUniverseSelectionConfig) -> UniverseSelectionArtifacts:
    source_root = Path(config.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Universe source root does not exist: {source_root}")

    start = _coerce_date(config.train_start)
    end = _coerce_date(config.train_end)
    if start is None or end is None:
        raise ValueError("Tight universe selection requires train_start and train_end.")

    source_format = _infer_source_format(source_root, config.source_format)
    by_symbol = _group_candidate_paths(source_root, source_format, config.etf_only)
    if not by_symbol:
        raise FileNotFoundError(f"No candidate symbols were found under {source_root}")

    benchmark_symbol = str(config.benchmark_symbol).upper()
    if benchmark_symbol not in by_symbol:
        raise ValueError(f"Benchmark symbol {benchmark_symbol} was not found in the candidate universe.")

    frames = _collect_symbol_frames(
        by_symbol,
        source_format=source_format,
        bar_minutes=config.bar_minutes,
        start=start,
        end=end,
    )
    if benchmark_symbol not in frames:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} does not contain bar_minutes={config.bar_minutes} data in the requested range."
        )

    benchmark_frame = frames[benchmark_symbol]
    benchmark_sessions = {pd.Timestamp(value).normalize() for value in benchmark_frame["session_date"].drop_duplicates().tolist()}
    benchmark_summary = {
        "symbol": benchmark_symbol,
        "observations": int(len(benchmark_frame.index)),
        "session_count": int(len(benchmark_sessions)),
        "median_dollar_volume": float(benchmark_frame["dollar_volume"].median()),
        "zero_volume_fraction": float((benchmark_frame["volume"].astype(float) <= 0.0).mean()) if len(benchmark_frame.index) else 1.0,
        "start_date": str(benchmark_frame["session_date"].min().date()),
        "end_date": str(benchmark_frame["session_date"].max().date()),
    }
    if pd.Timestamp(benchmark_summary["start_date"]) > start or pd.Timestamp(benchmark_summary["end_date"]) < end:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} does not span the requested training window "
            f"{start.date()} to {end.date()}."
        )

    rows: list[dict[str, Any]] = []
    returns_by_symbol: dict[str, pd.Series] = {}
    for symbol, frame in sorted(frames.items()):
        sessions = {pd.Timestamp(value).normalize() for value in frame["session_date"].drop_duplicates().tolist()}
        coverage_ratio = float(len(sessions & benchmark_sessions) / len(benchmark_sessions)) if benchmark_sessions else 0.0
        zero_volume_fraction = float((frame["volume"].astype(float) <= 0.0).mean()) if len(frame.index) else 1.0
        metrics_source = frame.loc[:, ["session_date", "close"]].copy()
        metrics_source["return"] = metrics_source["close"].astype(float).pct_change()
        returns = metrics_source.dropna(subset=["return"]).set_index("session_date")["return"].astype(float)
        returns_by_symbol[symbol] = returns
        symbol_metrics = _portfolio_metrics(returns)
        row = {
            "symbol": symbol,
            "observations": int(len(frame.index)),
            "session_count": int(len(sessions)),
            "coverage_ratio": coverage_ratio,
            "median_dollar_volume": float(frame["dollar_volume"].median()),
            "zero_volume_fraction": zero_volume_fraction,
            "start_date": str(frame["session_date"].min().date()),
            "end_date": str(frame["session_date"].max().date()),
            "train_annual_return": float(symbol_metrics["annual_return"]),
            "train_sharpe": float(symbol_metrics["sharpe"]),
            "train_max_drawdown": float(symbol_metrics["max_drawdown"]),
            "train_sortino": float(symbol_metrics["sortino"]),
            "train_calmar": float(symbol_metrics["calmar"]),
            "train_profit_factor": float(symbol_metrics["profit_factor"]),
        }
        rows.append(row)

    eligible_rows = [
        row
        for row in rows
        if row["coverage_ratio"] >= float(config.min_coverage_ratio)
        and row["median_dollar_volume"] >= float(config.min_median_dollar_volume)
        and row["zero_volume_fraction"] <= float(config.max_zero_volume_fraction)
    ]
    if not eligible_rows:
        raise ValueError("Tight universe selection produced no eligible symbols under the configured filters.")

    eligible_rows.sort(
        key=lambda row: (
            -float(row["train_sharpe"]),
            abs(float(row["train_max_drawdown"])),
            -float(row["median_dollar_volume"]),
            str(row["symbol"]),
        )
    )

    candidate_symbols = [str(row["symbol"]) for row in eligible_rows if str(row["symbol"]) != benchmark_symbol]
    pool_size = max(int(config.candidate_pool_size), int(config.target_size))
    candidate_symbols = candidate_symbols[:pool_size]

    selected_symbols = [benchmark_symbol]
    selected_details: list[dict[str, Any]] = [benchmark_summary]

    returns_frame = pd.DataFrame({symbol: series.reindex(sorted(benchmark_sessions)) for symbol, series in returns_by_symbol.items()})
    returns_frame.index = pd.Index(sorted(benchmark_sessions), name="session_date")

    current_score, current_metrics = _basket_score(
        returns_frame[selected_symbols],
        score_drawdown_weight=config.score_drawdown_weight,
        score_correlation_weight=config.score_correlation_weight,
        score_return_weight=config.score_return_weight,
    )
    min_selected_symbols = min(5, max(int(config.target_size), 1))

    remaining = [symbol for symbol in candidate_symbols if symbol not in selected_symbols]
    while remaining and len(selected_symbols) < max(int(config.target_size), 1):
        best_choice: str | None = None
        best_score = float("-inf")
        best_metrics: dict[str, float] | None = None
        best_row: dict[str, Any] | None = None
        for symbol in remaining:
            trial_symbols = selected_symbols + [symbol]
            trial_frame = returns_frame[trial_symbols]
            score, metrics = _basket_score(
                trial_frame,
                score_drawdown_weight=config.score_drawdown_weight,
                score_correlation_weight=config.score_correlation_weight,
                score_return_weight=config.score_return_weight,
            )
            if trial_frame.dropna(axis=0, how="any").empty:
                continue
            if len(trial_symbols) > 2 and metrics["avg_pairwise_correlation"] > float(config.max_pairwise_correlation):
                continue
            if score > best_score:
                best_choice = symbol
                best_score = score
                best_metrics = metrics
                best_row = next(row for row in eligible_rows if str(row["symbol"]) == symbol)
        if best_choice is None:
            break
        if len(selected_symbols) >= min_selected_symbols and best_score < current_score:
            break
        selected_symbols.append(best_choice)
        selected_details.append(dict(best_row or {}))
        current_score = best_score
        current_metrics = best_metrics or current_metrics
        remaining.remove(best_choice)

    selected_rows = [row for row in rows if str(row["symbol"]) in set(selected_symbols)]
    for row in selected_rows:
        if row["symbol"] == benchmark_symbol:
            row["selection_score"] = float(current_score)
        else:
            symbol_returns = returns_frame.loc[:, [benchmark_symbol, str(row["symbol"])]] .dropna(axis=0, how="any")
            symbol_score, symbol_metrics = _basket_score(
                symbol_returns,
                score_drawdown_weight=config.score_drawdown_weight,
                score_correlation_weight=config.score_correlation_weight,
                score_return_weight=config.score_return_weight,
            )
            row["selection_score"] = float(symbol_score)
            row["train_annual_return"] = float(symbol_metrics["annual_return"])
            row["train_sharpe"] = float(symbol_metrics["sharpe"])
            row["train_max_drawdown"] = float(symbol_metrics["max_drawdown"])
            row["avg_pairwise_correlation"] = float(symbol_metrics["avg_pairwise_correlation"])
    selected_rows.sort(
        key=lambda row: (
            str(row["symbol"]) != benchmark_symbol,
            -float(row.get("selection_score", row["train_sharpe"])),
            -float(row["median_dollar_volume"]),
            str(row["symbol"]),
        )
    )

    return UniverseSelectionArtifacts(
        config=config,  # type: ignore[arg-type]
        selected_symbols=tuple(row["symbol"] for row in selected_rows),
        summary=_finalize_summary_frame(selected_rows),
    )


def select_leveraged_sector_universe(config: LeveragedSectorUniverseSelectionConfig) -> UniverseSelectionArtifacts:
    source_root = Path(config.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Universe source root does not exist: {source_root}")

    start = _coerce_date(config.train_start)
    end = _coerce_date(config.train_end)
    if start is None or end is None:
        raise ValueError("Leveraged sector universe selection requires train_start and train_end.")

    source_format = _infer_source_format(source_root, config.source_format)
    by_symbol = _group_candidate_paths(source_root, source_format, config.etf_only)
    if not by_symbol:
        raise FileNotFoundError(f"No candidate symbols were found under {source_root}")

    benchmark_symbol = str(config.benchmark_symbol).upper()
    if benchmark_symbol not in by_symbol:
        raise ValueError(f"Benchmark symbol {benchmark_symbol} was not found in the candidate universe.")

    frames = _collect_symbol_frames(
        by_symbol,
        source_format=source_format,
        bar_minutes=config.bar_minutes,
        start=start,
        end=end,
    )
    if benchmark_symbol not in frames:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} does not contain bar_minutes={config.bar_minutes} data in the requested range."
        )

    benchmark_frame = frames[benchmark_symbol]
    benchmark_sessions = {pd.Timestamp(value).normalize() for value in benchmark_frame["session_date"].drop_duplicates().tolist()}
    benchmark_summary = {
        "symbol": benchmark_symbol,
        "observations": int(len(benchmark_frame.index)),
        "session_count": int(len(benchmark_sessions)),
        "median_dollar_volume": float(benchmark_frame["dollar_volume"].median()),
        "zero_volume_fraction": float((benchmark_frame["volume"].astype(float) <= 0.0).mean()) if len(benchmark_frame.index) else 1.0,
        "start_date": str(benchmark_frame["session_date"].min().date()),
        "end_date": str(benchmark_frame["session_date"].max().date()),
        "family": _symbol_family(benchmark_symbol),
    }
    if pd.Timestamp(benchmark_summary["start_date"]) > start or pd.Timestamp(benchmark_summary["end_date"]) < end:
        raise ValueError(
            f"Benchmark symbol {benchmark_symbol} does not span the requested training window "
            f"{start.date()} to {end.date()}."
        )

    rows: list[dict[str, Any]] = []
    returns_by_symbol: dict[str, pd.Series] = {}
    for symbol, frame in sorted(frames.items()):
        sessions = {pd.Timestamp(value).normalize() for value in frame["session_date"].drop_duplicates().tolist()}
        coverage_ratio = float(len(sessions & benchmark_sessions) / len(benchmark_sessions)) if benchmark_sessions else 0.0
        zero_volume_fraction = float((frame["volume"].astype(float) <= 0.0).mean()) if len(frame.index) else 1.0
        metrics_source = frame.loc[:, ["session_date", "close"]].copy()
        metrics_source["return"] = metrics_source["close"].astype(float).pct_change()
        returns = metrics_source.dropna(subset=["return"]).set_index("session_date")["return"].astype(float)
        returns_by_symbol[symbol] = returns
        symbol_metrics = _portfolio_metrics(returns)
        family = _symbol_family(symbol)
        row = {
            "symbol": symbol,
            "family": family,
            "observations": int(len(frame.index)),
            "session_count": int(len(sessions)),
            "coverage_ratio": coverage_ratio,
            "median_dollar_volume": float(frame["dollar_volume"].median()),
            "zero_volume_fraction": zero_volume_fraction,
            "start_date": str(frame["session_date"].min().date()),
            "end_date": str(frame["session_date"].max().date()),
            "train_annual_return": float(symbol_metrics["annual_return"]),
            "train_sharpe": float(symbol_metrics["sharpe"]),
            "train_max_drawdown": float(symbol_metrics["max_drawdown"]),
            "train_sortino": float(symbol_metrics["sortino"]),
            "train_calmar": float(symbol_metrics["calmar"]),
            "train_profit_factor": float(symbol_metrics["profit_factor"]),
        }
        rows.append(row)

    eligible_rows = [
        row
        for row in rows
        if row["coverage_ratio"] >= float(config.min_coverage_ratio)
        and row["median_dollar_volume"] >= float(config.min_median_dollar_volume)
        and row["zero_volume_fraction"] <= float(config.max_zero_volume_fraction)
        and row["family"] in {str(value) for value in config.allowed_families}
    ]
    if not eligible_rows:
        raise ValueError("Leveraged sector universe selection produced no eligible symbols under the configured filters.")

    eligible_rows.sort(
        key=lambda row: (
            0 if row["family"] == "leveraged" else 1,
            -float(row["train_sharpe"]),
            abs(float(row["train_max_drawdown"])),
            -float(row["median_dollar_volume"]),
            str(row["symbol"]),
        )
    )

    candidate_symbols = [str(row["symbol"]) for row in eligible_rows if str(row["symbol"]) != benchmark_symbol]
    pool_size = max(int(config.candidate_pool_size), int(config.target_size))
    candidate_symbols = candidate_symbols[:pool_size]

    leveraged_candidates = [symbol for symbol in candidate_symbols if _symbol_family(symbol) == "leveraged"]
    seed_pool = leveraged_candidates[: max(1, min(len(leveraged_candidates), int(config.min_seed_pool_size)))]
    if not seed_pool:
        seed_pool = candidate_symbols[: max(1, min(len(candidate_symbols), int(config.min_seed_pool_size)))]

    selected_symbols, selection_score, selection_metrics = _greedy_multi_start_basket(
        benchmark_symbol=benchmark_symbol,
        benchmark_sessions=benchmark_sessions,
        returns_by_symbol=returns_by_symbol,
        candidate_symbols=candidate_symbols,
        target_size=config.target_size,
        score_drawdown_weight=config.score_drawdown_weight,
        score_correlation_weight=config.score_correlation_weight,
        score_return_weight=config.score_return_weight,
        max_pairwise_correlation=config.max_pairwise_correlation,
        min_selected_symbols=config.min_selected_symbols,
        max_leveraged_symbols=config.max_leveraged_symbols,
        seed_symbols=seed_pool,
    )

    selected_rows = [row for row in rows if str(row["symbol"]) in set(selected_symbols)]
    for row in selected_rows:
        if row["symbol"] == benchmark_symbol:
            row["selection_score"] = float(selection_score)
            row["selection_family"] = row["family"]
        else:
            row["selection_family"] = row["family"]
            symbol_returns = pd.DataFrame(
                {
                    benchmark_symbol: returns_by_symbol[benchmark_symbol].reindex(sorted(benchmark_sessions)),
                    str(row["symbol"]): returns_by_symbol[str(row["symbol"])].reindex(sorted(benchmark_sessions)),
                }
            )
            symbol_score, symbol_metrics = _basket_score(
                symbol_returns,
                score_drawdown_weight=config.score_drawdown_weight,
                score_correlation_weight=config.score_correlation_weight,
                score_return_weight=config.score_return_weight,
            )
            row["selection_score"] = float(symbol_score)
            row["train_annual_return"] = float(symbol_metrics["annual_return"])
            row["train_sharpe"] = float(symbol_metrics["sharpe"])
            row["train_max_drawdown"] = float(symbol_metrics["max_drawdown"])
            row["avg_pairwise_correlation"] = float(symbol_metrics["avg_pairwise_correlation"])

    selected_rows.sort(
        key=lambda row: (
            str(row["symbol"]) != benchmark_symbol,
            0 if row.get("selection_family", row.get("family", "other")) == "leveraged" else 1,
            -float(row.get("selection_score", row["train_sharpe"])),
            -float(row["median_dollar_volume"]),
            str(row["symbol"]),
        )
    )

    selected_summary = _finalize_summary_frame(selected_rows)
    if "family" not in selected_summary.columns:
        selected_summary = selected_summary.with_columns(pl.lit(None).alias("family"))
    return UniverseSelectionArtifacts(
        config=config,  # type: ignore[arg-type]
        selected_symbols=tuple(row["symbol"] for row in selected_rows),
        summary=selected_summary,
    )
