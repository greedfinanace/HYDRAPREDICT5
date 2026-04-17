from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from quant_stack.module3 import AlphaDataset, AlphaModelArtifacts, CPCVConfig


@dataclass(frozen=True)
class Module4Config:
    cpcv: CPCVConfig = field(default_factory=CPCVConfig)
    probability_floor: float = 0.50
    bet_sizing_power: float = 1.0
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 25
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    feature_fraction: float | None = None
    reg_lambda: float = 1.0
    apply_regime_penalty: bool = True
    random_state: int = 42
    n_jobs: int = 1
    optimized_params_path: Path | None = None

    def effective_feature_fraction(self) -> float:
        if self.feature_fraction is not None:
            return float(self.feature_fraction)
        return float(self.colsample_bytree)

    def with_optimized_params(self) -> Module4Config:
        if self.optimized_params_path is None or not Path(self.optimized_params_path).exists():
            return self

        payload = json.loads(Path(self.optimized_params_path).read_text(encoding="utf-8"))
        params = payload.get("module4", {})
        allowed = {
            "probability_floor",
            "bet_sizing_power",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "feature_fraction",
            "reg_lambda",
            "apply_regime_penalty",
        }
        overrides = {key: params[key] for key in allowed if key in params}
        if not overrides:
            return self
        return replace(self, **overrides)


@dataclass(frozen=True)
class MetaDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    all_rows: pd.DataFrame


@dataclass(frozen=True)
class MetaFold:
    fold_id: int
    test_group_ids: tuple[int, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray
    test_windows: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_end: pd.Timestamp


@dataclass(frozen=True)
class MetaFoldModel:
    fold: MetaFold
    estimator: Any


@dataclass(frozen=True)
class MetaModelArtifacts:
    config: Module4Config
    dataset: MetaDataset
    trained_models: tuple[MetaFoldModel, ...]
    oof_trade_predictions: pl.DataFrame
    position_sizing: pl.DataFrame
    fold_metrics: pl.DataFrame
    overall_metrics: dict[str, float]
    feature_importance: pl.DataFrame


class _ConstantBinaryClassifier:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = np.full(X.shape[0], self.positive_probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def _safe_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _bet_size(probability: np.ndarray, floor: float, power: float) -> np.ndarray:
    floor = float(np.clip(floor, 0.0, 0.999999))
    scaled = np.where(probability <= floor, 0.0, (probability - floor) / (1.0 - floor))
    return np.power(np.clip(scaled, 0.0, 1.0), power)


def _build_meta_dataset_from_frames(
    alpha_dataset: AlphaDataset,
    alpha_predictions: pl.DataFrame | pd.DataFrame,
    config: Module4Config,
    *,
    require_labels: bool,
) -> MetaDataset:
    alpha_frame = alpha_dataset.frame.copy()
    prediction_frame = (
        alpha_predictions.copy() if isinstance(alpha_predictions, pd.DataFrame) else alpha_predictions.to_pandas()
    )
    prediction_frame["event_time"] = pd.to_datetime(prediction_frame["event_time"], utc=True)

    merged = alpha_frame.merge(
        prediction_frame,
        on=["row_id", "instrument_id", "event_time"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_alpha"),
    )
    if require_labels and "label" not in merged.columns:
        raise ValueError("Meta dataset requires realized alpha labels for training.")

    merged["meta_trade_active"] = (merged["pred_side"] != 0).astype(int)
    if "prediction_count" not in merged.columns:
        merged["prediction_count"] = 1
    if "label" in merged.columns:
        merged["meta_label"] = (merged["pred_side"] == merged["label"]).astype(int)
    merged["meta_primary_confidence"] = merged[
        ["pred_prob_sell", "pred_prob_hold", "pred_prob_buy"]
    ].max(axis=1)
    merged["meta_primary_entropy"] = _safe_entropy(
        merged[["pred_prob_sell", "pred_prob_hold", "pred_prob_buy"]].to_numpy(dtype=np.float64)
    )
    merged["meta_side_margin"] = np.where(
        merged["pred_side"] == 1,
        merged["pred_prob_buy"] - merged["pred_prob_sell"],
        np.where(
            merged["pred_side"] == -1,
            merged["pred_prob_sell"] - merged["pred_prob_buy"],
            merged["pred_prob_hold"] - merged[["pred_prob_sell", "pred_prob_buy"]].max(axis=1),
        ),
    )
    merged["meta_trade_side_prob"] = np.where(
        merged["pred_side"] == 1,
        merged["pred_prob_buy"],
        np.where(merged["pred_side"] == -1, merged["pred_prob_sell"], merged["pred_prob_hold"]),
    )
    merged["meta_opposite_side_prob"] = np.where(
        merged["pred_side"] == 1,
        merged["pred_prob_sell"],
        np.where(merged["pred_side"] == -1, merged["pred_prob_buy"], merged[["pred_prob_sell", "pred_prob_buy"]].max(axis=1)),
    )
    merged["meta_pred_side_flag"] = merged["pred_side"].astype(str).astype("category")

    trade_frame = merged.loc[merged["meta_trade_active"] == 1].copy()
    feature_columns = [
        column
        for column in alpha_dataset.feature_columns
        if column.startswith("feat_") or column.startswith("m2_") or column == "trgt"
    ]
    meta_feature_columns = [
        "pred_prob_sell",
        "pred_prob_hold",
        "pred_prob_buy",
        "prediction_count",
        "meta_primary_confidence",
        "meta_primary_entropy",
        "meta_side_margin",
        "meta_trade_side_prob",
        "meta_opposite_side_prob",
    ]
    categorical_columns = tuple(alpha_dataset.categorical_columns) + ("meta_pred_side_flag",)
    for column in categorical_columns:
        if column in trade_frame.columns:
            trade_frame[column] = trade_frame[column].astype("category")
    feature_columns = list(dict.fromkeys(feature_columns + meta_feature_columns + list(categorical_columns)))

    numeric_feature_columns = [column for column in feature_columns if column not in categorical_columns]
    finite_mask = np.isfinite(trade_frame[numeric_feature_columns]).all(axis=1)
    trade_frame = trade_frame.loc[finite_mask].reset_index(drop=True)

    return MetaDataset(
        frame=trade_frame,
        feature_columns=tuple(feature_columns),
        categorical_columns=tuple(categorical_columns),
        all_rows=merged.reset_index(drop=True),
    )


def build_meta_dataset(alpha_artifacts: AlphaModelArtifacts, config: Module4Config) -> MetaDataset:
    return _build_meta_dataset_from_frames(
        alpha_dataset=alpha_artifacts.dataset,
        alpha_predictions=alpha_artifacts.oof_predictions,
        config=config,
        require_labels=True,
    )


def build_meta_dataset_from_predictions(
    alpha_dataset: AlphaDataset,
    alpha_predictions: pl.DataFrame | pd.DataFrame,
    config: Module4Config,
) -> MetaDataset:
    return _build_meta_dataset_from_frames(
        alpha_dataset=alpha_dataset,
        alpha_predictions=alpha_predictions,
        config=config,
        require_labels=False,
    )


def _generate_meta_splits(dataset: MetaDataset, config: Module4Config) -> list[MetaFold]:
    frame = dataset.frame.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)
    unique_times = frame["event_time"].drop_duplicates().sort_values().to_list()
    if len(unique_times) < config.cpcv.n_groups:
        raise ValueError("Not enough unique meta-event timestamps for the requested CPCV groups.")

    grouped_indices = np.array_split(np.arange(len(unique_times), dtype=np.int64), config.cpcv.n_groups)
    time_to_group: dict[pd.Timestamp, int] = {}
    for group_id, indices_in_group in enumerate(grouped_indices):
        for idx in indices_in_group:
            time_to_group[pd.Timestamp(unique_times[int(idx)])] = group_id

    frame["cv_group"] = frame["event_time"].map(time_to_group)
    horizons = frame["t1"] - frame["event_time"]
    embargo_horizon = horizons.median() * config.cpcv.embargo_horizon_multiplier
    indices = np.arange(frame.shape[0], dtype=np.int64)
    folds: list[MetaFold] = []

    for fold_id, test_group_ids in enumerate(combinations(range(config.cpcv.n_groups), config.cpcv.test_groups)):
        test_mask = frame["cv_group"].isin(test_group_ids).to_numpy()
        if int(test_mask.sum()) < config.cpcv.min_test_samples:
            continue

        train_mask = ~test_mask.copy()
        test_frame = frame.loc[test_mask]
        test_start = test_frame["event_time"].min()
        test_end = test_frame["event_time"].max()
        embargo_end = test_end + embargo_horizon
        test_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        for group_id in test_group_ids:
            group_frame = frame.loc[frame["cv_group"] == group_id]
            group_start = group_frame["event_time"].min()
            group_end = group_frame["event_time"].max()
            test_windows.append((group_start, group_end))
            overlap_mask = (
                train_mask
                & (frame["event_time"] <= group_end + embargo_horizon).to_numpy()
                & (frame["t1"] >= group_start).to_numpy()
            )
            train_mask = train_mask & ~overlap_mask

        if int(train_mask.sum()) < config.cpcv.min_train_samples:
            continue

        folds.append(
            MetaFold(
                fold_id=fold_id,
                test_group_ids=tuple(int(group_id) for group_id in test_group_ids),
                train_indices=indices[train_mask],
                test_indices=indices[test_mask],
                purged_indices=indices[~train_mask & ~test_mask],
                test_windows=tuple(test_windows),
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
            )
        )

    if not folds:
        raise ValueError("No valid meta CPCV folds were produced.")
    return folds


def _prepare_features(dataset: MetaDataset) -> pd.DataFrame:
    features = dataset.frame.loc[:, dataset.feature_columns].copy()
    for column in dataset.categorical_columns:
        features[column] = features[column].astype("category")
    return features


def _fit_meta_estimator(X_train: pd.DataFrame, y_train: np.ndarray, config: Module4Config) -> Any:
    config = config.with_optimized_params()
    positive_rate = float(np.mean(y_train))
    if len(np.unique(y_train)) < 2:
        return _ConstantBinaryClassifier(positive_probability=positive_rate)
    feature_fraction = config.effective_feature_fraction()

    estimator = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=feature_fraction,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbose=-1,
    )
    estimator.fit(X_train, y_train)
    return estimator


def _predict_win_probability(estimator: Any, X_test: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(X_test)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        return probabilities
    return probabilities[:, -1]


def _binary_metrics(y_true: np.ndarray, prob_win: np.ndarray) -> dict[str, float]:
    y_pred = (prob_win >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - prob_win, prob_win]), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, prob_win)),
    }


def _feature_importance_frame(
    models: Sequence[MetaFoldModel],
    feature_columns: Sequence[str],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for model in models:
        if not hasattr(model.estimator, "feature_importances_"):
            continue
        importances = np.asarray(model.estimator.feature_importances_, dtype=np.float64)
        for feature, importance in zip(feature_columns, importances, strict=True):
            rows.append(
                {
                    "fold_id": model.fold.fold_id,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    if not rows:
        return pl.DataFrame({"feature": [], "mean_importance": []})

    return (
        pl.DataFrame(rows)
        .group_by("feature")
        .agg(
            [
                pl.col("importance").mean().alias("mean_importance"),
                pl.col("importance").sum().alias("total_importance"),
            ]
        )
        .sort("mean_importance", descending=True)
    )


def _build_position_frame(
    dataset: MetaDataset,
    oof_predictions: pl.DataFrame | pd.DataFrame,
    config: Module4Config,
) -> pl.DataFrame:
    config = config.with_optimized_params()
    all_rows = dataset.all_rows.copy()
    trade_predictions = (
        oof_predictions.copy() if isinstance(oof_predictions, pd.DataFrame) else oof_predictions.to_pandas()
    )
    merged = all_rows.merge(
        trade_predictions[["row_id", "pred_prob_win", "pred_prob_loss", "bet_size"]],
        on="row_id",
        how="left",
    )
    merged["pred_prob_win"] = merged["pred_prob_win"].fillna(0.0)
    merged["pred_prob_loss"] = merged["pred_prob_loss"].fillna(1.0)
    merged["bet_size"] = merged["bet_size"].fillna(0.0)
    if "m2_regime_penalty" in merged.columns:
        merged["regime_penalty"] = merged["m2_regime_penalty"].fillna(1.0).clip(lower=0.0, upper=1.0)
    else:
        merged["regime_penalty"] = 1.0
    if config.apply_regime_penalty:
        merged["bet_size"] = merged["bet_size"] * merged["regime_penalty"]
    merged.loc[merged["pred_side"] == 0, "bet_size"] = 0.0
    if "label" not in merged.columns:
        merged["label"] = 0
    if "meta_trade_active" not in merged.columns:
        merged["meta_trade_active"] = (merged["pred_side"] != 0).astype(int)

    rows = {
        "row_id": merged["row_id"].to_numpy(dtype=np.int64),
        "instrument_id": merged["instrument_id"].astype(str).to_list(),
        "symbol": merged["symbol"].astype(str).to_list(),
        "event_time": merged["event_time"].tolist(),
        "label": merged["label"].to_numpy(dtype=np.int64),
        "pred_side": merged["pred_side"].to_numpy(dtype=np.int64),
        "meta_trade_active": merged["meta_trade_active"].to_numpy(dtype=np.int64),
        "pred_prob_win": merged["pred_prob_win"].to_numpy(dtype=np.float64),
        "pred_prob_loss": merged["pred_prob_loss"].to_numpy(dtype=np.float64),
        "bet_size": merged["bet_size"].to_numpy(dtype=np.float64),
        "regime_penalty": merged["regime_penalty"].to_numpy(dtype=np.float64),
        "take_trade": (merged["bet_size"] > 0.0).astype(int).to_numpy(dtype=np.int64),
    }
    if "m2_regime_state" in merged.columns:
        rows["regime_state"] = merged["m2_regime_state"].fillna(-1).to_numpy(dtype=np.int64)
    if "m2_regime_label" in merged.columns:
        regime_labels = merged["m2_regime_label"].astype("object")
        rows["regime_label"] = regime_labels.where(regime_labels.notna(), "unknown").astype(str).to_list()

    return pl.DataFrame(rows).sort(["event_time", "instrument_id"])


def train_meta_model(dataset: MetaDataset, config: Module4Config) -> MetaModelArtifacts:
    config = config.with_optimized_params()
    frame = dataset.frame.reset_index(drop=True)
    features = _prepare_features(dataset)
    y = frame["meta_label"].to_numpy(dtype=np.int64)
    folds = _generate_meta_splits(dataset, config)

    probability_sum = np.zeros(frame.shape[0], dtype=np.float64)
    prediction_count = np.zeros(frame.shape[0], dtype=np.int64)
    models: list[MetaFoldModel] = []
    fold_metrics: list[dict[str, object]] = []

    for fold in folds:
        X_train = features.iloc[fold.train_indices].copy()
        y_train = y[fold.train_indices]
        X_test = features.iloc[fold.test_indices].copy()
        y_test = y[fold.test_indices]

        estimator = _fit_meta_estimator(X_train, y_train, config)
        prob_win = _predict_win_probability(estimator, X_test)
        probability_sum[fold.test_indices] += prob_win
        prediction_count[fold.test_indices] += 1

        metrics = _binary_metrics(y_test, prob_win)
        metrics.update(
            {
                "fold_id": fold.fold_id,
                "test_group_ids": ",".join(str(group_id) for group_id in fold.test_group_ids),
                "train_size": int(fold.train_indices.shape[0]),
                "test_size": int(fold.test_indices.shape[0]),
                "purged_size": int(fold.purged_indices.shape[0]),
            }
        )
        fold_metrics.append(metrics)
        models.append(MetaFoldModel(fold=fold, estimator=estimator))

    if not np.all(prediction_count > 0):
        missing = int(np.sum(prediction_count == 0))
        raise ValueError(f"Meta CPCV failed to produce predictions for {missing} trade samples.")

    avg_prob_win = probability_sum / prediction_count
    avg_prob_loss = 1.0 - avg_prob_win
    bet_sizes = _bet_size(avg_prob_win, config.probability_floor, config.bet_sizing_power)
    overall_metrics = _binary_metrics(y, avg_prob_win)

    trade_predictions_pd = pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(dtype=np.int64),
            "instrument_id": frame["instrument_id"].astype(str).to_list(),
            "symbol": frame["symbol"].astype(str).to_list(),
            "event_time": frame["event_time"].tolist(),
            "label": frame["label"].to_numpy(dtype=np.int64),
            "pred_side": frame["pred_side"].to_numpy(dtype=np.int64),
            "meta_label": y,
            "pred_prob_win": avg_prob_win,
            "pred_prob_loss": avg_prob_loss,
            "bet_size": bet_sizes,
        }
    ).sort_values(["event_time", "instrument_id"]).reset_index(drop=True)

    position_sizing = _build_position_frame(dataset, trade_predictions_pd, config)
    return MetaModelArtifacts(
        config=config,
        dataset=dataset,
        trained_models=tuple(models),
        oof_trade_predictions=pl.from_pandas(trade_predictions_pd),
        position_sizing=position_sizing,
        fold_metrics=pl.DataFrame(fold_metrics).sort("fold_id"),
        overall_metrics=overall_metrics,
        feature_importance=_feature_importance_frame(models, dataset.feature_columns),
    )


def predict_meta_bet_sizes(
    dataset: MetaDataset,
    artifacts: MetaModelArtifacts,
) -> pl.DataFrame:
    features = _prepare_features(dataset)
    prob_sum = np.zeros(dataset.frame.shape[0], dtype=np.float64)
    for model in artifacts.trained_models:
        prob_sum += _predict_win_probability(model.estimator, features)
    avg_prob_win = prob_sum / max(len(artifacts.trained_models), 1)
    bet_sizes = _bet_size(avg_prob_win, artifacts.config.probability_floor, artifacts.config.bet_sizing_power)
    regime_penalty = (
        dataset.frame["m2_regime_penalty"].fillna(1.0).to_numpy(dtype=np.float64)
        if "m2_regime_penalty" in dataset.frame.columns
        else np.ones(dataset.frame.shape[0], dtype=np.float64)
    )
    if artifacts.config.apply_regime_penalty:
        bet_sizes = bet_sizes * np.clip(regime_penalty, 0.0, 1.0)
    rows = {
        "row_id": dataset.frame["row_id"].to_numpy(dtype=np.int64),
        "instrument_id": dataset.frame["instrument_id"].astype(str).to_list(),
        "symbol": dataset.frame["symbol"].astype(str).to_list(),
        "event_time": dataset.frame["event_time"].tolist(),
        "pred_side": dataset.frame["pred_side"].to_numpy(dtype=np.int64),
        "pred_prob_win": avg_prob_win,
        "pred_prob_loss": 1.0 - avg_prob_win,
        "bet_size": bet_sizes,
        "regime_penalty": np.clip(regime_penalty, 0.0, 1.0),
    }
    if "m2_regime_state" in dataset.frame.columns:
        rows["regime_state"] = dataset.frame["m2_regime_state"].fillna(-1).to_numpy(dtype=np.int64)
    if "m2_regime_label" in dataset.frame.columns:
        regime_labels = dataset.frame["m2_regime_label"].astype("object")
        rows["regime_label"] = regime_labels.where(regime_labels.notna(), "unknown").astype(str).to_list()
    return pl.DataFrame(rows).sort(["event_time", "instrument_id"])


def run_meta_module(alpha_artifacts: AlphaModelArtifacts, config: Module4Config) -> MetaModelArtifacts:
    config = config.with_optimized_params()
    dataset = build_meta_dataset(alpha_artifacts, config)
    return train_meta_model(dataset, config)
