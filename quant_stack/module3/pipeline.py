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
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from quant_stack.module1 import Module1Artifacts
from quant_stack.module2 import Module2Config, build_module2_features


CLASS_ORDER = (-1, 0, 1)
PROBABILITY_COLUMNS = {
    -1: "pred_prob_sell",
    0: "pred_prob_hold",
    1: "pred_prob_buy",
}


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 5
    test_groups: int = 2
    embargo_horizon_multiplier: float = 1.0
    min_train_samples: int = 80
    min_test_samples: int = 20


@dataclass(frozen=True)
class Module3Config:
    module2: Module2Config = field(default_factory=Module2Config)
    use_module2_features: bool = True
    feature_lags: tuple[int, ...] = (1, 2, 3, 5)
    return_windows: tuple[int, ...] = (3, 8, 21)
    vol_windows: tuple[int, ...] = (5, 13, 34)
    volume_windows: tuple[int, ...] = (5, 20)
    cpcv: CPCVConfig = field(default_factory=CPCVConfig)
    n_estimators: int = 250
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    feature_fraction: float | None = None
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = 1
    optimized_params_path: Path | None = None

    def effective_feature_fraction(self) -> float:
        if self.feature_fraction is not None:
            return float(self.feature_fraction)
        return float(self.colsample_bytree)

    def with_optimized_params(self) -> Module3Config:
        if self.optimized_params_path is None or not Path(self.optimized_params_path).exists():
            return self

        payload = json.loads(Path(self.optimized_params_path).read_text(encoding="utf-8"))
        params = payload.get("module3", {})
        allowed = {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "feature_fraction",
            "reg_lambda",
        }
        overrides = {key: params[key] for key in allowed if key in params}
        if not overrides:
            return self
        return replace(self, **overrides)


@dataclass(frozen=True)
class AlphaDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


@dataclass(frozen=True)
class CPCVFold:
    fold_id: int
    test_group_ids: tuple[int, ...]
    test_windows: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_end: pd.Timestamp


@dataclass(frozen=True)
class FoldModel:
    fold: CPCVFold
    estimator: Any
    available_classes: tuple[int, ...]


@dataclass(frozen=True)
class AlphaModelArtifacts:
    config: Module3Config
    dataset: AlphaDataset
    trained_models: tuple[FoldModel, ...]
    oof_predictions: pl.DataFrame
    fold_metrics: pl.DataFrame
    overall_metrics: dict[str, float]
    feature_importance: pl.DataFrame


class _ConstantClassifier:
    def __init__(self, encoded_label: int) -> None:
        self.encoded_label = int(encoded_label)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.ones((X.shape[0], 1), dtype=np.float64)
        return probabilities


def _as_sorted_pandas(
    frame: pl.DataFrame | pd.DataFrame,
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame.sort(list(sort_columns)).to_pandas().reset_index(drop=True)
    return frame.sort_values(list(sort_columns)).reset_index(drop=True).copy()


def _series_group(
    frame: pd.DataFrame,
    column: str,
) -> pd.core.groupby.generic.SeriesGroupBy:
    return frame.groupby("instrument_id", sort=False)[column]


def _rolling_from_group(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    func: str,
) -> pd.Series:
    rolling = grouped.rolling(window=window)
    if func == "mean":
        values = rolling.mean()
    elif func == "std":
        values = rolling.std()
    elif func == "median":
        values = rolling.median()
    elif func == "sum":
        values = rolling.sum()
    else:
        raise ValueError(f"Unsupported rolling func: {func}")
    return values.reset_index(level=0, drop=True)


def _ewm_mean_from_group(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    span: int,
) -> pd.Series:
    return grouped.ewm(span=span, adjust=False).mean().reset_index(level=0, drop=True)


def _coerce_event_frame(
    events_df: pl.DataFrame | pd.DataFrame,
    *,
    require_labels: bool,
) -> pd.DataFrame:
    events = _as_sorted_pandas(events_df, ("instrument_id", "event_time"))
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    events["t1"] = pd.to_datetime(events["t1"], utc=True)
    if "label" not in events.columns:
        if require_labels:
            raise ValueError("Alpha dataset requires a label column for training.")
    else:
        events["label"] = events["label"].fillna(0).astype(int)
    return events


def _build_alpha_dataset_core(
    stationary_df: pl.DataFrame | pd.DataFrame,
    events_df: pl.DataFrame | pd.DataFrame,
    config: Module3Config,
    *,
    require_labels: bool,
) -> AlphaDataset:
    bars = _as_sorted_pandas(stationary_df, ("instrument_id", "timestamp"))
    labels = _coerce_event_frame(events_df, require_labels=require_labels)

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)

    bars["is_partial_bar"] = bars["is_partial_bar"].astype(int)
    bars["sampling_mode"] = bars["sampling_mode"].astype("category")
    bars["venue"] = bars["venue"].astype("category")
    bars["asset_group"] = bars["asset_group"].astype("category")
    bars["symbol"] = bars["symbol"].astype("category")

    if config.use_module2_features:
        module2_features = build_module2_features(bars, config.module2)
        bars = bars.merge(module2_features, on=["instrument_id", "timestamp"], how="left", validate="one_to_one")

    epsilon = 1e-12
    bars["feat_fracdiff_return"] = bars["fracdiff_return"]
    bars["feat_raw_log_return"] = bars["raw_log_return"]
    bars["feat_sigma_t"] = bars["sigma_t"]
    bars["feat_range_pct"] = (bars["high"] - bars["low"]) / bars["close"].clip(lower=epsilon)
    bars["feat_body_pct"] = (bars["close"] - bars["open"]) / bars["open"].clip(lower=epsilon)
    bars["feat_close_location"] = (bars["close"] - bars["low"]) / (
        (bars["high"] - bars["low"]).clip(lower=epsilon)
    )
    bars["feat_input_bar_count"] = bars["input_bar_count"].astype(float)
    bars["feat_bar_minutes"] = bars["bar_minutes"].astype(float)
    bars["feat_session_bar_index"] = bars["session_bar_index"].astype(float)
    bars["feat_partial_bar"] = bars["is_partial_bar"].astype(float)

    grouped_fracdiff = _series_group(bars, "fracdiff_return")
    grouped_raw = _series_group(bars, "raw_log_return")
    grouped_volume = _series_group(bars, "volume")
    grouped_sigma = _series_group(bars, "sigma_t")
    grouped_log_close = _series_group(bars, "log_close")

    for lag in config.feature_lags:
        bars[f"feat_fracdiff_return_lag_{lag}"] = grouped_fracdiff.shift(lag)
        bars[f"feat_raw_log_return_lag_{lag}"] = grouped_raw.shift(lag)
        bars[f"feat_volume_lag_{lag}"] = grouped_volume.shift(lag)

    for window in config.return_windows:
        bars[f"feat_raw_return_mean_{window}"] = _rolling_from_group(grouped_raw, window, "mean")
        bars[f"feat_fracdiff_return_mean_{window}"] = _rolling_from_group(grouped_fracdiff, window, "mean")

    for window in config.vol_windows:
        bars[f"feat_raw_return_std_{window}"] = _rolling_from_group(grouped_raw, window, "std")
        bars[f"feat_sigma_median_{window}"] = _rolling_from_group(grouped_sigma, window, "median")

    for window in config.volume_windows:
        bars[f"feat_volume_mean_{window}"] = _rolling_from_group(grouped_volume, window, "mean")
        bars[f"feat_volume_std_{window}"] = _rolling_from_group(grouped_volume, window, "std")

    for window in config.volume_windows:
        mean_col = f"feat_volume_mean_{window}"
        std_col = f"feat_volume_std_{window}"
        bars[f"feat_volume_ratio_{window}"] = bars["volume"] / bars[mean_col].clip(lower=epsilon)
        bars[f"feat_volume_z_{window}"] = (bars["volume"] - bars[mean_col]) / bars[std_col].clip(lower=epsilon)

    for window in config.vol_windows:
        median_col = f"feat_sigma_median_{window}"
        bars[f"feat_sigma_regime_{window}"] = bars["sigma_t"] / bars[median_col].clip(lower=epsilon)

    bars["feat_log_close_deviation_8"] = bars["log_close"] - _ewm_mean_from_group(grouped_log_close, 8)
    bars["feat_log_close_deviation_21"] = bars["log_close"] - _ewm_mean_from_group(grouped_log_close, 21)

    dataset = labels.merge(
        bars,
        left_on=["instrument_id", "event_time"],
        right_on=["instrument_id", "timestamp"],
        how="inner",
        validate="one_to_one",
        suffixes=("_label", ""),
    )

    if "label" in dataset.columns:
        dataset["label"] = dataset["label"].astype(int)
    dataset["horizon_minutes"] = (
        dataset["t1"] - dataset["event_time"]
    ).dt.total_seconds() / 60.0

    categorical_columns = ["symbol", "venue", "asset_group", "sampling_mode"]
    if "m2_regime_label" in dataset.columns:
        categorical_columns.append("m2_regime_label")

    feature_columns = [
        column
        for column in dataset.columns
        if column.startswith("feat_") or column.startswith("m2_")
    ]
    feature_columns.append("trgt")
    feature_columns.extend(categorical_columns)
    feature_columns = list(dict.fromkeys(feature_columns))
    missing_feature_columns = [column for column in feature_columns if column not in dataset.columns]
    if missing_feature_columns:
        missing = ", ".join(sorted(missing_feature_columns))
        raise ValueError(f"Alpha dataset is missing required feature columns: {missing}")

    dataset = dataset.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)
    numeric_feature_columns = [column for column in feature_columns if column not in categorical_columns]
    finite_mask = np.isfinite(dataset[numeric_feature_columns]).all(axis=1)
    dataset = dataset.loc[finite_mask].copy()

    for column in categorical_columns:
        dataset[column] = dataset[column].astype("category")

    dataset["row_id"] = np.arange(dataset.shape[0], dtype=np.int64)
    return AlphaDataset(
        frame=dataset.reset_index(drop=True),
        feature_columns=tuple(feature_columns),
        categorical_columns=tuple(categorical_columns),
    )


def build_alpha_dataset(
    stationary_df: pl.DataFrame | pd.DataFrame,
    labels_df: pl.DataFrame | pd.DataFrame,
    config: Module3Config,
) -> AlphaDataset:
    return _build_alpha_dataset_core(
        stationary_df=stationary_df,
        events_df=labels_df,
        config=config,
        require_labels=True,
    )


def build_alpha_inference_dataset(
    stationary_df: pl.DataFrame | pd.DataFrame,
    events_df: pl.DataFrame | pd.DataFrame,
    config: Module3Config,
) -> AlphaDataset:
    return _build_alpha_dataset_core(
        stationary_df=stationary_df,
        events_df=events_df,
        config=config,
        require_labels=False,
    )


def _compute_embargo_horizon(frame: pd.DataFrame, config: Module3Config) -> pd.Timedelta:
    horizons = frame["t1"] - frame["event_time"]
    median_horizon = horizons.median()
    if pd.isna(median_horizon):
        return pd.Timedelta(0)
    return median_horizon * config.cpcv.embargo_horizon_multiplier


def generate_cpcv_splits(dataset: AlphaDataset, config: Module3Config) -> list[CPCVFold]:
    frame = dataset.frame.sort_values(["event_time", "instrument_id"]).reset_index(drop=True)
    unique_times = frame["event_time"].drop_duplicates().sort_values().to_list()
    if len(unique_times) < config.cpcv.n_groups:
        raise ValueError("Not enough unique event timestamps for the requested CPCV groups.")

    grouped_indices = np.array_split(np.arange(len(unique_times), dtype=np.int64), config.cpcv.n_groups)
    time_to_group: dict[pd.Timestamp, int] = {}
    for group_id, indices_in_group in enumerate(grouped_indices):
        for idx in indices_in_group:
            time_to_group[pd.Timestamp(unique_times[int(idx)])] = group_id

    frame["cv_group"] = frame["event_time"].map(time_to_group)
    embargo_horizon = _compute_embargo_horizon(frame, config)
    indices = np.arange(frame.shape[0], dtype=np.int64)
    folds: list[CPCVFold] = []

    for fold_id, test_group_ids in enumerate(
        combinations(range(config.cpcv.n_groups), config.cpcv.test_groups)
    ):
        test_mask = frame["cv_group"].isin(test_group_ids).to_numpy()
        if int(test_mask.sum()) < config.cpcv.min_test_samples:
            continue

        train_mask = ~test_mask.copy()
        test_group_frame = frame.loc[test_mask]
        test_start = test_group_frame["event_time"].min()
        test_end = test_group_frame["event_time"].max()
        embargo_end = test_end + embargo_horizon
        test_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        for group_id in test_group_ids:
            group_frame = frame.loc[frame["cv_group"] == group_id]
            group_start = group_frame["event_time"].min()
            group_end = group_frame["event_time"].max()
            test_windows.append((group_start, group_end))
            expanded_end = group_end + embargo_horizon
            overlap_mask = (
                train_mask
                & (frame["event_time"] <= expanded_end).to_numpy()
                & (frame["t1"] >= group_start).to_numpy()
            )
            train_mask = train_mask & ~overlap_mask

        if int(train_mask.sum()) < config.cpcv.min_train_samples:
            continue

        train_indices = indices[train_mask]
        test_indices = indices[test_mask]
        purged_indices = indices[~train_mask & ~test_mask]

        folds.append(
            CPCVFold(
                fold_id=fold_id,
                test_group_ids=tuple(int(group_id) for group_id in test_group_ids),
                test_windows=tuple(test_windows),
                train_indices=train_indices,
                test_indices=test_indices,
                purged_indices=purged_indices,
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
            )
        )

    if not folds:
        raise ValueError("No valid CPCV folds were produced. Relax split constraints or add more data.")
    return folds


def _prepare_feature_frame(dataset: AlphaDataset) -> pd.DataFrame:
    features = dataset.frame.loc[:, dataset.feature_columns].copy()
    for column in dataset.categorical_columns:
        features[column] = features[column].astype("category")
    return features


def _fit_fold_estimator(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: Module3Config,
) -> tuple[Any, tuple[int, ...]]:
    config = config.with_optimized_params()
    available_classes = tuple(sorted(int(label) for label in np.unique(y_train)))
    encoded_map = {label: index for index, label in enumerate(available_classes)}
    y_encoded = np.asarray([encoded_map[int(label)] for label in y_train], dtype=np.int64)
    feature_fraction = config.effective_feature_fraction()

    if len(available_classes) == 1:
        estimator = _ConstantClassifier(encoded_label=0)
        return estimator, available_classes

    if len(available_classes) == 2:
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
    else:
        estimator = LGBMClassifier(
            objective="multiclass",
            num_class=len(available_classes),
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

    estimator.fit(X_train, y_encoded)
    return estimator, available_classes


def _expand_probabilities(
    probabilities: np.ndarray,
    available_classes: Sequence[int],
) -> np.ndarray:
    full = np.zeros((probabilities.shape[0], len(CLASS_ORDER)), dtype=np.float64)
    column_lookup = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    for source_index, label in enumerate(available_classes):
        full[:, column_lookup[int(label)]] = probabilities[:, source_index]
    row_sums = full.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return full / row_sums


def _predict_fold_probabilities(
    estimator: Any,
    available_classes: Sequence[int],
    X_test: pd.DataFrame,
) -> np.ndarray:
    if len(available_classes) == 1:
        full = np.zeros((X_test.shape[0], len(CLASS_ORDER)), dtype=np.float64)
        full[:, CLASS_ORDER.index(int(available_classes[0]))] = 1.0
        return full

    raw_probabilities = estimator.predict_proba(X_test)
    if isinstance(raw_probabilities, list):
        raw_probabilities = np.column_stack(raw_probabilities)
    if raw_probabilities.ndim == 1:
        raw_probabilities = np.column_stack([1.0 - raw_probabilities, raw_probabilities])
    return _expand_probabilities(np.asarray(raw_probabilities, dtype=np.float64), available_classes)


def _multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_pred = np.asarray([CLASS_ORDER[int(idx)] for idx in np.argmax(probabilities, axis=1)], dtype=np.int64)
    class_to_index = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    y_true_encoded = np.asarray([class_to_index[int(label)] for label in y_true], dtype=np.int64)
    unique_true = np.unique(y_true).astype(np.int64, copy=False)
    # Compute balanced accuracy as macro recall over observed true classes.
    # This avoids sklearn confusion-matrix warnings when a model predicts a class
    # that is absent in the fold's y_true (common in purged CPCV slices).
    balanced_accuracy = float(
        recall_score(
            y_true,
            y_pred,
            labels=unique_true.tolist(),
            average="macro",
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(CLASS_ORDER),
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_log_loss": float(
            log_loss(
                y_true_encoded,
                probabilities,
                labels=np.arange(len(CLASS_ORDER), dtype=np.int64),
            )
        ),
    }


def _feature_importance_frame(
    trained_models: Sequence[FoldModel],
    feature_columns: Sequence[str],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for model in trained_models:
        estimator = model.estimator
        if not hasattr(estimator, "feature_importances_"):
            continue
        importances = np.asarray(estimator.feature_importances_, dtype=np.float64)
        for column, importance in zip(feature_columns, importances, strict=True):
            rows.append(
                {
                    "fold_id": model.fold.fold_id,
                    "feature": column,
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


def train_alpha_model(dataset: AlphaDataset, config: Module3Config) -> AlphaModelArtifacts:
    config = config.with_optimized_params()
    frame = dataset.frame.reset_index(drop=True)
    features = _prepare_feature_frame(dataset)
    labels = frame["label"].to_numpy(dtype=np.int64)
    folds = generate_cpcv_splits(dataset, config)

    probability_sum = np.zeros((frame.shape[0], len(CLASS_ORDER)), dtype=np.float64)
    prediction_count = np.zeros(frame.shape[0], dtype=np.int64)
    trained_models: list[FoldModel] = []
    fold_metrics: list[dict[str, object]] = []

    for fold in folds:
        X_train = features.iloc[fold.train_indices].copy()
        y_train = labels[fold.train_indices]
        X_test = features.iloc[fold.test_indices].copy()
        y_test = labels[fold.test_indices]

        estimator, available_classes = _fit_fold_estimator(X_train, y_train, config)
        probabilities = _predict_fold_probabilities(estimator, available_classes, X_test)

        probability_sum[fold.test_indices] += probabilities
        prediction_count[fold.test_indices] += 1
        metrics = _multiclass_metrics(y_test, probabilities)
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
        trained_models.append(
            FoldModel(
                fold=fold,
                estimator=estimator,
                available_classes=tuple(int(label) for label in available_classes),
            )
        )

    if not np.all(prediction_count > 0):
        missing = int(np.sum(prediction_count == 0))
        raise ValueError(f"CPCV failed to produce predictions for {missing} samples.")

    averaged_probabilities = probability_sum / prediction_count[:, None]
    overall_metrics = _multiclass_metrics(labels, averaged_probabilities)

    oof_rows = {
        "row_id": frame["row_id"].to_numpy(dtype=np.int64),
        "instrument_id": frame["instrument_id"].astype(str).to_list(),
        "symbol": frame["symbol"].astype(str).to_list(),
        "event_time": frame["event_time"].tolist(),
        "t1": frame["t1"].tolist(),
        "label": labels,
        "prediction_count": prediction_count,
        "pred_side": np.asarray(
            [CLASS_ORDER[int(idx)] for idx in np.argmax(averaged_probabilities, axis=1)],
            dtype=np.int64,
        ),
    }
    for class_label, column_name in PROBABILITY_COLUMNS.items():
        oof_rows[column_name] = averaged_probabilities[:, CLASS_ORDER.index(class_label)]

    return AlphaModelArtifacts(
        config=config,
        dataset=dataset,
        trained_models=tuple(trained_models),
        oof_predictions=pl.DataFrame(oof_rows).sort(["event_time", "instrument_id"]),
        fold_metrics=pl.DataFrame(fold_metrics).sort("fold_id"),
        overall_metrics=overall_metrics,
        feature_importance=_feature_importance_frame(trained_models, dataset.feature_columns),
    )


def predict_alpha_probabilities(
    dataset: AlphaDataset,
    artifacts: AlphaModelArtifacts,
) -> pl.DataFrame:
    features = _prepare_feature_frame(dataset)
    probability_sum = np.zeros((dataset.frame.shape[0], len(CLASS_ORDER)), dtype=np.float64)

    for model in artifacts.trained_models:
        probability_sum += _predict_fold_probabilities(
            model.estimator,
            model.available_classes,
            features,
        )

    averaged_probabilities = probability_sum / max(len(artifacts.trained_models), 1)
    rows = {
        "row_id": dataset.frame["row_id"].to_numpy(dtype=np.int64),
        "instrument_id": dataset.frame["instrument_id"].astype(str).to_list(),
        "symbol": dataset.frame["symbol"].astype(str).to_list(),
        "event_time": dataset.frame["event_time"].tolist(),
        "pred_side": np.asarray(
            [CLASS_ORDER[int(idx)] for idx in np.argmax(averaged_probabilities, axis=1)],
            dtype=np.int64,
        ),
    }
    for class_label, column_name in PROBABILITY_COLUMNS.items():
        rows[column_name] = averaged_probabilities[:, CLASS_ORDER.index(class_label)]
    return pl.DataFrame(rows).sort(["event_time", "instrument_id"])


def run_alpha_module(module1_artifacts: Module1Artifacts, config: Module3Config) -> AlphaModelArtifacts:
    config = config.with_optimized_params()
    dataset = build_alpha_dataset(
        stationary_df=module1_artifacts.stationary,
        labels_df=module1_artifacts.labels,
        config=config,
    )
    return train_alpha_model(dataset, config)
