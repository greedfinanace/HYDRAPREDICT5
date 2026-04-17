from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class RegimeDetectorConfig:
    benchmark_symbol: str = "SPY"
    n_states: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    random_state: int = 42
    min_observations: int = 60
    return_column: str = "raw_log_return"
    vol_column: str = "sigma_t"
    bull_penalty: float = 1.0
    bear_penalty: float = 0.5
    sideways_penalty: float = 0.75


@dataclass(frozen=True)
class RegimeDetectionArtifacts:
    config: RegimeDetectorConfig
    timestamp_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    state_summary: pd.DataFrame


class RegimeDetector:
    def __init__(self, config: RegimeDetectorConfig = RegimeDetectorConfig()) -> None:
        self.config = config
        self.model_: GaussianHMM | None = None
        self.timestamp_regimes_: pd.DataFrame | None = None
        self.state_summary_: pd.DataFrame | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None

    def _prepare_frame(self, stationary_df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
        if isinstance(stationary_df, pl.DataFrame):
            frame = stationary_df.sort(["instrument_id", "timestamp"]).to_pandas()
        else:
            frame = stationary_df.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values(["timestamp", "instrument_id"]).reset_index(drop=True)

    def _market_observation_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        benchmark_mask = frame["symbol"].astype(str).str.upper() == self.config.benchmark_symbol.upper()
        benchmark_frame = frame.loc[benchmark_mask, ["timestamp", self.config.return_column, self.config.vol_column]].copy()
        benchmark_frame = benchmark_frame.dropna().sort_values("timestamp")

        if benchmark_frame.shape[0] >= self.config.min_observations:
            return benchmark_frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        aggregated = (
            frame.loc[:, ["timestamp", self.config.return_column, self.config.vol_column]]
            .groupby("timestamp", as_index=False)
            .mean()
            .dropna()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if aggregated.shape[0] < self.config.min_observations:
            raise ValueError("Not enough observations to fit the HMM regime detector.")
        return aggregated

    def _scaled_features(self, observation_frame: pd.DataFrame) -> np.ndarray:
        values = observation_frame.loc[:, [self.config.return_column, self.config.vol_column]].to_numpy(dtype=np.float64)
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0)
        stds = np.where(np.isfinite(stds) & (stds > 0.0), stds, 1.0)
        self.feature_means_ = means
        self.feature_stds_ = stds
        return (values - means) / stds

    def _state_summary(self, observation_frame: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
        summary = (
            observation_frame.assign(raw_state=states)
            .groupby("raw_state", as_index=False)
            .agg(
                mean_return=(self.config.return_column, "mean"),
                mean_vol=(self.config.vol_column, "mean"),
                observations=("timestamp", "count"),
            )
            .sort_values("raw_state")
            .reset_index(drop=True)
        )
        ordered_by_return = summary.sort_values(["mean_return", "mean_vol"], ascending=[False, True])
        observed_states = [int(value) for value in ordered_by_return["raw_state"].to_list()]
        if len(observed_states) == 1:
            raw_to_label = {observed_states[0]: "sideways"}
            raw_to_state_id = {observed_states[0]: 2}
        elif len(observed_states) == 2:
            raw_to_label = {
                observed_states[0]: "bull",
                observed_states[1]: "bear",
            }
            raw_to_state_id = {
                observed_states[0]: 0,
                observed_states[1]: 1,
            }
        else:
            bull_raw = observed_states[0]
            bear_raw = int(summary.sort_values(["mean_return", "mean_vol"], ascending=[True, False]).iloc[0]["raw_state"])
            remaining = [value for value in observed_states if value not in {bull_raw, bear_raw}]
            sideways_raw = remaining[0]
            raw_to_label = {
                bull_raw: "bull",
                bear_raw: "bear",
                sideways_raw: "sideways",
            }
            raw_to_state_id = {
                bull_raw: 0,
                bear_raw: 1,
                sideways_raw: 2,
            }

        summary["regime_label"] = summary["raw_state"].map(raw_to_label)
        summary["m2_regime_state"] = summary["raw_state"].map(raw_to_state_id)

        vol_rank = summary["mean_vol"].rank(method="dense", ascending=False)
        summary["m2_regime_penalty"] = np.where(
            summary["regime_label"] == "bull",
            self.config.bull_penalty,
            np.where(
                (summary["regime_label"] == "bear") & (vol_rank <= 1),
                self.config.bear_penalty,
                self.config.sideways_penalty,
            ),
        )
        return summary

    def fit(self, stationary_df: pl.DataFrame | pd.DataFrame) -> RegimeDetector:
        frame = self._prepare_frame(stationary_df)
        observation_frame = self._market_observation_frame(frame)
        scaled = self._scaled_features(observation_frame)

        model = GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            random_state=self.config.random_state,
        )
        model.fit(scaled)
        raw_states = model.predict(scaled)
        summary = self._state_summary(observation_frame, raw_states)

        timestamp_frame = observation_frame.loc[:, ["timestamp"]].copy()
        state_mapping = summary.set_index("raw_state")
        timestamp_frame["m2_regime_state"] = [int(state_mapping.loc[state, "m2_regime_state"]) for state in raw_states]
        timestamp_frame["m2_regime_label"] = [str(state_mapping.loc[state, "regime_label"]) for state in raw_states]
        timestamp_frame["m2_regime_penalty"] = [
            float(state_mapping.loc[state, "m2_regime_penalty"]) for state in raw_states
        ]
        timestamp_frame["feat_regime_bull"] = (timestamp_frame["m2_regime_label"] == "bull").astype(float)
        timestamp_frame["feat_regime_bear"] = (timestamp_frame["m2_regime_label"] == "bear").astype(float)
        timestamp_frame["feat_regime_sideways"] = (timestamp_frame["m2_regime_label"] == "sideways").astype(float)

        self.model_ = model
        self.timestamp_regimes_ = timestamp_frame
        self.state_summary_ = summary
        return self

    def transform(self, stationary_df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
        if self.timestamp_regimes_ is None:
            raise ValueError("RegimeDetector.fit() must be called before transform().")

        frame = self._prepare_frame(stationary_df)
        merged = frame.loc[:, ["instrument_id", "timestamp"]].merge(
            self.timestamp_regimes_,
            on="timestamp",
            how="left",
            validate="many_to_one",
        )
        merged = merged.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
        merged["m2_regime_state"] = merged["m2_regime_state"].ffill().bfill().fillna(-1).astype(int)
        merged["m2_regime_label"] = merged["m2_regime_label"].ffill().bfill().fillna("unknown")
        merged["m2_regime_penalty"] = merged["m2_regime_penalty"].ffill().bfill().fillna(1.0)
        for column in ("feat_regime_bull", "feat_regime_bear", "feat_regime_sideways"):
            merged[column] = merged[column].fillna(0.0).astype(float)
        return merged

    def fit_transform(self, stationary_df: pl.DataFrame | pd.DataFrame) -> RegimeDetectionArtifacts:
        self.fit(stationary_df)
        return RegimeDetectionArtifacts(
            config=self.config,
            timestamp_frame=self.timestamp_regimes_.copy(),  # type: ignore[union-attr]
            feature_frame=self.transform(stationary_df),
            state_summary=self.state_summary_.copy(),  # type: ignore[union-attr]
        )


def build_regime_feature_frame(
    stationary_df: pl.DataFrame | pd.DataFrame,
    config: RegimeDetectorConfig = RegimeDetectorConfig(),
) -> pd.DataFrame:
    detector = RegimeDetector(config)
    artifacts = detector.fit_transform(stationary_df)
    return artifacts.feature_frame
