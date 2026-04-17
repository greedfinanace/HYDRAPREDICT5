"""Module 6 optimization and regime-detection interface."""

from __future__ import annotations

from typing import Any


__all__ = [
    "HPOArtifacts",
    "HPOConfig",
    "apply_hpo_overrides",
    "run_hpo_study",
    "RegimeDetectionArtifacts",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "build_regime_feature_frame",
]


def __getattr__(name: str) -> Any:
    if name in {"HPOArtifacts", "HPOConfig", "apply_hpo_overrides", "run_hpo_study"}:
        from .optimization_engine import HPOArtifacts, HPOConfig, apply_hpo_overrides, run_hpo_study

        return {
            "HPOArtifacts": HPOArtifacts,
            "HPOConfig": HPOConfig,
            "apply_hpo_overrides": apply_hpo_overrides,
            "run_hpo_study": run_hpo_study,
        }[name]
    if name in {
        "RegimeDetectionArtifacts",
        "RegimeDetector",
        "RegimeDetectorConfig",
        "build_regime_feature_frame",
    }:
        from .regime_detector import (
            RegimeDetectionArtifacts,
            RegimeDetector,
            RegimeDetectorConfig,
            build_regime_feature_frame,
        )

        return {
            "RegimeDetectionArtifacts": RegimeDetectionArtifacts,
            "RegimeDetector": RegimeDetector,
            "RegimeDetectorConfig": RegimeDetectorConfig,
            "build_regime_feature_frame": build_regime_feature_frame,
        }[name]
    raise AttributeError(name)
