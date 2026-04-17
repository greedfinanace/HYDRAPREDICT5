"""Module 4 meta-labeling and bet sizing interface."""

from .pipeline import (
    MetaDataset,
    MetaModelArtifacts,
    Module4Config,
    build_meta_dataset,
    build_meta_dataset_from_predictions,
    predict_meta_bet_sizes,
    run_meta_module,
    train_meta_model,
)

__all__ = [
    "MetaDataset",
    "MetaModelArtifacts",
    "Module4Config",
    "build_meta_dataset",
    "build_meta_dataset_from_predictions",
    "predict_meta_bet_sizes",
    "run_meta_module",
    "train_meta_model",
]
