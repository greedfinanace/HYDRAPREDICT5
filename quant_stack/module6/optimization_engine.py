from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from quant_stack.module1 import Module1Config, build_module1_dataset
from quant_stack.module3 import Module3Config, run_alpha_module
from quant_stack.module4 import Module4Config, run_meta_module
from quant_stack.module5 import Module5Config, run_backtest


@dataclass(frozen=True)
class HPOConfig:
    study_name: str = "quant_stack_hpo"
    n_trials: int = 20
    timeout_seconds: int | None = None
    sampler_seed: int = 42
    n_startup_trials: int = 5
    pruner_warmup_steps: int = 1
    output_root: Path = Path("artifacts/module6_hpo")
    storage_url: str | None = None


@dataclass(frozen=True)
class HPOArtifacts:
    config: HPOConfig
    best_value: float
    best_params: dict[str, Any]
    best_params_path: Path
    trials_path: Path
    study_name: str
    completed_trials: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    return str(value)


def apply_hpo_overrides(
    module1_config: Module1Config,
    module3_config: Module3Config,
    module4_config: Module4Config,
    params_path: str | Path,
) -> tuple[Module1Config, Module3Config, Module4Config]:
    path = Path(params_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    module1_params = payload.get("module1", {})
    module1_overrides: dict[str, Any] = {}
    if "fracdiff_d" in module1_params:
        module1_overrides["fracdiff_d_grid"] = (float(module1_params["fracdiff_d"]),)
    if "weight_eps" in module1_params:
        module1_overrides["weight_eps"] = float(module1_params["weight_eps"])
        module1_overrides["adaptive_weight_eps"] = False

    resolved_module1 = replace(module1_config, **module1_overrides) if module1_overrides else module1_config
    resolved_module3 = replace(module3_config, optimized_params_path=path)
    resolved_module4 = replace(module4_config, optimized_params_path=path)
    return resolved_module1, resolved_module3, resolved_module4


def _trial_parameter_payload(
    trial: optuna.Trial,
) -> dict[str, dict[str, float | int]]:
    return {
        "module1": {
            "fracdiff_d": float(trial.suggest_float("fracdiff_d", 0.1, 0.9, step=0.05)),
            "weight_eps": float(trial.suggest_float("weight_eps", 1e-6, 1e-4, log=True)),
        },
        "module3": {
            "num_leaves": int(trial.suggest_int("alpha_num_leaves", 20, 150)),
            "learning_rate": float(trial.suggest_float("alpha_learning_rate", 0.001, 0.1, log=True)),
            "feature_fraction": float(trial.suggest_float("alpha_feature_fraction", 0.5, 0.9)),
        },
        "module4": {
            "num_leaves": int(trial.suggest_int("meta_num_leaves", 20, 150)),
            "learning_rate": float(trial.suggest_float("meta_learning_rate", 0.001, 0.1, log=True)),
            "feature_fraction": float(trial.suggest_float("meta_feature_fraction", 0.5, 0.9)),
        },
    }


def run_hpo_study(
    module1_config: Module1Config,
    module3_config: Module3Config,
    module4_config: Module4Config,
    module5_config: Module5Config,
    config: HPOConfig = HPOConfig(),
) -> HPOArtifacts:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=config.sampler_seed)
    pruner = MedianPruner(
        n_startup_trials=config.n_startup_trials,
        n_warmup_steps=config.pruner_warmup_steps,
    )
    study = optuna.create_study(
        study_name=config.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=config.storage_url,
        load_if_exists=bool(config.storage_url),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _trial_parameter_payload(trial)
        trial_root = output_root / f"trial_{trial.number:04d}"

        trial_module1 = replace(
            module1_config,
            fracdiff_d_grid=(float(params["module1"]["fracdiff_d"]),),
            weight_eps=float(params["module1"]["weight_eps"]),
            adaptive_weight_eps=False,
            output_root=trial_root / "module1",
        )
        trial_module3 = replace(
            module3_config,
            num_leaves=int(params["module3"]["num_leaves"]),
            learning_rate=float(params["module3"]["learning_rate"]),
            feature_fraction=float(params["module3"]["feature_fraction"]),
            optimized_params_path=None,
        )
        trial_module4 = replace(
            module4_config,
            num_leaves=int(params["module4"]["num_leaves"]),
            learning_rate=float(params["module4"]["learning_rate"]),
            feature_fraction=float(params["module4"]["feature_fraction"]),
            optimized_params_path=None,
        )
        trial_module5 = replace(
            module5_config,
            output_root=trial_root / "module5",
            save_plots=False,
        )

        try:
            module1_artifacts = build_module1_dataset(trial_module1)
            alpha_artifacts = run_alpha_module(module1_artifacts, trial_module3)
            alpha_accuracy = float(alpha_artifacts.overall_metrics.get("accuracy", 0.0))
            trial.report(alpha_accuracy, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned after weak alpha-stage accuracy.")

            meta_artifacts = run_meta_module(alpha_artifacts, trial_module4)
            meta_precision = float(meta_artifacts.overall_metrics.get("precision", 0.0))
            trial.report(meta_precision, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned after weak meta-stage precision.")

            backtest_artifacts = run_backtest(
                meta_artifacts.position_sizing,
                module1_artifacts.stationary,
                trial_module5,
            )
            sharpe = float(backtest_artifacts.tearsheet["strategies"]["meta_strategy"]["sharpe"])
            if not np.isfinite(sharpe := sharpe):
                raise optuna.TrialPruned("Backtest produced a non-finite Sharpe ratio.")

            trial.report(sharpe, step=2)
            if trial.should_prune():
                raise optuna.TrialPruned("Pruned after weak backtest Sharpe.")

            trial.set_user_attr("meta_strategy_mdd", backtest_artifacts.tearsheet["strategies"]["meta_strategy"]["max_drawdown"])
            trial.set_user_attr("alpha_accuracy", alpha_accuracy)
            trial.set_user_attr("meta_precision", meta_precision)
            trial.set_user_attr("events", module1_artifacts.events.height)
            return sharpe
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            raise optuna.TrialPruned(str(exc)) from exc

    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise ValueError("The HPO study finished without any completed trials.")

    best_trial = study.best_trial
    best_params = {
        "objective": {
            "name": "module5_meta_strategy_sharpe",
            "best_value": float(best_trial.value),
            "trial_number": int(best_trial.number),
        },
        "module1": {
            "fracdiff_d": float(best_trial.params["fracdiff_d"]),
            "weight_eps": float(best_trial.params["weight_eps"]),
        },
        "module3": {
            "num_leaves": int(best_trial.params["alpha_num_leaves"]),
            "learning_rate": float(best_trial.params["alpha_learning_rate"]),
            "feature_fraction": float(best_trial.params["alpha_feature_fraction"]),
        },
        "module4": {
            "num_leaves": int(best_trial.params["meta_num_leaves"]),
            "learning_rate": float(best_trial.params["meta_learning_rate"]),
            "feature_fraction": float(best_trial.params["meta_feature_fraction"]),
        },
        "trial_user_attributes": _json_safe(best_trial.user_attrs),
    }

    best_params_path = output_root / "best_params.json"
    best_params_path.write_text(json.dumps(_json_safe(best_params), indent=2), encoding="utf-8")

    trials_path = output_root / "study_trials.csv"
    trials_frame = study.trials_dataframe()
    trials_frame.to_csv(trials_path, index=False)

    return HPOArtifacts(
        config=config,
        best_value=float(best_trial.value),
        best_params=_json_safe(best_params),
        best_params_path=best_params_path,
        trials_path=trials_path,
        study_name=study.study_name,
        completed_trials=len(completed_trials),
    )
