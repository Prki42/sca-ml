import sys

import optuna
import torch
import yaml

from datasets import DATASET_CLASSES
from evaluate import compute_rank_curve
from models import TRIAL_MODELS
from train import split_profiling, train_model


def run_search(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    ds = DATASET_CLASSES[ds_cfg["name"]](**ds_cfg.get("params", {}))

    profiling_key = ds.profiling_key

    # Currently we only use ascad v1 which has fixed profiling key.
    # In the case profiling key not being constant we should split validation out of attack set.
    assert profiling_key is not None

    profile_traces, profile_labels, profile_plaintexts = ds.get_profiling()
    train_traces, train_labels, val_traces, val_labels, val_plaintexts = (
        split_profiling(profile_traces, profile_labels, profile_plaintexts)
    )

    model_builder = TRIAL_MODELS[cfg["model"]]
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("evaluation", {})
    study_cfg = cfg["study"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial: optuna.Trial) -> float:
        model = model_builder(trial, ds.input_dim, ds.num_classes)

        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        model = train_model(
            model,
            train_traces,
            train_labels,
            val_traces,
            val_labels,
            epochs=train_cfg.get("epochs", 200),
            batch_size=batch_size,
            lr=lr,
            patience=train_cfg.get("patience", 20),
            device=device,
        )

        curve = compute_rank_curve(
            model,
            val_traces,
            val_plaintexts,
            profiling_key,
            ds.leakage,
            eval_cfg.get("traces", 500),
            eval_cfg.get("rank_repeats", 10),
            device,
        )
        return curve.mean()

    study_name = study_cfg["name"]
    db_path = f"results/optuna/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=study_cfg.get("trials", 50))

    print(f"\nBest trial: rank={study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")
    return study


if __name__ == "__main__":
    run_search(sys.argv[1])
