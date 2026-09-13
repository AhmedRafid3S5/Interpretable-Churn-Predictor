"""TabNet -- Candidate C, the canonical "interpretable tabular NN" baseline.

Per the proposal's §2: intrinsic sparse masks are the selling point, but it is "notoriously fiddly
to tune and slow to converge" and typically underperforms both GBDTs and simple MLPs. Uses
`pytorch-tabnet`'s own sklearn-style TabNetClassifier rather than train_utils.train_torch_model,
because its .fit() API (and its early stopping / masking internals) doesn't match the generic
forward(x)->logit contract the other three baselines share.

Companion documentation: docs/baselines.md

Usage:
    python baseline_tabnet.py --tune --n-trials 12
    python baseline_tabnet.py
    pip install pytorch-tabnet --break-system-packages   # if not already installed
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import data_loader as dl
import train_utils as tu

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch-tabnet is required for the TabNet baseline. "
        "Install with: pip install pytorch-tabnet --break-system-packages"
    ) from e

DEFAULT_CONFIG = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "lr": 2e-2,
    "batch_size": 4096,
    "virtual_batch_size": 256,
}


def make_tabnet(config: dict) -> TabNetClassifier:
    return TabNetClassifier(
        n_d=config["n_d"],
        n_a=config["n_a"],
        n_steps=config["n_steps"],
        gamma=config["gamma"],
        lambda_sparse=config["lambda_sparse"],
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": config["lr"]},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        seed=tu.SEED,
        verbose=0,
        device_name=tu.DEVICE.type,
    )


def run_fold(fold: int, config: dict, max_epochs: int, patience: int,
             sample_rows, model_name: str = "tabnet", log: bool = True):
    Xtr, ytr, Xva, yva, feature_cols, _, _ = tu.load_fold_arrays(fold, sample_rows)

    clf = make_tabnet(config)
    t0 = time.time()
    clf.fit(
        Xtr, ytr.astype(np.int64),
        eval_set=[(Xva, yva.astype(np.int64))],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=config["batch_size"],
        virtual_batch_size=config["virtual_batch_size"],
        drop_last=True,
    )
    train_seconds = time.time() - t0

    val_prob = clf.predict_proba(Xva)[:, 1]
    metrics = tu.compute_metrics(yva, val_prob)
    n_params = sum(p.numel() for p in clf.network.parameters())
    best_epoch = int(np.argmax(clf.history["val_auc"])) if "val_auc" in clf.history else -1

    row = {
        "model": model_name, "fold": fold, **metrics, "n_params": n_params,
        "best_epoch": best_epoch, "train_seconds": round(train_seconds, 1),
        "config_json": json.dumps(config),
    }
    if log:
        tu.log_result(row)
    return row, clf, feature_cols


def save_masks(clf: TabNetClassifier, X: np.ndarray, ids: np.ndarray, feature_cols: list[str], out_path):
    """Aggregated *and* per-decision-step instance-level feature masks (§4/§5/§10 payoff of this
    model) -- hand off to whoever does the XAI faithfulness comparison, per the team guide's Step 5.
    Per-step masks matter beyond the aggregate: they're the "internal layer values" for TabNet --
    each of the n_steps sequential attention steps picks a different feature subset, and §5's
    rank-agreement / stability checks are more informative on the raw per-step sequence than on
    the already-aggregated mask alone.
    """
    masks, masks_per_step = clf.explain(X)
    np.savez_compressed(
        out_path,
        ids=np.array(ids),
        feature_cols=np.array(feature_cols),
        masks_aggregated=masks,
        **{f"masks_step{k}": v for k, v in masks_per_step.items()},
    )
    print(f"[tabnet] saved aggregated + {len(masks_per_step)}-step masks for "
          f"{len(X):,} rows -> {out_path}")


# --------------------------------------------------------------------------
# Optuna tuning (fold 0 only; TabNet converges slowly so keep max_epochs/patience tighter here
# than in the final run, per the guide's "~10-15 trials" note)
# --------------------------------------------------------------------------

def objective(trial, sample_rows):
    config = {
        "n_d": trial.suggest_categorical("n_d_n_a", [8, 16, 32, 64]),
        "n_steps": trial.suggest_int("n_steps", 3, 7),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
        "lr": trial.suggest_float("lr", 1e-3, 4e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "virtual_batch_size": 256,
    }
    config["n_a"] = config["n_d"]  # kept equal, per the guide's note

    row, _, _ = run_fold(0, config, max_epochs=60, patience=8, sample_rows=sample_rows,
                          model_name="tabnet_tuning", log=False)
    trial.set_user_attr("config", config)
    return row["auc"]


def tune(n_trials: int, sample_rows) -> dict:
    import optuna

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tu.SEED))
    study.optimize(lambda t: objective(t, sample_rows), n_trials=n_trials)

    best_config = study.best_trial.user_attrs["config"]
    print(f"[tabnet] best fold-0 AUC {study.best_value:.5f}  config={best_config}")
    return best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save-explanations", action="store_true",
                         help="save fold-0 checkpoint + aggregated/per-step masks on the fixed "
                              "§5 evaluation sample (results/explanations/)")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.tune:
        config = tune(args.n_trials, args.sample_rows)

    rows = []
    for fold in args.folds:
        row, clf, feature_cols = run_fold(
            fold, config, max_epochs=args.max_epochs, patience=args.patience,
            sample_rows=args.sample_rows,
        )
        rows.append(row)
        print(f"[tabnet] fold {fold}  auc={row['auc']:.5f}  "
              f"epochs_trained={row['best_epoch']}  {row['train_seconds']}s")

        if args.save_explanations and fold == tu.EXPLAIN_FOLD:
            tu.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
            tu.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            clf.save_model(str(tu.CHECKPOINT_DIR / f"tabnet_fold{fold}"))

            X_eval, _, ids_eval, feature_cols, _ = tu.get_explanation_sample(args.sample_rows)
            save_masks(clf, X_eval, ids_eval, feature_cols, tu.EXPLAIN_DIR / "tabnet_masks.npz")

    aucs = [r["auc"] for r in rows]
    mean_auc = sum(aucs) / len(aucs)
    print(f"\n[tabnet] {len(rows)} fold(s)  AUC = {mean_auc:.5f} "
          f"± {(sum((a - mean_auc) ** 2 for a in aucs) / len(aucs)) ** 0.5:.5f}")
    print(f"[tabnet] config: {json.dumps(config)}")
    print(f"[tabnet] results appended to {tu.RESULTS_CSV}")


if __name__ == "__main__":
    main()
