"""Plain concatenated-feature MLP -- the "normal neural net" baseline.

Per the proposal's §7: "the critical ablation-as-baseline: any multi-view gain must beat this,
or the architecture is decoration." No view splitting -- all 127 features go into one vector.

Companion documentation: docs/baselines.md

Usage:
    python baseline_mlp.py --tune --n-trials 20          # Optuna search on fold 0, then full 5-fold
    python baseline_mlp.py                                # full 5-fold with the default config below
    python baseline_mlp.py --sample-rows 50000 --folds 0  # fast smoke test
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import data_loader as dl
import train_utils as tu

DEFAULT_CONFIG = {
    "widths": (128, 64, 32),
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 2048,
}


class PlainMLP(nn.Module):
    def __init__(self, in_dim: int, widths=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.GELU(), nn.Dropout(dropout)]
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def model_builder(in_dim: int, config: dict):
    return lambda: PlainMLP(in_dim, widths=tuple(config["widths"]), dropout=config["dropout"])


# --------------------------------------------------------------------------
# Internal-layer extraction for faithfulness evaluation (§5).
#
# Plain MLP has no intrinsic explanation, so it's the "control" for the §5 comparison: every
# faithfulness number for it has to come from a post-hoc method (IG/DeepSHAP) run against the
# saved checkpoint. What's captured here is the raw per-layer activations on the fixed evaluation
# sample -- useful for sanity-checking those post-hoc attributions (e.g. does an IG-important
# feature actually move the hidden representation much?) and as a cheap baseline for representation
# -level stability/sensitivity checks alongside the other three models' intrinsic explanations.
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(model: PlainMLP, X_eval, batch_size: int = 4096) -> dict:
    device = next(model.parameters()).device
    model.eval()
    layer_outputs: dict[str, list] = {}
    hooks = []

    def make_hook(name):
        def hook(_module, _inp, out):
            layer_outputs.setdefault(name, []).append(out.detach().cpu())
        return hook

    for name, module in model.net.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(f"linear_{name}")))

    for start in range(0, len(X_eval), batch_size):
        xb = torch.from_numpy(X_eval[start:start + batch_size]).to(device)
        model(xb)

    for h in hooks:
        h.remove()
    return {k: torch.cat(v).numpy() for k, v in layer_outputs.items()}


def save_activations(activations: dict, ids_eval, out_path) -> None:
    tu.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=np.array(ids_eval), **activations)
    print(f"[mlp] saved {len(activations)} layer(s) of activations for {len(ids_eval):,} rows -> {out_path}")


# --------------------------------------------------------------------------
# Optuna tuning (fold 0 only, per the proposal's "<=25 trials on 1 fold" budget)
# --------------------------------------------------------------------------

ARCH_CHOICES = ["256-128-64", "128-64-32", "128-64", "64-32", "256-128-64-32"]


def objective(trial, in_dim: int, sample_rows):
    import optuna  # local import so the module still loads without optuna installed

    arch = trial.suggest_categorical("arch", ARCH_CHOICES)
    widths = tuple(int(w) for w in arch.split("-"))
    config = {
        "widths": widths,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    }

    in_dim_local = in_dim
    rows = tu.train_torch_model(
        model_builder(in_dim_local, config),
        model_name="mlp_tuning",
        epochs=40,
        patience=5,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        sample_rows=sample_rows,
        folds_to_run=[0],
        config=config,
        log=False,
        verbose=False,
    )
    trial.set_user_attr("config", config)
    return rows[0]["auc"]


def tune(in_dim: int, n_trials: int, sample_rows) -> dict:
    import optuna

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tu.SEED))
    study.optimize(lambda t: objective(t, in_dim, sample_rows), n_trials=n_trials)

    best_config = study.best_trial.user_attrs["config"]
    print(f"[mlp] best fold-0 AUC {study.best_value:.5f}  config={best_config}")
    return best_config


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true", help="run Optuna search on fold 0 first")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--sample-rows", type=int, default=None, help="truncate rows for a fast smoke test")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save-explanations", action="store_true",
                         help="save fold-0 checkpoint + internal-layer activations on the fixed "
                              "§5 evaluation sample (results/explanations/)")
    args = parser.parse_args()

    in_dim = len(dl.load_train(n_rows=1).feature_cols)

    config = dict(DEFAULT_CONFIG)
    if args.tune:
        config = tune(in_dim, args.n_trials, args.sample_rows)

    rows, models = tu.train_torch_model(
        model_builder(in_dim, config),
        model_name="mlp",
        epochs=args.epochs,
        patience=args.patience,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        sample_rows=args.sample_rows,
        folds_to_run=args.folds,
        config=config,
        return_models=True,
        save_checkpoints=args.save_explanations,
    )

    if args.save_explanations and tu.EXPLAIN_FOLD in args.folds:
        model = models[list(args.folds).index(tu.EXPLAIN_FOLD)]
        X_eval, _, ids_eval, _, _ = tu.get_explanation_sample(args.sample_rows)
        activations = extract_activations(model, X_eval)
        tu.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
        save_activations(activations, ids_eval, tu.EXPLAIN_DIR / "mlp_activations.npz")

    aucs = [r["auc"] for r in rows]
    print(f"\n[mlp] {len(rows)} fold(s)  AUC = {sum(aucs) / len(aucs):.5f} "
          f"± {(sum((a - sum(aucs) / len(aucs)) ** 2 for a in aucs) / len(aucs)) ** 0.5:.5f}")
    print(f"[mlp] config: {json.dumps(config)}")
    print(f"[mlp] results appended to {tu.RESULTS_CSV}")


if __name__ == "__main__":
    main()
