"""Neural Additive Model -- Candidate D, the interpretability anchor.

Per the proposal's §2: f(x) = beta_0 + sum_i g_i(x_i), one tiny MLP per feature. "The gold
standard -- fully glass-box... its explanations are faithful by construction." Expected to be the
weakest of the four baselines on accuracy (§2: "accuracy ceiling typically well below GBDTs on
interaction-heavy data") -- that's the correct, expected outcome, not a bug. Its real payoff is the
per-feature shape-function curves used in §10's qualitative figures.

No off-the-shelf library is standard for NAM (unlike rtdl / pytorch-tabnet), so this implements it
directly. All 127 per-feature subnetworks are batched into a handful of einsum ops rather than a
Python loop over 127 tiny nn.Linear modules -- see NAM.feature_contributions.

Companion documentation: docs/baselines.md

Usage:
    python baseline_nam.py --tune --n-trials 25
    python baseline_nam.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_loader as dl
import train_utils as tu

DEFAULT_CONFIG = {
    "hidden": 32,
    "depth": 2,
    "dropout": 0.1,
    "output_penalty": 1e-3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 2048,
}


class NAM(nn.Module):
    """One small MLP per feature, summed at the end: logit = bias + sum_f g_f(x_f).

    Implemented as batched per-feature weight tensors (shape (n_features, ...)) combined with
    torch.einsum, so the whole ensemble of tiny per-feature nets runs as a few big matmuls instead
    of a 127-iteration Python loop.
    """

    def __init__(self, n_features: int, hidden: int = 32, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.depth = depth

        self.w1 = nn.Parameter(torch.randn(n_features, hidden) * (1.0 / hidden ** 0.5))
        self.b1 = nn.Parameter(torch.zeros(n_features, hidden))

        self.w_hidden = nn.ParameterList([
            nn.Parameter(torch.randn(n_features, hidden, hidden) * (1.0 / hidden ** 0.5))
            for _ in range(max(0, depth - 1))
        ])
        self.b_hidden = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features, hidden)) for _ in range(max(0, depth - 1))
        ])

        self.w_out = nn.Parameter(torch.randn(n_features, hidden) * (1.0 / hidden ** 0.5))
        self.b_out = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def feature_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F) standardized features -> contributions g_f(x_f): (B, F)."""
        h = torch.einsum("bf,fh->bfh", x, self.w1) + self.b1
        h = self.dropout(F.gelu(h))
        for w, b in zip(self.w_hidden, self.b_hidden):
            h = torch.einsum("bfh,fhk->bfk", h, w) + b
            h = self.dropout(F.gelu(h))
        contrib = torch.einsum("bfh,fh->bf", h, self.w_out) + self.b_out
        return contrib

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_contributions(x).sum(dim=1) + self.bias

    def output_penalty_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Mean squared per-feature output -- discourages any one shape function from dominating
        and keeps curves smooth (§6: "NAM output penalty + weight decay... keeps shape functions
        smooth"). Weight decay itself is handled by the optimizer's weight_decay arg.
        """
        return (self.feature_contributions(x) ** 2).mean()


def model_builder(n_features: int, config: dict):
    return lambda: NAM(n_features, hidden=config["hidden"], depth=config["depth"], dropout=config["dropout"])


# --------------------------------------------------------------------------
# Shape-function extraction -- the deliverable §10 actually wants from this model
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_shape_functions(model: NAM, Xtr: np.ndarray, feature_cols: list[str],
                             n_points: int = 41) -> dict:
    """For each feature, evaluate g_f over a grid spanning its observed (standardized) range with
    every other feature held at 0 (i.e. its mean, since inputs are standardized) -- this is exact
    for an additive model, unlike a GBDT partial-dependence plot which has to marginalize.
    """
    model.eval()
    device = next(model.parameters()).device
    n_features = Xtr.shape[1]

    lo = Xtr.min(axis=0)
    hi = Xtr.max(axis=0)
    grids = np.linspace(lo, hi, n_points, axis=0).astype(np.float32)  # (n_points, n_features)

    # Build a (n_points, n_features, n_features) batch: for grid point p and feature f, every
    # column is 0 except column f, which sweeps grids[:, f]. Evaluate all features in one batched
    # einsum call by reusing feature_contributions on a block-diagonal-ish input.
    curves = np.zeros((n_features, n_points), dtype=np.float32)
    batch = torch.zeros((n_points, n_features), device=device)
    for f in range(n_features):
        batch.zero_()
        batch[:, f] = torch.from_numpy(grids[:, f]).to(device)
        contrib = model.feature_contributions(batch)[:, f]  # only feature f's own curve is meaningful here
        curves[f] = contrib.cpu().numpy()

    return {
        "feature_cols": feature_cols,
        "grid": grids,          # (n_points, n_features), standardized x-values
        "curves": curves,       # (n_features, n_points), g_f(x) values
    }


@torch.no_grad()
def extract_instance_contributions(model: NAM, X_eval, ids_eval, batch_size: int = 4096) -> dict:
    """Per-instance, per-feature g_f(x_i) on the fixed §5 evaluation sample -- NAM's *local*
    explanation, exact by construction (no approximation, unlike IG/SHAP on the other 3 models).
    These are the "internal layer values" for NAM: the actual output of each feature's
    subnetwork for real customers, not just the grid used for the global shape-function plots.
    """
    model.eval()
    device = next(model.parameters()).device
    contribs = []
    for start in range(0, len(X_eval), batch_size):
        xb = torch.from_numpy(X_eval[start:start + batch_size]).to(device)
        contribs.append(model.feature_contributions(xb).cpu().numpy())
    return {"ids": np.array(ids_eval), "contributions": np.concatenate(contribs)}


def save_instance_contributions(data: dict, feature_cols: list[str], out_path) -> None:
    tu.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=data["ids"], contributions=data["contributions"],
                         feature_cols=np.array(feature_cols))
    print(f"[nam] saved per-instance contributions for {len(data['ids']):,} rows -> {out_path}")


def save_shape_functions(shape_data: dict, out_path) -> None:
    np.savez_compressed(
        out_path,
        feature_cols=np.array(shape_data["feature_cols"]),
        grid=shape_data["grid"],
        curves=shape_data["curves"],
    )
    print(f"[nam] saved shape functions for {len(shape_data['feature_cols'])} features -> {out_path}")


# --------------------------------------------------------------------------
# Optuna tuning (cheap model -- afford more trials than the other 3 baselines)
# --------------------------------------------------------------------------

def objective(trial, n_features: int, sample_rows):
    config = {
        "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
        "depth": trial.suggest_int("depth", 2, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "output_penalty": trial.suggest_float("output_penalty", 1e-5, 1e-1, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
    }

    rows = tu.train_torch_model(
        model_builder(n_features, config),
        model_name="nam_tuning",
        epochs=50,
        patience=6,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        sample_rows=sample_rows,
        folds_to_run=[0],
        config=config,
        aux_loss_fn=lambda m, xb: m.output_penalty_loss(xb),
        aux_loss_weight=config["output_penalty"],
        log=False,
        verbose=False,
    )
    trial.set_user_attr("config", config)
    return rows[0]["auc"]


def tune(n_features: int, n_trials: int, sample_rows) -> dict:
    import optuna

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tu.SEED))
    study.optimize(lambda t: objective(t, n_features, sample_rows), n_trials=n_trials)

    best_config = study.best_trial.user_attrs["config"]
    print(f"[nam] best fold-0 AUC {study.best_value:.5f}  config={best_config}")
    return best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--shape-fold", type=int, default=0,
                         help="fold whose trained model is used to extract shape functions")
    parser.add_argument("--save-explanations", action="store_true",
                         help="also save the fold-0 checkpoint + per-instance feature "
                              "contributions on the fixed §5 evaluation sample (results/explanations/)")
    args = parser.parse_args()

    train_meta = dl.load_train(n_rows=1)
    n_features = len(train_meta.feature_cols)

    config = dict(DEFAULT_CONFIG)
    if args.tune:
        config = tune(n_features, args.n_trials, args.sample_rows)

    rows, models = tu.train_torch_model(
        model_builder(n_features, config),
        model_name="nam",
        epochs=args.epochs,
        patience=args.patience,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        sample_rows=args.sample_rows,
        folds_to_run=args.folds,
        config=config,
        aux_loss_fn=lambda m, xb: m.output_penalty_loss(xb),
        aux_loss_weight=config["output_penalty"],
        return_models=True,
        save_checkpoints=args.save_explanations,
    )

    if args.shape_fold in args.folds:
        pos = list(args.folds).index(args.shape_fold)
        Xtr, _, _, _, feature_cols, _, _ = tu.load_fold_arrays(args.shape_fold, args.sample_rows)
        shape_data = extract_shape_functions(models[pos], Xtr, feature_cols)
        tu.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_shape_functions(shape_data, tu.RESULTS_DIR / "nam_shape_functions.npz")

    if args.save_explanations and tu.EXPLAIN_FOLD in args.folds:
        model = models[list(args.folds).index(tu.EXPLAIN_FOLD)]
        X_eval, _, ids_eval, feature_cols, _ = tu.get_explanation_sample(args.sample_rows)
        contrib_data = extract_instance_contributions(model, X_eval, ids_eval)
        tu.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
        save_instance_contributions(contrib_data, feature_cols, tu.EXPLAIN_DIR / "nam_instance_contributions.npz")

    aucs = [r["auc"] for r in rows]
    mean_auc = sum(aucs) / len(aucs)
    print(f"\n[nam] {len(rows)} fold(s)  AUC = {mean_auc:.5f} "
          f"± {(sum((a - mean_auc) ** 2 for a in aucs) / len(aucs)) ** 0.5:.5f}")
    print(f"[nam] config: {json.dumps(config)}")
    print(f"[nam] results appended to {tu.RESULTS_CSV}")


if __name__ == "__main__":
    main()
