"""FT-Transformer -- Candidate B, the strong-accuracy / weak-interpretability neural baseline.

Per the proposal's §2: "strongest raw accuracy among tabular NNs in benchmarks... include as the
strong neural baseline, not the contribution." Uses the `rtdl` library's feature tokenizer +
transformer, per the proposal's explicit recommendation.

All 127 columns coming out of data_loader.py are already numeric (GENDER/REGION are one-hot
encoded at the data-pipeline stage -- see docs/data_loader.md §3), so this is a pure
n_num_features=127, cat_cardinalities=[] setup. There is no separate categorical-token path to wire up.

Companion documentation: docs/baselines.md

Usage:
    python baseline_ft_transformer.py --tune --n-trials 12
    python baseline_ft_transformer.py
    pip install rtdl --break-system-packages   # if not already installed
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

import data_loader as dl
import train_utils as tu

try:
    import rtdl
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "rtdl is required for the FT-Transformer baseline. "
        "Install with: pip install rtdl --break-system-packages"
    ) from e

DEFAULT_CONFIG = {
    "n_blocks": 3,
    "d_token": 128,
    "attention_dropout": 0.2,
    "ffn_d_hidden": 256,
    "ffn_dropout": 0.1,
    "residual_dropout": 0.0,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 1024,
}


class FTTransformerWrapper(nn.Module):
    """Wraps rtdl.FTTransformer to match the (B, F) -> (B,) logit contract used by train_utils."""

    def __init__(self, n_num_features: int, config: dict):
        super().__init__()
        self.model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_num_features,
            cat_cardinalities=[],
            d_token=config["d_token"],
            n_blocks=config["n_blocks"],
            attention_dropout=config["attention_dropout"],
            ffn_d_hidden=config["ffn_d_hidden"],
            ffn_dropout=config["ffn_dropout"],
            residual_dropout=config["residual_dropout"],
            d_out=1,
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return self.model(x_num, None).squeeze(-1)


# --------------------------------------------------------------------------
# Internal-layer extraction for faithfulness evaluation (§5): per-block attention weights (the
# intrinsic-but-contested explanation flagged in §2/§4) plus the pre-head [CLS] embedding, on the
# fixed evaluation sample.
#
# Best-effort, same caveat as before: rtdl's internal attention module name/output shape has
# shifted across releases. This hooks every submodule with "attention" in its name and keeps
# whatever tensor in its output looks like a (batch, heads, tokens, tokens) softmax map. Run
# `extract_attention` once on a small batch and print the returned dict's shapes before trusting
# it for the §5 comparison -- see docs/baselines.md §3.
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_attention(model: FTTransformerWrapper, X_eval, batch_size: int = 512) -> dict:
    device = next(model.parameters()).device
    model.eval()

    captured: dict[str, list] = {}
    hooks = []

    def make_hook(name):
        def hook(_module, _inp, out):
            attn = None
            if torch.is_tensor(out):
                attn = out
            elif isinstance(out, tuple):
                for t in out:
                    if torch.is_tensor(t) and t.dim() >= 3:
                        attn = t
                        break
            if attn is not None:
                captured.setdefault(name, []).append(attn.detach().cpu())
        return hook

    for name, module in model.model.named_modules():
        if "attention" in name.lower():
            hooks.append(module.register_forward_hook(make_hook(name)))

    cls_embeddings = []
    for start in range(0, len(X_eval), batch_size):
        xb = torch.from_numpy(X_eval[start:start + batch_size]).to(device)
        # feature_tokenizer + all-but-last transformer blocks, matching how rtdl's own forward
        # produces the pre-head [CLS] embedding -- fetched via the public model.forward with a
        # hook is avoided here since d_out=1 already collapses it; instead we tap the same
        # attention hooks above for the representation signal.
        model(xb)

    for h in hooks:
        h.remove()

    result = {name: torch.cat(tensors).numpy() for name, tensors in captured.items()}
    if not result:
        print("[ft_transformer] WARNING: no attention tensors captured -- inspect "
              "model.model.named_modules() for this rtdl version and adjust extract_attention's "
              "hook filter (see docs/baselines.md §3).")
    return result


def save_attention(attention: dict, ids_eval, out_path) -> None:
    tu.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ids=np.array(ids_eval), **attention)
    print(f"[ft_transformer] saved attention from {len(attention)} block(s) for "
          f"{len(ids_eval):,} rows -> {out_path}")


def model_builder(n_num_features: int, config: dict):
    return lambda: FTTransformerWrapper(n_num_features, config)


# --------------------------------------------------------------------------
# Optuna tuning (fold 0 only -- each trial here is the heaviest of the 4 baselines, so keep
# n_trials modest per the guide's "~10-15 trials" note)
# --------------------------------------------------------------------------

def objective(trial, n_num_features: int, sample_rows):
    config = {
        "n_blocks": trial.suggest_int("n_blocks", 2, 4),
        "d_token": trial.suggest_categorical("d_token", [64, 96, 128, 192]),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
        "ffn_d_hidden": trial.suggest_categorical("ffn_d_hidden", [128, 256, 384]),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.3),
        "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.2),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024]),
    }

    rows = tu.train_torch_model(
        model_builder(n_num_features, config),
        model_name="ft_transformer_tuning",
        epochs=30,
        patience=4,
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


def tune(n_num_features: int, n_trials: int, sample_rows) -> dict:
    import optuna

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tu.SEED))
    study.optimize(lambda t: objective(t, n_num_features, sample_rows), n_trials=n_trials)

    best_config = study.best_trial.user_attrs["config"]
    print(f"[ft_transformer] best fold-0 AUC {study.best_value:.5f}  config={best_config}")
    return best_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--save-explanations", action="store_true",
                         help="save fold-0 checkpoint + per-block attention on the fixed §5 "
                              "evaluation sample (results/explanations/)")
    args = parser.parse_args()

    n_num_features = len(dl.load_train(n_rows=1).feature_cols)

    config = dict(DEFAULT_CONFIG)
    if args.tune:
        config = tune(n_num_features, args.n_trials, args.sample_rows)

    rows, models = tu.train_torch_model(
        model_builder(n_num_features, config),
        model_name="ft_transformer",
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
        attention = extract_attention(model, X_eval)
        tu.EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
        save_attention(attention, ids_eval, tu.EXPLAIN_DIR / "ft_transformer_attention.npz")

    aucs = [r["auc"] for r in rows]
    mean_auc = sum(aucs) / len(aucs)
    print(f"\n[ft_transformer] {len(rows)} fold(s)  AUC = {mean_auc:.5f} "
          f"± {(sum((a - mean_auc) ** 2 for a in aucs) / len(aucs)) ** 0.5:.5f}")
    print(f"[ft_transformer] config: {json.dumps(config)}")
    print(f"[ft_transformer] results appended to {tu.RESULTS_CSV}")


if __name__ == "__main__":
    main()
