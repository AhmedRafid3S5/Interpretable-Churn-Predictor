"""Shared training harness for the four baseline neural models (MLP, FT-Transformer, TabNet, NAM).

Companion documentation: docs/baselines.md

NOTE: per the XMV-Net skeleton notebook, Siam owns the canonical `train_utils.py` for XMV-Net
(AMP + early-stopping-on-fold-val-AUC + config logging). This module covers the same
responsibilities for the four *baseline* models so their numbers land in one shared results
schema. Reconcile the two once Siam's lands -- don't let two training loops silently diverge.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    brier_score_loss,
)

import data_loader as dl

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "baseline_results.csv"
RESULTS_SCHEMA = [
    "model", "fold", "auc", "ap", "p_at_10", "r_at_10", "mcc_at_10", "f1_at_10",
    "ece", "brier", "n_params", "best_epoch", "train_seconds", "config_json",
]

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
EXPLAIN_DIR = RESULTS_DIR / "explanations"

# §5 calls for "a fixed 5-10K-customer evaluation sample" shared across every model, so faithfulness
# metrics (deletion/insertion, infidelity, sensitivity, stability, rank agreement) are computed on
# identical rows for every architecture. Fixed to fold EXPLAIN_FOLD's validation split, first
# EVAL_SAMPLE_SIZE rows -- deterministic because the frozen folds + CSV row order never change, so
# every script that calls get_explanation_sample() gets the exact same customers.
EXPLAIN_FOLD = 0
EVAL_SAMPLE_SIZE = 10_000

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------
# Metrics (§8 of the proposal)
# --------------------------------------------------------------------------

def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.10) -> tuple[float, float]:
    """Precision/Recall at a fixed-budget top decile -- the operational metric (§8)."""
    n = len(y_true)
    k = max(1, int(round(n * frac)))
    order = np.argsort(-y_score)
    top = order[:k]
    tp = float(y_true[top].sum())
    precision = tp / k
    recall = tp / y_true.sum() if y_true.sum() > 0 else float("nan")
    return precision, recall


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_prob, bins[1:-1])
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Full §8 metric set for one fold's validation predictions."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)

    p10, r10 = precision_recall_at_k(y_true, y_prob, 0.10)
    k = max(1, int(round(len(y_prob) * 0.10)))
    thresh = np.sort(y_prob)[::-1][k - 1]
    y_pred_at_10 = (y_prob >= thresh).astype(np.int64)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
        "p_at_10": p10,
        "r_at_10": r10,
        "mcc_at_10": float(matthews_corrcoef(y_true, y_pred_at_10)),
        "f1_at_10": float(f1_score(y_true, y_pred_at_10)),
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


# --------------------------------------------------------------------------
# Results logging -- one shared schema for all 4 baselines (and comparable to XMV-Net's)
# --------------------------------------------------------------------------

def log_result(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_SCHEMA)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULTS_SCHEMA})


# --------------------------------------------------------------------------
# Data plumbing shared by every torch baseline
# --------------------------------------------------------------------------

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).to(DEVICE)
        logit = model(xb)
        out.append(torch.sigmoid(logit).detach().cpu().numpy())
    model.train()
    return np.concatenate(out)


def load_fold_arrays(fold: int, sample_rows: Optional[int] = None):
    """One fold's standardized train/val arrays, using the team's frozen split (§Step 0.2 of the guide).

    Returns (Xtr, ytr, Xva, yva, feature_cols, view_groups, ids_va). `ids_va` lets any saved
    explanation artifact be traced back to the account it belongs to.
    """
    train = dl.load_train(n_rows=sample_rows)
    folds = dl.make_folds(
        train.y, frozen_path=dl.frozen_folds_path(), ids=train.ids,
        allow_placeholder=sample_rows is not None,  # a truncated sample can't verify against the full-set hash
    )
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]

    mean, std = dl.fit_standardizer(train.X, tr_idx)
    Xtr = dl.apply_standardizer(np.asarray(train.X[tr_idx]), mean, std)
    Xva = dl.apply_standardizer(np.asarray(train.X[va_idx]), mean, std)
    ytr = train.y[tr_idx].astype(np.float32)
    yva = train.y[va_idx].astype(np.float32)
    ids_va = train.ids[va_idx]
    return Xtr, ytr, Xva, yva, train.feature_cols, train.view_groups, ids_va


def get_explanation_sample(sample_rows: Optional[int] = None, n: int = EVAL_SAMPLE_SIZE):
    """The fixed, shared evaluation sample every model's explanation extractor should use (§5).

    Returns (X_eval, y_eval, ids_eval, feature_cols, view_groups). Always the first `n` rows of
    EXPLAIN_FOLD's validation split -- same customers regardless of which script calls this, so
    attention-vs-IG-vs-SHAP-vs-masks rank-agreement comparisons (§5, item 5) are apples-to-apples.
    """
    _, _, Xva, yva, feature_cols, view_groups, ids_va = load_fold_arrays(EXPLAIN_FOLD, sample_rows)
    n = min(n, len(Xva))
    return Xva[:n], yva[:n], ids_va[:n], feature_cols, view_groups


def save_checkpoint(model: torch.nn.Module, model_name: str, fold: int, config: Optional[dict] = None) -> Path:
    """Saves trained weights so post-hoc explainers (IG, DeepSHAP -- §4) can load the model later
    without retraining. Called automatically by train_torch_model when save_checkpoints=True.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{model_name}_fold{fold}.pt"
    torch.save({"state_dict": model.state_dict(), "config": config or {}}, path)
    return path


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------

@dataclass
class EarlyStopper:
    patience: int = 8
    best: float = float("-inf")
    counter: int = 0
    best_state: Optional[dict] = None

    def step(self, score: float, model: torch.nn.Module) -> bool:
        """Returns True iff this is a new best (and snapshots the state)."""
        if score > self.best:
            self.best = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience


# --------------------------------------------------------------------------
# Generic AMP training loop, shared by MLP / FT-Transformer / NAM
# (TabNet is fit through its own sklearn-style API -- see baseline_tabnet.py)
# --------------------------------------------------------------------------

def train_torch_model(
    model_fn: Callable[[], torch.nn.Module],
    model_name: str,
    epochs: int = 60,
    patience: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 2048,
    sample_rows: Optional[int] = None,
    folds_to_run: Iterable[int] = range(5),
    config: Optional[dict] = None,
    aux_loss_fn: Optional[Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]] = None,
    aux_loss_weight: float = 0.0,
    log: bool = True,
    verbose: bool = True,
    return_models: bool = False,
    save_checkpoints: bool = False,
):
    """Trains `model_fn()` (a 0-arg constructor so each fold gets a fresh model) across the given
    folds. Model.forward(x) must return a raw logit of shape (B,). Every model in this file's
    companion scripts (MLP, FT-Transformer wrapper, NAM) matches that contract.

    Returns `all_rows` (metric dicts), or `(all_rows, models)` if `return_models=True` -- useful
    when a caller needs the trained weights afterward (e.g. NAM's shape-function extraction)
    without a second, wasted training pass.
    """
    all_rows = []
    models = []
    for fold in folds_to_run:
        Xtr, ytr, Xva, yva, _, _, _ = load_fold_arrays(fold, sample_rows)

        model = model_fn().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        use_amp = DEVICE.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        loader = make_loader(Xtr, ytr, batch_size)
        stopper = EarlyStopper(patience=patience)

        t0 = time.time()
        best_epoch = 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                    logit = model(xb)
                    loss = F.binary_cross_entropy_with_logits(logit, yb)
                    if aux_loss_fn is not None and aux_loss_weight > 0:
                        loss = loss + aux_loss_weight * aux_loss_fn(model, xb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            val_prob = predict_proba(model, Xva, batch_size=max(batch_size, 4096))
            val_auc = roc_auc_score(yva, val_prob)
            improved = stopper.step(val_auc, model)
            if improved:
                best_epoch = epoch
            if verbose:
                marker = "  *" if improved else ""
                print(f"[{model_name}] fold {fold} epoch {epoch:>3}  val_auc {val_auc:.5f}{marker}")
            if stopper.should_stop:
                if verbose:
                    print(f"[{model_name}] fold {fold} early stop at epoch {epoch} "
                          f"(best {stopper.best:.5f} @ epoch {best_epoch})")
                break

        model.load_state_dict(stopper.best_state)
        train_seconds = time.time() - t0

        if save_checkpoints:
            save_checkpoint(model, model_name, fold, config)

        val_prob = predict_proba(model, Xva, batch_size=max(batch_size, 4096))
        metrics = compute_metrics(yva, val_prob)
        n_params = sum(p.numel() for p in model.parameters())

        row = {
            "model": model_name, "fold": fold, **metrics, "n_params": n_params,
            "best_epoch": best_epoch, "train_seconds": round(train_seconds, 1),
            "config_json": json.dumps(config or {}),
        }
        if log:
            log_result(row)
        all_rows.append(row)
        if return_models:
            models.append(model)
    return (all_rows, models) if return_models else all_rows
