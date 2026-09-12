# `baseline_mlp.py`, `baseline_ft_transformer.py`, `baseline_tabnet.py`, `baseline_nam.py`, `train_utils.py` — companion documentation

The four "not-XMV-Net" neural baselines from the proposal's §2/§7: the **Plain MLP** ("normal neural
net"), **FT-Transformer** (Candidate B), **TabNet** (Candidate C), and **NAM** (Candidate D). All
four consume the same frozen split committed at `folds/frozen_folds_5.npy` and report into one
shared results table, so their numbers are directly comparable to each other and to XMV-Net.

---

## 1. Shared contract (`train_utils.py`)

Three of the four models (MLP, FT-Transformer, NAM) are plain `nn.Module`s whose `forward(x)`
returns a raw logit of shape `(B,)`. `train_utils.train_torch_model(...)` is the one training loop
all three share: AMP on CUDA, `AdamW`, early stopping on fold-validation AUC (matching the
convention set in `xmvnet_skeleton.ipynb`), and a snapshot-and-restore of the best-epoch weights.

TabNet does **not** fit this contract — `pytorch_tabnet.TabNetClassifier` is a sklearn-style
estimator with its own `.fit()`/early-stopping internals — so `baseline_tabnet.py` drives it
directly and only *reuses* `train_utils.compute_metrics` / `train_utils.log_result` so its output
row has the same shape as everyone else's.

**Why a separate module from Siam's `train_utils.py`?** The skeleton notebook flags that Siam owns
the canonical harness for XMV-Net (temperature/λ schedules, diagnostics hooks, the whole-output-dict
loss). The four baselines here don't need any of that — just BCE, optionally plus one auxiliary
term (NAM's output penalty) — so a lighter, separate module avoids overloading Siam's interface with
baseline-only concerns. **Reconcile before the results table is finalized**: if Siam's harness ends
up covering plain-logit models too, point these four scripts at it instead of duplicating logic.

### Results schema (`results/baseline_results.csv`)

One row per `(model, fold)`: `auc, ap, p_at_10, r_at_10, mcc_at_10, f1_at_10, ece, brier` (§8 of the
proposal) plus `n_params, best_epoch, train_seconds, config_json` for bookkeeping. Every script
appends to the same file — safe to run all four independently, in any order, any number of times.

### Metrics (`train_utils.compute_metrics`)

- `precision_recall_at_k` — fixed-budget top-decile precision/recall, matching the datathon's
  Phase-8 operational framing.
- `expected_calibration_error` — 15-bin ECE; needed because the proposal's loss ablation (A6) is
  partly justified by calibration cost, and this is the metric that shows it.
- Everything else is a direct `sklearn.metrics` call.

### `load_fold_arrays(fold, sample_rows)`

The one function every script calls to get standardized train/val arrays for a fold: loads
`data_loader.load_train`, resolves folds via `data_loader.make_folds` against the **frozen** split
(`folds/frozen_folds_5.npy` is already committed in this repo — nobody needs to run
`freeze_folds()`), fits the standardizer on train-fold rows only, and applies it to both splits.
Pass `sample_rows` for a fast smoke test on a prefix of the data (used automatically with
`allow_placeholder=True` in that case, since a truncated sample can't be hash-verified against the
full-data fold manifest).

---

## 2. `baseline_mlp.py` — Plain MLP

No view splitting: all 127 features concatenated into one vector, 2–3 `Linear→BatchNorm→GELU→Dropout`
blocks, sigmoid output. Deliberately kept near XMV-Net's parameter budget (~100–300K) so a
comparison isn't confounded by raw capacity — per §7, this is the number any multi-view claim must beat.

`--tune` runs an Optuna search (TPE sampler, fold 0 only) over architecture (5 preset width strings
rather than a free per-layer search, to keep the search space simple), dropout, LR, weight decay,
batch size. Default trial budget 20, matching the guide's "~15-20 trials" recommendation.

## 3. `baseline_ft_transformer.py` — FT-Transformer (Candidate B)

Thin wrapper around `rtdl.FTTransformer.make_baseline`. Because `data_loader.py` already one-hot
encodes `GENDER`/`REGION` into the numeric matrix (see `docs/data_loader.md` §3), this is a pure
`n_num_features=127, cat_cardinalities=[]` setup — there's no separate categorical-embedding path
to configure.

**Version caveat:** `rtdl`'s constructor kwargs and internal module names have shifted across
releases. The parameter names used here (`attention_dropout`, `ffn_d_hidden`, `ffn_dropout`,
`residual_dropout`, `d_out`) match the commonly-documented `make_baseline` signature, but **run
`python -c "import rtdl, inspect; print(inspect.signature(rtdl.FTTransformer.make_baseline))"`
against your installed version before the first real run** and adjust if it's drifted.
`FTTransformerWrapper._register_attention_hooks` is similarly best-effort: it greps for any
submodule with "attention" in its name and caches whatever looks like an attention-weight tensor
in its output. Verify `model._last_attn` is actually populated after one forward pass before relying
on it for the §4/§5 attention-as-attribution comparison — if it's `None`, inspect
`model.model.named_modules()` and adjust the hook's name filter.

`--tune` trial budget defaults to 12 (lower than the MLP/NAM) since this is the most expensive
model per trial — check wall-clock against the team's shared GPU-hour budget before widening it.

## 4. `baseline_tabnet.py` — TabNet (Candidate C)

Drives `pytorch_tabnet.tab_model.TabNetClassifier` directly rather than through
`train_utils.train_torch_model`. `n_d == n_a` is enforced (kept equal per convention). Expect slower
convergence and more variance across trials than the other three models — this is the documented,
expected behavior per §2, not a configuration bug.

`save_masks(clf, X, feature_cols, out_path)` calls `clf.explain(X)` for the aggregated per-instance
feature masks and writes them to `results/tabnet_masks.npz` — the intrinsic-explanation artifact
that feeds the §5 faithfulness comparison and §10's qualitative figures. By default this runs once,
on fold 0's validation split (`--save-masks-fold`).

## 5. `baseline_nam.py` — Neural Additive Model (Candidate D)

`f(x) = bias + Σ_f g_f(x_f)`, one tiny MLP per feature. Implemented with **batched per-feature
weight tensors** (`nn.Parameter` of shape `(n_features, hidden, ...)`) combined via `torch.einsum`,
so all 127 per-feature subnetworks train as a handful of big matmuls rather than a 127-iteration
Python loop over `nn.Linear` modules — see `NAM.feature_contributions`.

Loss is BCE plus `output_penalty_loss` (mean-squared per-feature contribution), matching §6's "NAM
output penalty + weight decay" row; weight decay itself goes through the optimizer's
`weight_decay` argument, not a manual penalty term.

`extract_shape_functions(model, Xtr, feature_cols)` is the model's real deliverable for §10: for
each feature, it sweeps a 41-point grid across that feature's observed (standardized) range with
every other feature held at its mean (= 0, since inputs are standardized) and records `g_f(x)`.
Because the model is exactly additive, this is the *true* shape function, not a GBDT-style
partial-dependence approximation. Saved to `results/nam_shape_functions.npz`
(`feature_cols`, `grid`, `curves` arrays) — hand this file to whoever builds the §10 shape-function
figure.

`--tune` gets the largest default trial budget (25) of the four scripts since NAM is by far the
cheapest to train per trial.

---

## 5b. Saving internal layer values for faithfulness evaluation (§5)

Every script now takes a `--save-explanations` flag. When set, it:

1. Trains fold `EXPLAIN_FOLD` (0) as usual, then saves a checkpoint to
   `results/checkpoints/{model}_fold0.pt` (TabNet uses its own `.save_model()` zip format instead).
   Checkpoints exist so post-hoc explainers (IG, DeepSHAP -- §4) can be computed later without
   retraining.
2. Runs the model once more over `train_utils.get_explanation_sample()` -- the **first 10,000 rows
   of fold 0's validation split**, fixed and identical across all four scripts (and matching the
   "5-10K-customer evaluation sample" §5 asks for) -- and saves each model's internal/intrinsic
   values to `results/explanations/`:

| Model | File | Contents |
|---|---|---|
| MLP | `mlp_activations.npz` | Post-`Linear` activation at every layer, per row |
| FT-Transformer | `ft_transformer_attention.npz` | Per-block attention tensors, per row (best-effort -- verify shapes against your `rtdl` version first, see §3 above) |
| TabNet | `tabnet_masks.npz` | Aggregated mask **and** each of the `n_steps` per-step masks, per row |
| NAM | `nam_instance_contributions.npz` | `g_f(x_i)` for every feature *f* and row *i* (exact local explanation) |

Every file also stores the row `ids` so artifacts can be joined back to accounts, and to each other
across models, for the §5 rank-agreement check.

**Why this is a small change:** none of the four architectures needed new layers or a different
training procedure -- the values already exist inside a forward pass, this just hooks or exposes
them on a fixed, shared sample and writes them to disk once, right after the model that's going in
the results table finishes training.

## 6. Expected outcome, per §2/§7 of the proposal

Roughly: `LightGBM > FT-Transformer ≳ XMV-Net > Plain MLP ≳ TabNet > NAM`. If your numbers wildly
contradict this ordering, treat it as a signal to debug (standardization, a leaked column, or a
fold mismatch against the frozen split) before reporting — not automatically a novel finding. NAM's
low AUC in particular is expected and *is* the point: it anchors the accuracy–interpretability
frontier plot in §12, it isn't a bug to chase away by over-tuning.

---

## 7. Running everything

```bash
# one-off environment prep
pip install rtdl pytorch-tabnet optuna --break-system-packages

# fast smoke test on a 50k-row prefix, fold 0 only, before committing GPU-hours
python baseline_mlp.py           --sample-rows 50000 --folds 0
python baseline_ft_transformer.py --sample-rows 50000 --folds 0
python baseline_tabnet.py         --sample-rows 50000 --folds 0
python baseline_nam.py            --sample-rows 50000 --folds 0

# real runs: tune on fold 0, then train all 5 folds with the winning config
python baseline_mlp.py            --tune --n-trials 20
python baseline_ft_transformer.py --tune --n-trials 12
python baseline_tabnet.py         --tune --n-trials 12
python baseline_nam.py            --tune --n-trials 25
```

Each run appends to `results/baseline_results.csv`; rerunning a model just adds more rows (dedupe
by `model`+`fold`+`config_json` downstream if you re-run with the same config). `run_all_baselines.py`
runs all four back-to-back with tuning enabled, for a single unattended pass.
