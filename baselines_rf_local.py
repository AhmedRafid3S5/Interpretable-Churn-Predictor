import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
import sklearn
import data_loader as dl

print("code:", CODE_DIR)
print("numpy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__)

SEED = 42
RUN_TAG = "local_rf"                   # names the output files; the Kaggle run uses "kaggle"

MODELS = ["rf"]
DROP_VIEWS = []                        # ["recency"] -> ablation A7
SAMPLE_ROWS = None                     # e.g. 50_000 while debugging; None = all 595,000 rows

RF_TREES = 300
RF_MIN_LEAF = 20
N_JOBS = -1                            # all cores
SAVE_CHECKPOINTS = False

TOP_FRACTION = 0.10                    # retention budget: the top 10% of customers

OUT_DIR = CODE_DIR / "outputs" / "baselines"
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
print("outputs ->", OUT_DIR)


RF_PARAMS = dict(n_estimators=RF_TREES, max_features="sqrt", min_samples_leaf=RF_MIN_LEAF,
                 n_jobs=N_JOBS, random_state=SEED)
RUN_PARAMS = {"rf": RF_PARAMS}
print(RF_PARAMS)

import json

train = dl.load_train(n_rows=SAMPLE_ROWS)


def find_file(name):
    """Works for the repo layout, a flat dataset upload, or any attached dataset."""
    return next((Path(p) for p in [
        CODE_DIR / "folds" / name,
        CODE_DIR / name,
        *(sorted(Path("/kaggle/input").glob(f"**/{name}")) if Path("/kaggle/input").exists() else []),
    ] if Path(p).exists()), None)


FOLDS_FILE = find_file("frozen_folds_5.npy")
if FOLDS_FILE is None:
    raise FileNotFoundError("frozen_folds_5.npy not found -- attach the repo or the folds dataset")

folds = dl.make_folds(train.y, ids=train.ids, frozen_path=FOLDS_FILE)
y = train.y.astype(np.int32)
N_SPLITS = int(folds.max()) + 1

CHECKS_FILE = find_file("frozen_folds_5.checks.json")
if CHECKS_FILE is None:
    print("FEATURE_COLS check skipped: frozen_folds_5.checks.json is not attached")
else:
    expected = json.loads(CHECKS_FILE.read_text())["matrix"]["feature_cols"]
    assert train.feature_cols == expected, "FEATURE_COLS differ from the verified list"
    print("FEATURE_COLS match the verified list:", CHECKS_FILE)

print(f"{len(y):,} rows | {train.n_features} columns | {N_SPLITS} folds | churn {y.mean():.5f}")
print(train.summary().to_string(index=False))


from sklearn.metrics import roc_auc_score, average_precision_score

METRIC_NAMES = ["auc", "ap", "p_at_10", "r_at_10"]


def evaluate(y_true, scores):
    k = max(1, int(round(TOP_FRACTION * len(y_true))))
    top_k = np.argpartition(-scores, k - 1)[:k]
    hits = int(y_true[top_k].sum())
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ap": float(average_precision_score(y_true, scores)),
        "p_at_10": hits / k,
        "r_at_10": hits / max(1, int(y_true.sum())),
    }




def select_columns(train, drop_views):
    keep = np.ones(train.n_features, dtype=bool)
    for view in drop_views:
        if view not in train.view_groups:
            raise KeyError(f"unknown view {view!r}; available: {list(train.view_groups)}")
        keep[train.view_groups[view]] = False
    idx = np.where(keep)[0]
    return idx, [train.feature_cols[i] for i in idx]


def standardize(Xtr, *others):
    mean = Xtr.mean(axis=0, dtype=np.float64)
    std = Xtr.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return [((X - mean) / std).astype(np.float32) for X in (Xtr, *others)]


COL_IDX, COLS = select_columns(train, DROP_VIEWS)
print(f"{len(COLS)} feature columns"
      + (f"  (dropped views: {', '.join(DROP_VIEWS)})" if DROP_VIEWS else ""))


from sklearn.ensemble import RandomForestClassifier


def fit_rf(Xtr, ytr, Xva, yva):
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(Xtr, ytr)
    assert model.n_features_in_ == len(COLS), "random forest feature count drifted"
    return model, model.predict_proba(Xva)[:, 1], None


TRAINERS = {"rf": fit_rf}


def save_model(model, name, fold):
    if model is None or not SAVE_CHECKPOINTS:
        return None
    models_dir = OUT_DIR / "models"
    if name == "lgbm":
        path = models_dir / f"lgbm_fold{fold}.txt"
        model.save_model(str(path), num_iteration=model.best_iteration)
    elif name == "catboost":
        path = models_dir / f"catboost_fold{fold}.cbm"
        model.save_model(str(path))
    else:
        import joblib
        path = models_dir / f"{name}_fold{fold}.joblib"
        joblib.dump(model, path, compress=3)
    return path.name



import time

rows, oof = [], {m: np.full(len(y), np.nan) for m in MODELS}

for fold in range(N_SPLITS):
    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]
    Xtr = np.asarray(train.X[tr_idx])[:, COL_IDX]
    Xva = np.asarray(train.X[va_idx])[:, COL_IDX]
    ytr, yva = y[tr_idx], y[va_idx]

    for name in MODELS:
        t0 = time.perf_counter()
        model, scores, best_iter = TRAINERS[name](Xtr, ytr, Xva, yva)
        seconds = time.perf_counter() - t0

        oof[name][va_idx] = scores
        metrics = evaluate(yva, scores)
        rows.append({"model": name, "fold": fold, **metrics, "seconds": round(seconds, 1),
                     "best_iteration": best_iter, "n_features": len(COLS),
                     "checkpoint": save_model(model, name, fold)})
        print(f"fold {fold} {name:<9} auc {metrics['auc']:.5f}  ap {metrics['ap']:.5f}  "
              f"p@10 {metrics['p_at_10']:.4f}  r@10 {metrics['r_at_10']:.4f}  {seconds:7.1f}s")

    del Xtr, Xva

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / f"results_per_fold_{RUN_TAG}.csv", index=False)
for name, preds in oof.items():
    np.save(OUT_DIR / f"oof_{name}.npy", preds)

print(results.to_string(index=False))



import json

agg = results.groupby("model")[METRIC_NAMES].agg(["mean", "std"])
table = pd.DataFrame({m: agg[(m, "mean")].map("{:.5f}".format) + " ± "
                      + agg[(m, "std")].map("{:.5f}".format) for m in METRIC_NAMES})
table["total_seconds"] = results.groupby("model")["seconds"].sum().round(0)

manifest_path = Path(FOLDS_FILE).with_suffix(".json")
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

(OUT_DIR / f"run_{RUN_TAG}.json").write_text(json.dumps({
    "run_tag": RUN_TAG, "models": MODELS, "drop_views": DROP_VIEWS, "sample_rows": SAMPLE_ROWS,
    "seed": SEED, "n_features": len(COLS), "params": RUN_PARAMS,
    "folds": {"file": Path(FOLDS_FILE).name, "seed": manifest.get("seed"),
              "ids_sha256": manifest.get("ids_sha256")},
    "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                 "scikit-learn": sklearn.__version__},
    "summary": {m: {k: {"mean": float(agg[(m, "mean")][k]), "std": float(agg[(m, "std")][k])}
                    for k in agg.index} for m in METRIC_NAMES},
}, indent=2))

print(table.to_string())
print("\nsaved ->", OUT_DIR)



parts = sorted(OUT_DIR.glob("results_per_fold_*.csv"))
combined = pd.concat([pd.read_csv(p).assign(run=p.stem.replace("results_per_fold_", ""))
                      for p in parts], ignore_index=True)
combined.to_csv(OUT_DIR / "results_all.csv", index=False)

agg_all = combined.groupby("model")[METRIC_NAMES].agg(["mean", "std"])
table_all = pd.DataFrame({m: agg_all[(m, "mean")].map("{:.5f}".format) + " ± "
                          + agg_all[(m, "std")].map("{:.5f}".format) for m in METRIC_NAMES})
table_all["folds"] = combined.groupby("model")["fold"].count()
table_all["run"] = combined.groupby("model")["run"].first()

print(f"\nmerged {[p.name for p in parts]}")
print(table_all.to_string())
