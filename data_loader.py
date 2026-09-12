"""Memory-bounded loading + preprocessing for the FictiPay top88 churn CSVs.

Companion documentation: docs/data_loader.md
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TRAIN_FILE = "account_features_train_top88.csv"
TEST_FILE = "account_features_test_top88.csv"

ID_COL = "ACCOUNT_ID"
TARGET_COL = "CHURN"

GENDER_LEVELS = ["Female", "Male", "Other"]
REGION_LEVELS = [
    "Barishal", "Chattogram", "Dhaka", "Khulna",
    "Mymensingh", "Rajshahi", "Rangpur", "Sylhet",
]

N_SPLITS = 5
FOLD_SEED = 42

DROP_CONSTANT_COLS = True
ADD_NAN_FLAGS = True
APPLY_LOG1P = True
APPLY_CLIP = True
ONEHOT_DEC_CLUSTER = False

SKEW_THRESHOLD = 2.0
CLIP_QUANTILES = (0.001, 0.999)
PROBE_ROWS = 200_000
CHUNK_ROWS = 200_000

SCHEMA_VERSION = 3

# --------------------------------------------------------------------------
# Semantic view partition
# --------------------------------------------------------------------------

VIEW_RULES: dict[str, list[str]] = {
    "recency": [
        "recency_days", "region_recency_deviation", "recency_pct_of_tenure",
        "mean_gap_days", "zero_days_7d", "zero_days_14d",
        "rec_le7", "rec_8_29", "rec_ge30", "span_days",
    ],
    "frequency": [
        "n_trx_7d", "n_trx_14d", "n_trx_30d", "n_trx_60d", "n_trx_90d", "n_trx_total",
        "out_trx_cnt_7d", "out_trx_cnt_30d", "out_trx_cnt_90d",
        "active_days_7d", "active_days_14d", "active_days_30d",
        "active_days_60d", "active_days_90d",
        "cnt_m1", "cnt_m2", "cnt_m3",
        "vel_ratio_7_30", "vel_ratio_14_60", "vel_ratio_30_90",
        "freq_recent_share_7d", "freq_recent_share_30d",
        "freq_momentum_7d_vs_30d", "freq_momentum_30d_vs_90d",
        "activity_drop_ratio", "momentum_ratio", "avg_trx_per_active_day_90d",
        "n_limit_7d", "n_limit_14d", "n_limit_30d",
    ],
    "monetary": [
        "total_amt_7d", "total_amt_30d", "total_amt_60d", "total_amt_90d",
        "avg_amt_7d", "out_amt_std_30d",
        "salary_amt_7d", "salary_amt_14d",
        "salary_amt_ratio_7d", "salary_amt_ratio_14d",
        "salary_day_ratio_7d", "salary_day_ratio_14d",
        "n_salary_7d", "n_salary_14d", "n_salary_30d",
    ],
    "network": [
        "uniq_cp_30d", "recency_p2p", "recency_billpay",
    ],
    "calendar": [
        "n_weekend_7d", "n_weekend_14d", "n_weekend_90d",
        "weekend_ratio_14d", "weekend_ratio_30d", "n_sunday_7d",
        "active_days_mar", "active_days_mar_1_15", "active_days_mar_16_31",
        "n_trx_mar_16_31", "total_amt_mar_1_15", "total_amt_mar_16_31",
        "late_march_share", "late_march_active_share", "march_second_half_drop",
    ],
    "balance": [
        "last_balance", "mean_bal_7d", "max_bal_7d", "min_bal_7d",
        "std_bal_7d", "std_bal_14d", "balance_std_7d",
        "balance_first_90d", "last_bal_to_mean",
    ],
    "dec_segment": [
        "dec_cluster", "dec_latent_3", "dec_latent_4",
        "dec_latent_5", "dec_latent_9", "dec_prob_4",
    ],
    "demographic": [
        "GENDER", "REGION",
    ],
}

VIEW_NAMES = list(VIEW_RULES.keys())
CATEGORICAL_COLS = ["GENDER", "REGION"]
RAW_NUMERIC_COLS = [c for v, cols in VIEW_RULES.items() for c in cols if c not in CATEGORICAL_COLS]
_COL_TO_VIEW = {c: v for v, cols in VIEW_RULES.items() for c in cols}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def data_dir() -> Path:
    env = os.environ.get("CHURN_DATA_DIR")
    if env:
        return Path(env)
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.glob(f"**/{TRAIN_FILE}"))
        if hits:
            return hits[0].parent
    return Path(__file__).resolve().parent / "Dataset"


def folds_dir() -> Path:
    """Committed, unlike cache_dir() -- the frozen split is a shared team artifact."""
    d = Path(__file__).resolve().parent / "folds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def frozen_folds_path(n_splits: int = N_SPLITS) -> Path:
    return folds_dir() / f"frozen_folds_{n_splits}.npy"


def cache_dir() -> Path:
    env = os.environ.get("CHURN_CACHE_DIR")
    root = Path(env) if env else (
        Path("/kaggle/working/churn_cache") if Path("/kaggle/working").exists()
        else Path(__file__).resolve().parent / "Dataset" / "_cache"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------

@dataclass
class ChurnData:
    X: np.ndarray
    y: np.ndarray | None
    ids: np.ndarray
    feature_cols: list[str]
    view_groups: dict[str, np.ndarray]

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def view(self, name: str) -> np.ndarray:
        return self.X[:, self.view_groups[name]]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(v, len(idx)) for v, idx in self.view_groups.items()],
            columns=["view", "n_features"],
        )


# --------------------------------------------------------------------------
# Pass 1 -- schema discovery (train only)
# --------------------------------------------------------------------------

def _read_dtypes(header: list[str]) -> dict:
    dt = {c: "float32" for c in header}
    dt[ID_COL] = "string"
    for c in CATEGORICAL_COLS:
        dt[c] = "string"
    if TARGET_COL in header:
        dt[TARGET_COL] = "int8"
    return dt


def _csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _discover_schema(train_path: Path) -> dict:
    header = _csv_header(train_path)
    missing = set(RAW_NUMERIC_COLS + CATEGORICAL_COLS) - set(header)
    if missing:
        raise ValueError(f"VIEW_RULES references columns absent from the CSV: {sorted(missing)}")
    unassigned = set(header) - set(RAW_NUMERIC_COLS) - set(CATEGORICAL_COLS) - {ID_COL, TARGET_COL}
    if unassigned:
        raise ValueError(f"CSV columns not assigned to any view: {sorted(unassigned)}")

    dtypes = _read_dtypes(header)
    n_rows = 0
    nan_counts = np.zeros(len(RAW_NUMERIC_COLS), dtype=np.int64)
    col_min = np.full(len(RAW_NUMERIC_COLS), np.inf, dtype=np.float64)
    col_max = np.full(len(RAW_NUMERIC_COLS), -np.inf, dtype=np.float64)
    probe = None

    reader = pd.read_csv(train_path, chunksize=CHUNK_ROWS, dtype=dtypes,
                         usecols=RAW_NUMERIC_COLS)
    for chunk in reader:
        chunk = chunk[RAW_NUMERIC_COLS]
        n_rows += len(chunk)
        nan_counts += chunk.isna().sum().to_numpy()
        col_min = np.fmin(col_min, chunk.min(numeric_only=True).to_numpy(dtype=np.float64))
        col_max = np.fmax(col_max, chunk.max(numeric_only=True).to_numpy(dtype=np.float64))
        if probe is None:
            probe = chunk.head(PROBE_ROWS).copy()

    filled = probe.fillna(0.0)
    mu = filled.mean().to_numpy(dtype=np.float64)
    sd = filled.std(ddof=0).to_numpy(dtype=np.float64)
    sd_safe = np.where(sd > 0, sd, 1.0)
    skew = (((filled.to_numpy(dtype=np.float64) - mu) / sd_safe) ** 3).mean(axis=0)

    q_lo = filled.quantile(CLIP_QUANTILES[0]).to_numpy(dtype=np.float64)
    q_hi = filled.quantile(CLIP_QUANTILES[1]).to_numpy(dtype=np.float64)

    constant = (col_max - col_min) == 0
    keep = ~constant if DROP_CONSTANT_COLS else np.ones_like(constant)

    kept_cols, nan_flag_cols, log1p_cols = [], [], []
    clip_lo, clip_hi = {}, {}
    for i, col in enumerate(RAW_NUMERIC_COLS):
        if not keep[i]:
            continue
        kept_cols.append(col)
        if ADD_NAN_FLAGS and nan_counts[i] > 0:
            nan_flag_cols.append(col)
        if APPLY_LOG1P and col_min[i] >= 0 and skew[i] > SKEW_THRESHOLD:
            log1p_cols.append(col)
        if APPLY_CLIP:
            clip_lo[col], clip_hi[col] = float(q_lo[i]), float(q_hi[i])

    feature_cols, feature_view = _build_feature_cols(kept_cols, nan_flag_cols)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_train_rows": int(n_rows),
        "raw_numeric_cols": RAW_NUMERIC_COLS,
        "kept_cols": kept_cols,
        "dropped_constant_cols": [c for i, c in enumerate(RAW_NUMERIC_COLS) if not keep[i]],
        "nan_flag_cols": nan_flag_cols,
        "log1p_cols": log1p_cols,
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "onehot_dec_cluster": ONEHOT_DEC_CLUSTER,
        "feature_cols": feature_cols,
        "feature_view": feature_view,
        "nan_counts": {c: int(nan_counts[i]) for i, c in enumerate(RAW_NUMERIC_COLS)},
    }


def _build_feature_cols(kept_cols: list[str], nan_flag_cols: list[str]):
    """Final column order, grouped by view so each view is a contiguous slice."""
    kept = set(kept_cols)
    flags = set(nan_flag_cols)
    feature_cols, feature_view = [], []

    def emit(name, view):
        feature_cols.append(name)
        feature_view.append(view)

    for view in VIEW_NAMES:
        for col in VIEW_RULES[view]:
            if col == "GENDER":
                for lvl in GENDER_LEVELS:
                    emit(f"GENDER={lvl}", view)
            elif col == "REGION":
                for lvl in REGION_LEVELS:
                    emit(f"REGION={lvl}", view)
            elif col == "dec_cluster" and ONEHOT_DEC_CLUSTER:
                for k in range(5):
                    emit(f"dec_cluster={k}", view)
            elif col in kept:
                emit(col, view)
            if col in flags:
                emit(f"{col}__isna", view)
    return feature_cols, feature_view


# --------------------------------------------------------------------------
# Pass 2 -- chunked transform into a memmapped .npy
# --------------------------------------------------------------------------

def _transform_chunk(chunk: pd.DataFrame, schema: dict) -> np.ndarray:
    parts: dict[str, np.ndarray] = {}
    kept = schema["kept_cols"]
    log1p_set = set(schema["log1p_cols"])
    flags = set(schema["nan_flag_cols"])

    for col in flags:
        parts[f"{col}__isna"] = chunk[col].isna().to_numpy(dtype=np.float32)

    num = chunk[kept].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(num, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    for j, col in enumerate(kept):
        if APPLY_CLIP and col in schema["clip_lo"]:
            np.clip(num[:, j], schema["clip_lo"][col], schema["clip_hi"][col], out=num[:, j])
        if col in log1p_set:
            num[:, j] = np.log1p(num[:, j])
        parts[col] = num[:, j]

    gender = chunk["GENDER"].fillna("__NA__").to_numpy()
    region = chunk["REGION"].fillna("__NA__").to_numpy()
    for lvl in GENDER_LEVELS:
        parts[f"GENDER={lvl}"] = (gender == lvl).astype(np.float32)
    for lvl in REGION_LEVELS:
        parts[f"REGION={lvl}"] = (region == lvl).astype(np.float32)
    if schema["onehot_dec_cluster"]:
        dc = chunk["dec_cluster"].fillna(-1).to_numpy()
        for k in range(5):
            parts[f"dec_cluster={k}"] = (dc == k).astype(np.float32)

    return np.column_stack([parts[c] for c in schema["feature_cols"]]).astype(np.float32)


def _count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        while block := fh.read(8 << 20):
            n += block.count(b"\n")
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 1))
        if fh.read(1) not in (b"\n", b""):
            n += 1
    return n - 1


def _build_cache(csv_path: Path, split: str, schema: dict, out_dir: Path) -> None:
    header = _csv_header(csv_path)
    dtypes = _read_dtypes(header)
    usecols = [ID_COL] + schema["kept_cols"] + CATEGORICAL_COLS
    usecols += [c for c in schema["nan_flag_cols"] if c not in usecols]
    if "dec_cluster" not in usecols and schema["onehot_dec_cluster"]:
        usecols.append("dec_cluster")
    has_target = TARGET_COL in header
    if has_target:
        usecols.append(TARGET_COL)

    n_rows = _count_rows(csv_path)
    n_feat = len(schema["feature_cols"])

    X = np.lib.format.open_memmap(
        out_dir / f"{split}_X.npy", mode="w+", dtype=np.float32, shape=(n_rows, n_feat)
    )
    ids = np.empty(n_rows, dtype="<U16")
    y = np.empty(n_rows, dtype=np.int8) if has_target else None

    t0, pos = time.time(), 0
    reader = pd.read_csv(csv_path, chunksize=CHUNK_ROWS, dtype=dtypes, usecols=usecols)
    for chunk in reader:
        k = len(chunk)
        X[pos:pos + k] = _transform_chunk(chunk, schema)
        ids[pos:pos + k] = chunk[ID_COL].to_numpy()
        if has_target:
            y[pos:pos + k] = chunk[TARGET_COL].to_numpy()
        pos += k
        print(f"  {split}: {pos:,}/{n_rows:,} rows", end="\r", flush=True)

    if pos != n_rows:
        raise RuntimeError(f"row count mismatch for {split}: counted {n_rows}, wrote {pos}")

    X.flush()
    del X
    np.save(out_dir / f"{split}_ids.npy", ids)
    if has_target:
        np.save(out_dir / f"{split}_y.npy", y)
    print(f"  {split}: {n_rows:,} rows x {n_feat} features in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def build_cache(force: bool = False, verbose: bool = True) -> dict:
    """Discover the schema from train, then write train/test caches. Idempotent."""
    out = cache_dir()
    meta_path = out / "schema.json"

    if meta_path.exists() and not force:
        schema = json.loads(meta_path.read_text())
        expected = [out / f"{s}_{p}.npy" for s in ("train", "test") for p in ("X", "ids")]
        if schema.get("schema_version") == SCHEMA_VERSION and all(p.exists() for p in expected):
            if verbose:
                print(f"cache hit: {out}")
            return schema

    src = data_dir()
    train_path, test_path = src / TRAIN_FILE, src / TEST_FILE
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (set CHURN_DATA_DIR to override)")

    if verbose:
        print(f"source: {src}\ncache:  {out}\npass 1/2  schema discovery...")
    schema = _discover_schema(train_path)

    if verbose:
        print(f"  dropped {len(schema['dropped_constant_cols'])} constant cols: "
              f"{schema['dropped_constant_cols']}")
        print(f"  {len(schema['nan_flag_cols'])} nan-flag cols, "
              f"{len(schema['log1p_cols'])} log1p cols")
        print(f"  {len(schema['feature_cols'])} final features")
        print("pass 2/2  writing caches...")

    _build_cache(train_path, "train", schema, out)
    _build_cache(test_path, "test", schema, out)
    meta_path.write_text(json.dumps(schema, indent=2))
    return schema


def _view_groups(schema: dict) -> dict[str, np.ndarray]:
    fv = np.array(schema["feature_view"])
    return {v: np.where(fv == v)[0] for v in VIEW_NAMES}


def _load_split(split: str, n_rows: int | None, mmap: bool) -> ChurnData:
    schema = build_cache(verbose=False)
    out = cache_dir()

    X = np.load(out / f"{split}_X.npy", mmap_mode="r" if mmap else None)
    ids = np.load(out / f"{split}_ids.npy", allow_pickle=False)
    y_path = out / f"{split}_y.npy"
    y = np.load(y_path) if y_path.exists() else None

    if n_rows is not None and n_rows < X.shape[0]:
        X, ids = np.ascontiguousarray(X[:n_rows]), ids[:n_rows]
        y = y[:n_rows] if y is not None else None

    feature_cols = schema["feature_cols"]
    if X.shape[1] != len(feature_cols):
        raise AssertionError(f"width {X.shape[1]} != {len(feature_cols)} feature_cols")
    if len(schema["feature_view"]) != len(feature_cols):
        raise AssertionError("feature_view / feature_cols length mismatch")

    return ChurnData(X, y, ids, feature_cols, _view_groups(schema))


def load_train(n_rows: int | None = None, mmap: bool = True) -> ChurnData:
    return _load_split("train", n_rows, mmap)


def load_test(n_rows: int | None = None, mmap: bool = True) -> ChurnData:
    return _load_split("test", n_rows, mmap)


def _ids_digest(ids: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.asarray(ids).astype("<U16").tobytes()).hexdigest()


def freeze_folds(n_splits: int = N_SPLITS, seed: int = FOLD_SEED, force: bool = False) -> np.ndarray:
    """Generate the shared 5-fold split over the FULL train set and commit it to disk.

    Run once, commit folds/, never run again. Regenerating with a different seed
    silently invalidates every number the team has already reported.
    """
    import json as _json

    path = frozen_folds_path(n_splits)
    manifest_path = path.with_suffix(".json")
    if path.exists() and not force:
        print(f"frozen folds already exist: {path}  (pass force=True to regenerate)")
        return np.load(path)

    from sklearn.model_selection import StratifiedKFold

    train = load_train()
    y, ids = train.y, train.ids

    folds = np.empty(len(y), dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = k

    np.save(path, folds)
    manifest = {
        "n_rows": int(len(y)),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "stratified_on": TARGET_COL,
        "sklearn_splitter": "StratifiedKFold(shuffle=True)",
        "row_order": f"as-read from {TRAIN_FILE}",
        "ids_sha256": _ids_digest(ids),
        "first_id": str(ids[0]),
        "last_id": str(ids[-1]),
        "overall_churn": float(y.mean()),
        "fold_sizes": np.bincount(folds, minlength=n_splits).tolist(),
        "fold_churn": [float(y[folds == k].mean()) for k in range(n_splits)],
    }
    manifest_path.write_text(_json.dumps(manifest, indent=2))

    print(f"FROZE {n_splits}-fold split -> {path}")
    print(f"  {len(y):,} rows | seed {seed} | overall churn {y.mean():.6f}")
    for k in range(n_splits):
        m = folds == k
        print(f"  fold {k}: n={m.sum():,}  churn={y[m].mean():.6f}")
    print(f"  ids sha256 {manifest['ids_sha256'][:16]}...")
    print("  -> commit folds/ so the whole team shares this split")
    return folds


def make_folds(y: np.ndarray, n_splits: int = N_SPLITS, seed: int = FOLD_SEED,
               frozen_path: str | Path | None = None, ids: np.ndarray | None = None,
               allow_placeholder: bool = False) -> np.ndarray:
    """Per-row validation-fold assignment, from the frozen split by default.

    Verifies the fold file against the row identity it was built on, so reordered
    or regenerated data cannot silently mis-align folds. Pass `ids` to enable it.
    """
    import json as _json

    path = Path(frozen_path) if frozen_path is not None else frozen_folds_path(n_splits)

    if not path.exists():
        if not allow_placeholder:
            raise FileNotFoundError(
                f"frozen fold file not found: {path}\n"
                f"Run  python -c \"import data_loader; data_loader.freeze_folds()\"  once, "
                f"then commit folds/. Pass allow_placeholder=True only for throwaway experiments."
            )
        from sklearn.model_selection import StratifiedKFold
        print(f"WARNING: PLACEHOLDER folds (seed={seed}) -- NOT comparable across the team")
        folds = np.empty(len(y), dtype=np.int8)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            folds[val_idx] = k
        return folds

    folds = np.load(path)
    manifest_path = path.with_suffix(".json")
    manifest = _json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if len(folds) < len(y):
        raise ValueError(f"frozen folds cover {len(folds):,} rows but y has {len(y):,}")

    sampled = len(y) < len(folds)
    if sampled:
        folds = folds[:len(y)]

    if ids is not None:
        ids = np.asarray(ids)
        if not sampled and "ids_sha256" in manifest:
            if _ids_digest(ids) != manifest["ids_sha256"]:
                raise AssertionError(
                    "ACCOUNT_ID order does not match the frozen split -- folds would be "
                    "mis-assigned. Rebuild the cache, or re-freeze if the data really changed."
                )
        elif manifest.get("first_id") and str(ids[0]) != manifest["first_id"]:
            raise AssertionError(
                f"row 0 is {ids[0]}, frozen split was built on {manifest['first_id']}"
            )

    tag = f"prefix sample of {len(y):,} rows" if sampled else f"all {len(y):,} rows"
    print(f"FROZEN folds: {path.name} | seed {manifest.get('seed', '?')} | {tag}"
          + (" | ids verified" if ids is not None and not sampled else ""))
    if sampled:
        print("  NOTE: sample mode -- fold sizes below are a prefix, not the frozen fold sizes")
    return folds


def fit_standardizer(X: np.ndarray, rows: np.ndarray | None = None, chunk: int = 200_000):
    """Mean/std over `rows` only -- call with the TRAIN fold indices, never all rows."""
    idx = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    n, s1 = 0, np.zeros(X.shape[1], dtype=np.float64)
    s2 = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(idx), chunk):
        block = np.asarray(X[idx[start:start + chunk]], dtype=np.float64)
        n += len(block)
        s1 += block.sum(axis=0)
        s2 += (block ** 2).sum(axis=0)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float32) - mean) / std).astype(np.float32)


if __name__ == "__main__":
    s = build_cache(force=True)
    tr, te = load_train(), load_test()
    print(tr.summary().to_string(index=False))
    print(f"train {tr.X.shape}  churn={tr.y.mean():.5f}   test {te.X.shape}")
