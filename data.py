from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

import data_loader as dl

N_TRAIN_ROWS = 595_000
N_TEST_ROWS = 255_000
N_RAW_FEATURES = 88
CHURN_TOLERANCE = 0.001
FINITE_CHUNK = 200_000


CHECKS_NAME = f"frozen_folds_{dl.N_SPLITS}.checks.json"


def checks_path(for_writing: bool = False) -> Path:
    """Where the verification record lives.

    Reading must not go through data_loader.folds_dir(): that mkdirs, which fails on
    Kaggle's read-only /kaggle/input.
    """
    repo = Path(__file__).resolve().parent / "folds" / CHECKS_NAME
    if for_writing or repo.exists():
        return repo
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.glob(f"**/{CHECKS_NAME}"))
        if hits:
            return hits[0]
    return repo


def ids_digest(ids) -> str:
    """Mirrors data_loader._ids_digest: SHA-256 over the fixed-width ID array bytes."""
    return hashlib.sha256(np.asarray(ids).astype("<U16").tobytes()).hexdigest()


def text_digest(items) -> str:
    return hashlib.sha256("\n".join(map(str, items)).encode()).hexdigest()


def file_sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class Report:
    checks: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.checks.append({"check": name, "ok": ok, "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} {detail}")
        return ok

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]


def verify_csvs(rep: Report):
    src = dl.data_dir()
    train_csv, test_csv = src / dl.TRAIN_FILE, src / dl.TEST_FILE
    print(f"\n1. Source CSVs  ({src})")

    train_cols = list(pd.read_csv(train_csv, nrows=0).columns)
    test_cols = list(pd.read_csv(test_csv, nrows=0).columns)
    head = [dl.ID_COL, "GENDER", "REGION"]
    features = train_cols[3:-1]

    rep.check("train layout", train_cols[:3] == head and train_cols[-1] == dl.TARGET_COL,
              f"{len(train_cols)} cols = ID + GENDER + REGION + {len(features)} features + CHURN")
    rep.check("test layout", test_cols == train_cols[:-1], f"{len(test_cols)} cols, no CHURN")
    rep.check("feature count", len(features) == len(set(features)) == N_RAW_FEATURES,
              f"{len(features)} unique feature columns")

    wanted = {dl.ID_COL, "GENDER", "REGION", dl.TARGET_COL}
    dtypes = {dl.ID_COL: "string", "GENDER": "string", "REGION": "string"}
    train = pd.read_csv(train_csv, usecols=lambda c: c in wanted, dtype=dtypes)
    test = pd.read_csv(test_csv, usecols=lambda c: c in wanted, dtype=dtypes)

    rep.check("row counts", len(train) == N_TRAIN_ROWS and len(test) == N_TEST_ROWS,
              f"train {len(train):,} | test {len(test):,}")
    rep.check("ids unique", train[dl.ID_COL].is_unique and test[dl.ID_COL].is_unique, "no duplicates")
    overlap = set(train[dl.ID_COL]) & set(test[dl.ID_COL])
    rep.check("train/test disjoint", not overlap, f"{len(overlap)} shared accounts")

    y = train[dl.TARGET_COL].to_numpy(dtype=np.int8)
    rep.check("labels binary", train[dl.TARGET_COL].isin([0, 1]).all(),
              f"churn {y.mean():.6f} ({int(y.sum()):,} of {len(y):,})")

    bad = {}
    for frame, name in ((train, "train"), (test, "test")):
        for col, levels in (("GENDER", dl.GENDER_LEVELS), ("REGION", dl.REGION_LEVELS)):
            unseen = set(frame[col].dropna()) - set(levels)
            if unseen or frame[col].isna().any():
                bad[f"{name}.{col}"] = sorted(unseen) or "has NaN"
    rep.check("categorical levels", not bad,
              "GENDER/REGION within data_loader levels" if not bad else str(bad))

    digests = {name: file_sha256(p) for name, p in (("train", train_csv), ("test", test_csv))}
    print(f"         train sha256 {digests['train'][:16]}...  test sha256 {digests['test'][:16]}...")

    rep.facts["csv"] = {
        "dir": str(src),
        "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
        "n_raw_features": len(features), "churn_rate": float(y.mean()),
        "sha256": digests,
    }
    return train[dl.ID_COL].to_numpy(dtype=str), y


# --------------------------------------------------------------------------
# 2. data_loader matrix
# --------------------------------------------------------------------------

def _all_finite(X, chunk: int = FINITE_CHUNK) -> bool:
    return all(np.isfinite(np.asarray(X[s:s + chunk])).all() for s in range(0, X.shape[0], chunk))


def verify_loader(rep: Report, ids: np.ndarray, y: np.ndarray):
    print("\n2. data_loader matrix")
    dl.build_cache(verbose=False)
    train, test = dl.load_train(), dl.load_test()

    rep.check("row order", np.array_equal(train.ids, ids), "loader rows match CSV order")
    rep.check("labels", np.array_equal(train.y, y), "loader labels match CSV")
    rep.check("FEATURE_COLS", train.feature_cols == test.feature_cols,
              f"{train.n_features} columns, identical for train and test")

    columns = np.concatenate(list(train.view_groups.values()))
    rep.check("view partition", len(columns) == len(np.unique(columns)) == train.n_features,
              f"{len(train.view_groups)} views cover every column exactly once")
    rep.check("no NaN/inf", _all_finite(train.X) and _all_finite(test.X),
              f"train {train.X.shape} | test {test.X.shape}")

    print("\n   per-view feature counts")
    for view, idx in train.view_groups.items():
        print(f"     {view:<12} {len(idx):>3}")

    rep.facts["matrix"] = {
        "schema_version": dl.SCHEMA_VERSION,
        "n_features": int(train.n_features),
        "views": {v: int(len(i)) for v, i in train.view_groups.items()},
        "feature_cols": train.feature_cols,
        "feature_cols_sha256": text_digest(train.feature_cols),
    }
    return train


def verify_folds(rep: Report, train, ids: np.ndarray, y: np.ndarray):
    path = dl.frozen_folds_path()
    manifest_path = path.with_suffix(".json")
    print(f"\n3. Frozen folds  ({path.name})")

    if not path.exists():
        rep.check("fold file exists", False, f"{path} not found -- run data_loader.freeze_folds() once")
        return None
    folds = np.load(path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    n_splits = int(manifest.get("n_splits", dl.N_SPLITS))
    seed = int(manifest.get("seed", dl.FOLD_SEED))

    rep.check("fold file exists", True, f"{len(folds):,} values, dtype {folds.dtype}")
    rep.check("manifest present", bool(manifest), f"seed {seed}, {n_splits} splits")
    rep.check("covers train set", len(folds) == len(y), f"{len(folds):,} rows")
    rep.check("fold values", set(np.unique(folds)) == set(range(n_splits)),
              f"folds {sorted(set(np.unique(folds).tolist()))}")

    summary = (pd.DataFrame({"fold": folds, "churn": y})
               .groupby("fold")["churn"].agg(n_rows="size", churn_rate="mean"))
    rep.check("fold sizes", summary["n_rows"].max() - summary["n_rows"].min() <= 1,
              f"{summary['n_rows'].min():,}-{summary['n_rows'].max():,} rows per fold")
    rep.check("stratified", (summary["churn_rate"] - y.mean()).abs().max() < CHURN_TOLERANCE,
              f"max deviation {float((summary['churn_rate'] - y.mean()).abs().max()):.2e}")

    rep.check("ids fingerprint", ids_digest(ids) == manifest.get("ids_sha256"),
              "fold file was built on this exact row order")

    replay = np.full(len(y), -1, dtype=np.int8)
    for k, (_, val_idx) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                                     random_state=seed).split(np.zeros(len(y)), y)):
        replay[val_idx] = k
    rep.check("recipe reproduces", np.array_equal(replay, folds),
              f"StratifiedKFold({n_splits}, shuffle=True, random_state={seed}) on CSV order")

    rep.check("make_folds agrees", np.array_equal(dl.make_folds(train.y, ids=train.ids), folds),
              "data_loader.make_folds returns this array")

    print("\n   fold      val rows   train rows   val churn   churners")
    for fold, row in summary.iterrows():
        n_val = int(row.n_rows)
        print(f"     {int(fold)}       {n_val:>8,}   {len(y) - n_val:>10,}    {row.churn_rate:.6f}"
              f"   {int(y[folds == fold].sum()):>6,}")

    rep.facts["folds"] = {
        "file": path.name,
        "sha256": file_sha256(path),
        "n_splits": n_splits, "seed": seed,
        "ids_sha256": manifest.get("ids_sha256"),
        "per_fold": [{"fold": int(f), "n_rows": int(r.n_rows), "churn_rate": float(r.churn_rate)}
                     for f, r in summary.iterrows()],
    }
    return folds


# --------------------------------------------------------------------------
# Public API for training scripts
# --------------------------------------------------------------------------

def verified_folds(train=None) -> np.ndarray:
    """Frozen folds for the FULL train split, with the ID check switched on."""
    train = dl.load_train() if train is None else train
    return dl.make_folds(train.y, ids=train.ids)


def fold_indices(folds: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.where(folds != k)[0], np.where(folds == k)[0]


def assert_feature_cols(feature_cols: list[str]) -> None:
    """Fail if data_loader's config drifted since the last verification run."""
    path = checks_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{CHECKS_NAME} not found -- run `python data.py` locally, and include the whole "
            "folds/ directory in the code dataset you upload to Kaggle"
        )
    record = json.loads(path.read_text())
    expected = record["matrix"]["feature_cols"]
    if list(feature_cols) != expected:
        raise AssertionError(
            f"FEATURE_COLS differ from the verified list ({len(expected)} columns) -- a data_loader "
            "config change would make your numbers incomparable to the rest of the team"
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    print("XMV-Net dataset verification")
    ids, y = verify_csvs(rep := Report())
    train = verify_loader(rep, ids, y)
    verify_folds(rep, train, ids, y)

    rep.facts["environment"] = {
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
    }
    rep.facts["all_passed"] = not rep.failures
    rep.facts["checks"] = rep.checks
    record_path = checks_path(for_writing=True)
    record_path.write_text(json.dumps(rep.facts, indent=2))

    print(f"\n{len(rep.checks) - len(rep.failures)}/{len(rep.checks)} checks passed"
          f"  ->  {record_path}")
    if rep.failures:
        for c in rep.failures:
            print(f"  FAILED: {c['check']} -- {c['detail']}")
        return 1

    print("\nUse the frozen split in every training script:\n"
          "    import data_loader as dl, data\n"
          "    train = dl.load_train()\n"
          "    folds = data.verified_folds(train)\n"
          "    for k in range(5):\n"
          "        tr_idx, va_idx = data.fold_indices(folds, k)\n"
          "        mean, std = dl.fit_standardizer(train.X, tr_idx)   # train rows only\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
