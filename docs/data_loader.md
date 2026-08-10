# `data_loader.py` — companion documentation

Loading and preprocessing for the FictiPay `top88` churn CSVs. Written so that the 410 MB train file and
175 MB test file are parsed **once**, into a format that later loads in ~20 ms.

---

## 1. Why it is built this way

Naively, `pd.read_csv(train)` reads every numeric column as `float64` and builds a `DataFrame` with a Python
object column for `ACCOUNT_ID`. That is ~450 MB resident for a file we re-read on every kernel restart.

This module instead does a **two-pass build, then caches**:

| | What happens | Cost |
|---|---|---|
| Pass 1 | Stream the train CSV, discover the schema (row count, per-column NaN counts, constant columns, clip quantiles, skew) | ~7 s, O(1) memory |
| Pass 2 | Stream both CSVs again, writing the transformed matrix straight into a memmapped `.npy` on disk | ~11 s, O(chunk) memory |
| Every later load | `np.load(..., mmap_mode="r")` | ~0.02 s, O(1) memory |

The memmap matters beyond speed: `X` behaves like an array but pages in from disk on demand, so a 595,000 × 127
`float32` matrix (288 MB) never has to be resident all at once. When you slice it (`X[idx]`) NumPy materialises
only those rows.

**Why not parquet?** `pyarrow` is not installed in this environment, and `.npy` is the natural format anyway —
the model consumes a dense float matrix, not a table.

---

## 2. Configuration constants

| Constant | Default | Meaning |
|---|---|---|
| `DROP_CONSTANT_COLS` | `True` | Drop columns with zero variance across the whole train file. Eight qualify (all in the *monetary* view). A constant column's gate weight has no gradient signal, so it would sit at its init value and be counted as a "used feature" by the `§5` sparsity metrics. |
| `ADD_NAN_FLAGS` | `True` | For each column with any NaN in train, emit a `<col>__isna` binary column into the **same view**. |
| `APPLY_LOG1P` | `True` | `log1p` any non-negative column whose sample skew exceeds `SKEW_THRESHOLD`. 27 columns qualify. |
| `APPLY_CLIP` | `True` | Winsorise to the train `[0.1%, 99.9%]` quantiles before `log1p`. |
| `ONEHOT_DEC_CLUSTER` | `False` | ⚠️ **Open decision.** `dec_cluster ∈ {0..4}` is a *nominal* cluster ID — the ordering is meaningless, so a linear gate on it is arguably nonsense. Default `False` keeps the feature exactly as supplied; flip to `True` to one-hot it. Worth an ablation row. |
| `SKEW_THRESHOLD` | `2.0` | Standard "substantially skewed" cutoff. |
| `CHUNK_ROWS` | `200_000` | Rows per streaming chunk. Lower it if memory is tight. |
| `SCHEMA_VERSION` | `3` | Bump this whenever you change any option above — it invalidates the cache automatically. **If you edit a config constant and forget to bump, you will silently keep using the old cache.** |

---

## 3. The view partition

`VIEW_RULES` maps each of the 88 raw numeric columns plus `GENDER`/`REGION` to exactly one of 8 views.
It is the single source of truth; the model never hardcodes indices.

| View | Raw | Final | Note |
|---|---|---|---|
| recency | 10 | 11 | The near-definitional cluster. Ablation A7 drops this whole view. |
| frequency | 30 | **50** | Widest view by far — 39% of all features. |
| monetary | 15 | 13 | 8 of its 15 raw columns are constant-zero and get dropped, leaving 7 real features. |
| network | 3 | **4** | Thinnest view. `top88` simply kept very few network features. |
| calendar | 15 | 21 | |
| balance | 9 | 11 | |
| dec_segment | 6 | 6 | |
| demographic | 2 | 11 | `GENDER` (3 levels) + `REGION` (8 levels), one-hot. |
| **Total** | **90** | **127** | |

"Final" counts include `__isna` flags and one-hot expansion.

> **The proposal says 7 views.** `GENDER`/`REGION` fit none of the RFM-LIR domains, so they form an 8th
> *demographic* view. `§3` of the report needs updating to match.

> **The 50-vs-4 width gap has a modelling consequence.** An L1 gate penalty written as
> `Σ_v Σ_j m_j^(v)` penalises *frequency* ~12× harder than *network* at equal average gate value, so
> ablation A5 would be measuring view width rather than the cost of sparsity. Prefer a width-normalised
> (`mean`) reduction, or weight per view.

### Adding or moving a feature

Edit `VIEW_RULES`, bump `SCHEMA_VERSION`, re-run. `_discover_schema` raises if a rule names a column that is
not in the CSV, **or** if a CSV column is not assigned to any view — the partition cannot silently drift out of sync.

---

## 4. Function reference

### Paths

**`data_dir()`** — where the CSVs live. Resolution order: `$CHURN_DATA_DIR` → a recursive glob under
`/kaggle/input` for `account_features_train_top88.csv` → `./Dataset`. The Kaggle glob exists because the
dataset slug in `/kaggle/input/<slug>/` differs per upload and hardcoding it breaks on re-upload.

**`cache_dir()`** — where the `.npy` caches go. `$CHURN_CACHE_DIR` → `/kaggle/working/churn_cache` →
`./Dataset/_cache`. On Kaggle this **must** differ from `data_dir()`, because `/kaggle/input` is mounted read-only.

### Schema discovery

**`_csv_header(path)`** — `pd.read_csv(path, nrows=0)` reads only the header line and returns an empty frame.
Cheapest way to get column names.

**`_read_dtypes(header)`** — builds the `dtype=` dict passed to `read_csv`: `float32` for everything numeric,
`string` for the ID and the two categoricals, `int8` for the target. Declaring dtypes up front halves peak memory
versus letting pandas infer `float64` and then downcasting, because inference materialises the wide version first.

**`_discover_schema(train_path)`** — the pass-1 workhorse.

- First validates the partition in both directions (rules → CSV, CSV → rules) and raises on any mismatch.
- Streams the train file accumulating exact `n_rows`, per-column NaN counts, and per-column min/max.
- `np.fmin` / `np.fmax` (not `np.minimum` / `np.maximum`) — the `f` variants **ignore NaN** rather than
  propagating it. With 44 NaN-bearing columns, `np.minimum` would return NaN for all of them and constant
  detection would fail silently.
- Keeps the first 200,000 rows as `probe` for statistics that need the distribution, not just extremes:
  - **skew** — computed manually as `mean(((x - μ)/σ)³)` rather than `pd.skew()`, which applies a small-sample
    bias correction that is meaningless at n=200k and costs an extra pass.
  - **clip quantiles** — the 0.1% / 99.9% points.
- `constant = (col_max - col_min) == 0` over the **full** file, not the probe — a column that happens to be
  constant in the first 200k rows but varies later must not be dropped.

**`_build_feature_cols(kept_cols, nan_flag_cols)`** — produces the final ordered column list, **grouped by view**,
so each view is a contiguous index range. Not required (the model looks up indices by name), but contiguous
slices make `X[:, view_idx]` a cheap strided view instead of a gather.

### Transform and cache

**`_transform_chunk(chunk, schema)`** — applies, in this order:

1. `__isna` flags captured **first**, before filling — capture them after the fill and they are all zero.
2. `nan_to_num(..., nan=0.0)` — fill NaN with 0. Semantically correct here: NaN means "no transactions in the
   window", so the true value *is* zero. The flag preserves the "was dormant" signal that plain zero-filling loses.
3. Clip to the train quantiles. Computed on **train only** and reused for test — recomputing on test would be
   test-set leakage.
4. `log1p` on the selected columns. `log1p(x)` is `log(1+x)` computed accurately for small `x`, and is defined
   at `x=0` — which matters because zero-filling just created a lot of zeros.
5. One-hot `GENDER`/`REGION` against **fixed level lists**, not `pd.get_dummies`. `get_dummies` derives columns
   from the values present, so a chunk missing a rare region would produce a different column set and the chunks
   would not stack. Fixed lists guarantee identical width everywhere. An unseen level yields an all-zero row
   rather than a crash.
6. `np.column_stack` in `schema["feature_cols"]` order — the ordering contract.

**`_count_rows(path)`** — counts `\n` in 8 MB binary blocks, then adds one if the file lacks a trailing newline,
then subtracts the header. Needed *before* writing because `open_memmap` must be given the final shape. Reading
raw bytes is ~50× faster than parsing rows just to count them.

**`_build_cache(csv_path, split, schema, out_dir)`** — pass 2.

- `np.lib.format.open_memmap(..., mode="w+", shape=(n_rows, n_feat))` creates a properly-headed `.npy` on disk
  and returns a writable memmap. Chunks are written into row slices, so peak memory is one chunk (~100 MB),
  independent of file size. This is what makes the loader scale to the 2 M-row `account_features_top88.csv`.
- Asserts `pos == n_rows` at the end — catches any disagreement between the byte-level line count and what
  pandas actually parsed (embedded newlines, malformed rows).
- `X.flush(); del X` before saving the sidecars, to force the OS to commit dirty pages.
- IDs are stored as `<U16` fixed-width Unicode. `CUST00000001` is 12 chars, so 16 is safe with headroom, and a
  fixed-width array avoids `allow_pickle=True` on load (which is both slower and a security footgun).

### Public API

**`build_cache(force=False, verbose=True)`** — idempotent. Returns the schema dict. Re-builds only if `force=True`,
`schema.json` is missing, `schema_version` no longer matches, or any expected `.npy` is absent.

**`load_train(n_rows=None, mmap=True)`** / **`load_test(...)`** — return a `ChurnData`. Calls `build_cache` first,
so the first call in a fresh clone just works. `n_rows` truncates for fast iteration; when it is set, the slice is
`ascontiguousarray`'d so the result is a real in-memory array rather than a memmap view.

`load_test().y` is `None` — **the test set has no labels.** Every metric in `§8` comes from cross-validation on
train. There is no way to compute a test-set score locally.

**`ChurnData`** — a dataclass holding `X`, `y`, `ids`, `feature_cols`, `view_groups`.

- `.view(name)` → the column block for one view.
- `.summary()` → the per-view feature-count table.
- `view_groups` maps view name → `np.ndarray` of column indices into `X`.

**`make_folds(y, n_splits=5, seed=42, frozen_path=None)`**

Returns an `int8` array of length `n` giving each row's **validation** fold.

- With `frozen_path` pointing at an existing file, loads it and length-checks it.
- Otherwise generates a seeded `StratifiedKFold` and prints a loud warning.

> 🔁 **This is the swap point for the team's frozen split.** Until `frozen_path` is set, every number produced
> is on placeholder folds and is **not comparable** to Siam's or Tausif's results. Per the shared conventions,
> nobody re-splits the data themselves.

**`fit_standardizer(X, rows=None, chunk=200_000)`**

Returns `(mean, std)` computed **only over `rows`** — pass the training-fold indices. Uses streaming sums of
`x` and `x²` in `float64`, so it never materialises the full matrix and does not lose precision on the wide
columns. `std` below `1e-6` is clamped to `1.0` to avoid dividing by ~0 on near-constant columns (the `__isna`
flags for rare-NaN columns are close to this).

Fitting on *all* rows would leak the validation distribution into training. Mild, but free to avoid — and this
project is partly about leakage discipline.

**`apply_standardizer(X, mean, std)`** — `(X - mean) / std` in `float32`.

---

## 5. Known data facts worth remembering

| Fact | Consequence |
|---|---|
| Train 595,000 rows, 12.68% churn | Mild imbalance — plain BCE is a legitimate default; no reweighting needed for a rank metric like AUC. |
| Test 255,000 rows, no label | All reported metrics are CV-on-train. |
| Train IDs 1,3,5…; test IDs 2,4,6… | The split is **account-level random, not temporal**. Both sets cover Jan–Mar behaviour; the label is Apr churn. |
| `rec_ge30 == 1` → 94.2% churn (vs 5.4%) | The recency view is close to definitional. Every model will score high for this reason. **A7 is the honest headline.** |
| `recency_days` r = 0.79, `span_days` r = −0.78 with label | Same. |
| 8 constant-zero columns, all monetary | The monetary view has only 7 real features. |
| 44 columns with structural NaN, worst 18.9% | Handled by zero-fill + `__isna` flags. |

---

## 6. Usage

```python
import data_loader as dl

dl.build_cache()                        # ~11 s once, then a no-op

train = dl.load_train(n_rows=50_000)    # None for the full 595k
test  = dl.load_test()

folds = dl.make_folds(train.y, frozen_path=None)   # set this when the frozen split lands
tr_idx = np.where(folds != 0)[0]

mean, std = dl.fit_standardizer(train.X, tr_idx)   # train rows only
Xtr = dl.apply_standardizer(np.asarray(train.X[tr_idx]), mean, std)
```

Rebuild from scratch (after changing a config constant — and remember to bump `SCHEMA_VERSION`):

```bash
python data_loader.py
```

Point at a different data location:

```bash
CHURN_DATA_DIR=/kaggle/input/fictipay-top88 python data_loader.py
```
