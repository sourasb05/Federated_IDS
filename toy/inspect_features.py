# inspect_features.py
#
# Scans all CSV files under attack_data/ and reports, for each feature column:
#   - categorical  : integer-valued with few unique values (n_unique <= CAT_THRESHOLD)
#   - continuous   : all other numeric columns
#
# Output: summary table printed to console + saved as inspect_features.csv

import os
import sys
import pandas as pd
import numpy as np

# ── config ────────────────────────────────────────────────────
CAT_THRESHOLD  = 20      # n_unique <= this → treated as categorical
MAX_FILES      = 3       # CSVs to sample per domain (faster scan)
ATTACK_DATA    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'attack_data'
)
DROP_COLS      = ['Unnamed: 0', 'label']
# ──────────────────────────────────────────────────────────────


def load_sample(attack_data_root: str, max_files_per_domain: int) -> pd.DataFrame:
    """Read up to max_files_per_domain CSVs from every domain and concat."""
    frames = []
    for domain in sorted(os.listdir(attack_data_root)):
        domain_path = os.path.join(attack_data_root, domain)
        if not os.path.isdir(domain_path):
            continue
        csv_files = sorted(
            f for f in os.listdir(domain_path) if f.endswith('.csv')
        )[:max_files_per_domain]
        for fname in csv_files:
            fpath = os.path.join(domain_path, fname)
            try:
                df = pd.read_csv(fpath, encoding='utf-8',
                                 encoding_errors='ignore')
                frames.append(df)
            except Exception as e:
                print(f"  [skip] {fpath}: {e}")
    if not frames:
        sys.exit("No CSV files found under attack_data/")
    return pd.concat(frames, ignore_index=True)


def classify_features(df: pd.DataFrame,
                       cat_threshold: int,
                       drop_cols: list[str]) -> pd.DataFrame:
    """
    For each feature column compute:
        n_unique   : number of distinct values
        dtype      : pandas dtype
        kind       : 'categorical' | 'continuous'
        sample_vals: a few example values
    """
    feat_cols = [c for c in df.columns if c not in drop_cols]

    rows = []
    for col in feat_cols:
        series   = df[col].dropna()
        n_unique = int(series.nunique())
        dtype    = str(series.dtype)

        # categorical if: object dtype  OR  integer-like with few unique values
        is_object  = series.dtype == object
        is_int     = pd.api.types.is_integer_dtype(series)
        is_float   = pd.api.types.is_float_dtype(series)

        # float column that only contains whole numbers → treat as categorical
        if is_float and n_unique <= cat_threshold:
            all_whole = (series.dropna() % 1 == 0).all()
        else:
            all_whole = False

        if is_object or is_int or all_whole:
            kind = 'categorical' if n_unique <= cat_threshold else 'continuous'
        else:
            kind = 'continuous'

        sample = series.unique()[:5].tolist()

        rows.append({
            'feature'    : col,
            'dtype'      : dtype,
            'n_unique'   : n_unique,
            'kind'       : kind,
            'sample_vals': sample,
        })

    return pd.DataFrame(rows).sort_values(
        ['kind', 'n_unique'], ascending=[True, True]
    ).reset_index(drop=True)


def main():
    print(f"\nScanning: {os.path.abspath(ATTACK_DATA)}")
    print(f"Sampling up to {MAX_FILES} CSV(s) per domain ...\n")

    df_all = load_sample(ATTACK_DATA, MAX_FILES)
    print(f"Loaded {len(df_all):,} rows × {len(df_all.columns)} columns "
          f"from {df_all.shape[0]} combined rows.\n")

    result = classify_features(df_all, CAT_THRESHOLD, DROP_COLS)

    categorical = result[result['kind'] == 'categorical']
    continuous  = result[result['kind'] == 'continuous']

    # ── print summary ──────────────────────────────────────────
    print("=" * 65)
    print(f"  CATEGORICAL features  ({len(categorical)})  "
          f"[n_unique <= {CAT_THRESHOLD}]")
    print("=" * 65)
    if len(categorical):
        print(categorical[['feature', 'dtype', 'n_unique', 'sample_vals']]
              .to_string(index=False))
    else:
        print("  (none)")

    print()
    print("=" * 65)
    print(f"  CONTINUOUS features  ({len(continuous)})")
    print("=" * 65)
    if len(continuous):
        print(continuous[['feature', 'dtype', 'n_unique', 'sample_vals']]
              .to_string(index=False))
    else:
        print("  (none)")

    print()
    print(f"  Total feature columns : {len(result)}")
    print(f"  Categorical           : {len(categorical)}")
    print(f"  Continuous            : {len(continuous)}")
    print()

    # ── save CSV ───────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'inspect_features.csv')
    result.to_csv(out_path, index=False)
    print(f"Full results saved to: {out_path}\n")


if __name__ == '__main__':
    main()
