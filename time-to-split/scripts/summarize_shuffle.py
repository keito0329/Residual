"""
BSARec / DuoRec shuffle experiment summary.

Reads original / partial_shuffle_keep_last / full_shuffle results and prints
NDCG@10 with drop rates (Δ%) relative to original.

Usage:
  python scripts/summarize_shuffle.py [--data_path PATH]

Default data path: $SEQ_SPLITS_DATA_PATH or ./data.
Also checks /home/kozaki/mount/time-to-split/data as fallback.
"""

import os
import glob
import argparse
import pandas as pd

DATASETS = ["Beauty", "Sports", "Video", "Diginetica", "Toys",
            "Steam", "BeerAdvocate", "Movielens-1m", "Zvuk"]

DATASET_DISPLAY = {
    "Beauty":       "Beauty",
    "Sports":       "Sports",
    "Video":        "Video",
    "Diginetica":   "Diginetica",
    "Toys":         "Toys",
    "Steam":        "Steam",
    "BeerAdvocate": "BeerAdv",
    "Movielens-1m": "ML-1M",
    "Zvuk":         "Zvuk",
}

MODES = {
    "original":   "test_last",
    "partial":    "test_last_input_partial_shuffle_keep_last",
    "full":       "test_last_input_full_shuffle",
}

BSAREC_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]


def find_data_root():
    candidates = [
        os.environ.get("SEQ_SPLITS_DATA_PATH", ""),
        "/home/kozaki/mount/time-to-split/data",
        os.path.join(os.path.dirname(__file__), "..", "data"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "results")):
            return c
    return candidates[-1]


def read_metric(path, key):
    try:
        df = pd.read_csv(path, index_col=0)
        row = df[df["metric_name"] == key]
        return float(row["metric_value"].iloc[0]) if not row.empty else float("nan")
    except Exception:
        return float("nan")


def find_csv(result_dir, exclude_patterns=("hrli", "hr_rank", "swap", "per_user", "hist")):
    if not os.path.isdir(result_dir):
        return None
    files = glob.glob(os.path.join(result_dir, "*.csv"))
    for f in files:
        bn = os.path.basename(f).lower()
        if any(p in bn for p in exclude_patterns):
            continue
        return f
    return None


def get_bsarec_best_alpha_csv(data_root, dataset):
    """Return (best_alpha, csv_path) for test_last by max NDCG@10."""
    base_dir = os.path.join(
        data_root, "results", "global_timesplit", "val_by_time",
        dataset, "q09", "BSARec", "test_last"
    )
    best_ndcg, best_alpha, best_path = -1.0, None, None
    for alpha in BSAREC_ALPHAS:
        pattern = os.path.join(base_dir, f"*_{alpha}_*_17.csv")
        matches = [f for f in glob.glob(pattern)
                   if not any(p in os.path.basename(f).lower()
                              for p in ("hrli", "hr_rank", "swap", "per_user"))]
        for f in matches:
            ndcg = read_metric(f, "test_last_NDCG@10")
            if ndcg > best_ndcg:
                best_ndcg, best_alpha, best_path = ndcg, alpha, f
    return best_alpha, best_path


def get_bsarec_shuffle_csv(data_root, dataset, mode_dir):
    """Return CSV in mode_dir whose alpha matches the best alpha."""
    if mode_dir == "test_last":
        _, p = get_bsarec_best_alpha_csv(data_root, dataset)
        return p
    d = os.path.join(data_root, "results", "global_timesplit", "val_by_time",
                     dataset, "q09", "BSARec", mode_dir)
    return find_csv(d)


def get_duorec_csv(data_root, dataset, mode_dir):
    d = os.path.join(data_root, "results", "global_timesplit", "val_by_time",
                     dataset, "q09", "DuoRec", mode_dir)
    # prefer ssl=un
    if os.path.isdir(d):
        files = glob.glob(os.path.join(d, "*_un_*_17.csv"))
        if files:
            return files[0]
    return find_csv(d)


def fmt(v):
    return f"{v:.4f}" if not pd.isna(v) else "   N/A"


def fmt_drop(orig, v):
    if pd.isna(orig) or pd.isna(v) or orig == 0:
        return "    N/A"
    drop = (orig - v) / orig * 100
    sign = "-" if drop >= 0 else "+"
    return f"{sign}{abs(drop):5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--metric", default="NDCG@10")
    args = parser.parse_args()

    data_root = args.data_path or find_data_root()
    metric_key_template = "test_last_{metric}"
    metric = args.metric
    metric_key = metric_key_template.format(metric=metric.replace("@", "@"))

    print(f"Data root : {data_root}")
    print(f"Metric    : {metric}")
    print()

    # ── BSARec ────────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{'BSARec':^80}")
    print("=" * 80)
    header = f"{'Dataset':<13} {'BestAlpha':>9}  {'Original':>9}  {'Partial(Δ)':>12}  {'Full(Δ)':>12}"
    print(header)
    print("-" * 80)

    bsarec_rows = []
    for ds in DATASETS:
        best_alpha, orig_path = get_bsarec_best_alpha_csv(data_root, ds)
        orig = read_metric(orig_path, metric_key) if orig_path else float("nan")

        partial_path = get_bsarec_shuffle_csv(data_root, ds, MODES["partial"])
        partial = read_metric(partial_path, metric_key.replace("test_last_", "test_last_input_partial_shuffle_keep_last_")) if partial_path else float("nan")
        # fallback: same metric key prefix
        if pd.isna(partial) and partial_path:
            partial = read_metric(partial_path, metric_key)

        full_path = get_bsarec_shuffle_csv(data_root, ds, MODES["full"])
        full = read_metric(full_path, metric_key.replace("test_last_", "test_last_input_full_shuffle_")) if full_path else float("nan")
        if pd.isna(full) and full_path:
            full = read_metric(full_path, metric_key)

        alpha_str = str(best_alpha) if best_alpha is not None else "N/A"
        print(f"{DATASET_DISPLAY[ds]:<13} {alpha_str:>9}  {fmt(orig):>9}  {fmt(partial):>9}{fmt_drop(orig,partial):>8}  {fmt(full):>9}{fmt_drop(orig,full):>8}")
        bsarec_rows.append({"dataset": ds, "best_alpha": best_alpha,
                             "original": orig, "partial": partial, "full": full})

    # ── DuoRec ────────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"{'DuoRec':^80}")
    print("=" * 80)
    header2 = f"{'Dataset':<13}  {'Original':>9}  {'Partial(Δ)':>12}  {'Full(Δ)':>12}"
    print(header2)
    print("-" * 80)

    duorec_rows = []
    for ds in DATASETS:
        orig_path   = get_duorec_csv(data_root, ds, MODES["original"])
        partial_path = get_duorec_csv(data_root, ds, MODES["partial"])
        full_path   = get_duorec_csv(data_root, ds, MODES["full"])

        orig    = read_metric(orig_path,    metric_key) if orig_path else float("nan")
        partial_key = metric_key.replace("test_last_", "test_last_input_partial_shuffle_keep_last_")
        full_key    = metric_key.replace("test_last_", "test_last_input_full_shuffle_")
        partial = read_metric(partial_path, partial_key) if partial_path else float("nan")
        full    = read_metric(full_path,    full_key)    if full_path    else float("nan")

        print(f"{DATASET_DISPLAY[ds]:<13}  {fmt(orig):>9}  {fmt(partial):>9}{fmt_drop(orig,partial):>8}  {fmt(full):>9}{fmt_drop(orig,full):>8}")
        duorec_rows.append({"dataset": ds, "original": orig,
                             "partial": partial, "full": full})

    # ── Drop rate summary ─────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("Drop rate summary (mean over available datasets)")
    print("=" * 80)
    for label, rows in [("BSARec", bsarec_rows), ("DuoRec", duorec_rows)]:
        partial_drops, full_drops = [], []
        for r in rows:
            if not pd.isna(r["original"]) and r["original"] > 0:
                if not pd.isna(r["partial"]):
                    partial_drops.append((r["original"] - r["partial"]) / r["original"] * 100)
                if not pd.isna(r["full"]):
                    full_drops.append((r["original"] - r["full"]) / r["original"] * 100)
        pd_mean = sum(partial_drops) / len(partial_drops) if partial_drops else float("nan")
        fd_mean = sum(full_drops)   / len(full_drops)    if full_drops    else float("nan")
        n_p = len(partial_drops)
        n_f = len(full_drops)
        print(f"  {label:<8}  partial_drop={pd_mean:+.1f}% (n={n_p})   full_drop={fd_mean:+.1f}% (n={n_f})")


if __name__ == "__main__":
    main()
