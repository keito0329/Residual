import glob
import re
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent


def seed_from_path(path: Path):
    """Extract trailing integer before .csv, return None if not found."""
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def load_metrics_long_format(path: Path):
    """Files with columns [metric_name, metric_value]."""
    df = pd.read_csv(path)
    return df.set_index("metric_name")["metric_value"]


def load_metrics_wide_format(path: Path):
    """Files with first column grid_point and metrics as remaining columns."""
    df = pd.read_csv(path)
    df = df.set_index("grid_point")
    return df


def bucket_summary(sas_all: pd.DataFrame, light_all: pd.DataFrame, alpha=0.05):
    metrics = sorted(set(sas_all.columns) & set(light_all.columns))
    # HR@1/NDCG@1/MRR@1 are identical, so treat them as one for aggregation
    aliases = {
        "test_HitRate@1": "rank@1",
        "test_NDCG@1": "rank@1",
        "test_MRR@1": "rank@1",
        "test_last_HitRate@1": "rank@1",
        "test_last_NDCG@1": "rank@1",
        "test_last_MRR@1": "rank@1",
    }
    seen_canonical = set()
    buckets = {
        "sig_SASRec": 0,
        "sig_LightSASRec": 0,
        "ns_SASRec": 0,
        "ns_LightSASRec": 0,
        "tie": 0,
    }
    details = []

    def cohens_d(x, y):
        if len(x) < 2 or len(y) < 2:
            return float("nan")
        diff = x.mean() - y.mean()
        pooled_var = (x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2
        return diff / (pooled_var ** 0.5) if pooled_var > 0 else float("nan")

    for m in metrics:
        canonical = aliases.get(m, m)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        sa = sas_all[m]
        la = light_all[m]
        diff = sa.mean() - la.mean()
        d = cohens_d(sa, la)
        if len(sa) >= 2:
            t, p = stats.ttest_rel(sa, la)
        else:
            t, p = float("nan"), float("nan")
        if diff == 0:
            buckets["tie"] += 1
            better = "tie"
        elif pd.notna(p) and p < alpha:
            if diff > 0:
                buckets["sig_SASRec"] += 1
                better = "SASRec"
            else:
                buckets["sig_LightSASRec"] += 1
                better = "LightSASRec"
        else:
            if diff > 0:
                buckets["ns_SASRec"] += 1
                better = "SASRec"
            else:
                buckets["ns_LightSASRec"] += 1
                better = "LightSASRec"
        details.append((canonical, diff, d, better, p))

    return buckets, sorted(details, key=lambda x: -abs(x[1]))


def summarize_target_metrics(sas_all, light_all, alpha=0.05, targets=None):
    """Return summary dict for selected metrics (e.g., HR/NDCG@10,20)."""
    if targets is None:
        targets = []
    filtered = [m for m in sas_all.columns if m in targets]
    summary = []
    for m in filtered:
        sa = sas_all[m]
        la = light_all[m]
        diff = sa.mean() - la.mean()
        if len(sa) >= 2 and len(la) >= 2:
            pooled_var = (sa.std(ddof=1) ** 2 + la.std(ddof=1) ** 2) / 2
            d = diff / (pooled_var ** 0.5) if pooled_var > 0 else float("nan")
        else:
            d = float("nan")
        if len(sa) >= 2:
            t, p = stats.ttest_rel(sa, la)
        else:
            t, p = float("nan"), float("nan")
        if diff > 0:
            winner = "SASRec"
        elif diff < 0:
            winner = "LightSASRec"
        else:
            winner = "tie"
        sig = pd.notna(p) and p < alpha
        summary.append((m, sa.mean(), la.mean(), winner, sig, p, d))
    return summary


def analyze_leave_one_out(dataset="Sports"):
    sas_paths = [Path(p) for p in glob.glob(str(ROOT / f"data/results/leave-one-out/{dataset}/SASRec/test/X_64_2_2_0.5_200_*.csv"))]
    light_paths = [Path(p) for p in glob.glob(str(ROOT / f"data/results/leave-one-out/{dataset}/LightSASRec/test/X_64_2_2_0.5_200_*.csv"))]

    sas_df = pd.DataFrame({"path": sas_paths})
    sas_df["seed"] = sas_df["path"].apply(seed_from_path)
    light_df = pd.DataFrame({"path": light_paths})
    light_df["seed"] = light_df["path"].apply(seed_from_path)

    sas_df = sas_df.dropna(subset=["seed"])
    light_df = light_df.dropna(subset=["seed"])
    pairs = sas_df.merge(light_df, on="seed", suffixes=("_sas", "_light")).sort_values("seed")

    sas_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_sas) for _, row in pairs.iterrows()}).T
    light_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_light) for _, row in pairs.iterrows()}).T

    buckets, details = bucket_summary(sas_all, light_all)

    print(f"=== Leave-one-out results {dataset} ===")
    print("seeds:", list(pairs["seed"]))
    print("metrics:", len(set(sas_all.columns) & set(light_all.columns)))
    print("bucket counts:", buckets)
    print("mean SASRec:\n", sas_all.mean())
    print("mean LightSASRec:\n", light_all.mean())
    print("top diffs:")
    for m, d, eff, b, p in details[:10]:
        print(f"  {m}: diff={d:.4g} d={eff:.4g} ({b}), p={p:.3g}")
    targets = [m for m in sas_all.columns if (m.endswith("@10") or m.endswith("@20")) and ("HitRate" in m or "NDCG" in m)]
    summary = summarize_target_metrics(sas_all, light_all, targets=targets)
    print("HR/NDCG @10,@20 summary (mean_SASRec, mean_Light, winner, significant, p, d):")
    for m, sa_mean, la_mean, winner, sig, p, d in summary:
        print(f"  {m}: {sa_mean:.4g} vs {la_mean:.4g} -> {winner}, sig={sig}, p={p:.3g}, d={d:.4g}")


def analyze_global_timesplit(dataset="Sports", quantile="q09", file_name="test_last.csv"):
    base = ROOT / f"data/results/global_timesplit/val_by_time/{dataset}/{quantile}"

    # 1) Try X_*.csv under test_last/ (same style as leave-one-out)
    sas_glob = list((base / "SASRec" / "test_last").glob("X_64_2_2_0.5_200_*.csv"))
    light_glob = list((base / "LightSASRec" / "test_last").glob("X_64_2_2_0.5_200_*.csv"))

    if sas_glob and light_glob:
        sas_df = pd.DataFrame({"path": sas_glob})
        sas_df["seed"] = sas_df["path"].apply(seed_from_path)
        light_df = pd.DataFrame({"path": light_glob})
        light_df["seed"] = light_df["path"].apply(seed_from_path)
        sas_df = sas_df.dropna(subset=["seed"])
        light_df = light_df.dropna(subset=["seed"])
        pairs = sas_df.merge(light_df, on="seed", suffixes=("_sas", "_light")).sort_values("seed")

        if pairs.empty:
            print("Global timesplit X_*.csv found but no matching seeds.")
            return

        sas_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_sas) for _, row in pairs.iterrows()}).T
        light_all = pd.DataFrame({row.seed: load_metrics_long_format(row.path_light) for _, row in pairs.iterrows()}).T
        index_info = list(pairs["seed"])
    else:
        # 2) Fallback to aggregated final_results/<file_name>
        sas_path = base / f"SASRec/final_results/{file_name}"
        light_path = base / f"LightSASRec/final_results/{file_name}"
        if not sas_path.exists() or not light_path.exists():
            print("Global timesplit files not found:", sas_path, light_path)
            return

        sas_df = load_metrics_wide_format(sas_path)
        light_df = load_metrics_wide_format(light_path)
        common_idx = sas_df.index.intersection(light_df.index)
        sas_all = sas_df.loc[common_idx]
        light_all = light_df.loc[common_idx]
        index_info = list(common_idx)

    buckets, details = bucket_summary(sas_all, light_all)
    print(f"\n=== Global timesplit val_by_time results {dataset} ===")
    print("pairs:", index_info)
    print("metrics:", len(set(sas_all.columns) & set(light_all.columns)))
    print("bucket counts:", buckets)
    print("mean SASRec:\n", sas_all.mean())
    print("mean LightSASRec:\n", light_all.mean())
    print("top diffs:")
    for m, d, eff, b, p in details[:10]:
        print(f"  {m}: diff={d:.4g} d={eff:.4g} ({b}), p={p:.3g}")
    targets = [m for m in sas_all.columns if (m.endswith("@10") or m.endswith("@20")) and ("HitRate" in m or "NDCG" in m)]
    summary = summarize_target_metrics(sas_all, light_all, targets=targets)
    print("HR/NDCG @10,@20 summary (mean_SASRec, mean_Light, winner, significant, p, d):")
    for m, sa_mean, la_mean, winner, sig, p, d in summary:
        print(f"  {m}: {sa_mean:.4g} vs {la_mean:.4g} -> {winner}, sig={sig}, p={p:.3g}, d={d:.4g}")


if __name__ == "__main__":
    dataset = "Zvuk"
    analyze_leave_one_out(dataset=dataset)
    analyze_global_timesplit(dataset=dataset, quantile="q09", file_name="X_64_2_2_0.5_200_*.csv")
