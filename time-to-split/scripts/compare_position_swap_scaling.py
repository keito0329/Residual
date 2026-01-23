#!/usr/bin/env python3
import argparse
from typing import List

import numpy as np
import pandas as pd


def parse_k_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def hit(rank: pd.Series, k: int) -> pd.Series:
    return (rank > 0) & (rank <= k)


def summarize(base_df: pd.DataFrame, scaled_df: pd.DataFrame, ks: List[int]) -> pd.DataFrame:
    merged = base_df.merge(
        scaled_df,
        on=["user_id", "input_len"],
        how="inner",
        suffixes=("_base", "_scaled"),
    )

    if merged.empty:
        raise ValueError("No overlapping users between base and scaled results.")

    rows = []
    for k in ks:
        rank_L_base = merged["rank_L_base"]
        rank_Lm1_base = merged["rank_Lm1_base"]
        rank_L_scaled = merged["rank_L_scaled"]

        valid = rank_L_base.notna() & rank_Lm1_base.notna() & rank_L_scaled.notna()
        base_miss = ~hit(rank_L_base, k)
        base_lm1_hit = hit(rank_Lm1_base, k)
        cond = valid & base_miss & base_lm1_hit

        recovered = hit(rank_L_scaled, k)
        recovered_rate = float(recovered[cond].mean()) if cond.any() else 0.0

        rows.append(
            {
                "k": k,
                "n_condition": int(cond.sum()),
                "n_recovered": int((recovered & cond).sum()),
                "recovered_rate": recovered_rate,
                "base_hit_rate_all": float(hit(rank_L_base, k).mean()),
                "scaled_hit_rate_all": float(hit(rank_L_scaled, k).mean()),
                "scaled_mean_rank_L_in_condition": float(rank_L_scaled[cond].mean())
                if cond.any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare position-swap ranks between base and scaled residual settings."
    )
    ap.add_argument("--base", required=True, help="Path to *_position_swap_ranks.csv (scale=1)")
    ap.add_argument("--scaled", required=True, help="Path to *_position_swap_ranks.csv (scale!=1)")
    ap.add_argument(
        "--k",
        default="1,5,10,20,50,100,200",
        help="Comma-separated list of top-K values.",
    )
    ap.add_argument("--out", default=None, help="Optional output CSV path.")
    args = ap.parse_args()

    base_df = pd.read_csv(args.base)
    scaled_df = pd.read_csv(args.scaled)

    required = {"user_id", "input_len", "rank_L", "rank_Lm1"}
    if not required.issubset(base_df.columns):
        raise ValueError(f"Base file missing columns: {required - set(base_df.columns)}")
    if not required.issubset(scaled_df.columns):
        raise ValueError(f"Scaled file missing columns: {required - set(scaled_df.columns)}")

    ks = parse_k_list(args.k)
    summary = summarize(base_df, scaled_df, ks)

    print(summary.to_string(index=False))
    if args.out:
        summary.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
