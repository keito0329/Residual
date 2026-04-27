#!/usr/bin/env python3
"""
Compute swap scaling recovered rates from position_swap_ranks.csv files.
Outputs a CSV compatible with plot_swap_scaling_recovered_overlay.py:
  residual_scale,k,offset,value
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def parse_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_scale_from_name(path: str) -> float:
    base = os.path.basename(path)
    parts = base.split("_")
    try:
        pos_idx = parts.index("position")
    except ValueError as exc:
        raise ValueError(f"Cannot parse scale from filename: {base}") from exc
    if pos_idx < 3:
        raise ValueError(f"Filename too short to parse scale: {base}")
    scale_token = parts[pos_idx - 3]
    try:
        return float(scale_token)
    except ValueError as exc:
        raise ValueError(f"Invalid scale token '{scale_token}' in {base}") from exc


def hit(rank: pd.Series, k: int) -> pd.Series:
    return (rank > 0) & (rank <= k)


def load_and_merge(base_path: str, scaled_path: str) -> pd.DataFrame:
    base_df = pd.read_csv(base_path)
    scaled_df = pd.read_csv(scaled_path)
    return base_df.merge(
        scaled_df,
        on=["user_id", "input_len"],
        how="inner",
        suffixes=("_base", "_scaled"),
    )


def recovered_rate(
    merged: pd.DataFrame, k: int, offset: int
) -> float:
    base_rank = merged["rank_L_base"]
    scaled_rank = merged["rank_L_scaled"]
    lm_col = f"rank_Lm{offset + 1}_base"
    if lm_col not in merged.columns:
        raise ValueError(f"Missing column for offset={offset}: {lm_col}")

    base_lm = merged[lm_col]
    valid = base_rank.notna() & base_lm.notna() & scaled_rank.notna()
    cond = valid & (~hit(base_rank, k)) & hit(base_lm, k)
    recovered = hit(scaled_rank, k)
    return float(recovered[cond].mean()) if cond.any() else 0.0


def collect_paths(input_dir: str, pattern: str) -> Dict[float, str]:
    paths = sorted(glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(input_dir, pattern)}")
    scale_map: Dict[float, str] = {}
    for path in paths:
        scale = parse_scale_from_name(path)
        if scale in scale_map:
            print(f"[WARN] Duplicate scale {scale} for {path}, keeping {scale_map[scale]}")
            continue
        scale_map[scale] = path
    return scale_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        default="data/results/global_timesplit/val_by_time/Beauty/q09/SASRecAnalyze/test_last",
        help="Directory containing *_position_swap_ranks.csv files.",
    )
    ap.add_argument(
        "--pattern",
        default="*_position_swap_ranks.csv",
        help="Glob pattern for position swap rank files.",
    )
    ap.add_argument(
        "--base-scale",
        type=float,
        default=1.0,
        help="Scale to use as base (default=1.0).",
    )
    ap.add_argument(
        "--k",
        default="10",
        help="Comma-separated list of top-K values.",
    )
    ap.add_argument(
        "--offset",
        default="0",
        help="Comma-separated list of offsets (0 -> rank_Lm1, 1 -> rank_Lm2, ...).",
    )
    ap.add_argument(
        "--output",
        default="swap_scaling_recovered.csv",
        help="Output CSV path.",
    )
    args = ap.parse_args()

    ks = parse_list(args.k)
    offsets = parse_list(args.offset)

    scale_map = collect_paths(args.input_dir, args.pattern)
    if args.base_scale not in scale_map:
        raise ValueError(f"Base scale {args.base_scale} not found in {args.input_dir}")

    base_path = scale_map[args.base_scale]
    rows: List[Tuple[float, int, int, float]] = []
    for scale in sorted(scale_map.keys()):
        merged = load_and_merge(base_path, scale_map[scale])
        if merged.empty:
            raise ValueError(f"No overlapping users between {base_path} and {scale_map[scale]}")
        for k in ks:
            for offset in offsets:
                value = recovered_rate(merged, k, offset)
                rows.append((scale, k, offset, value))

    out_df = pd.DataFrame(rows, columns=["residual_scale", "k", "offset", "value"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
