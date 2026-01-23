"""
Compute item-embedding adjacency features for position-swap cases.
Added (Changed) by Author
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.prepr import last_item_split


def _load_item_embeddings(ckpt_path: str) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    key = None
    for k in state_dict:
        if k.endswith("item_emb.weight"):
            key = k
            break
    if key is None:
        raise KeyError("Could not find item_emb.weight in checkpoint.")
    return state_dict[key].cpu().numpy()


def _load_inputs(data_path: str) -> pd.DataFrame:
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    inputs, _ = last_item_split(test)
    return inputs


def _build_sequences(inputs: pd.DataFrame) -> dict:
    seqs = {}
    sorted_df = inputs.sort_values(["user_id", "timestamp"], kind="stable")
    for uid, grp in sorted_df.groupby("user_id"):
        seqs[uid] = grp["item_id"].tolist()
    return seqs


def _case_label(rank_L, rank_Lm1, k):
    if rank_L > k or rank_L == 0:
        if 0 < rank_Lm1 <= k:
            return "Lm1_hit_L_miss"
        return "both_miss"
    return "L_hit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--position-swap-csv", required=True, help="position_swap_ranks.csv path")
    parser.add_argument("--data-path", required=True, help="Path to split data folder with test.csv")
    parser.add_argument("--k", type=int, default=10, help="K for case labeling")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    emb = _load_item_embeddings(args.checkpoint)
    inputs = _load_inputs(args.data_path)
    seqs = _build_sequences(inputs)

    ranks = pd.read_csv(args.position_swap_csv)
    if "rank_L" not in ranks.columns or "rank_Lm1" not in ranks.columns:
        raise ValueError("position_swap_ranks.csv must contain rank_L and rank_Lm1.")

    rows = []
    for _, row in ranks.iterrows():
        uid = row["user_id"]
        seq = seqs.get(uid)
        if not seq or len(seq) < 2:
            continue
        L = len(seq)
        i_L = seq[-1]
        i_Lm1 = seq[-2]
        i_Lm2 = seq[-3] if L >= 3 else None
        i_Lm3 = seq[-4] if L >= 4 else None

        if i_L >= emb.shape[0] or i_Lm1 >= emb.shape[0]:
            continue

        dot_Lm1_L = float(np.dot(emb[i_Lm1], emb[i_L]))
        dot_Lm2_Lm1 = (
            float(np.dot(emb[i_Lm2], emb[i_Lm1])) if i_Lm2 is not None else np.nan
        )
        dot_Lm3_Lm2 = (
            float(np.dot(emb[i_Lm3], emb[i_Lm2])) if i_Lm3 is not None else np.nan
        )
        delta_last = (
            dot_Lm1_L - dot_Lm2_Lm1 if not np.isnan(dot_Lm2_Lm1) else np.nan
        )
        case = _case_label(row["rank_L"], row["rank_Lm1"], args.k)

        rows.append(
            {
                "user_id": uid,
                "input_len": L,
                "rank_L": row["rank_L"],
                "rank_Lm1": row["rank_Lm1"],
                "case": case,
                "dot_Lm1_L": dot_Lm1_L,
                "dot_Lm2_Lm1": dot_Lm2_Lm1,
                "dot_Lm3_Lm2": dot_Lm3_Lm2,
                "delta_last": delta_last,
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
