"""
Average contribution heatmap analysis (STRICT, PAPER-SAFE)

Pipeline (Heatmap):
1. For each sequence:
   - remove padding (input_ids == 0) BEFORE computation
   - reverse sequence (newer -> larger index)
   - restrict to most recent K items
   - build KxK contribution heatmap with NaN padding
2. Average all per-sequence heatmaps (NaN ignored)
3. Visualize dataset-level average heatmap
   - source = Y-axis (down = recent)
   - target = X-axis (right = recent)

Statistics (Latest mixing ratio):
- latest item is defined as the RIGHTMOST non-padding item in the ORIGINAL sequence
- do NOT apply reverse/recentK/to_fixed_kxk for statistics (avoid mixing pipelines)

Properties:
- padding NEVER affects computation
- short sequences are handled correctly
- representative dataset-level heatmap
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

# =========================
# Config
# =========================
PADDING_IDX = 0
RECENT_K = 15
FIGSIZE = (5, 5)


# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_dataset_name(analysis_dir: str) -> str:
    parts = os.path.normpath(analysis_dir).split(os.sep)
    if "SASRecAnalyze" in parts:
        idx = parts.index("SASRecAnalyze")
        if idx + 1 < len(parts):
            name = parts[idx + 1]
            if name == "Movielens-1m":
                return "ML-1M"
            return name
    name = os.path.basename(os.path.normpath(analysis_dir))
    if name == "Movielens-1m":
        return "ML-1M"
    return name


def load_all_batches(analysis_dir: str):
    paths = sorted(glob.glob(os.path.join(analysis_dir, "batch*.npz")))
    if not paths:
        raise RuntimeError(f"No batch files in {analysis_dir}")
    return [np.load(p) for p in paths]


def detect_num_layers(analysis_dir: str) -> int:
    """
    Detect number of layers based on saved keys in the first batch file.
    """
    first = load_all_batches(analysis_dir)[0]
    keys = [k for k in first.files if k.startswith("layer") and k.endswith("_post_ln_norm")]
    # keys like: layer0_post_ln_norm, layer1_post_ln_norm, ...
    # robustly find max layer index
    layer_ids = []
    for k in keys:
        # "layer{idx}_post_ln_norm"
        try:
            idx = int(k.split("_")[0].replace("layer", ""))
            layer_ids.append(idx)
        except Exception:
            continue
    if not layer_ids:
        raise RuntimeError("Could not detect layers from npz keys.")
    return max(layer_ids) + 1


def extract_valid(ids: np.ndarray, x: np.ndarray):
    """
    Remove padding BEFORE any computation.
    ids: [L]
    x:   [L] or [L,L]
    """
    idx = np.where(ids != PADDING_IDX)[0]

    if x.ndim == 1:
        return x[idx]
    if x.ndim == 2:
        return x[np.ix_(idx, idx)]
    raise ValueError("Unsupported ndim")


def reverse_seq(x: np.ndarray):
    """
    Reverse sequence so newer items have larger indices.
    - after reverse: right/end is most recent
    """
    if x.ndim == 1:
        return x[::-1]
    if x.ndim == 2:
        return x[::-1, ::-1]
    raise ValueError("Unsupported ndim")


def restrict_to_recent(x: np.ndarray, k: int):
    """
    Keep only the most recent k items (after reverse, this means take tail).
    """
    if x.ndim == 1:
        return x[-k:]
    if x.ndim == 2:
        return x[-k:, -k:]
    raise ValueError("Unsupported ndim")


def to_fixed_kxk(mat: np.ndarray, k: int):
    """
    Convert [L', L'] -> [k, k] by NaN padding (top-left).
    Assumes L' <= k after restriction (or uses min).
    """
    out = np.full((k, k), np.nan, dtype=np.float32)
    Lp = min(mat.shape[0], k)
    out[:Lp, :Lp] = mat[:Lp, :Lp]
    return out


# =========================
# Latest mixing ratio (STATISTICS) — strict definition
# =========================
def collect_latest_mixing_ratio_strict(analysis_dir: str, layer: int = 0) -> np.ndarray:
    """
    Latest item is defined as the RIGHTMOST non-padding item in the ORIGINAL sequence.
    That is: padding removed -> take last element.

    IMPORTANT:
    - Do NOT apply reverse/recentK/to_fixed_kxk here.
    - This avoids mixing the visualization pipeline into statistics.

    Returns:
        np.ndarray shape [N]
    """
    batches = load_all_batches(analysis_dir)
    vals = []

    key = f"layer{layer}_mixing_ratio"
    for batch in batches:
        ids_all = batch["input_ids"]  # [B, L]
        mix_all = batch[key]          # [B, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            mix = mix_all[b]

            v = extract_valid(ids, mix)  # remove padding ONLY
            if v.size == 0:
                continue

            vals.append(float(v[-1]))  # rightmost non-padding = latest

    return np.asarray(vals, dtype=np.float64)


def latest_mixing_stats(latest_vals: np.ndarray, atol: float = 1e-8) -> dict:
    """
    Summary stats + near-zero check.
    """
    if latest_vals.size == 0:
        return {"N": 0}

    return {
        "N": int(latest_vals.size),
        "min": float(np.min(latest_vals)),
        "max": float(np.max(latest_vals)),
        "mean": float(np.mean(latest_vals)),
        "median": float(np.median(latest_vals)),   # added
        "std": float(np.std(latest_vals)),
        "num_nonzero": int(np.sum(np.abs(latest_vals) > atol)),
        "fraction_nonzero": float(np.mean(np.abs(latest_vals) > atol)),
    }


def mixing_stats(vals: np.ndarray, atol: float = 1e-8) -> dict:
    if vals.size == 0:
        return {"N": 0}

    return {
        "N": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "num_nonzero": int(np.sum(np.abs(vals) > atol)),
        "fraction_nonzero": float(np.mean(np.abs(vals) > atol)),
    }


# =========================
# Contribution heatmaps (VISUALIZATION)
# =========================
def collect_contribution_heatmaps_recent_k(analysis_dir: str, layer: int = 0, k: int = RECENT_K) -> np.ndarray:
    """
    For each sequence:
    - remove padding
    - reverse (so recent -> larger index)
    - restrict to recent K
    - make fixed KxK with NaN padding

    Returns:
        heatmaps: [N, K, K] (NaN padded)
    """
    batches = load_all_batches(analysis_dir)
    maps = []

    key = f"layer{layer}_post_ln_norm"
    for batch in batches:
        ids_all = batch["input_ids"]   # [B, L]
        contrib_all = batch[key]       # [B, L, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            contrib = contrib_all[b]

            c = extract_valid(ids, contrib)      # drop padding rows/cols
            c = reverse_seq(c)                   # recent -> end
            c = restrict_to_recent(c, k)         # keep last k
            c = to_fixed_kxk(c, k)               # [k,k] with NaN padding

            maps.append(c)

    return np.stack(maps, axis=0).astype(np.float32)  # [N,k,k]


def average_heatmap_nan(heatmaps: np.ndarray) -> np.ndarray:
    """
    NaN-safe average over sequences.
    """
    return np.nanmean(heatmaps, axis=0)


# def plot_average_contribution_heatmap(avg_map: np.ndarray, fig_dir: str, layer: int = 0, k: int = RECENT_K):
#     """
#     source = Y-axis, target = X-axis.
#     Requirements:
#       - target recent -> right
#       - source recent -> down

#     Our internal map is indexed as [target, source] after the strict pipeline.
#     We use transpose to show source on Y and target on X.
#     Then we set origin='upper' to make Y increase downward (recent down).
#     """
#     ensure_dir(fig_dir)

#     # source=y, target=x
#     avg_vis = avg_map.T

#     plt.figure(figsize=FIGSIZE)
#     im = plt.imshow(avg_vis, cmap="viridis", origin="upper")
#     plt.colorbar(im, fraction=0.046, pad=0.04, label="Post-LN norm")

#     plt.xlabel("Target position i (recent → right)")
#     plt.ylabel("Source position j (recent → down)")
#     plt.title(f"Average Contribution Heatmap\nLayer {layer}, Recent {k}")

#     plt.tight_layout()
#     out_path = os.path.join(fig_dir, f"avg_contribution_layer{layer}_recent{k}_sourceDown.png")
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     print(f"[Saved] {out_path}")

def plot_average_contribution_heatmap(
    avg_map: np.ndarray,
    fig_dir: str,
    dataset_name: str,
    layer: int = 0,
    k: int = RECENT_K,
):
    ensure_dir(fig_dir)

    # source=y, target=x
    avg_vis = avg_map.T

    plt.figure(figsize=FIGSIZE)
    im = plt.imshow(avg_vis, origin="upper")
    # No colorbar (remove the right bar entirely)

    plt.xlabel("To")
    plt.ylabel("From")
    plt.title(f"{dataset_name} AttnResLN", fontsize=18)

    plt.tight_layout()
    out_path = os.path.join(
        fig_dir,
        f"avg_contribution_layer{layer}_recent{k}_sourceDown.pdf",
    )
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")




def collect_attention_heatmaps_recent_k(
    analysis_dir: str,
    layer: int = 0,
    k: int = RECENT_K,
):
    """
    Collect strict attention heatmaps.

    Assumes:
      layer{layer}_attention is saved in npz
      shape: [B, L, L] (head-averaged)

    Returns:
      heatmaps: np.ndarray [N, K, K] (NaN padded)
    """
    batches = load_all_batches(analysis_dir)
    maps = []

    key = f"layer{layer}_attention"

    for batch in batches:
        if key not in batch.files:
            raise RuntimeError(
                f"{key} not found in npz. "
                "Make sure attention is saved during analysis."
            )

        ids_all = batch["input_ids"]     # [B, L]
        attn_all = batch[key]            # [B, L, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            attn = attn_all[b]

            # ---- STRICT pipeline ----
            a = extract_valid(ids, attn)     # remove padding
            a = reverse_seq(a)               # recent -> larger index
            a = restrict_to_recent(a, k)     # keep recent K
            a = to_fixed_kxk(a, k)            # KxK with NaN

            maps.append(a)

    return np.stack(maps, axis=0).astype(np.float32)


def average_attention_heatmap(heatmaps: np.ndarray) -> np.ndarray:
    """
    NaN-safe average over all sequences, then row-normalize.
    """
    # Count valid rows per target position to keep a consistent denominator.
    row_valid = ~np.all(np.isnan(heatmaps), axis=2)  # [N, K]
    row_counts = row_valid.sum(axis=0)               # [K]

    sum_map = np.nansum(heatmaps, axis=0)            # [K, K]
    avg_map = sum_map / np.maximum(row_counts[:, None], 1)

    # Preserve NaN where no data exists for a cell.
    cell_counts = np.sum(~np.isnan(heatmaps), axis=0)
    avg_map[cell_counts == 0] = np.nan

    # Row-normalize to make each row sum to 1 (ignoring NaN).
    row_sums = np.nansum(avg_map, axis=1, keepdims=True)
    avg_map = avg_map / np.where(row_sums == 0, 1.0, row_sums)
    avg_map[row_counts == 0] = np.nan

    return avg_map



# def plot_average_attention_heatmap(
#     avg_map: np.ndarray,
#     fig_dir: str,
#     layer: int,
#     k: int = RECENT_K,
# ):
#     """
#     Visualization:
#       - source = Y-axis (down = recent)
#       - target = X-axis (right = recent)
#     """
#     ensure_dir(fig_dir)

#     # source = y, target = x
#     avg_vis = avg_map.T

#     plt.figure(figsize=FIGSIZE)
#     im = plt.imshow(avg_vis, cmap="magma", origin="upper")
#     plt.colorbar(im, fraction=0.046, pad=0.04, label="Attention weight")

#     plt.xlabel("Target position i (recent → right)")
#     plt.ylabel("Source position j (recent → down)")
#     plt.title(f"Average Attention Heatmap\nLayer {layer}, Recent {k}")

#     plt.tight_layout()

#     path = os.path.join(
#         fig_dir,
#         f"avg_attention_layer{layer}_recent{k}_sourceDown.png",
#     )
#     plt.savefig(path, dpi=200, bbox_inches="tight")
#     plt.close()

#     print(f"[Saved] {path}")

def plot_average_attention_heatmap(
    avg_map: np.ndarray,
    fig_dir: str,
    dataset_name: str,
    layer: int,
    k: int = RECENT_K,
):
    ensure_dir(fig_dir)

    # source = y, target = x
    avg_vis = avg_map.T
    # Normalize rows in the visualization (source rows) to sum to 1.
    row_sums = np.nansum(avg_vis, axis=1, keepdims=True)
    avg_vis = avg_vis / np.where(row_sums == 0, 1.0, row_sums)

    plt.figure(figsize=FIGSIZE)
    im = plt.imshow(avg_vis, origin="upper")
    # No colorbar (remove the right bar entirely)

    plt.xlabel("To")
    plt.ylabel("From")
    plt.title(f"{dataset_name} Attn", fontsize=18)

    plt.tight_layout()
    path = os.path.join(
        fig_dir,
        f"avg_attention_layer{layer}_recent{k}_sourceDown.pdf",
    )
    plt.savefig(path)
    plt.close()

    print(f"[Saved] {path}")



def compute_attention_mixing_ratio(attn: np.ndarray) -> np.ndarray:
    """
    Compute attention mixing ratio per target position.

    Args:
        attn: [L, L] attention map (row i sums to 1)

    Returns:
        mixing_ratio: [L]
          mixing_ratio[i] = sum_{j != i} attn[i, j]
                          = 1 - attn[i, i]
    """
    if attn.ndim != 2:
        raise ValueError("attn must be [L, L]")

    diag = np.diag(attn)          # [L]
    mixing_ratio = 1.0 - diag     # since sum_j attn[i,j] = 1

    return mixing_ratio


def collect_attention_mixing_ratio_strict(
    analysis_dir: str,
    layer: int = 0,
    k: int = RECENT_K,
):
    """
    Collect attention mixing ratios for the most recent K items.

    Returns:
        values: np.ndarray [N * <=K]
    """
    batches = load_all_batches(analysis_dir)
    vals = []

    key = f"layer{layer}_attention"

    for batch in batches:
        ids_all = batch["input_ids"]   # [B, L]
        attn_all = batch[key]          # [B, L, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            attn = attn_all[b]

            # ---- STRICT pipeline ----
            a = extract_valid(ids, attn)
            a = reverse_seq(a)
            a = restrict_to_recent(a, k)

            if a.size == 0:
                continue

            mr = compute_attention_mixing_ratio(a)  # [<=K]
            vals.append(mr)

    if not vals:
        return np.array([])

    return np.concatenate(vals)



def collect_latest_attention_mixing_ratio_strict(
    analysis_dir: str,
    layer: int = 0,
):
    """
    Latest-item attention mixing ratio:
      = 1 - attention[i,i] at the rightmost non-padding position.

    Returns:
        np.ndarray [N]
    """
    batches = load_all_batches(analysis_dir)
    vals = []

    key = f"layer{layer}_attention"

    for batch in batches:
        ids_all = batch["input_ids"]   # [B, L]
        attn_all = batch[key]          # [B, L, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            attn = attn_all[b]

            a = extract_valid(ids, attn)
            if a.size == 0:
                continue

            # rightmost non-padding = latest
            i = a.shape[0] - 1
            vals.append(1.0 - a[i, i])

    return np.asarray(vals, dtype=np.float64)


def collect_all_mixing_ratio_strict(
    analysis_dir: str,
    layer: int = 0,
) -> np.ndarray:
    """
    Collect ALL valid (non-padding) mixing ratios across all sequences.

    STRICT definition:
    - remove padding ONLY
    - do NOT apply reverse / recentK / heatmap pipeline

    Returns:
        np.ndarray [total_valid_positions]
    """
    batches = load_all_batches(analysis_dir)
    vals = []

    key = f"layer{layer}_mixing_ratio"

    for batch in batches:
        ids_all = batch["input_ids"]  # [B, L]
        mix_all = batch[key]          # [B, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            mix = mix_all[b]

            v = extract_valid(ids, mix)  # remove padding only
            if v.size == 0:
                continue

            vals.append(v)

    if not vals:
        return np.array([])

    return np.concatenate(vals, axis=0).astype(np.float64)


def collect_all_generic_mixing_ratio_strict(
    analysis_dir: str,
    layer: int,
    key_suffix: str,
) -> np.ndarray:
    """
    Collect ALL valid (non-padding) mixing ratios across all sequences.

    STRICT definition:
    - remove padding ONLY
    - do NOT apply reverse / recentK / heatmap pipeline

    key_suffix examples:
      - "attn_mixing_ratio"        (Attn-N)
      - "attnres_mixing_ratio"     (AttnRes-N)
      - "mixing_ratio"             (AttnResLN-N)

    Returns:
        np.ndarray [total_valid_positions]
    """
    batches = load_all_batches(analysis_dir)
    vals = []

    key = f"layer{layer}_{key_suffix}"

    for batch in batches:
        if key not in batch.files:
            raise RuntimeError(f"{key} not found in npz.")

        ids_all = batch["input_ids"]   # [B, L]
        mix_all = batch[key]           # [B, L]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            mix = mix_all[b]

            v = extract_valid(ids, mix)   # remove padding only
            if v.size == 0:
                continue

            vals.append(v)

    if not vals:
        return np.array([])

    return np.concatenate(vals, axis=0).astype(np.float64)


def collect_all_layers_mixing_ratio_strict(
    analysis_dir: str,
    layers: List[int],
    key_suffix: str,
) -> np.ndarray:
    """
    Collect ALL valid mixing ratios from multiple layers and concatenate them.

    Example:
      layers=[0,1], key_suffix="attn_mixing_ratio"
    """
    all_vals = []

    for layer in layers:
        vals = collect_all_generic_mixing_ratio_strict(
            analysis_dir,
            layer,
            key_suffix,
        )
        if vals.size > 0:
            all_vals.append(vals)

    if not all_vals:
        return np.array([])

    return np.concatenate(all_vals, axis=0)



def compute_row_entropy(attn: np.ndarray, eps: float = 1e-12):
    """
    Normalized row-wise entropy.
    Returns 0 when sequence length is 1.
    """
    if attn.ndim != 2:
        raise ValueError("attn must be [L, L]")

    L = attn.shape[1]

    if L <= 1:
        return np.zeros(attn.shape[0], dtype=np.float64)

    p = np.clip(attn, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)

    max_entropy = np.log(L)

    return entropy / max_entropy



def collect_all_attention_entropy_strict(
    analysis_dir: str,
    layer: int = 0,
):
    """
    Collect normalized row-wise entropy for all valid positions.
    STRICT:
      - remove padding only
      - no reverse / recentK
    """
    batches = load_all_batches(analysis_dir)
    vals = []

    key = f"layer{layer}_attention"

    for batch in batches:
        ids_all = batch["input_ids"]
        attn_all = batch[key]

        for b in range(ids_all.shape[0]):
            ids = ids_all[b]
            attn = attn_all[b]

            a = extract_valid(ids, attn)
            if a.size == 0:
                continue

            ent = compute_row_entropy(a)
            vals.append(ent)

    if not vals:
        return np.array([])

    return np.concatenate(vals).astype(np.float64)




# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, help="dataset name (e.g., Sports)")
    ap.add_argument("--model", type=str, default="SASRecAnalyze")
    ap.add_argument("--split", type=str, default="global_timesplit")
    ap.add_argument("--seed", type=str, default="seed_1")
    args = ap.parse_args()

    # Preferred path (most common)
    analysis_dir = os.path.join(
        "data",
        "results",
        "analysis",
        args.model,
        args.dataset,
        args.split,
        args.seed,
    )

    if not os.path.isdir(analysis_dir):
        # Fallback: try to find a unique match for the dataset
        pattern = os.path.join(
            "data",
            "results",
            "analysis",
            "*",
            args.dataset,
            args.split,
            "seed_*",
        )
        matches = sorted(glob.glob(pattern))
        if len(matches) == 1:
            analysis_dir = matches[0]
        elif len(matches) == 0:
            raise RuntimeError(
                f"No analysis directory found for dataset '{args.dataset}'. "
                f"Tried: {analysis_dir} and pattern {pattern}"
            )
        else:
            raise RuntimeError(
                "Multiple analysis directories found. "
                f"Please specify --model/--seed. Candidates: {matches}"
            )

    fig_dir = os.path.join(analysis_dir, f"figures_avg_recent{RECENT_K}_sourceDown")
    ensure_dir(fig_dir)
    dataset_name = infer_dataset_name(analysis_dir)

    num_layers = detect_num_layers(analysis_dir)
    print(f"[INFO] Detected {num_layers} layers")

    for layer in range(num_layers):
      # latest
      latest_vals = collect_latest_mixing_ratio_strict(analysis_dir, layer=layer)
      latest_stats = mixing_stats(latest_vals)

      # overall
      all_vals = collect_all_mixing_ratio_strict(analysis_dir, layer=layer)
      all_stats = mixing_stats(all_vals)

      print(f"[Layer {layer}] Mixing ratio statistics (STRICT)")
      print("  Latest item (rightmost non-padding):")
      for k, v in latest_stats.items():
          print(f"    {k}: {v}")

      print("  All valid positions (padding removed):")
      for k, v in all_stats.items():
          print(f"    {k}: {v}")

    # ---- (B) Average contribution heatmap per layer ----
    for layer in range(num_layers):
        print(f"[INFO] Processing contribution heatmaps for layer {layer}")

        heatmaps = collect_contribution_heatmaps_recent_k(
            analysis_dir,
            layer=layer,
            k=RECENT_K,
        )
        print(f"  collected {heatmaps.shape[0]} heatmaps, shape per map={heatmaps.shape[1:]}")

        avg_map = average_heatmap_nan(heatmaps)

        plot_average_contribution_heatmap(
            avg_map,
            fig_dir,
            dataset_name,
            layer=layer,
            k=RECENT_K,
        )

        print(
            f"  avg_map stats: "
            f"min={np.nanmin(avg_map):.4f}, "
            f"max={np.nanmax(avg_map):.4f}, "
            f"mean={np.nanmean(avg_map):.4f}"
        )

    for layer in range(num_layers):
      print(f"[INFO] Processing attention heatmaps for layer {layer}")

      attn_maps = collect_attention_heatmaps_recent_k(
          analysis_dir,
          layer=layer,
          k=RECENT_K,
      )

      print(f"  collected {attn_maps.shape[0]} attention maps")

      avg_attn = average_attention_heatmap(attn_maps)

      plot_average_attention_heatmap(
          avg_attn,
          fig_dir,
          dataset_name,
          layer=layer,
          k=RECENT_K,
      )

      print(
          f"  attention stats: "
          f"min={np.nanmin(avg_attn):.4f}, "
          f"max={np.nanmax(avg_attn):.4f}, "
          f"mean={np.nanmean(avg_attn):.4f}"
      )

    
    for layer in range(num_layers):
      latest_attn_mr = collect_latest_attention_mixing_ratio_strict(
          analysis_dir,
          layer=layer,
      )

      print(
          f"[Layer {layer}] latest-item ATTENTION mixing ratio: "
          f"mean={latest_attn_mr.mean():.4f}, "
          f"median={np.median(latest_attn_mr):.4f}, "   # added
          f"std={latest_attn_mr.std():.4f}, "
          f"N={len(latest_attn_mr)}"
      )

      for layer in range(num_layers):
        print(f"\n[Layer {layer}] GLOBAL mixing ratio statistics (STRICT)")

        attn_vals = collect_all_generic_mixing_ratio_strict(
            analysis_dir, layer, "attn_mixing_ratio"
        )
        attnres_vals = collect_all_generic_mixing_ratio_strict(
            analysis_dir, layer, "attnres_mixing_ratio"
        )
        postln_vals = collect_all_generic_mixing_ratio_strict(
            analysis_dir, layer, "mixing_ratio"
        )

        print("  Attn-N:")
        for k, v in mixing_stats(attn_vals).items():
            print(f"    {k}: {v}")

        print("  AttnRes-N (pre-LN):")
        for k, v in mixing_stats(attnres_vals).items():
            print(f"    {k}: {v}")

        print("  AttnResLN-N (post-LN):")
        for k, v in mixing_stats(postln_vals).items():
            print(f"    {k}: {v}")

    layers_01 = [0, 1]

    print("\n[Layer0+1] GLOBAL mixing ratio statistics")

    attn_vals_01 = collect_all_layers_mixing_ratio_strict(
        analysis_dir, layers_01, "attn_mixing_ratio"
    )
    attnres_vals_01 = collect_all_layers_mixing_ratio_strict(
        analysis_dir, layers_01, "attnres_mixing_ratio"
    )
    postln_vals_01 = collect_all_layers_mixing_ratio_strict(
        analysis_dir, layers_01, "mixing_ratio"
    )

    print("  Attn-N (Layer0+1):")
    for k, v in mixing_stats(attn_vals_01).items():
        print(f"    {k}: {v}")

    print("  AttnRes-N (Layer0+1):")
    for k, v in mixing_stats(attnres_vals_01).items():
        print(f"    {k}: {v}")

    print("  AttnResLN-N (Layer0+1):")
    for k, v in mixing_stats(postln_vals_01).items():
        print(f"    {k}: {v}")


    print("\n[Attention Entropy Statistics]")

    for layer in range(num_layers):
        ent_vals = collect_all_attention_entropy_strict(
            analysis_dir,
            layer=layer,
        )

        stats = mixing_stats(ent_vals)

        print(f"[Layer {layer}] Row-wise Attention Entropy")
        for k, v in stats.items():
            print(f"    {k}: {v}")

