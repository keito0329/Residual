"""
Analyze effective sequence length distribution from analysis npz files.

- padding (input_ids == 0) is ignored
- sequence length = number of non-padding tokens
- saves histogram figure
- prints detailed statistics

SAFE for paper use.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

PADDING_IDX = 0


# ======================================================
# Utilities
# ======================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_all_batches(analysis_dir):
    paths = sorted(glob.glob(os.path.join(analysis_dir, "batch*.npz")))
    if not paths:
        raise RuntimeError(f"No batch files in {analysis_dir}")
    return [np.load(p) for p in paths]


# ======================================================
# Core logic
# ======================================================
def collect_sequence_lengths(analysis_dir):
    """
    Returns:
        lengths: np.ndarray of shape [N]
    """
    batches = load_all_batches(analysis_dir)

    lengths = []
    for batch in batches:
        input_ids = batch["input_ids"]  # [B, L]
        # count non-padding per sequence
        lens = (input_ids != PADDING_IDX).sum(axis=1)
        lengths.append(lens)

    return np.concatenate(lengths)


def summarize_lengths(lengths):
    return {
        "count": len(lengths),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
    }


# ======================================================
# Visualization
# ======================================================
def plot_length_histogram(
    lengths,
    fig_dir,
    max_len: int = 50,
):
    ensure_dir(fig_dir)

    plt.figure(figsize=(6, 4))
    plt.hist(
        lengths,
        bins=np.arange(0, max_len + 2) - 0.5,
        edgecolor="black",
    )
    plt.xlabel("Effective sequence length (non-padding)")
    plt.ylabel("Count")
    plt.title("Sequence Length Distribution")
    plt.xticks(range(0, max_len + 1, 5))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "sequence_length_histogram.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {path}")


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    # >>> CHANGE HERE <<<
    analysis_dir = "/app/data/results/analysis/LightSASRecAnalyze/Beauty/leave-one-out/seed_17"
    # >>>>>>>>>>>>>>>>>>

    fig_dir = os.path.join(analysis_dir, "figures_strict_50x50")
    ensure_dir(fig_dir)

    lengths = collect_sequence_lengths(analysis_dir)
    stats = summarize_lengths(lengths)

    print("=== Sequence Length Statistics ===")
    for k, v in stats.items():
        print(f"{k:>8}: {v}")

    plot_length_histogram(lengths, fig_dir, max_len=50)
