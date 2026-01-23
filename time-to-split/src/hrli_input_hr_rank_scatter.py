"""
Scatter plot: HRLI_input@1 vs HR_rank (per-user).
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with HRLI_input@1 and HR_rank")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "HRLI_input@1" not in df.columns or "HR_rank" not in df.columns:
        raise ValueError("Input CSV must contain 'HRLI_input@1' and 'HR_rank'.")

    miss_mask = df["HR_rank"] == 0
    hit_mask = ~miss_mask

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    if hit_mask.any():
        hb = ax.hexbin(
            df.loc[hit_mask, "HRLI_input@1"],
            df.loc[hit_mask, "HR_rank"],
            gridsize=40,
            cmap="Blues",
            mincnt=1,
        )
        fig.colorbar(hb, ax=ax, label="User count")
    if miss_mask.any():
        ax.scatter(
            df.loc[miss_mask, "HRLI_input@1"],
            df.loc[miss_mask, "HR_rank"],
            s=20,
            alpha=0.7,
            color="#d95f02",
            marker="x",
            label="miss (rank=0)",
        )
        ax.legend(frameon=False)

    ax.set_xlabel("HRLI_input@1 (per user)")
    ax.set_ylabel("HR_rank")
    ax.set_title("HRLI_input@1 vs HR_rank (density view)")
    ax.grid(True, alpha=0.2)

    ax_hist = axes[1]
    bins = 20
    if hit_mask.any():
        ax_hist.hist(
            df.loc[hit_mask, "HRLI_input@1"],
            bins=bins,
            alpha=0.9,
            color="#1f6f8b",
            label="hit",
            histtype="step",
            linewidth=1.8,
        )
    if miss_mask.any():
        ax_hist.hist(
            df.loc[miss_mask, "HRLI_input@1"],
            bins=bins,
            alpha=0.9,
            color="#d95f02",
            label="miss (rank=0)",
            histtype="step",
            linewidth=1.8,
        )
    ax_hist.set_xlabel("HRLI_input@1 (per user)")
    ax_hist.set_ylabel("User count")
    ax_hist.legend(frameon=False)
    ax_hist.grid(True, alpha=0.2)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
