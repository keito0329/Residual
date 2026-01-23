"""
Plot case-wise distributions for item-embedding features.
Added (Changed) by Author
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


CASES = ["Lm1_hit_L_miss", "L_hit", "both_miss"]
FEATURES = ["dot_Lm1_L", "dot_Lm2_Lm1", "delta_last"]


def _collect(df, feature):
    data = []
    labels = []
    for c in CASES:
        vals = df.loc[df["case"] == c, feature].dropna().astype(float).values
        data.append(vals)
        labels.append(c)
    return data, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV from position_swap_item_emb_features.py")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    fig, axes = plt.subplots(2, len(FEATURES), figsize=(12, 6), sharey="row")

    for j, feat in enumerate(FEATURES):
        data, labels = _collect(df, feat)

        ax = axes[0, j]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"{feat} (box)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.2)

        ax = axes[1, j]
        parts = ax.violinplot(data, showmeans=False, showextrema=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
            pc.set_facecolor("#1f6f8b")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f"{feat} (violin)")
        ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
