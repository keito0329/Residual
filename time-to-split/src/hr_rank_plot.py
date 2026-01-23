"""
Plot histogram for HR rank distribution.
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram(input_csv: str, output_png: str, max_k: int = None):
    df = pd.read_csv(input_csv)
    if "HR_rank" not in df.columns:
        raise ValueError("Input CSV must contain column 'HR_rank'.")

    ranks = df["HR_rank"].fillna(0).astype(int)
    if max_k is None:
        max_k = int(ranks.max()) if not ranks.empty else 0

    counts = (
        ranks.value_counts()
        .reindex(range(0, max_k + 1), fill_value=0)
        .sort_index()
    )

    labels = ["miss"] + [str(i) for i in range(1, max_k + 1)]
    values = counts.to_list()

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(values)), values, color="#2f6f8f")
    plt.xticks(range(len(values)), labels)
    plt.xlabel("Rank of ground-truth item in top-K (0=miss)")
    plt.ylabel("User count")
    plt.title("HR Rank Distribution")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Per-user rank CSV path")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--max-k", type=int, default=None, help="Max K to plot")
    args = parser.parse_args()

    plot_histogram(args.input, args.output, max_k=args.max_k)


if __name__ == "__main__":
    main()
