"""
Scatter plot for HR@10 vs HRLI@1 from metrics CSVs.
"""

import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_metric(csv_path: str, metric_name: str):
    df = pd.read_csv(csv_path)
    row = df.loc[df["metric_name"] == metric_name, "metric_value"]
    if row.empty:
        return None
    return float(row.iloc[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True, help="Glob for metrics CSVs")
    parser.add_argument("--x-metric", default="test_last_HitRate@10")
    parser.add_argument("--y-metric", default="test_last_HRLI@1")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--points-csv", default=None, help="Optional CSV to save points")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.glob)):
        x = load_metric(path, args.x_metric)
        y = load_metric(path, args.y_metric)
        if x is None or y is None:
            continue
        rows.append({"path": path, "x": x, "y": y})

    if not rows:
        raise RuntimeError("No points found. Check glob or metric names.")

    points = pd.DataFrame(rows)
    plt.figure(figsize=(6, 5))
    plt.scatter(points["x"], points["y"], s=30, alpha=0.8, color="#1f6f8b")
    plt.xlabel(args.x_metric)
    plt.ylabel(args.y_metric)
    plt.title("HR@10 vs HRLI@1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)

    if args.points_csv:
        points.to_csv(args.points_csv, index=False)


if __name__ == "__main__":
    main()
