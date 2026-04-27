#!/usr/bin/env python
"""Summarize Attn/AttnRes/AttnResLN mixing ratios from analysis npz files."""

import argparse
import csv
import glob
import os
import re
from typing import Dict, Iterable, List

import numpy as np


RATIO_KEYS = {
    "Attn": "attn_mixing_ratio",
    "AttnRes": "attnres_mixing_ratio",
    "AttnResLN": "mixing_ratio",
}


def load_batches(analysis_dir: str):
    paths = sorted(glob.glob(os.path.join(analysis_dir, "batch*.npz")))
    if not paths:
        raise RuntimeError(f"No batch files found in {analysis_dir}")
    return [(path, np.load(path)) for path in paths]


def detect_layers(batches) -> List[int]:
    layer_ids = set()
    pattern = re.compile(r"^layer(\d+)_(attn_mixing_ratio|attnres_mixing_ratio|mixing_ratio)$")
    for _, batch in batches:
        for key in batch.files:
            match = pattern.match(key)
            if match:
                layer_ids.add(int(match.group(1)))
    if not layer_ids:
        raise RuntimeError("Could not detect layer ratio keys from npz files.")
    return sorted(layer_ids)


def valid_mask(input_ids: np.ndarray) -> np.ndarray:
    return input_ids != 0


def collect_all(batches, layer: int, key_suffix: str) -> np.ndarray:
    key = f"layer{layer}_{key_suffix}"
    vals = []
    for path, batch in batches:
        if "input_ids" not in batch.files:
            raise RuntimeError(f"input_ids not found in {path}")
        if key not in batch.files:
            raise RuntimeError(f"{key} not found in {path}")
        ids_all = batch["input_ids"]
        ratio_all = batch[key]
        vals.append(ratio_all[valid_mask(ids_all)])
    if not vals:
        return np.array([], dtype=np.float64)
    return np.concatenate(vals).astype(np.float64)


def collect_latest(batches, layer: int, key_suffix: str) -> np.ndarray:
    key = f"layer{layer}_{key_suffix}"
    vals = []
    for path, batch in batches:
        if "input_ids" not in batch.files:
            raise RuntimeError(f"input_ids not found in {path}")
        if key not in batch.files:
            raise RuntimeError(f"{key} not found in {path}")
        ids_all = batch["input_ids"]
        ratio_all = batch[key]
        lengths = valid_mask(ids_all).sum(axis=1)
        keep = lengths > 0
        if np.any(keep):
            rows = np.nonzero(keep)[0]
            cols = lengths[keep] - 1
            vals.append(ratio_all[rows, cols])
    if not vals:
        return np.array([], dtype=np.float64)
    return np.concatenate(vals).astype(np.float64)


def stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "zero_rate": np.nan,
            "one_rate": np.nan,
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "zero_rate": float(np.mean(np.isclose(values, 0.0))),
        "one_rate": float(np.mean(np.isclose(values, 1.0))),
    }


def add_row(rows: List[Dict], dataset: str, layer: str, variant: str, scope: str, values: np.ndarray):
    row = {
        "dataset": dataset,
        "layer": layer,
        "variant": variant,
        "scope": scope,
    }
    row.update(stats(values))
    rows.append(row)


def print_rows(rows: Iterable[Dict]):
    for row in rows:
        print(
            f"{row['dataset']}\t{row['layer']}\t{row['variant']}\t{row['scope']}\t"
            f"n={row['n']}\tmean={row['mean']:.6f}\tmedian={row['median']:.6f}\t"
            f"std={row['std']:.6f}"
        )


def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "dataset",
        "layer",
        "variant",
        "scope",
        "n",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "zero_rate",
        "one_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    batches = load_batches(args.analysis_dir)
    layers = detect_layers(batches)
    dataset = args.dataset or os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(args.analysis_dir)))
    )

    rows = []
    for layer in layers:
        for variant, key_suffix in RATIO_KEYS.items():
            add_row(rows, dataset, str(layer), variant, "latest", collect_latest(batches, layer, key_suffix))
            add_row(rows, dataset, str(layer), variant, "all", collect_all(batches, layer, key_suffix))

    for variant, key_suffix in RATIO_KEYS.items():
        latest_vals = [collect_latest(batches, layer, key_suffix) for layer in layers]
        all_vals = [collect_all(batches, layer, key_suffix) for layer in layers]
        add_row(rows, dataset, "all_layers", variant, "latest", np.concatenate(latest_vals))
        add_row(rows, dataset, "all_layers", variant, "all", np.concatenate(all_vals))

    print_rows(rows)

    if args.out_csv:
        write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()
