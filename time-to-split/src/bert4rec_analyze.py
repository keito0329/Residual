"""
BERT4Rec norm analysis utilities (for norm-analysis transformers output).

Supported metrics (saved in npz):
  head_attn_n         -> layer{L}_weighted_norm        [B,H,L,L]
  attn_n              -> layer{L}_summed_weighted_norm [B,L,L]
  attnres_n           -> layer{L}_residual_weighted_norm [B,L,L]
  attnresln_n         -> layer{L}_post_ln_norm         [B,L,L]
  attn_n_ratio        -> layer{L}_attn_mixing_ratio    [B,L]
  attnres_n_ratio     -> layer{L}_attnres_mixing_ratio [B,L]
  attnresln_n_ratio   -> layer{L}_mixing_ratio         [B,L]
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


PADDING_IDX = 0
RECENT_K = 15
FIGSIZE = (5, 5)


METRIC_KEYS: Dict[str, str] = {
    "head_attn_n": "weighted_norm",
    "attn_n": "summed_weighted_norm",
    "attnres_n": "residual_weighted_norm",
    "attnresln_n": "post_ln_norm",
    "attn_n_ratio": "attn_mixing_ratio",
    "attnres_n_ratio": "attnres_mixing_ratio",
    "attnresln_n_ratio": "mixing_ratio",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_all_batches(analysis_dir: str):
    paths = sorted(glob.glob(os.path.join(analysis_dir, "batch*.npz")))
    if not paths:
        raise RuntimeError(f"No batch files in {analysis_dir}")
    return [np.load(p) for p in paths]


def detect_num_layers(analysis_dir: str) -> int:
    first = load_all_batches(analysis_dir)[0]
    keys = [k for k in first.files if k.startswith("layer") and k.endswith("_post_ln_norm")]
    layer_ids = []
    for k in keys:
        try:
            idx = int(k.split("_")[0].replace("layer", ""))
            layer_ids.append(idx)
        except Exception:
            continue
    if not layer_ids:
        raise RuntimeError("Could not detect layers from npz keys.")
    return max(layer_ids) + 1


def extract_valid(ids: np.ndarray, x: np.ndarray):
    idx = np.where(ids != PADDING_IDX)[0]
    if x.ndim == 1:
        return x[idx]
    if x.ndim == 2:
        return x[np.ix_(idx, idx)]
    raise ValueError("Unsupported ndim")


def reverse_seq(x: np.ndarray):
    if x.ndim == 1:
        return x[::-1]
    if x.ndim == 2:
        return x[::-1, ::-1]
    raise ValueError("Unsupported ndim")


def restrict_to_recent(x: np.ndarray, k: int):
    if x.ndim == 1:
        return x[-k:]
    if x.ndim == 2:
        return x[-k:, -k:]
    raise ValueError("Unsupported ndim")


def to_fixed_kxk(mat: np.ndarray, k: int):
    out = np.full((k, k), np.nan, dtype=np.float32)
    lp = min(mat.shape[0], k)
    out[:lp, :lp] = mat[:lp, :lp]
    return out


def _key(layer: int, metric: str) -> str:
    if metric not in METRIC_KEYS:
        raise ValueError(f"Unsupported metric: {metric}")
    return f"layer{layer}_{METRIC_KEYS[metric]}"


def collect_all_ratio_strict(analysis_dir: str, layer: int, metric: str) -> np.ndarray:
    if metric not in ("attn_n_ratio", "attnres_n_ratio", "attnresln_n_ratio"):
        raise ValueError("All-item ratio collection supports only *_ratio metrics.")

    batches = load_all_batches(analysis_dir)
    vals: List[float] = []
    key = _key(layer, metric)
    for d in batches:
        ids = d["input_ids"]
        arr = d[key]
        for i in range(ids.shape[0]):
            idx = np.where(ids[i] != PADDING_IDX)[0]
            if len(idx) == 0:
                continue
            vals.extend(arr[i, idx].tolist())
    return np.array(vals, dtype=np.float32)


def collect_heatmaps_recent_k(
    analysis_dir: str,
    layer: int,
    metric: str,
    k: int = RECENT_K,
    head: Optional[int] = None,
    avg_heads: bool = False,
) -> np.ndarray:
    if metric == "head_attn_n":
        if head is None and not avg_heads:
            raise ValueError("Specify head or set avg_heads=True for head_attn_n.")
    key = _key(layer, metric)
    batches = load_all_batches(analysis_dir)
    heatmaps = []

    for d in batches:
        ids = d["input_ids"]
        arr = d[key]

        if metric == "head_attn_n":
            if avg_heads:
                arr = arr.mean(axis=1)  # [B,L,L]
            else:
                arr = arr[:, head, :, :]  # [B,L,L]

        for i in range(ids.shape[0]):
            mat = arr[i]
            mat = extract_valid(ids[i], mat)
            if mat.size == 0:
                continue
            mat = reverse_seq(mat)
            mat = restrict_to_recent(mat, k)
            mat = to_fixed_kxk(mat, k)
            heatmaps.append(mat)

    if not heatmaps:
        raise RuntimeError("No heatmaps collected.")
    return np.stack(heatmaps, axis=0)


def average_heatmap(heatmaps: np.ndarray) -> np.ndarray:
    return np.nanmean(heatmaps, axis=0)


def plot_average_heatmap(
    heatmap: np.ndarray,
    out_path: str,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=FIGSIZE)
    im = plt.imshow(heatmap, origin="upper", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("To")
    plt.ylabel("From")
    # No colorbar (remove the right bar entirely)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _pretty_metric_title(metric: str, head: Optional[int] = None) -> str:
    mapping = {
        "attn_n": "Attn",
        "attnres_n": "AttnRes",
        "attnresln_n": "AttnResLN",
        "head_attn_n": "HeadAttn",
    }
    base = mapping.get(metric, metric)
    if metric == "head_attn_n" and head is not None:
        return f"BERT4Rec {base} head{head}"
    return f"BERT4Rec {base}"


def summarize_all_ratio(analysis_dir: str, layer: int, metric: str) -> Tuple[float, float, int]:
    vals = collect_all_ratio_strict(analysis_dir, layer, metric)
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    return float(vals.mean()), float(vals.std()), int(vals.size)


def run_all_summaries(
    analysis_dir: str,
    layer: int,
    recent_k: int,
    out_dir: str,
    head: Optional[int] = None,
    avg_heads: bool = True,
):
    print(f"[INFO] analysis_dir={analysis_dir}")
    print(f"[INFO] layer={layer} recent_k={recent_k}")

    ratio_metrics = ["attn_n_ratio", "attnres_n_ratio", "attnresln_n_ratio"]
    for metric in ratio_metrics:
        mean, std, n = summarize_all_ratio(analysis_dir, layer, metric)
        print(f"[all_ratio] metric={metric} mean={mean:.6f} std={std:.6f} n={n}")

    heatmap_metrics = ["attn_n", "attnres_n", "attnresln_n"]
    for metric in heatmap_metrics:
        heatmaps = collect_heatmaps_recent_k(
            analysis_dir,
            layer,
            metric,
            k=recent_k,
        )
        avg = average_heatmap(heatmaps)
        out_path = os.path.join(out_dir, f"{metric}_layer{layer}.pdf")
        title = _pretty_metric_title(metric)
        plot_average_heatmap(avg, out_path, title=title)
        print(f"[Saved] {out_path}")

    head_metric = "head_attn_n"
    heatmaps = collect_heatmaps_recent_k(
        analysis_dir,
        layer,
        head_metric,
        k=recent_k,
        head=head,
        avg_heads=avg_heads,
    )
    avg = average_heatmap(heatmaps)
    head_tag = "avg_heads" if avg_heads else f"head{head}"
    out_path = os.path.join(out_dir, f"{head_metric}_{head_tag}_layer{layer}.pdf")
    title = _pretty_metric_title(head_metric, head=head if not avg_heads else None)
    plot_average_heatmap(avg, out_path, title=title)
    print(f"[Saved] {out_path}")


def run_all_layers_summaries(
    analysis_dir: str,
    recent_k: int,
    out_dir: str,
    head: Optional[int] = None,
    avg_heads: bool = True,
):
    num_layers = detect_num_layers(analysis_dir)
    print(f"[INFO] analysis_dir={analysis_dir}")
    print(f"[INFO] layers=0..{num_layers - 1} recent_k={recent_k}")

    ratio_metrics = ["attn_n_ratio", "attnres_n_ratio", "attnresln_n_ratio"]
    layer_means: Dict[str, List[float]] = {m: [] for m in ratio_metrics}

    for layer in range(num_layers):
        print(f"[INFO] layer={layer}")
        for metric in ratio_metrics:
            mean, std, n = summarize_all_ratio(analysis_dir, layer, metric)
            layer_means[metric].append(mean)
            print(f"[all_ratio] metric={metric} mean={mean:.6f} std={std:.6f} n={n}")

        heatmap_metrics = ["attn_n", "attnres_n", "attnresln_n"]
        for metric in heatmap_metrics:
            heatmaps = collect_heatmaps_recent_k(
                analysis_dir,
                layer,
                metric,
                k=recent_k,
            )
            avg = average_heatmap(heatmaps)
            out_path = os.path.join(out_dir, f"{metric}_layer{layer}.pdf")
            title = _pretty_metric_title(metric)
            plot_average_heatmap(avg, out_path, title=title)
            print(f"[Saved] {out_path}")

        head_metric = "head_attn_n"
        heatmaps = collect_heatmaps_recent_k(
            analysis_dir,
            layer,
            head_metric,
            k=recent_k,
            head=head,
            avg_heads=avg_heads,
        )
        avg = average_heatmap(heatmaps)
        head_tag = "avg_heads" if avg_heads else f"head{head}"
        out_path = os.path.join(out_dir, f"{head_metric}_{head_tag}_layer{layer}.pdf")
        title = _pretty_metric_title(head_metric, head=head if not avg_heads else None)
        plot_average_heatmap(avg, out_path, title=title)
        print(f"[Saved] {out_path}")

    for metric in ratio_metrics:
        vals = np.array(layer_means[metric], dtype=np.float32)
        if vals.size == 0:
            continue
        overall_mean = float(vals.mean())
        print(f"[layer_avg] metric={metric} mean_over_layers={overall_mean:.6f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="BERT4Rec norm analysis utilities")
    ap.add_argument("--analysis_dir", type=str, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--metric", type=str, default="attnresln_n",
                    choices=list(METRIC_KEYS.keys()))
    ap.add_argument("--recent_k", type=int, default=RECENT_K)
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--avg_heads", action="store_true")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--all_ratio", action="store_true")
    ap.add_argument("--all", action="store_true", help="Run all metrics and plots at once")
    ap.add_argument("--all_layers", action="store_true", help="Run all layers (metrics + plots)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.analysis_dir, "figures_bert4rec")

    if args.all_layers:
        run_all_layers_summaries(
            args.analysis_dir,
            args.recent_k,
            out_dir,
            head=args.head,
            avg_heads=args.avg_heads or args.head is None,
        )
    elif args.all:
        run_all_summaries(
            args.analysis_dir,
            args.layer,
            args.recent_k,
            out_dir,
            head=args.head,
            avg_heads=args.avg_heads or args.head is None,
        )
    elif args.all_ratio:
        if not args.metric.endswith("_ratio"):
            raise ValueError("Use *_ratio metric for --all_ratio.")
        mean, std, n = summarize_all_ratio(args.analysis_dir, args.layer, args.metric)
        print(f"[all_ratio] metric={args.metric} layer={args.layer} mean={mean:.6f} std={std:.6f} n={n}")
    else:
        heatmaps = collect_heatmaps_recent_k(
            args.analysis_dir,
            args.layer,
            args.metric,
            k=args.recent_k,
            head=args.head,
            avg_heads=args.avg_heads,
        )
        avg = average_heatmap(heatmaps)
        if args.plot:
            title = _pretty_metric_title(args.metric, head=args.head)
            out_path = os.path.join(out_dir, f"{args.metric}_layer{args.layer}.pdf")
            plot_average_heatmap(avg, out_path, title=title)
            print(f"[Saved] {out_path}")
        else:
            print(f"[INFO] avg heatmap shape: {avg.shape}")
