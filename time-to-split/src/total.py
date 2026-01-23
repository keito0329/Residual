# -*- coding: utf-8 -*-
"""
Compute and visualize M_total from LightSASRecAnalyze saved npz.

Definition (recommended):
  A_tilde^{l} = RowNorm(I + A^{l})
  M_total     = A_tilde^{L-1} ... A_tilde^{0}

Works with keys:
  - num_layers
  - layer{i}_attention : [B, Hh, L, L]  (or [B, 1, L, L] in fallback)
  - input_ids          : [B, L] (optional but recommended)

Usage:
  python src/analyze_mtotal.py \
    --npz data/results/.../analysis_batch0.npz \
    --sample 0 --head -1 --mode residual_rownorm \
    --out_dir ./mtotal_out --prefix demo

Notes:
- If you want head-specific, set --head 0..Hh-1
- If you want head-avg, use --head -1
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_npz(path: str):
    d = np.load(path, allow_pickle=False)
    return d


def get_num_layers(d):
    if "num_layers" in d:
        return int(d["num_layers"].reshape(-1)[0])
    # fallback: infer
    i = 0
    while f"layer{i}_attention" in d:
        i += 1
    return i


def row_normalize(mat: np.ndarray, eps: float = 1e-12):
    """
    mat: [..., L, L]
    """
    s = mat.sum(axis=-1, keepdims=True)
    return mat / (s + eps)


def make_residual_rownorm_attention(A: np.ndarray, add_identity: bool = True):
    """
    A: [B, L, L] attention probs (row-stochastic-ish already, but we add I then renorm)
    returns: [B, L, L] row-normalized (I + A)
    """
    B, L, _ = A.shape
    if add_identity:
        I = np.eye(L, dtype=A.dtype)[None, :, :]
        M = A + I
    else:
        M = A
    return row_normalize(M)


def make_residual_weighted_attention(A: np.ndarray, alpha: float = 1.0):
    """
    Another option:
      M = RowNorm(I + alpha * A)
    alpha can emphasize attention vs residual.
    """
    B, L, _ = A.shape
    I = np.eye(L, dtype=A.dtype)[None, :, :]
    M = I + alpha * A
    return row_normalize(M)


def pick_head_attention(attn: np.ndarray, head: int):
    """
    attn: [B, Hh, L, L] or [B, 1, L, L] or [B, L, L]
    head:
      -1 -> average over heads
       k -> use that head
    return: [B, L, L]
    """
    if attn.ndim == 3:
        return attn
    if attn.ndim != 4:
        raise ValueError(f"Unexpected attention ndim={attn.ndim}, shape={attn.shape}")

    B, Hh, L, L2 = attn.shape
    assert L == L2

    if head < 0:
        return attn.mean(axis=1)
    if head >= Hh:
        raise ValueError(f"head={head} out of range (Hh={Hh})")
    return attn[:, head, :, :]


def compute_m_total_from_npz(d, head: int = -1, mode: str = "residual_rownorm", alpha: float = 1.0):
    """
    Returns:
      M_total: [B, L, L]
      Atilde_list: list of [B, L, L] per layer (after residual+norm)
      A_list: list of [B, L, L] per layer (raw attention chosen)
    """
    num_layers = get_num_layers(d)
    # attention shapes are derived from layer0
    attn0 = d["layer0_attention"]
    A0 = pick_head_attention(attn0, head=head)  # [B,L,L]
    B, L, _ = A0.shape

    A_list = []
    Atilde_list = []

    for i in range(num_layers):
        attn = d[f"layer{i}_attention"]
        A = pick_head_attention(attn, head=head).astype(np.float64)  # stable multiply
        A_list.append(A)

        if mode == "residual_rownorm":
            Atilde = make_residual_rownorm_attention(A, add_identity=True)
        elif mode == "residual_weighted":
            Atilde = make_residual_weighted_attention(A, alpha=alpha)
        elif mode == "plain_rownorm":
            # no residual: just renorm A (usually already row-sum=1, but safe)
            Atilde = row_normalize(A)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        Atilde_list.append(Atilde)

    # Multiply across layers: M_total = Atilde_{L-1} ... Atilde_0
    M = np.tile(np.eye(L, dtype=np.float64)[None, :, :], (B, 1, 1))
    for Atilde in Atilde_list:
        # apply in forward order: after layer0 then layer1 ...
        # If you want exact: M = Atilde_{k} ... Atilde_0, start from I and left-multiply:
        # newM = Atilde @ M
        M = np.matmul(Atilde, M)

    return M.astype(np.float32), [x.astype(np.float32) for x in Atilde_list], [x.astype(np.float32) for x in A_list]


def diag_mass(M: np.ndarray):
    """M: [B,L,L] -> [B] mean diagonal weight"""
    B, L, _ = M.shape
    d = np.stack([np.diag(M[b]) for b in range(B)], axis=0)  # [B,L]
    return d.mean(axis=1)  # [B]


def row_entropy(M: np.ndarray, eps: float = 1e-12):
    """
    entropy per row then average over rows:
      H = -sum p log p
    M: [B,L,L] row-stochastic
    returns: [B]
    """
    P = np.clip(M, eps, 1.0)
    H = -(P * np.log(P)).sum(axis=-1)  # [B,L]
    return H.mean(axis=1)


def plot_heatmap(mat2d: np.ndarray, title: str, out_path: str, vmax: float = None):
    plt.figure(figsize=(6, 5))
    if vmax is None:
        plt.imshow(mat2d, aspect="auto")
    else:
        plt.imshow(mat2d, aspect="auto", vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("source position s")
    plt.ylabel("target position t")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./mtotal_out")
    ap.add_argument("--prefix", type=str, default="mtotal")
    ap.add_argument("--sample", type=int, default=0, help="batch index to visualize")
    ap.add_argument("--head", type=int, default=-1, help="-1=head avg, else head id")
    ap.add_argument("--mode", type=str, default="residual_rownorm",
                    choices=["residual_rownorm", "residual_weighted", "plain_rownorm"])
    ap.add_argument("--alpha", type=float, default=1.0, help="used if mode=residual_weighted")
    ap.add_argument("--vmax", type=float, default=None, help="fixed vmax for heatmaps")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    d = load_npz(args.npz)
    num_layers = get_num_layers(d)

    M_total, Atilde_list, A_list = compute_m_total_from_npz(
        d, head=args.head, mode=args.mode, alpha=args.alpha
    )

    B, L, _ = M_total.shape
    s = args.sample
    if not (0 <= s < B):
        raise ValueError(f"--sample {s} out of range, B={B}")

    # metrics
    dm = diag_mass(M_total)[s]
    ent = row_entropy(M_total)[s]

    # save matrices
    np.save(os.path.join(args.out_dir, f"{args.prefix}_M_total.npy"), M_total)
    # also save sample matrix
    np.save(os.path.join(args.out_dir, f"{args.prefix}_M_total_sample{s}.npy"), M_total[s])

    # plot per-layer residual-aware maps
    for i in range(num_layers):
        plot_heatmap(
            Atilde_list[i][s],
            title=f"A_tilde layer {i} (mode={args.mode}, head={args.head})",
            out_path=os.path.join(args.out_dir, f"{args.prefix}_Atilde_layer{i}_sample{s}.png"),
            vmax=args.vmax
        )

    # plot M_total
    plot_heatmap(
        M_total[s],
        title=f"M_total (layers={num_layers}, mode={args.mode}, head={args.head})\n"
              f"diag_mass={dm:.4f}, row_entropy={ent:.4f}",
        out_path=os.path.join(args.out_dir, f"{args.prefix}_M_total_sample{s}.png"),
        vmax=args.vmax
    )

    # also plot "distance profile" from each target t (expected source index)
    src_idx = np.arange(L, dtype=np.float32)[None, :]  # [1,L]
    expected_src = (M_total[s] * src_idx).sum(axis=-1)  # [L]
    plt.figure(figsize=(7, 3))
    plt.plot(np.arange(L), expected_src)
    plt.title("Expected source position per target t (from M_total)")
    plt.xlabel("target position t")
    plt.ylabel("E[source s]")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.prefix}_expected_source_sample{s}.png"), dpi=200)
    plt.close()

    print(f"Loaded: {args.npz}")
    print(f"num_layers: {num_layers}, B={B}, L={L}, head={args.head}, mode={args.mode}")
    print(f"Saved to: {args.out_dir}")
    print(f"Sample {s}: diag_mass={dm:.6f}, row_entropy={ent:.6f}")


if __name__ == "__main__":
    main()
