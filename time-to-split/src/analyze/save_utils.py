# src/analysis/save_utils.py
import os
import numpy as np
import torch


def save_analysis_batch(
    analysis: dict,
    input_ids: torch.Tensor,
    out_dir: str,
    batch_idx: int,
):
    """
    analysis:
        {
          "layer_attn_raw": List[Tensor[B,L,L]],
          "giant_attn_raw": Tensor[B,L,L],
          "last_ln_scale":  Tensor[B,L]
        }
    """

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "layer_attn"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "giant_attn"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "last_ln"), exist_ok=True)

    pad_mask = (input_ids == 0)  # [B, L]

    # ===============================
    # layer-wise attention
    # ===============================
    for l, A in enumerate(analysis["layer_attn_raw"]):
        # A: [B, L, L]
        A = A.masked_fill(pad_mask.unsqueeze(1), 0.0)
        A_mean = A.mean(dim=0)  # batch mean → [L, L]
        np.save(
            os.path.join(out_dir, "layer_attn", f"layer{l}_batch{batch_idx}.npy"),
            A_mean.detach().cpu().numpy(),
        )

    # ===============================
    # giant attention
    # ===============================
    G = analysis["giant_attn_raw"]  # [B, L, L]
    G = G.masked_fill(pad_mask.unsqueeze(1), 0.0)
    G_mean = G.mean(dim=0)
    np.save(
        os.path.join(out_dir, "giant_attn", f"giant_batch{batch_idx}.npy"),
        G_mean.detach().cpu().numpy(),
    )

    # ===============================
    # last LayerNorm scale
    # ===============================
    ln = analysis["last_ln_scale"]  # [B, L]
    ln = ln.masked_fill(pad_mask, 0.0)
    ln_mean = ln.mean(dim=0)
    np.save(
        os.path.join(out_dir, "last_ln", f"ln_scale_batch{batch_idx}.npy"),
        ln_mean.detach().cpu().numpy(),
    )
