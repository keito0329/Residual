"""
validationデータを使ってresidual scaling係数 alpha を自動選択する。

【選択基準】
  alpha* = argmax_alpha Recovery_val(alpha)
  subject to: HR@K_val(alpha) >= (1 - epsilon) * HR@K_val(alpha=1.0)

【recoverable_base固定の理由】
  alpha=1.0 を基準として「前の位置の情報で回収できるシーケンス」を定義する。
  alphaごとに再定義すると比較基準がずれ、各alphaの改善量を公平に評価できない。

【testを使ってはいけない理由】
  testはalpha選択後に1度だけ最終評価に使う。
  testでalpha選択するとtest setへの過適合（情報漏洩）が生じる。

【このalpha選択の目的】
  精度（HR@K）を保ちつつ、前の位置（L-1, L-2, L-3）の情報を活用するalphaを選ぶ。
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _hit_series(hr_rank_df: pd.DataFrame, k: int) -> pd.Series:
    """user_idをindexとした hit@k (bool) Series。rank==0 はmiss。"""
    s = hr_rank_df.set_index("user_id")["HR_rank"]
    return (s > 0) & (s <= k)


def build_recoverable_base(
    hit_L: pd.Series,
    hit_Lm1: pd.Series,
    hit_Lm2: pd.Series,
    hit_Lm3: pd.Series,
) -> Set:
    """
    recoverable_base を構築する（alpha=1.0 で1度だけ定義し、以降固定する）。

    条件:
      - alpha=1.0 で L 位置が miss
      - かつ L-1, L-2, L-3 のいずれかで hit

    short sequence（長さ <= offset）の場合、対応する hit_Lm* には NaN が入る。
    fillna(False) で安全に False として扱う。
    """
    miss_L = ~hit_L.fillna(False)
    idx = hit_L.index
    hit_any_prev = (
        hit_Lm1.reindex(idx).fillna(False) |
        hit_Lm2.reindex(idx).fillna(False) |
        hit_Lm3.reindex(idx).fillna(False)
    )
    return set(idx[miss_L & hit_any_prev])


def compute_recovery(
    hit_L_alpha: pd.Series,
    recoverable_base: Set,
    reference_index: pd.Index,
) -> float:
    """
    Recovery(alpha) = recoverable_base 内ユーザーのうち、
                      このalphaでL位置がhitになった割合。

    recoverable_base が空の場合は 0.0 を返す。
    recoverable_base は固定なので異なるalpha間で公平に比較できる。
    精度を保ちながら前の位置の情報を活用するalphaを選ぶための指標。
    """
    if not recoverable_base:
        return 0.0
    base_in_ref = [u for u in recoverable_base if u in reference_index]
    if not base_in_ref:
        return 0.0
    hits = hit_L_alpha.reindex(base_in_ref).fillna(False)
    return float(hits.mean())


def select_alpha_with_recovery_constraint(
    predict_at_alpha_offset: Callable[[float, int], pd.DataFrame],
    compute_hr_rank_per_user_fn: Callable,
    val_ground_truth: pd.DataFrame,
    input_lengths: pd.Series,
    alpha_candidates: Optional[List[float]] = None,
    epsilon: float = 0.01,
    top_k: int = 10,
) -> Tuple[float, Dict]:
    """
    validationデータのみを使ってalpha*を選択する。

    Args:
        predict_at_alpha_offset: (alpha, offset) -> recs DataFrame のcallable
        compute_hr_rank_per_user_fn: (recs, ground_truth, top_k_list) -> DataFrame
        val_ground_truth: validation ground truth DataFrame
        input_lengths: user_id をindexとしたシーケンス長 Series
        alpha_candidates: alphaの候補（デフォルト: 0.0〜1.0, 0.1刻み）
        epsilon: HR制約の許容劣化幅（デフォルト: 0.01）
        top_k: 評価するK

    Returns:
        best_alpha: 選ばれたalpha
        log: ログ辞書
    """
    if alpha_candidates is None:
        alpha_candidates = [round(i * 0.1, 1) for i in range(11)]

    top_k_list = [top_k]

    print(f"\n[AlphaSelection] ===== alpha自動選択開始 =====")
    print(f"[AlphaSelection] 候補: {alpha_candidates}")
    print(f"[AlphaSelection] epsilon={epsilon}, top_k={top_k}")

    # ===== Step1: alpha=1.0 で L, L-1, L-2, L-3 の hit情報を収集 =====
    print("[AlphaSelection] Step1: alpha=1.0 でbase評価（L, L-1, L-2, L-3）")

    recs_base = predict_at_alpha_offset(1.0, 0)
    rank_L_base = compute_hr_rank_per_user_fn(recs_base, val_ground_truth, top_k_list)
    hit_L_base = _hit_series(rank_L_base, top_k)
    all_users = hit_L_base.index

    def _hit_at_offset(offset: int) -> pd.Series:
        """offset位置のhit情報を取得。short sequenceはNaN処理。"""
        recs_off = predict_at_alpha_offset(1.0, offset)
        rank_off = compute_hr_rank_per_user_fn(recs_off, val_ground_truth, top_k_list)
        hit_off = _hit_series(rank_off, top_k).reindex(all_users)
        # short sequence（シーケンス長 <= offset）のユーザーをNaNに
        if input_lengths is not None:
            short_users = input_lengths[input_lengths <= offset].index
            hit_off.loc[hit_off.index.isin(short_users)] = np.nan
        return hit_off

    hit_Lm1_base = _hit_at_offset(1)
    hit_Lm2_base = _hit_at_offset(2)
    hit_Lm3_base = _hit_at_offset(3)

    # ===== Step2: recoverable_base を1度だけ定義（以降固定） =====
    recoverable_base = build_recoverable_base(
        hit_L_base, hit_Lm1_base, hit_Lm2_base, hit_Lm3_base
    )
    hr_base = float(hit_L_base.fillna(False).mean())
    hr_threshold = (1.0 - epsilon) * hr_base

    print(f"[AlphaSelection] alpha=1.0: HR@{top_k}={hr_base:.4f}")
    print(f"[AlphaSelection] HR制約閾値: {hr_threshold:.4f}  (= (1-{epsilon}) x {hr_base:.4f})")
    print(f"[AlphaSelection] recoverable_base: {len(recoverable_base)} / {len(all_users)} ユーザー")

    # ===== Step3: 各alphaでfinal position（offset=0）のみ評価 =====
    alpha_logs: Dict[float, Dict] = {}

    for alpha in alpha_candidates:
        if alpha == 1.0:
            # 既に計算済みの結果を再利用（無駄な再計算を避ける）
            hit_L_alpha = hit_L_base
            hr_alpha = hr_base
        else:
            recs_alpha = predict_at_alpha_offset(alpha, 0)
            rank_L_alpha = compute_hr_rank_per_user_fn(recs_alpha, val_ground_truth, top_k_list)
            hit_L_alpha = _hit_series(rank_L_alpha, top_k).reindex(all_users)
            hr_alpha = float(hit_L_alpha.fillna(False).mean())

        recovery = compute_recovery(hit_L_alpha, recoverable_base, all_users)
        constraint_ok = hr_alpha >= hr_threshold

        alpha_logs[alpha] = {
            f"HR@{top_k}_val": hr_alpha,
            "Recovery_val": recovery,
            "constraint_ok": constraint_ok,
        }
        status = "OK" if constraint_ok else "NG"
        print(f"  alpha={alpha:.1f}: HR@{top_k}={hr_alpha:.4f}, Recovery={recovery:.4f} [{status}]")

    # ===== Step4: 制約を満たすalphaの中でRecovery最大を選択 =====
    feasible = {a: v for a, v in alpha_logs.items() if v["constraint_ok"]}

    if feasible:
        best_alpha = max(feasible, key=lambda a: feasible[a]["Recovery_val"])
        print(f"[AlphaSelection] 制約OKのalpha: {sorted(feasible.keys())}")
        print(f"[AlphaSelection] 選択alpha*={best_alpha}  Recovery={feasible[best_alpha]['Recovery_val']:.4f}")
    else:
        best_alpha = 1.0
        print(f"[AlphaSelection] 制約を満たすalphaなし → alpha=1.0 を使用")

    log = {
        "best_alpha": best_alpha,
        "epsilon": epsilon,
        "top_k": top_k,
        "hr_base": hr_base,
        "hr_threshold": hr_threshold,
        "recoverable_base_size": len(recoverable_base),
        "total_users": len(all_users),
        "alpha_logs": alpha_logs,
    }
    print("[AlphaSelection] ===== 選択完了 =====\n")

    return best_alpha, log
