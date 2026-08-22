"""
==============================================================================
[Fusion 전략 & 임계값 통합 비교]
Multi-Strategy Fusion Comparison for Precision Optimization
==============================================================================

[목적]
Precision을 높이기 위한 여러 Fusion 전략과 임계값을 데이터 기반으로 비교.
기존 추론 결과(sample_predictions.csv)를 재활용하므로 재추론 없이 즉시 실행.

[비교 전략 6종]

1. OR (기존) : 1 - (1-p_v)(1-p_a)        — Recall 최대화
2. AND       : p_v × p_a                  — Precision 최대화
3. Mean      : (p_v + p_a) / 2           — 평균
4. Weighted  : α·p_v + (1-α)·p_a         — AUC 기반 가중평균
5. Max       : max(p_v, p_a)             — 더 확신하는 쪽 선택
6. OR+AND    : 둘 다 높으면 Fake, 둘 다 낮으면 Real, 애매하면 OR

[임계값 스윕]
각 전략마다 0.3 ~ 0.9를 0.05 간격으로 스윕하여
- 최대 F1 지점
- Precision 85% 이상 구간
- Recall 90% 이상 구간
을 모두 리포트.

[출력]
- 전략별 성능 비교 표 (markdown)
- Precision/Recall 트레이드오프 곡선 데이터 (CSV)
- 추천 구성 탑 5
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Fusion 전략 정의
# ══════════════════════════════════════════════════════════════════════════════
def fuse_or(p_v: np.ndarray, p_a: np.ndarray) -> np.ndarray:
    """확률적 OR: 1 - (1-p_v)(1-p_a) — 한쪽이라도 가짜면 가짜"""
    return 1.0 - (1.0 - p_v) * (1.0 - p_a)


def fuse_and(p_v: np.ndarray, p_a: np.ndarray) -> np.ndarray:
    """확률적 AND: p_v × p_a — 둘 다 가짜여야 가짜"""
    return p_v * p_a


def fuse_mean(p_v: np.ndarray, p_a: np.ndarray) -> np.ndarray:
    """단순 평균"""
    return (p_v + p_a) / 2.0


def fuse_weighted(p_v: np.ndarray, p_a: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """가중 평균 — alpha는 영상 가중치 (AUC 기반으로 0.7 추천)"""
    return alpha * p_v + (1 - alpha) * p_a


def fuse_max(p_v: np.ndarray, p_a: np.ndarray) -> np.ndarray:
    """더 확신 있는 쪽 선택 — OR과 유사하지만 더 보수적"""
    return np.maximum(p_v, p_a)


def fuse_or_and_hybrid(p_v: np.ndarray, p_a: np.ndarray,
                       high: float = 0.7, low: float = 0.3) -> np.ndarray:
    """
    OR + AND 하이브리드:
      - 둘 다 high 이상 → p_v × p_a 로 강하게 확정 (높은 Precision)
      - 둘 다 low 이하 → 작은 값으로 Real 확정
      - 애매한 영역 → OR 사용 (Recall 확보)
    """
    result = np.zeros_like(p_v)
    both_high = (p_v >= high) & (p_a >= high)
    both_low  = (p_v <= low)  & (p_a <= low)
    ambiguous = ~(both_high | both_low)

    result[both_high] = np.maximum(p_v[both_high], p_a[both_high])  # 더 강한 쪽
    result[both_low]  = np.minimum(p_v[both_low],  p_a[both_low])   # 더 약한 쪽
    result[ambiguous] = fuse_or(p_v[ambiguous], p_a[ambiguous])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. 지표 계산
# ══════════════════════════════════════════════════════════════════════════════
def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray,
                         threshold: float) -> dict:
    preds = (probs > threshold).astype(int)
    labels_int = labels.astype(int)

    try:
        auc = roc_auc_score(labels, probs) * 100
    except Exception:
        auc = 0.0

    acc  = accuracy_score(labels_int, preds) * 100
    prec = precision_score(labels_int, preds, zero_division=0) * 100
    rec  = recall_score(labels_int, preds, zero_division=0) * 100
    f1   = f1_score(labels_int, preds, zero_division=0) * 100

    cm = confusion_matrix(labels_int, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return dict(
        threshold=threshold, auc=auc, acc=acc,
        precision=prec, recall=rec, f1=f1,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. 전략별 Best 찾기
# ══════════════════════════════════════════════════════════════════════════════
def find_best_configs(probs: np.ndarray, labels: np.ndarray,
                      strategy_name: str,
                      thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(probs, labels, t)
        m['strategy'] = strategy_name
        rows.append(m)
    return pd.DataFrame(rows)


def print_strategy_summary(name: str, df: pd.DataFrame):
    """전략별 최적 구성을 찾아 출력."""
    auc = df['auc'].iloc[0]  # AUC는 threshold 무관

    best_f1       = df.loc[df['f1'].idxmax()]
    best_at_p85   = df[df['precision'] >= 85]
    best_at_r90   = df[df['recall']    >= 90]

    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  📊 {name}   (AUC: {auc:.2f}%)")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    print(f"  [최대 F1]     t={best_f1['threshold']:.2f} → "
          f"P={best_f1['precision']:.1f}%  "
          f"R={best_f1['recall']:.1f}%  "
          f"F1={best_f1['f1']:.1f}%  "
          f"Acc={best_f1['acc']:.1f}%")

    if len(best_at_p85) > 0:
        # Precision 85% 이상 중 F1 최대
        pick = best_at_p85.loc[best_at_p85['f1'].idxmax()]
        print(f"  [P≥85%]      t={pick['threshold']:.2f} → "
              f"P={pick['precision']:.1f}%  "
              f"R={pick['recall']:.1f}%  "
              f"F1={pick['f1']:.1f}%  "
              f"Acc={pick['acc']:.1f}%")
    else:
        print(f"  [P≥85%]      달성 불가")

    if len(best_at_r90) > 0:
        pick = best_at_r90.loc[best_at_r90['f1'].idxmax()]
        print(f"  [R≥90%]      t={pick['threshold']:.2f} → "
              f"P={pick['precision']:.1f}%  "
              f"R={pick['recall']:.1f}%  "
              f"F1={pick['f1']:.1f}%  "
              f"Acc={pick['acc']:.1f}%")
    else:
        print(f"  [R≥90%]      달성 불가")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    INPUT_CSV  = "multimodal_report/sample_predictions.csv"
    OUTPUT_DIR = "multimodal_report"

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 입력 파일 없음: {INPUT_CSV}")
        print(f"   먼저 Multimodal_emtion_fusion.py 를 실행하세요.")
        sys.exit(1)

    # ── 기존 추론 결과 로드 ─────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)
    labels = df['실제_레이블(0:Real 1:Fake)'].values
    p_v    = df['영상_Fake확률(%)'].values / 100.0
    p_a    = df['오디오_Fake확률(%)'].values / 100.0

    print(f"📂 로드: {len(df)}개 샘플 "
          f"(Real={int((labels==0).sum())}, Fake={int((labels==1).sum())})")

    # ── 임계값 스윕 범위 ────────────────────────────────────────────────
    thresholds = np.arange(0.30, 0.95, 0.05)

    # ── 6개 전략 모두 평가 ──────────────────────────────────────────────
    strategies = {
        'OR (1-(1-pv)(1-pa))':     fuse_or(p_v, p_a),
        'AND (pv × pa)':            fuse_and(p_v, p_a),
        'Mean ((pv+pa)/2)':         fuse_mean(p_v, p_a),
        'Weighted (α=0.7 영상)':    fuse_weighted(p_v, p_a, alpha=0.7),
        'Weighted (α=0.6 영상)':    fuse_weighted(p_v, p_a, alpha=0.6),
        'Max (max(pv, pa))':        fuse_max(p_v, p_a),
        'OR+AND 하이브리드':         fuse_or_and_hybrid(p_v, p_a, high=0.7, low=0.3),
    }

    # 단일 모델도 비교용으로 포함
    strategies_with_single = {
        '영상 단독':               p_v,
        '오디오 단독':             p_a,
        **strategies,
    }

    print("\n" + "=" * 60)
    print("🔍 전략별 최적 구성 분석")
    print("=" * 60)

    all_results = []
    for name, probs in strategies_with_single.items():
        strat_df = find_best_configs(probs, labels, name, thresholds)
        all_results.append(strat_df)
        print_strategy_summary(name, strat_df)

    full_df = pd.concat(all_results, ignore_index=True)

    # ── 상위 추천 구성 (F1 기준) ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏆 전체 구성 중 F1 상위 Top 5")
    print("=" * 60)

    top5_f1 = full_df.sort_values('f1', ascending=False).head(5)
    for i, r in enumerate(top5_f1.itertuples(), 1):
        print(f"  {i}. [{r.strategy}] t={r.threshold:.2f}")
        print(f"     P={r.precision:.1f}%  R={r.recall:.1f}%  "
              f"F1={r.f1:.1f}%  Acc={r.acc:.1f}%  AUC={r.auc:.1f}%")

    # ── Precision ≥ 85% & Recall ≥ 85% 만족하는 구성 ─────────────────
    print("\n" + "=" * 60)
    print("⚖  균형 좋은 구성 (P≥85% AND R≥85%)")
    print("=" * 60)

    balanced = full_df[
        (full_df['precision'] >= 85) &
        (full_df['recall']    >= 85)
    ].sort_values('f1', ascending=False)

    if len(balanced) > 0:
        print(f"  총 {len(balanced)}개 구성이 조건 만족. F1 상위 3개:")
        for i, r in enumerate(balanced.head(3).itertuples(), 1):
            print(f"  {i}. [{r.strategy}] t={r.threshold:.2f} → "
                  f"P={r.precision:.1f}% R={r.recall:.1f}% F1={r.f1:.1f}%")
    else:
        print("  조건(P≥85 AND R≥85) 만족 구성 없음. "
              "완화된 기준(P≥80 AND R≥85)으로 재검색:")
        balanced = full_df[
            (full_df['precision'] >= 80) &
            (full_df['recall']    >= 85)
        ].sort_values('f1', ascending=False)
        for i, r in enumerate(balanced.head(3).itertuples(), 1):
            print(f"  {i}. [{r.strategy}] t={r.threshold:.2f} → "
                  f"P={r.precision:.1f}% R={r.recall:.1f}% F1={r.f1:.1f}%")

    # ── 전체 결과 CSV 저장 ─────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, "fusion_strategy_comparison.csv")
    full_df.round(2).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 전체 비교 결과: {output_path}")

    # ── 간단한 요약 테이블 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 전략 요약 (t=0.5 고정 기준)")
    print("=" * 60)
    print(f"  {'전략':<30} {'AUC':>6} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for name, probs in strategies_with_single.items():
        m = metrics_at_threshold(probs, labels, 0.5)
        print(f"  {name:<30} {m['auc']:>5.1f}% "
              f"{m['precision']:>5.1f}% "
              f"{m['recall']:>5.1f}% "
              f"{m['f1']:>5.1f}%")

    print("\n✅ 완료")


if __name__ == "__main__":
    main()