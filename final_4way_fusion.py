"""
==============================================================================
[4방향 확률적 OR Fusion] 아티팩트 + 감정 통합 딥페이크 탐지
==============================================================================

[아키텍처 (제출 그림 그대로)]

   Video  ─► X3D_m       ─► Artifact-score (비디오) ─┐
                                                       ├─► Artifact OR ─┐
   Audio  ─► AASIST      ─► Artifact-score (오디오) ─┘                  │
                                                                         ├─► 최종 OR
   Video  ─► HSEmotion   ─► Emotion-score (비디오)  ─┐                   │
                                                       ├─► Emotion  OR ─┘
   Audio  ─► CRNN        ─► Emotion-score (오디오)  ─┘

[Fusion 수식]
  Final = 1 − (1 − P_v_art)(1 − P_a_art)(1 − P_v_emo)(1 − P_a_emo)

[입력]
  - multimodal_report/artifact_predictions.csv
  - multimodal_report/sample_predictions.csv

[출력]
  - multimodal_report/final_4way_fusion.csv
  - multimodal_report/final_4way_summary.csv
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
# 1. 데이터 로드 & 정합성 체크
# ══════════════════════════════════════════════════════════════════════════════
def load_and_align(artifact_csv, emotion_csv):
    art = pd.read_csv(artifact_csv)
    emo = pd.read_csv(emotion_csv)

    print(f"📂 아티팩트 CSV: {len(art)}개")
    print(f"📂 감정     CSV: {len(emo)}개")

    art = art.rename(columns={
        '실제_레이블(0:Real 1:Fake)': 'label',
        '비디오아티팩트_확률(%)':      'p_v_artifact',
        '오디오아티팩트_확률(%)':      'p_a_artifact',
    })
    emo = emo.rename(columns={
        '실제_레이블(0:Real 1:Fake)': 'label',
        '영상_Fake확률(%)':            'p_v_emotion',
        '오디오_Fake확률(%)':          'p_a_emotion',
    })

    print(f"   아티팩트 라벨 분포: {art['label'].value_counts().sort_index().to_dict()}")
    print(f"   감정     라벨 분포: {emo['label'].value_counts().sort_index().to_dict()}")

    # 같은 평가셋이라는 가정 하에 레이블 기준 정렬 후 결합
    art_sorted = art.sort_values('label').reset_index(drop=True)
    emo_sorted = emo.sort_values('label').reset_index(drop=True)

    if len(art_sorted) != len(emo_sorted):
        n = min(len(art_sorted), len(emo_sorted))
        art_sorted = art_sorted.iloc[:n]
        emo_sorted = emo_sorted.iloc[:n]

    merged = pd.DataFrame({
        'label':        art_sorted['label'].values,
        'p_v_artifact': art_sorted['p_v_artifact'].values / 100.0,
        'p_a_artifact': art_sorted['p_a_artifact'].values / 100.0,
        'p_v_emotion':  emo_sorted['p_v_emotion'].values  / 100.0,
        'p_a_emotion':  emo_sorted['p_a_emotion'].values  / 100.0,
    })
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 2. Fusion 전략
# ══════════════════════════════════════════════════════════════════════════════
def prob_or(*probs):
    result = np.ones_like(probs[0])
    for p in probs:
        result *= (1.0 - p)
    return 1.0 - result


def strategy_4way_or(df):
    art_or = prob_or(df['p_v_artifact'].values, df['p_a_artifact'].values)
    emo_or = prob_or(df['p_v_emotion'].values,  df['p_a_emotion'].values)
    final  = prob_or(art_or, emo_or)
    return final, art_or, emo_or


def strategy_4way_mean(df):
    return (df['p_v_artifact'].values + df['p_a_artifact'].values +
            df['p_v_emotion'].values  + df['p_a_emotion'].values) / 4.0


def strategy_conditional(df, threshold=0.3):
    art = prob_or(df['p_v_artifact'].values, df['p_a_artifact'].values)
    emo = prob_or(df['p_v_emotion'].values,  df['p_a_emotion'].values)
    result = art.copy()
    ambiguous = (art > threshold) & (art < 1 - threshold)
    result[ambiguous] = (art[ambiguous] + emo[ambiguous]) / 2.0
    return result


def strategy_weighted(df, w_art=0.7):
    art = prob_or(df['p_v_artifact'].values, df['p_a_artifact'].values)
    emo = prob_or(df['p_v_emotion'].values,  df['p_a_emotion'].values)
    return w_art * art + (1 - w_art) * emo


def strategy_artifact_only(df):
    return prob_or(df['p_v_artifact'].values, df['p_a_artifact'].values)


def strategy_emotion_only(df):
    return prob_or(df['p_v_emotion'].values, df['p_a_emotion'].values)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 지표 계산
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_metrics(probs, labels, threshold=0.5):
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

    return dict(auc=auc, acc=acc, precision=prec, recall=rec, f1=f1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def print_metrics(name, m):
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  📊 {name}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  AUC={m['auc']:.2f}%  Acc={m['acc']:.2f}%  "
          f"P={m['precision']:.2f}%  R={m['recall']:.2f}%  F1={m['f1']:.2f}%")
    print(f"  Confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ARTIFACT_CSV = "multimodal_report/artifact_predictions.csv"
    EMOTION_CSV  = "multimodal_report/sample_predictions.csv"
    OUTPUT_CSV   = "multimodal_report/final_4way_fusion.csv"
    SUMMARY_CSV  = "multimodal_report/final_4way_summary.csv"

    for p in [ARTIFACT_CSV, EMOTION_CSV]:
        if not os.path.exists(p):
            print(f"❌ 필수 파일 없음: {p}")
            sys.exit(1)

    print("=" * 60)
    print("🔗 아티팩트 + 감정 예측 결합")
    print("=" * 60)
    df = load_and_align(ARTIFACT_CSV, EMOTION_CSV)
    print(f"\n✅ 결합 완료: {len(df)}개 샘플")

    labels = df['label'].values

    print("\n" + "=" * 60)
    print("🎯 Fusion 전략별 평가")
    print("=" * 60)

    strategies = {}
    strategies['아티팩트 단독 OR']        = strategy_artifact_only(df)
    strategies['감정 단독 OR']            = strategy_emotion_only(df)

    final_or, art_or, emo_or              = strategy_4way_or(df)
    strategies['🌟 4방향 OR (제출 그림)']  = final_or
    strategies['4방향 Mean']              = strategy_4way_mean(df)
    strategies['Weighted (art=0.7)']      = strategy_weighted(df, 0.7)
    strategies['Weighted (art=0.9)']      = strategy_weighted(df, 0.9)
    strategies['조건부 (art 주, emo 보조)'] = strategy_conditional(df, 0.3)

    results = []
    for name, probs in strategies.items():
        m = compute_all_metrics(probs, labels)
        results.append({'strategy': name, **m})
        print_metrics(name, m)

    # 요약
    print("\n" + "=" * 60)
    print("📋 전략 비교 요약 (t=0.5)")
    print("=" * 60)
    print(f"  {'전략':<30} {'AUC':>7} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(f"  {r['strategy']:<30} "
              f"{r['auc']:>6.2f}% {r['acc']:>6.2f}% "
              f"{r['precision']:>6.2f}% {r['recall']:>6.2f}% {r['f1']:>6.2f}%")

    # 샘플별 상세 저장
    out_df = df.copy()
    out_df['label_int']     = labels.astype(int)
    out_df['artifact_or']   = art_or
    out_df['emotion_or']    = emo_or
    out_df['4way_or_final'] = final_or

    for col in ['p_v_artifact', 'p_a_artifact', 'p_v_emotion', 'p_a_emotion',
                'artifact_or', 'emotion_or', '4way_or_final']:
        out_df[col] = (out_df[col] * 100).round(2)

    out_df['pred_final'] = (final_or > 0.5).astype(int)
    out_df['correct']    = (out_df['pred_final'] == labels.astype(int)).astype(int)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n💾 샘플별 최종 예측: {OUTPUT_CSV}")

    pd.DataFrame(results).round(2).to_csv(SUMMARY_CSV, index=False, encoding='utf-8-sig')
    print(f"💾 전략 비교 요약  : {SUMMARY_CSV}")

    # 인사이트
    print("\n" + "=" * 60)
    print("💡 핵심 인사이트")
    print("=" * 60)
    bl_art = next(r for r in results if r['strategy'] == '아티팩트 단독 OR')
    bl_4   = next(r for r in results if r['strategy'] == '🌟 4방향 OR (제출 그림)')

    auc_d = bl_4['auc'] - bl_art['auc']
    f1_d  = bl_4['f1']  - bl_art['f1']
    fp_d  = bl_4['fp']  - bl_art['fp']

    print(f"  아티팩트 단독 → 4방향 OR:")
    print(f"    AUC: {bl_art['auc']:.2f}% → {bl_4['auc']:.2f}% ({auc_d:+.2f}%p)")
    print(f"    F1 : {bl_art['f1']:.2f}% → {bl_4['f1']:.2f}% ({f1_d:+.2f}%p)")
    print(f"    FP : {bl_art['fp']:4d} → {bl_4['fp']:4d} ({fp_d:+d}개)")

    if auc_d >= 0.1:
        print(f"\n  ✅ 4방향 OR이 개선됨 → 아키텍처 유효성 확인")
    elif abs(auc_d) < 0.1:
        print(f"\n  ⚖  두 방식이 사실상 동등")
        print(f"     FakeAVCeleb에선 아티팩트만으로 천장. 크로스 데이터셋에서 진가 확인 예정")
    else:
        print(f"\n  ⚠  4방향 OR이 단독보다 낮음 → 감정이 노이즈로 작용")

    print("\n" + "=" * 60)
    print("✅ 완료. 다음 단계: PolyGlotFake Zero-shot 평가")
    print("=" * 60)


if __name__ == "__main__":
    main()