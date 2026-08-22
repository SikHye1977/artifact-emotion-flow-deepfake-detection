"""
==============================================================================
[AV-Deepfake1M 합성기법별 분석]
기존 predictions.csv + AVDF1M 메타데이터를 결합하여
합성기법(VITS/YourTTS/TalkLip)별 + operation(replace/insert/delete)별
모델 성능을 측정합니다.

[입력 파일]
1. predictions.csv  (이전 evaluate_avdf1m_zeroshot.py 결과)
2. AVDF1M val metadata json  (val_metadata.json 또는 val.json)

[메타데이터 JSON 구조]
[
  {
    "file": "id00012/21Uxsk56VDQ/00001/fake_video_real_audio.mp4",
    "original": "...",
    "modify_type": "visual_modified",         <- ["real", "visual_modified", "audio_modified", "both_modified"]
    "audio_model": "vits" or "yourtts",        <- 음성 생성 모델
    "visual_model": "talklip",                 <- 영상 생성 모델 (단일)
    "fake_segments": [[start, end], ...],      <- 시간 구간
    "audio_fake_segments": [...],
    "visual_fake_segments": [...],
    "operations": ["replace", "insert", "delete"]  <- ChatGPT가 적용한 LLM operation
  }, ...
]

[출력]
- per_audio_model.csv: VITS vs YourTTS 별 모델 성능
- per_visual_model.csv: TalkLip 단일이지만 변조 유형별 분석
- per_operation.csv: replace/insert/delete 별 (가능한 경우)
- per_synthesis_combined.csv: 위 3개를 종합

[사용법]
  cd ~/hsh/AIApplication
  python analyze_avdf1m_synthesis.py \\
      --predictions avdf1m_zeroshot_report/predictions.csv \\
      --metadata    AV-Deepfake1M_RootFiles/val_metadata.json \\
      --output      avdf1m_zeroshot_report/synthesis_analysis/
==============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)


# ──────────────────────────────────────────────────────────────────────
# 1. 유틸리티
# ──────────────────────────────────────────────────────────────────────
def prob_or(*ps):
    """확률 OR 융합."""
    r = np.ones_like(ps[0])
    for p in ps:
        r *= (1.0 - p)
    return 1.0 - r


def compute_metrics(probs, labels, t=0.5):
    """기본 지표 계산."""
    preds = (probs > t).astype(int)
    li = labels.astype(int)
    
    # AUC는 양쪽 클래스가 있을 때만 계산 가능
    try:
        auc = roc_auc_score(labels, probs) * 100 if len(np.unique(labels)) > 1 else 0.0
    except Exception:
        auc = 0.0
    
    return dict(
        n=len(labels),
        n_fake=int(li.sum()),
        n_real=int((1 - li).sum()),
        auc=round(auc, 2),
        acc=round(accuracy_score(li, preds) * 100, 2),
        precision=round(precision_score(li, preds, zero_division=0) * 100, 2),
        recall=round(recall_score(li, preds, zero_division=0) * 100, 2),
        f1=round(f1_score(li, preds, zero_division=0) * 100, 2),
    )


# ──────────────────────────────────────────────────────────────────────
# 2. 메타데이터 로드 및 predictions와 매칭
# ──────────────────────────────────────────────────────────────────────
def load_metadata(metadata_path: str):
    """AVDF1M 메타데이터 JSON 로드."""
    print(f"📂 메타데이터 로드: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta_list = json.load(f)
    
    print(f"   총 엔트리: {len(meta_list)}")
    
    # 첫 엔트리의 키 확인
    if meta_list:
        sample_keys = list(meta_list[0].keys())
        print(f"   필드: {sample_keys}")
    
    return meta_list


def extract_video_key(file_path: str):
    """
    파일 경로에서 식별 가능한 key 추출.
    
    metadata: "id00012/21Uxsk56VDQ/00001/fake_video_real_audio.mp4"
    predictions.csv: speaker=id00012, youtube_id=21Uxsk56VDQ, seq_id=00001, fake_type=fake_video_only
    
    매칭을 위해 (speaker, youtube_id, seq_id, label_filename) 튜플 생성.
    """
    parts = file_path.replace('\\', '/').split('/')
    if len(parts) < 4:
        return None
    speaker = parts[-4]
    youtube_id = parts[-3]
    seq_id = parts[-2]
    label_file = parts[-1]
    return (speaker, youtube_id, seq_id, label_file)


# fake_type (predictions.csv) <-> label_filename (metadata) 매핑
FAKE_TYPE_TO_FILENAME = {
    'real':            'real.mp4',
    'fake_video_only': 'fake_video_real_audio.mp4',
    'fake_audio_only': 'real_video_fake_audio.mp4',
    'fake_both':       'fake_video_fake_audio.mp4',
}


def build_metadata_index(meta_list):
    """메타데이터를 (speaker, youtube_id, seq_id, label_file) -> entry로 인덱싱."""
    index = {}
    no_audio_model_count = 0
    no_visual_model_count = 0
    
    for entry in meta_list:
        file_path = entry.get('file', '')
        key = extract_video_key(file_path)
        if key is None:
            continue
        index[key] = entry
        
        if not entry.get('audio_model'):
            no_audio_model_count += 1
        if not entry.get('visual_model'):
            no_visual_model_count += 1
    
    print(f"   인덱싱 완료: {len(index)} 엔트리")
    if no_audio_model_count > 0:
        print(f"   ⚠️  audio_model 없는 엔트리: {no_audio_model_count}")
    if no_visual_model_count > 0:
        print(f"   ⚠️  visual_model 없는 엔트리: {no_visual_model_count}")
    
    return index


def merge_predictions_with_metadata(df_pred, meta_index):
    """predictions.csv에 메타데이터의 합성기법 정보를 join."""
    audio_models = []
    visual_models = []
    operations_list = []
    
    missed = 0
    for _, row in df_pred.iterrows():
        speaker = row['speaker']
        youtube_id = row['youtube_id']
        seq_id = row['seq_id']
        fake_type = row['fake_type']
        
        label_file = FAKE_TYPE_TO_FILENAME.get(fake_type, '')
        key = (speaker, youtube_id, seq_id, label_file)
        
        meta_entry = meta_index.get(key)
        if meta_entry is None:
            audio_models.append(None)
            visual_models.append(None)
            operations_list.append(None)
            missed += 1
            continue
        
        # 합성 모델
        audio_models.append(meta_entry.get('audio_model'))
        visual_models.append(meta_entry.get('visual_model'))
        
        # Operation 추출 (있으면)
        ops = meta_entry.get('operations')
        if ops:
            if isinstance(ops, list):
                operations_list.append(','.join(sorted(set(ops))))
            else:
                operations_list.append(str(ops))
        else:
            # fake_segments에서 operation 정보를 추출하는 fallback
            # (메타데이터 버전에 따라 형식이 다를 수 있음)
            operations_list.append(None)
    
    df_pred = df_pred.copy()
    df_pred['audio_model'] = audio_models
    df_pred['visual_model'] = visual_models
    df_pred['operations'] = operations_list
    
    if missed > 0:
        pct = missed * 100.0 / len(df_pred)
        print(f"   ⚠️  메타데이터 매칭 실패: {missed}/{len(df_pred)} ({pct:.1f}%)")
    
    return df_pred


# ──────────────────────────────────────────────────────────────────────
# 3. 합성기법별 분석
# ──────────────────────────────────────────────────────────────────────
def compute_all_strategies(df_subset):
    """주어진 subset에 대해 7가지 전략의 성능을 계산."""
    if len(df_subset) == 0:
        return None
    
    labels = df_subset['video_label'].values.astype(float)
    p_v_art = df_subset['p_v_artifact'].values / 100
    p_a_art = df_subset['p_a_artifact'].values / 100
    p_v_emo = df_subset['p_v_emotion'].values / 100
    p_a_emo = df_subset['p_a_emotion'].values / 100
    
    art_or = prob_or(p_v_art, p_a_art)
    emo_or = prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)
    
    strategies = {
        'X3D':           p_v_art,
        'AASIST':        p_a_art,
        'HSEmotion':     p_v_emo,
        'CRNN':          p_a_emo,
        'Score_art':     art_or,
        'Score_emo':     emo_or,
        'Score_final':   final_or,
    }
    
    results = {}
    for name, probs in strategies.items():
        m = compute_metrics(probs, labels)
        results[name] = m
    
    return results


def analyze_by_audio_model(df_merged, output_dir):
    """
    VITS vs YourTTS 별 분석.
    오디오가 변조된 샘플 (fake_audio_only, fake_both) + 모든 Real 비교.
    """
    print("\n" + "=" * 70)
    print("🎙️  Audio Generation Model 분석 (VITS vs YourTTS)")
    print("=" * 70)
    
    # 오디오가 변조된 fake만 추출 (audio_model이 있는 것)
    fake_audio = df_merged[
        df_merged['fake_type'].isin(['fake_audio_only', 'fake_both']) &
        df_merged['audio_model'].notna()
    ]
    real_df = df_merged[df_merged['fake_type'] == 'real']
    
    print(f"\n분포:")
    print(f"  Real: {len(real_df)}")
    print(f"  Fake (audio_model 보유): {len(fake_audio)}")
    if len(fake_audio) > 0:
        print(fake_audio['audio_model'].value_counts().to_string())
    
    rows = []
    for am in ['vits', 'yourtts']:
        # 케이스 무시
        sub_fake = fake_audio[
            fake_audio['audio_model'].str.lower() == am.lower()
        ]
        if len(sub_fake) == 0:
            print(f"\n  ⚠️  {am}: 0개, skip")
            continue
        
        # Real + 해당 audio_model fake로 평가
        sub = pd.concat([real_df, sub_fake]).reset_index(drop=True)
        results = compute_all_strategies(sub)
        if results is None:
            continue
        
        print(f"\n[audio_model = {am}]  N_fake={len(sub_fake)}, N_real={len(real_df)}")
        print(f"  {'전략':<14} {'AUC':>7} {'F1':>7} {'Recall':>8}")
        print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*8}")
        for strat, m in results.items():
            print(f"  {strat:<14} {m['auc']:>6.2f}% {m['f1']:>6.2f}% {m['recall']:>7.2f}%")
        
        for strat, m in results.items():
            rows.append({
                'audio_model': am,
                'n_fake': m['n_fake'],
                'n_real': m['n_real'],
                'strategy': strat,
                **{k: m[k] for k in ['auc', 'acc', 'precision', 'recall', 'f1']}
            })
    
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(output_dir, 'per_audio_model.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {os.path.join(output_dir, 'per_audio_model.csv')}")
    
    return rows


def analyze_by_modify_type_x_audio_model(df_merged, output_dir):
    """
    변조 유형(modify_type) x audio_model 교차 분석.
    fake_audio_only:VITS vs fake_audio_only:YourTTS vs fake_both:VITS vs ...
    """
    print("\n" + "=" * 70)
    print("🔀 Cross-Analysis: fake_type x audio_model")
    print("=" * 70)
    
    real_df = df_merged[df_merged['fake_type'] == 'real']
    rows = []
    
    for fake_type in ['fake_audio_only', 'fake_both']:
        for am in ['vits', 'yourtts']:
            sub_fake = df_merged[
                (df_merged['fake_type'] == fake_type) &
                (df_merged['audio_model'].notna()) &
                (df_merged['audio_model'].str.lower() == am.lower())
            ]
            if len(sub_fake) == 0:
                continue
            
            sub = pd.concat([real_df, sub_fake]).reset_index(drop=True)
            results = compute_all_strategies(sub)
            if results is None:
                continue
            
            print(f"\n[{fake_type} × {am}]  N_fake={len(sub_fake)}")
            for strat in ['AASIST', 'CRNN', 'Score_art', 'Score_emo', 'Score_final']:
                m = results[strat]
                print(f"  {strat:<14} AUC={m['auc']:>6.2f}%  F1={m['f1']:>6.2f}%")
            
            for strat, m in results.items():
                rows.append({
                    'fake_type': fake_type,
                    'audio_model': am,
                    'n_fake': m['n_fake'],
                    'strategy': strat,
                    **{k: m[k] for k in ['auc', 'acc', 'precision', 'recall', 'f1']}
                })
    
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(output_dir, 'per_modify_x_audio.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {os.path.join(output_dir, 'per_modify_x_audio.csv')}")
    
    return rows


def analyze_by_operation(df_merged, output_dir):
    """
    LLM operation (replace/insert/delete) 별 분석.
    operations 필드가 있을 때만 동작.
    """
    print("\n" + "=" * 70)
    print("✂️  Manipulation Operation 분석 (replace/insert/delete)")
    print("=" * 70)
    
    has_op = df_merged[df_merged['operations'].notna()]
    if len(has_op) == 0:
        print("⚠️  operations 필드가 메타데이터에 없습니다. Skip.")
        return []
    
    real_df = df_merged[df_merged['fake_type'] == 'real']
    print(f"\noperations 보유 fake: {len(has_op)}")
    
    # 단일 operation만 가진 샘플로 한정 (혼합 샘플은 별도 처리)
    rows = []
    for op in ['replace', 'insert', 'delete']:
        # 정확히 해당 operation만 포함 (단순화)
        sub_fake = has_op[
            (has_op['operations'].str.contains(op, na=False, case=False)) &
            (has_op['video_label'] == 1)
        ]
        if len(sub_fake) == 0:
            continue
        
        sub = pd.concat([real_df, sub_fake]).reset_index(drop=True)
        results = compute_all_strategies(sub)
        if results is None:
            continue
        
        print(f"\n[operation contains '{op}']  N_fake={len(sub_fake)}")
        for strat in ['Score_art', 'Score_emo', 'Score_final']:
            m = results[strat]
            print(f"  {strat:<14} AUC={m['auc']:>6.2f}%  F1={m['f1']:>6.2f}%")
        
        for strat, m in results.items():
            rows.append({
                'operation': op,
                'n_fake': m['n_fake'],
                'strategy': strat,
                **{k: m[k] for k in ['auc', 'acc', 'precision', 'recall', 'f1']}
            })
    
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(output_dir, 'per_operation.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {os.path.join(output_dir, 'per_operation.csv')}")
    
    return rows


# ──────────────────────────────────────────────────────────────────────
# 4. 논문용 요약 테이블 생성
# ──────────────────────────────────────────────────────────────────────
def create_paper_table(audio_rows, output_dir):
    """
    논문 §4.5.2 에 들어갈 표 형태로 정리.
    AASIST가 핵심: VITS vs YourTTS 별 AUC/F1
    """
    if not audio_rows:
        return
    
    df = pd.DataFrame(audio_rows)
    
    # 논문용 요약: 주요 전략만 추출
    key_strats = ['AASIST', 'Score_art', 'CRNN', 'Score_emo', 'Score_final']
    summary = df[df['strategy'].isin(key_strats)].pivot(
        index='audio_model', columns='strategy', values='f1'
    )
    
    auc_summary = df[df['strategy'].isin(key_strats)].pivot(
        index='audio_model', columns='strategy', values='auc'
    )
    
    print("\n" + "=" * 70)
    print("📋 논문용 요약 표 (Per-Audio-Model)")
    print("=" * 70)
    print("\n[F1 (%)]")
    print(summary.round(2).to_string())
    print("\n[AUC (%)]")
    print(auc_summary.round(2).to_string())
    
    # LaTeX 형태 출력
    print("\n" + "=" * 70)
    print("📝 LaTeX 표 (Score_final 기준):")
    print("=" * 70)
    
    final_summary = df[df['strategy'] == 'Score_final']
    if len(final_summary) > 0:
        print("\n\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Audio Model & N (fake) & AUC & F1 & Recall \\\\")
        print("\\midrule")
        for _, row in final_summary.iterrows():
            print(f"{row['audio_model'].upper():<10} & "
                  f"{row['n_fake']:>4d} & "
                  f"{row['auc']:.2f} & "
                  f"{row['f1']:.2f} & "
                  f"{row['recall']:.2f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    
    # CSV로도 저장
    summary.round(2).to_csv(
        os.path.join(output_dir, 'paper_summary_audio_model_f1.csv'),
        encoding='utf-8-sig'
    )
    auc_summary.round(2).to_csv(
        os.path.join(output_dir, 'paper_summary_audio_model_auc.csv'),
        encoding='utf-8-sig'
    )


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AVDF1M 합성기법별 (VITS/YourTTS/Operation) 분석"
    )
    parser.add_argument('--predictions', type=str, required=True,
                        help="이전 evaluate_avdf1m_zeroshot.py 결과 predictions.csv")
    parser.add_argument('--metadata', type=str, required=True,
                        help="AVDF1M val metadata JSON 파일 경로")
    parser.add_argument('--output', type=str,
                        default='avdf1m_zeroshot_report/synthesis_analysis',
                        help="결과 출력 디렉토리")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 AVDF1M Per-Synthesis-Technique 분석")
    print("=" * 70)
    
    # 입력 파일 확인
    if not os.path.exists(args.predictions):
        print(f"❌ predictions.csv 없음: {args.predictions}")
        sys.exit(1)
    if not os.path.exists(args.metadata):
        print(f"❌ 메타데이터 JSON 없음: {args.metadata}")
        print(f"   AVDF1M val_metadata.json 또는 val.json 파일 경로를 지정하세요.")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Predictions 로드
    print(f"\n📂 Predictions 로드: {args.predictions}")
    df_pred = pd.read_csv(args.predictions, encoding='utf-8-sig')
    print(f"   총 샘플: {len(df_pred)}")
    print(f"   fake_type 분포:")
    print(df_pred['fake_type'].value_counts().to_string())
    
    # 2. 메타데이터 로드 및 인덱싱
    meta_list = load_metadata(args.metadata)
    meta_index = build_metadata_index(meta_list)
    
    # 3. Join
    print(f"\n🔗 Predictions × Metadata Join...")
    df_merged = merge_predictions_with_metadata(df_pred, meta_index)
    
    # 매칭 통계
    has_audio_model = df_merged['audio_model'].notna().sum()
    has_visual_model = df_merged['visual_model'].notna().sum()
    print(f"   audio_model 매칭: {has_audio_model}/{len(df_merged)}")
    print(f"   visual_model 매칭: {has_visual_model}/{len(df_merged)}")
    
    # 합본 저장
    merged_path = os.path.join(args.output, 'predictions_merged.csv')
    df_merged.to_csv(merged_path, index=False, encoding='utf-8-sig')
    print(f"💾 통합 데이터 저장: {merged_path}")
    
    # 4. 분석 수행
    audio_rows = analyze_by_audio_model(df_merged, args.output)
    cross_rows = analyze_by_modify_type_x_audio_model(df_merged, args.output)
    op_rows = analyze_by_operation(df_merged, args.output)
    
    # 5. 논문용 요약 표
    create_paper_table(audio_rows, args.output)
    
    print("\n" + "=" * 70)
    print(f"✅ 분석 완료. 결과 위치: {os.path.abspath(args.output)}/")
    print("=" * 70)
    print("\n📊 다음 단계:")
    print("  1. per_audio_model.csv → 논문 §4.5.2 표")
    print("  2. paper_summary_*.csv → 본문에 인용할 수치")
    print("  3. 결과 공유해주시면 v6 논문에 통합하겠습니다.")


if __name__ == "__main__":
    main()
