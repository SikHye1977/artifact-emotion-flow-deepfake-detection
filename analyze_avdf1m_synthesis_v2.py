"""
==============================================================================
[AV-Deepfake1M 합성기법별 분석 — v2 FIXED]

수정 사항 (v1 → v2):
  - seq_id zero-padding 처리 (467 → "00467" 등)
  - audio_model 4종 처리: vits, vits_word, yourtts, yourtts_word
  - vits_word/yourtts_word를 별도 카테고리로 분석 (논문 강조점)

[사용법]
  cd ~/hsh/AIApplication
  python analyze_avdf1m_synthesis_v2.py \\
      --predictions avdf1m_zeroshot_report/predictions.csv \\
      --metadata    AV-Deepfake1M_RootFiles/val_metadata.json \\
      --output      avdf1m_zeroshot_report/synthesis_analysis_v2/
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
    recall_score, f1_score
)


def prob_or(*ps):
    r = np.ones_like(ps[0])
    for p in ps:
        r *= (1.0 - p)
    return 1.0 - r


def compute_metrics(probs, labels, t=0.5):
    preds = (probs > t).astype(int)
    li = labels.astype(int)
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
# 핵심 수정: seq_id 정규화 함수
# ──────────────────────────────────────────────────────────────────────
def normalize_seq_id(seq_id):
    """
    seq_id를 5자리 zero-padded 문자열로 정규화.
    
    Examples:
        467     → "00467"
        "5"     → "00005"
        "00118" → "00118"  (이미 정규화됨)
    """
    s = str(seq_id).strip()
    # 숫자가 아닌 문자 제거 후 정수 변환
    try:
        n = int(s)
        return f"{n:05d}"
    except ValueError:
        return s  # 변환 실패하면 원본 반환


# 메타데이터의 file path에서 (speaker, youtube_id, seq_id, label_file) 추출
def extract_meta_key(file_path):
    """
    "id02432/OmzNYuwUO4o/00041/fake_video_real_audio.mp4"
    → ('id02432', 'OmzNYuwUO4o', '00041', 'fake_video_real_audio.mp4')
    """
    parts = file_path.replace('\\', '/').split('/')
    if len(parts) < 4:
        return None
    return (parts[-4], parts[-3], parts[-2], parts[-1])


# fake_type → label filename 매핑
FAKE_TYPE_TO_FILENAME = {
    'real':            'real.mp4',
    'fake_video_only': 'fake_video_real_audio.mp4',
    'fake_audio_only': 'real_video_fake_audio.mp4',
    'fake_both':       'fake_video_fake_audio.mp4',
}


def build_metadata_index(meta_list):
    """메타데이터를 (speaker, youtube_id, seq_id_padded, label_file)로 인덱싱."""
    index = {}
    for entry in meta_list:
        key = extract_meta_key(entry.get('file', ''))
        if key is None:
            continue
        index[key] = entry
    return index


def merge_predictions_with_metadata(df_pred, meta_index):
    """predictions × metadata join (seq_id zero-padding 적용)."""
    audio_models = []
    modify_types = []
    fake_seg_starts = []
    fake_seg_ends = []
    fake_seg_durations = []
    
    missed = 0
    for _, row in df_pred.iterrows():
        speaker = row['speaker']
        youtube_id = row['youtube_id']
        seq_id_padded = normalize_seq_id(row['seq_id'])  # ⭐ 핵심 수정
        fake_type = row['fake_type']
        
        label_file = FAKE_TYPE_TO_FILENAME.get(fake_type, '')
        key = (speaker, youtube_id, seq_id_padded, label_file)
        
        meta_entry = meta_index.get(key)
        if meta_entry is None:
            audio_models.append(None)
            modify_types.append(None)
            fake_seg_starts.append(None)
            fake_seg_ends.append(None)
            fake_seg_durations.append(None)
            missed += 1
            continue
        
        audio_models.append(meta_entry.get('audio_model'))
        modify_types.append(meta_entry.get('modify_type'))
        
        # fake_segments에서 변조 구간 길이 계산 (operation 대용)
        fs = meta_entry.get('fake_segments', [])
        if fs and len(fs) > 0:
            start = fs[0][0]
            end = fs[0][1]
            fake_seg_starts.append(start)
            fake_seg_ends.append(end)
            fake_seg_durations.append(end - start)
        else:
            fake_seg_starts.append(None)
            fake_seg_ends.append(None)
            fake_seg_durations.append(None)
    
    df_pred = df_pred.copy()
    df_pred['audio_model'] = audio_models
    df_pred['modify_type'] = modify_types
    df_pred['fake_seg_start'] = fake_seg_starts
    df_pred['fake_seg_end'] = fake_seg_ends
    df_pred['fake_seg_duration'] = fake_seg_durations
    
    if missed > 0:
        pct = missed * 100.0 / len(df_pred)
        print(f"   ⚠️  매칭 실패: {missed}/{len(df_pred)} ({pct:.1f}%)")
    else:
        print(f"   ✅ 매칭 100% 성공: {len(df_pred)}/{len(df_pred)}")
    
    return df_pred


def compute_all_strategies(df_subset):
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
        'X3D':         p_v_art,
        'AASIST':      p_a_art,
        'HSEmotion':   p_v_emo,
        'CRNN':        p_a_emo,
        'Score_art':   art_or,
        'Score_emo':   emo_or,
        'Score_final': final_or,
    }
    return {name: compute_metrics(probs, labels) for name, probs in strategies.items()}


# ──────────────────────────────────────────────────────────────────────
# 분석 1: audio_model 4종 (vits/vits_word/yourtts/yourtts_word)
# ──────────────────────────────────────────────────────────────────────
def analyze_by_audio_model(df_merged, output_dir):
    print("\n" + "=" * 70)
    print("🎙️  Audio Generation Model 분석 (VITS/YourTTS × full-sentence/word-level)")
    print("=" * 70)
    
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
    audio_models = ['vits', 'vits_word', 'yourtts', 'yourtts_word']
    for am in audio_models:
        sub_fake = fake_audio[fake_audio['audio_model'] == am]
        if len(sub_fake) == 0:
            print(f"\n  ⚠️  {am}: 0개, skip")
            continue
        
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
        print(f"\n💾 저장: per_audio_model.csv")
    
    return rows


# ──────────────────────────────────────────────────────────────────────
# 분석 2: full-sentence vs word-level (집계)
# ──────────────────────────────────────────────────────────────────────
def analyze_by_synthesis_granularity(df_merged, output_dir):
    """vits + yourtts (full-sentence) vs vits_word + yourtts_word (word-level) 비교."""
    print("\n" + "=" * 70)
    print("📏 Synthesis Granularity 분석 (Full-sentence vs Word-level)")
    print("=" * 70)
    
    real_df = df_merged[df_merged['fake_type'] == 'real']
    fake_audio = df_merged[
        df_merged['fake_type'].isin(['fake_audio_only', 'fake_both']) &
        df_merged['audio_model'].notna()
    ]
    
    # 그룹화: full-sentence (vits, yourtts) vs word-level (vits_word, yourtts_word)
    full_models = ['vits', 'yourtts']
    word_models = ['vits_word', 'yourtts_word']
    
    groups = {
        'Full-sentence (VITS+YourTTS)': fake_audio[fake_audio['audio_model'].isin(full_models)],
        'Word-level (VITS_word+YourTTS_word)': fake_audio[fake_audio['audio_model'].isin(word_models)],
    }
    
    rows = []
    for group_name, sub_fake in groups.items():
        if len(sub_fake) == 0:
            continue
        sub = pd.concat([real_df, sub_fake]).reset_index(drop=True)
        results = compute_all_strategies(sub)
        if results is None:
            continue
        
        print(f"\n[{group_name}]  N_fake={len(sub_fake)}")
        print(f"  {'전략':<14} {'AUC':>7} {'F1':>7} {'Recall':>8}")
        print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*8}")
        for strat, m in results.items():
            print(f"  {strat:<14} {m['auc']:>6.2f}% {m['f1']:>6.2f}% {m['recall']:>7.2f}%")
        
        for strat, m in results.items():
            rows.append({
                'granularity': group_name,
                'n_fake': m['n_fake'],
                'strategy': strat,
                **{k: m[k] for k in ['auc', 'acc', 'precision', 'recall', 'f1']}
            })
    
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(output_dir, 'per_granularity.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: per_granularity.csv")
    return rows


# ──────────────────────────────────────────────────────────────────────
# 분석 3: fake_seg_duration 별 (변조 구간 길이)
# ──────────────────────────────────────────────────────────────────────
def analyze_by_fake_duration(df_merged, output_dir):
    """변조 구간 길이별 분석 (짧을수록 어려움)."""
    print("\n" + "=" * 70)
    print("⏱️  Fake Segment Duration 분석 (변조 구간 길이별)")
    print("=" * 70)
    
    real_df = df_merged[df_merged['fake_type'] == 'real']
    fake_with_dur = df_merged[
        (df_merged['video_label'] == 1) &
        df_merged['fake_seg_duration'].notna()
    ]
    
    if len(fake_with_dur) == 0:
        print("⚠️  fake_segments 정보 없음")
        return []
    
    # 길이 분포 출력
    durs = fake_with_dur['fake_seg_duration'].values
    print(f"\nFake segment 길이 통계:")
    print(f"  count: {len(durs)}")
    print(f"  min:   {durs.min():.3f}s")
    print(f"  max:   {durs.max():.3f}s")
    print(f"  mean:  {durs.mean():.3f}s")
    print(f"  median:{np.median(durs):.3f}s")
    
    # 3구간 분할 (short / medium / long)
    q33, q67 = np.percentile(durs, [33, 67])
    
    bins = [
        ('Short  (<{:.2f}s)'.format(q33),  fake_with_dur[fake_with_dur['fake_seg_duration'] < q33]),
        ('Medium ({:.2f}-{:.2f}s)'.format(q33, q67), 
         fake_with_dur[(fake_with_dur['fake_seg_duration'] >= q33) & (fake_with_dur['fake_seg_duration'] < q67)]),
        ('Long   (≥{:.2f}s)'.format(q67), fake_with_dur[fake_with_dur['fake_seg_duration'] >= q67]),
    ]
    
    rows = []
    for bin_name, sub_fake in bins:
        if len(sub_fake) == 0:
            continue
        sub = pd.concat([real_df, sub_fake]).reset_index(drop=True)
        results = compute_all_strategies(sub)
        if results is None:
            continue
        
        print(f"\n[{bin_name}]  N_fake={len(sub_fake)}")
        for strat in ['AASIST', 'HSEmotion', 'Score_art', 'Score_emo', 'Score_final']:
            m = results[strat]
            print(f"  {strat:<14} AUC={m['auc']:>6.2f}%  F1={m['f1']:>6.2f}%")
        
        for strat, m in results.items():
            rows.append({
                'duration_bin': bin_name,
                'n_fake': m['n_fake'],
                'strategy': strat,
                **{k: m[k] for k in ['auc', 'acc', 'precision', 'recall', 'f1']}
            })
    
    if rows:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(os.path.join(output_dir, 'per_duration.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: per_duration.csv")
    return rows


# ──────────────────────────────────────────────────────────────────────
# 논문용 요약 표
# ──────────────────────────────────────────────────────────────────────
def create_paper_tables(audio_rows, granularity_rows, output_dir):
    print("\n" + "=" * 70)
    print("📋 논문용 요약 표 1: Per-Audio-Model (Score_final)")
    print("=" * 70)
    
    if audio_rows:
        df = pd.DataFrame(audio_rows)
        final = df[df['strategy'] == 'Score_final'].copy()
        if len(final) > 0:
            print("\n\\begin{tabular}{lccccc}")
            print("\\toprule")
            print("Audio Model & N (fake) & AUC & F1 & Recall \\\\")
            print("\\midrule")
            BS = "\\"  # backslash 변수로 분리 (f-string 제한 회피)
            for _, row in final.iterrows():
                model_escaped = row['audio_model'].replace('_', BS + '_')
                print(f"{model_escaped:<14} & "
                      f"{row['n_fake']:>4d} & "
                      f"{row['auc']:>5.2f} & "
                      f"{row['f1']:>5.2f} & "
                      f"{row['recall']:>5.2f} {BS}{BS}")
            print("\\bottomrule")
            print("\\end{tabular}")
        
        # AASIST 위주 표 (논문 핵심)
        print("\n\nAASIST 단독 (논문 강조점):")
        aasist = df[df['strategy'] == 'AASIST']
        for _, row in aasist.iterrows():
            print(f"  {row['audio_model']:<15} AUC={row['auc']:>5.2f}%  F1={row['f1']:>5.2f}%")
    
    print("\n" + "=" * 70)
    print("📋 논문용 요약 표 2: Full-sentence vs Word-level")
    print("=" * 70)
    
    if granularity_rows:
        df = pd.DataFrame(granularity_rows)
        key_strats = ['AASIST', 'Score_art', 'Score_emo', 'Score_final']
        sub = df[df['strategy'].isin(key_strats)]
        pivot_auc = sub.pivot(index='granularity', columns='strategy', values='auc')
        pivot_f1 = sub.pivot(index='granularity', columns='strategy', values='f1')
        
        print("\n[AUC]")
        print(pivot_auc.round(2).to_string())
        print("\n[F1]")
        print(pivot_f1.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description="AVDF1M 합성기법별 분석 v2")
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--output', type=str,
                        default='avdf1m_zeroshot_report/synthesis_analysis_v2/')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 AVDF1M Per-Synthesis-Technique 분석 v2 (seq_id 정규화 적용)")
    print("=" * 70)
    
    os.makedirs(args.output, exist_ok=True)
    
    # 로드
    print(f"\n📂 Predictions 로드: {args.predictions}")
    df_pred = pd.read_csv(args.predictions, encoding='utf-8-sig')
    print(f"   총 샘플: {len(df_pred)}")
    
    print(f"\n📂 메타데이터 로드: {args.metadata}")
    with open(args.metadata, 'r') as f:
        meta_list = json.load(f)
    print(f"   엔트리: {len(meta_list)}")
    
    meta_index = build_metadata_index(meta_list)
    print(f"   인덱싱: {len(meta_index)}")
    
    # Join (seq_id zero-padding 적용)
    print(f"\n🔗 Join (seq_id 정규화: 467 → 00467)...")
    df_merged = merge_predictions_with_metadata(df_pred, meta_index)
    
    # 매칭 통계
    has_am = df_merged['audio_model'].notna().sum()
    has_mt = df_merged['modify_type'].notna().sum()
    print(f"   audio_model 매칭: {has_am}/{len(df_merged)}")
    print(f"   modify_type 매칭: {has_mt}/{len(df_merged)}")
    
    # 저장
    df_merged.to_csv(os.path.join(args.output, 'predictions_merged.csv'),
                     index=False, encoding='utf-8-sig')
    
    # 분석
    audio_rows = analyze_by_audio_model(df_merged, args.output)
    granularity_rows = analyze_by_synthesis_granularity(df_merged, args.output)
    duration_rows = analyze_by_fake_duration(df_merged, args.output)
    
    # 요약 표
    create_paper_tables(audio_rows, granularity_rows, args.output)
    
    print("\n" + "=" * 70)
    print(f"✅ 완료. 결과: {os.path.abspath(args.output)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
