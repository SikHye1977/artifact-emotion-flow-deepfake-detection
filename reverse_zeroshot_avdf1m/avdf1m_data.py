"""
==============================================================================
[공통 데이터 로더] AV-Deepfake1M 학습용 데이터 분할
polyglotfake_data.py / FakeAVCeleb 학습과 동일한 패턴
==============================================================================

[목적]
- AV-Deepfake1M extracted_train/ 데이터를 FAV/PGF 학습 방식과 동일하게 처리
- 라벨링: 모든 fake_* 파일 = Fake(1), real.mp4 = Real(0)
- Real n_real개 + Fake n_fake개 (3가지 변조 유형 균등) → train/val 분할

[데이터셋 구조]
  ../AV-Deepfake1M_RootFiles/
  └── extracted_train/train/{speaker}/{youtube_id}/{seq}/
      ├── real.mp4                       (Label 0)
      ├── fake_video_real_audio.mp4      (Label 1)
      ├── real_video_fake_audio.mp4      (Label 1)
      └── fake_video_fake_audio.mp4      (Label 1)

[제공]
- build_avdf1m_train_val(n_real, n_fake, val_ratio, seed)
==============================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 라벨 매핑 (FAV 방식: 모든 fake = 1)
# ══════════════════════════════════════════════════════════════════════════════
LABEL_TYPES = {
    'real.mp4':                       ('real',              0),
    'fake_video_real_audio.mp4':      ('fake_video_only',   1),
    'real_video_fake_audio.mp4':      ('fake_audio_only',   1),
    'fake_video_fake_audio.mp4':      ('fake_both',         1),
}


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 루트 자동 탐색
# ══════════════════════════════════════════════════════════════════════════════
def find_avdf1m_root():
    """현재 폴더와 상위 폴더에서 AV-Deepfake1M 루트 탐색."""
    candidates = [
        'AV-Deepfake1M_RootFiles',
        'AV_Deepfake1M_RootFiles',
        'AV-Deepfake1M',
        'AV_Deepfake1M',
        'AVDeepfake1M',
    ]
    search_bases = ['.', '..', '../..']

    for base in search_bases:
        for c in candidates:
            full = os.path.abspath(os.path.join(base, c))
            if os.path.isdir(full):
                return full

    raise FileNotFoundError(
        "AV-Deepfake1M 폴더 못 찾음. 후보:\n  "
        + "\n  ".join(f"{b}/{c}" for b in search_bases for c in candidates)
    )


def find_train_path(data_root: str, split: str = 'train'):
    """extracted_train/train 또는 train 자동 인식."""
    base_candidates = [
        os.path.join(data_root, f'extracted_{split}', split),
        os.path.join(data_root, f'extracted_{split}'),
        os.path.join(data_root, split),
    ]
    for c in base_candidates:
        if not os.path.isdir(c):
            continue
        try:
            entries = os.listdir(c)
        except PermissionError:
            continue
        if not entries:
            continue
        if any(e.startswith('id') for e in entries):
            return c
        first = os.path.join(c, entries[0])
        if os.path.isdir(first):
            try:
                sub = os.listdir(first)
                if any(s.endswith('.mp4') or os.path.isdir(os.path.join(first, s)) for s in sub):
                    return c
            except Exception:
                continue
    raise FileNotFoundError(
        f"{split} 폴더 못 찾음. 시도:\n  " + "\n  ".join(base_candidates)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 학습 데이터셋 빌더 (PGF의 build_polyglotfake_train_val과 동일 패턴)
# ══════════════════════════════════════════════════════════════════════════════
def build_avdf1m_train_val(n_real=1000, n_fake=4000,
                            val_ratio=0.1, seed=42):
    """
    AV-Deepfake1M 학습/검증 분할.

    Args:
        n_real:    사용할 real 영상 개수 (default 1000)
        n_fake:    사용할 fake 영상 개수 (3가지 유형 균등 분배, default 4000)
        val_ratio: 검증셋 비율 (default 0.1)
        seed:      랜덤 시드 (default 42)

    Returns:
        (train_df, val_df): pandas DataFrame
        각 row는 'video_path', 'video_label', 'fake_type', 'speaker', ...
    """
    data_root = find_avdf1m_root()
    base_path = find_train_path(data_root, split='train')

    print(f"📂 데이터 루트: {data_root}")
    print(f"📂 train 경로 : {base_path}")
    print(f"\n📊 파일 스캔 중...")

    # 전체 스캔
    all_files = []
    for label_fname, (type_name, label) in LABEL_TYPES.items():
        pattern = os.path.join(base_path, '**', label_fname)
        matches = glob.glob(pattern, recursive=True)
        for path in matches:
            rel = os.path.relpath(path, base_path)
            parts = rel.split(os.sep)
            if len(parts) >= 4:
                speaker, youtube_id, seq_id = parts[0], parts[1], parts[2]
            else:
                speaker = youtube_id = seq_id = 'unknown'
            all_files.append({
                'video_path':  path,
                'video_label': float(label),
                'fake_type':   type_name,
                'speaker':     speaker,
                'youtube_id':  youtube_id,
                'seq_id':      seq_id,
            })

    df = pd.DataFrame(all_files)
    print(f"   전체 발견: {len(df)}개")
    print(f"   유형별 분포:")
    print(df['fake_type'].value_counts().to_string())

    if len(df) == 0:
        raise RuntimeError("스캔 결과 없음. 데이터 경로 확인 필요.")

    # Real n_real개 샘플링
    real_df = df[df['video_label'] == 0.0]
    n_real_actual = min(n_real, len(real_df))
    sampled_real = real_df.sample(n=n_real_actual, random_state=seed)
    print(f"\n📊 샘플링 결과:")
    print(f"   Real: {n_real_actual}개")

    # Fake n_fake개 균등 샘플링 (3유형)
    fake_per_type = n_fake // 3
    remainder = n_fake - fake_per_type * 3
    fake_dfs = []
    for i, ft in enumerate(['fake_video_only', 'fake_audio_only', 'fake_both']):
        sub = df[df['fake_type'] == ft]
        n_take = fake_per_type + (1 if i < remainder else 0)
        n_actual = min(n_take, len(sub))
        sampled = sub.sample(n=n_actual, random_state=seed + i + 1)
        fake_dfs.append(sampled)
        print(f"   {ft}: {n_actual}개")
    sampled_fake = pd.concat(fake_dfs)

    # 합치고 셔플
    full_df = pd.concat([sampled_real, sampled_fake]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    print(f"\n   총 샘플: {len(full_df)}개 "
          f"(Real {(full_df['video_label']==0).sum()}, "
          f"Fake {(full_df['video_label']==1).sum()})")

    # train/val 분할
    n_val = int(len(full_df) * val_ratio)
    val_df = full_df.iloc[:n_val].reset_index(drop=True)
    train_df = full_df.iloc[n_val:].reset_index(drop=True)
    print(f"\n📊 학습/검증 분할 (val_ratio={val_ratio}):")
    print(f"   Train: {len(train_df)}개 "
          f"(Real {(train_df['video_label']==0).sum()}, "
          f"Fake {(train_df['video_label']==1).sum()})")
    print(f"   Val  : {len(val_df)}개 "
          f"(Real {(val_df['video_label']==0).sum()}, "
          f"Fake {(val_df['video_label']==1).sum()})")

    return train_df, val_df


# ══════════════════════════════════════════════════════════════════════════════
# 단독 실행 시 데이터 확인
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("[avdf1m_data] 단독 실행 - 데이터 확인")
    print("=" * 60)
    train_df, val_df = build_avdf1m_train_val(
        n_real=1000, n_fake=4000, val_ratio=0.1, seed=42
    )
    print(f"\n✅ 데이터 로더 동작 정상")
    print(f"   train_df 컬럼: {train_df.columns.tolist()}")
    print(f"   샘플 (train_df.head(3)):")
    print(train_df.head(3).to_string())