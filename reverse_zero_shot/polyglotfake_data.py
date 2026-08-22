"""
==============================================================================
[공통 유틸] PolyGlotFake 학습용 데이터 준비
==============================================================================

[폴더 구조]
~/hsh/AIApplication/
├── PolyGlotFake/                ← 데이터셋 (상위 폴더)
└── reverse_zero_shot/           ← 이 파일이 위치
    └── polyglotfake_data.py

[경로 처리]
이 스크립트가 상위 폴더(AIApplication)에서 실행되든, 본인 폴더(reverse_zero_shot)
에서 실행되든 모두 동작하도록 절대 경로 자동 탐지.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def find_polyglotfake_root():
    """
    실행 위치에 무관하게 PolyGlotFake 폴더 자동 탐지.
    
    탐색 우선순위:
        1. 현재 작업 디렉토리의 PolyGlotFake/
        2. 상위 디렉토리(../)의 PolyGlotFake/
        3. 부모의 부모(../../)의 PolyGlotFake/
    """
    candidates = [
        os.path.abspath("PolyGlotFake"),
        os.path.abspath("../PolyGlotFake"),
        os.path.abspath("../../PolyGlotFake"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            json_dir = os.path.join(path, 'json_file')
            if os.path.isdir(json_dir):
                return path
    raise FileNotFoundError(
        "PolyGlotFake 폴더를 찾을 수 없습니다. "
        "현재 위치 또는 상위 폴더에 PolyGlotFake/json_file/이 있어야 합니다."
    )


def build_polyglotfake_dataframe(base_dir: str = None) -> pd.DataFrame:
    """PolyGlotFake JSON에서 학습용 DataFrame 구성."""
    if base_dir is None:
        base_dir = find_polyglotfake_root()

    json_dir = os.path.join(base_dir, 'json_file')
    real_path = os.path.join(json_dir, 'real_json_file', 'all_real_video.json')
    fake_path = os.path.join(json_dir, 'fake_Json_file', 'all_fake_video.json')

    with open(real_path, 'r', encoding='utf-8') as f:
        real_data = json.load(f)
    with open(fake_path, 'r', encoding='utf-8') as f:
        fake_data = json.load(f)

    rows = []

    # Real 영상
    for v in real_data['videos']:
        rel_dir = f"real/{v['lang']}"
        # 절대 경로로 저장 (어디서 실행되든 작동)
        video_path = os.path.join(base_dir, rel_dir, v['filename'])
        rows.append({
            'filename':    v['filename'],
            'video_path':  video_path,
            'video_label': 0.0,
            'method':      'real',
            'rel_dir':     rel_dir,
            'lang':        v['lang'],
            'target_lang': v['lang'],
            'tts':         'real',
            'sync':        'real',
        })

    # Fake 영상
    for v in fake_data['video']:
        rel_dir = f"fake/to_{v['target_lang']}"
        video_path = os.path.join(base_dir, rel_dir, v['filename'])
        rows.append({
            'filename':    v['filename'],
            'video_path':  video_path,
            'video_label': 1.0,
            'method':      v['sync_tech'],
            'rel_dir':     rel_dir,
            'lang':        v['raw_lang'],
            'target_lang': v['target_lang'],
            'tts':         v['tts_technique'],
            'sync':        v['sync_tech'],
        })

    return pd.DataFrame(rows)


def build_polyglotfake_train_val(
    base_dir: str = None,
    num_fake_sample: int = 2000,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    학습/검증 데이터 분할.
    
    FakeAVCeleb 학습 코드와 동일한 방식:
      - Real 전량 사용 (766개)
      - Fake는 num_fake_sample개 무작위 샘플링
      - Train 90% / Val 10%, stratified by label
    """
    if base_dir is None:
        base_dir = find_polyglotfake_root()
        print(f"📁 PolyGlotFake 자동 탐지: {base_dir}")

    df = build_polyglotfake_dataframe(base_dir)

    real_df = df[df['video_label'] == 0.0]
    fake_df = df[df['video_label'] == 1.0]

    print(f"📊 PolyGlotFake 원본: Real={len(real_df)}, Fake={len(fake_df)}")

    n_fake = min(num_fake_sample, len(fake_df))
    sampled_fake = fake_df.sample(n=n_fake, random_state=seed)

    balanced_df = pd.concat([real_df, sampled_fake]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    print(f"✂️  학습 데이터: 총 {len(balanced_df)}개 "
          f"(Real {len(real_df)} : Fake {len(sampled_fake)})")

    train_df, val_df = train_test_split(
        balanced_df,
        test_size    = val_ratio,
        stratify     = balanced_df['video_label'],
        random_state = seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"📂 Train: {len(train_df)}  |  Val: {len(val_df)}")
    print(f"   Train Real/Fake: {(train_df['video_label']==0).sum()}/"
          f"{(train_df['video_label']==1).sum()}")
    print(f"   Val   Real/Fake: {(val_df['video_label']==0).sum()}/"
          f"{(val_df['video_label']==1).sum()}")

    return train_df, val_df


# 자가 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("PolyGlotFake 데이터 로더 테스트")
    print("=" * 60)
    train_df, val_df = build_polyglotfake_train_val()
    print("\n[샘플 확인]")
    print("Train 첫 3개 video_path:")
    for p in train_df['video_path'].head(3).tolist():
        print(f"  {p}")
        print(f"  존재 여부: {os.path.exists(p)}")