"""
==============================================================================
[avdf1m_train_data.py] AVDF1M 학습용 공통 데이터 모듈

4개 학습 스크립트가 공통으로 사용:
  - train_x3d_avdf1m.py
  - train_aasist_avdf1m.py
  - train_HSEmotion_avdf1m.py
  - train_CRNN_avdf1m.py

[제공 함수]
  build_avdf1m_train_val_split(): 학습/검증셋 DataFrame 생성
  load_video_for_x3d(): X3D 입력 영상 로드
  load_frames_for_hsemo(): HSEmotion 입력 프레임 로드
  load_audio_for_aasist(): AASIST 입력 오디오 로드
  (CRNN은 extract_audio_segments 사용 — 기존 모듈에서 import)

[데이터 분포]
  AVDF1M train val: ~286,547 entries (val의 5배)
  학습 10000 + 검증 1000 샘플링
  Fake type 균등 분포 (visual/audio/both 각 ~2667)
==============================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import av
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision import transforms as T

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import sys
    import torchvision.transforms.functional as Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = Ftv

from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale


# ═══════════════════════════════════════════════════════════════════
# 경로
# ═══════════════════════════════════════════════════════════════════
AVDF1M_ROOT  = os.path.expanduser("~/hsh/AIApplication/AV-Deepfake1M_RootFiles")
TRAIN_META   = os.path.join(AVDF1M_ROOT, "train_metadata.json")
VAL_META     = os.path.join(AVDF1M_ROOT, "val_metadata.json")
TRAIN_VIDEO  = os.path.join(AVDF1M_ROOT, "extracted_train/train")
VAL_VIDEO    = os.path.join(AVDF1M_ROOT, "extracted_val/val")


# ═══════════════════════════════════════════════════════════════════
# 1. 학습/검증셋 구성
# ═══════════════════════════════════════════════════════════════════
def build_avdf1m_train_val_split(
    n_train_real=2000, n_train_fake=8000,
    n_val_real=250, n_val_fake_per_type=250,
    seed=42, use_val_for_eval=True,
):
    """
    AVDF1M 학습/검증셋 분리.
    
    학습셋: extracted_train/train (train_metadata.json)
    검증셋: extracted_val/val (val_metadata.json) — 진짜 zero-shot 가능
    
    [학습셋 구성]
      Real: n_train_real (기본 2000)
      Fake: n_train_fake (기본 8000), val 분포대로 균등 샘플링 (~2667 × 3 types)
    
    [검증셋 구성]
      Real: n_val_real
      Fake: n_val_fake_per_type × 3 (visual/audio/both)
    
    Returns:
      train_df, val_df: DataFrame with columns
        - file_rel:    상대 경로 (id_xxx/yt_id/seq/filename.mp4)
        - video_path:  절대 경로
        - video_label: 0.0 (real) or 1.0 (fake)
        - fake_type:   real/fake_video_only/fake_audio_only/fake_both
        - modify_type: 원본 metadata 값
        - audio_model: vits/yourtts/etc (real은 None)
    """
    print(f"📂 학습 메타데이터 로드: {TRAIN_META}")
    with open(TRAIN_META, 'r') as f:
        train_meta = json.load(f)
    print(f"   전체 엔트리: {len(train_meta)}")
    
    # ⭐ 추출된 speaker로 필터링 (extracted_train/train/ 스캔)
    if os.path.isdir(TRAIN_VIDEO):
        extracted_speakers = set(
            d for d in os.listdir(TRAIN_VIDEO)
            if os.path.isdir(os.path.join(TRAIN_VIDEO, d)) and d.startswith('id')
        )
        print(f"   추출된 speaker 폴더: {len(extracted_speakers)}개")
        original_len = len(train_meta)
        train_meta = [e for e in train_meta
                      if e['file'].split('/')[0] in extracted_speakers]
        print(f"   필터링 후 엔트리: {len(train_meta)} / {original_len} "
              f"({100*len(train_meta)/original_len:.1f}%)")
    else:
        print(f"   ⚠️ {TRAIN_VIDEO} 폴더 없음 - 필터링 스킵")
    
    type_map = {
        'real':            'real',
        'visual_modified': 'fake_video_only',
        'audio_modified':  'fake_audio_only',
        'both_modified':   'fake_both',
    }
    
    # ─── 학습셋 ─────────────────────────────────────────────
    train_by_type = {k: [] for k in type_map.keys()}
    for entry in train_meta:
        mt = entry.get('modify_type', '')
        if mt in train_by_type:
            train_by_type[mt].append(entry)
    
    print(f"\n[train_metadata] modify_type 분포:")
    for k, v in train_by_type.items():
        print(f"   {k}: {len(v)}")
    
    rng = np.random.RandomState(seed)
    
    # Real 샘플링
    real_pool = train_by_type['real']
    real_idx = rng.choice(len(real_pool), size=min(n_train_real, len(real_pool)), replace=False)
    train_reals = [real_pool[i] for i in real_idx]
    
    # Fake 샘플링 (3 type 균등)
    n_per_fake_type = n_train_fake // 3
    train_fakes = []
    for mt in ['visual_modified', 'audio_modified', 'both_modified']:
        pool = train_by_type[mt]
        n_take = min(n_per_fake_type, len(pool))
        idx = rng.choice(len(pool), size=n_take, replace=False)
        train_fakes.extend([pool[i] for i in idx])
    
    train_rows = []
    for entry in train_reals + train_fakes:
        mt = entry['modify_type']
        ft = type_map[mt]
        video_label = 0.0 if mt == 'real' else 1.0
        file_rel = entry['file']
        train_rows.append({
            'file_rel':    file_rel,
            'video_path':  os.path.join(TRAIN_VIDEO, file_rel),
            'video_label': video_label,
            'fake_type':   ft,
            'modify_type': mt,
            'audio_model': entry.get('audio_model'),
        })
    
    train_df = pd.DataFrame(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # ─── 검증셋 ─────────────────────────────────────────────
    if use_val_for_eval:
        # val_metadata.json 사용 (진짜 unseen 데이터)
        print(f"\n📂 검증 메타데이터 로드: {VAL_META}")
        with open(VAL_META, 'r') as f:
            val_meta = json.load(f)
        val_root = VAL_VIDEO
    else:
        # train_metadata에서 학습에 안 쓴 것 사용
        val_meta = train_meta
        val_root = TRAIN_VIDEO
    
    val_by_type = {k: [] for k in type_map.keys()}
    used_files = set(r['file_rel'] for r in train_rows)
    for entry in val_meta:
        mt = entry.get('modify_type', '')
        if mt in val_by_type and entry['file'] not in used_files:
            val_by_type[mt].append(entry)
    
    val_rows = []
    rng_val = np.random.RandomState(seed + 1000)
    
    # Real
    real_idx = rng_val.choice(len(val_by_type['real']),
                              size=min(n_val_real, len(val_by_type['real'])),
                              replace=False)
    for i in real_idx:
        entry = val_by_type['real'][i]
        val_rows.append({
            'file_rel':    entry['file'],
            'video_path':  os.path.join(val_root, entry['file']),
            'video_label': 0.0,
            'fake_type':   'real',
            'modify_type': 'real',
            'audio_model': None,
        })
    
    # Fake type별 균등
    for mt in ['visual_modified', 'audio_modified', 'both_modified']:
        pool = val_by_type[mt]
        n_take = min(n_val_fake_per_type, len(pool))
        idx = rng_val.choice(len(pool), size=n_take, replace=False)
        for i in idx:
            entry = pool[i]
            val_rows.append({
                'file_rel':    entry['file'],
                'video_path':  os.path.join(val_root, entry['file']),
                'video_label': 1.0,
                'fake_type':   type_map[mt],
                'modify_type': mt,
                'audio_model': entry.get('audio_model'),
            })
    
    val_df = pd.DataFrame(val_rows).sample(frac=1, random_state=seed+2000).reset_index(drop=True)
    
    print(f"\n✅ 학습셋: {len(train_df)}개 (Real {(train_df.video_label==0).sum()} + Fake {(train_df.video_label==1).sum()})")
    print(f"✅ 검증셋: {len(val_df)}개 (Real {(val_df.video_label==0).sum()} + Fake {(val_df.video_label==1).sum()})")
    
    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════
# 2. 전처리 (각 모델 입력 준비)
# ═══════════════════════════════════════════════════════════════════
def _rescale_video(x): return x / 255.0
def _permute_to_tc(x): return x.permute(1, 0, 2, 3)
def _permute_to_ct(x): return x.permute(1, 0, 2, 3)

x3d_video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(_rescale_video),
    Lambda(_permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(_permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224)),
])

hsemo_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_video_for_x3d(path, max_frames=128):
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(frames) < 16: return None
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in idx]
        video = np.stack(frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


def load_frames_for_hsemo(path, num_frames=16):
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        total = stream.frames
        if total < num_frames:
            container.close(); return None
        targets = set(np.linspace(0, total-1, num_frames, dtype=int).tolist())
        sampled = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in targets:
                sampled[i] = frame.to_rgb().to_ndarray()
            if len(sampled) >= num_frames: break
        container.close()
        if len(sampled) < num_frames: return None
        keys = sorted(sampled.keys())
        return torch.stack([hsemo_frame_transform(sampled[k]) for k in keys])
    except Exception:
        return None


def load_audio_for_aasist(video_path, target_sr=16000, max_length=64000):
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close(); return None
        sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.dtype == np.int16:   arr = arr.astype(np.float32)/32768.0
            elif arr.dtype == np.int32: arr = arr.astype(np.float32)/2147483648.0
            else:                       arr = arr.astype(np.float32)
            if arr.ndim > 1 and arr.shape[0] > arr.shape[1]: arr = arr.T
            elif arr.ndim == 1: arr = arr[np.newaxis, :]
            frames.append(arr)
        container.close()
        if not frames: return None
        wav = torch.from_numpy(np.concatenate(frames, axis=-1))
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        if wav.shape[1] > max_length:
            wav = wav[:, :max_length]
        else:
            wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]))
        return wav.squeeze()
    except Exception:
        return None


if __name__ == "__main__":
    # 모듈 단독 실행 시 학습셋/검증셋 분할 테스트
    train_df, val_df = build_avdf1m_train_val_split()
    print("\n[학습셋 샘플]")
    print(train_df.head(3).to_string())
    print("\n[검증셋 샘플]")
    print(val_df.head(3).to_string())
    
    # 영상 존재 여부 sanity check
    print("\n[Sanity check]")
    n_missing = 0
    for path in train_df['video_path'].head(20):
        if not os.path.exists(path):
            n_missing += 1
            print(f"  ❌ {path}")
    if n_missing == 0:
        print(f"  ✅ 학습셋 첫 20개 영상 모두 존재")
    else:
        print(f"  ⚠️ {n_missing}/20 누락")