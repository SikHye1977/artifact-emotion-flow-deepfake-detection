"""
==============================================================================
[캐시 생성] X3D + AASIST 학습용 데이터 사전 추출
==============================================================================

[목적]
PolyGlotFake 영상에서 X3D용 비디오 텐서와 AASIST용 오디오 텐서를
사전 추출하여 .pt 파일로 저장. 학습 시 디코딩 비용 제거.

[캐시 위치]
  reverse_zero_shot/cache_pgf/
    ├── video/
    │   ├── 0000.pt   (X3D용: 224x224 영상 16프레임)
    │   ├── 0001.pt
    │   └── ...
    ├── audio/
    │   ├── 0000.pt   (AASIST용: 4초 16kHz 오디오)
    │   ├── 0001.pt
    │   └── ...
    └── manifest.csv  (인덱스 ↔ 메타데이터 매핑)

[디스크 사용량 예상]
  영상 2,766개 기준:
    - 비디오: 약 700MB (영상당 ~250KB, 16프레임 × 224 × 224 × 3 × float16)
    - 오디오: 약 350MB (영상당 ~128KB, 64,000 × float16)

[실행 시간] 약 30~40분 (1회만)

[사용법]
  cd ~/hsh/AIApplication/reverse_zero_shot
  python build_artifact_cache.py

[중간 끊김 처리]
  이미 생성된 파일은 자동 skip → 안전하게 재실행 가능
==============================================================================
"""

import sys
import os
import functools
import time
import gc

# 경로 설정
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torchaudio
import av
import numpy as np
import pandas as pd

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = Ftv

from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

torch.load = functools.partial(torch.load, weights_only=False)

from polyglotfake_data import build_polyglotfake_train_val


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리 함수 (학습 코드와 동일)
# ══════════════════════════════════════════════════════════════════════════════
def rescale_video(x): return x / 255.0
def permute_to_tc(x): return x.permute(1, 0, 2, 3)
def permute_to_ct(x): return x.permute(1, 0, 2, 3)

video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale_video),
    Lambda(permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224))
])


def load_and_transform_video(path, max_frames=128):
    """X3D용 비디오 텐서 추출 (전처리 완료된 상태)."""
    try:
        container = av.open(path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(all_frames) < 16: return None
        if len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
            all_frames = [all_frames[i] for i in indices]
        video = np.stack(all_frames)
        raw = torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
        # 변환까지 완료한 상태로 저장 (학습 시 그대로 사용)
        transformed = video_transform(raw)
        return transformed.to(torch.float16)  # 메모리 절약을 위해 fp16
    except Exception:
        return None


def load_audio(video_path, target_sr=16000, max_length=64000):
    """AASIST용 오디오 텐서 추출."""
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close()
            return None
        sample_rate = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            else:
                arr = arr.astype(np.float32)
            if len(arr.shape) > 1 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            elif len(arr.shape) == 1:
                arr = arr[np.newaxis, :]
            frames.append(arr)
        container.close()
        if not frames: return None
        waveform = np.concatenate(frames, axis=-1)
        waveform = torch.from_numpy(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )(waveform)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, max_length - waveform.shape[1])
            )
        return waveform.squeeze().to(torch.float16)  # fp16 절약
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 캐시 생성
# ══════════════════════════════════════════════════════════════════════════════
def build_cache(df, cache_dir, prefix="train"):
    """
    DataFrame 행 순서대로 캐시 생성.
    파일명: {prefix}_{idx:05d}.pt
    """
    video_dir = os.path.join(cache_dir, 'video')
    audio_dir = os.path.join(cache_dir, 'audio')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    manifest_rows = []
    n_total   = len(df)
    n_done    = 0
    n_skipped = 0
    n_failed  = 0
    t_start   = time.time()

    print(f"\n📦 {prefix} 캐시 생성 시작 ({n_total}개)")

    for idx in range(n_total):
        row = df.iloc[idx]
        video_path = row['video_path']

        v_path = os.path.join(video_dir, f"{prefix}_{idx:05d}.pt")
        a_path = os.path.join(audio_dir, f"{prefix}_{idx:05d}.pt")

        # 이미 둘 다 있으면 skip
        if os.path.exists(v_path) and os.path.exists(a_path):
            n_skipped += 1
            manifest_rows.append({
                'index':       idx,
                'video_cache': v_path,
                'audio_cache': a_path,
                'video_label': row['video_label'],
                'method':      row['method'],
                'orig_path':   video_path,
            })
            continue

        # 영상 처리
        video_ok = os.path.exists(v_path)
        if not video_ok:
            v_tensor = load_and_transform_video(video_path)
            if v_tensor is not None:
                torch.save(v_tensor, v_path)
                video_ok = True
                del v_tensor

        # 오디오 처리
        audio_ok = os.path.exists(a_path)
        if not audio_ok:
            a_tensor = load_audio(video_path)
            if a_tensor is not None:
                torch.save(a_tensor, a_path)
                audio_ok = True
                del a_tensor

        if video_ok and audio_ok:
            n_done += 1
            manifest_rows.append({
                'index':       idx,
                'video_cache': v_path,
                'audio_cache': a_path,
                'video_label': row['video_label'],
                'method':      row['method'],
                'orig_path':   video_path,
            })
        else:
            n_failed += 1
            # 부분 실패 파일 정리
            if os.path.exists(v_path) and not audio_ok:
                os.remove(v_path)
            if os.path.exists(a_path) and not video_ok:
                os.remove(a_path)

        # 메모리 관리
        if (idx + 1) % 50 == 0:
            gc.collect()
            elapsed = time.time() - t_start
            speed = (idx + 1) / elapsed
            eta = (n_total - idx - 1) / speed
            print(f"  [{idx+1:5d}/{n_total}] "
                  f"속도: {speed:.1f}/s  ETA: {eta/60:.1f}분  "
                  f"(완료 {n_done} / skip {n_skipped} / 실패 {n_failed})")

    # manifest 저장
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(cache_dir, f'manifest_{prefix}.csv')
    manifest_df.to_csv(manifest_path, index=False, encoding='utf-8-sig')

    elapsed = time.time() - t_start
    print(f"\n✅ {prefix} 캐시 완료")
    print(f"   처리 시간: {elapsed/60:.1f}분")
    print(f"   성공: {n_done + n_skipped} (신규 {n_done} + skip {n_skipped})")
    print(f"   실패: {n_failed}")
    print(f"   manifest: {manifest_path}")

    return manifest_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        CACHE_DIR       = "cache_pgf",
        NUM_FAKE_SAMPLE = 2000,
        VAL_RATIO       = 0.1,
        SEED            = 42,
    )

    print(f"📁 Working dir: {os.getcwd()}")
    print(f"💾 Cache dir  : {os.path.abspath(CFG['CACHE_DIR'])}")

    # 디스크 공간 확인
    import shutil
    stat = shutil.disk_usage(os.getcwd())
    free_gb = stat.free / 1024**3
    print(f"💿 디스크 여유: {free_gb:.1f} GB")
    if free_gb < 2.0:
        print("⚠️  디스크 공간이 2GB 미만입니다. 캐시 생성 전 정리 권장.")

    os.makedirs(CFG['CACHE_DIR'], exist_ok=True)

    # 데이터 분할 (학습 코드와 동일한 seed)
    train_df, val_df = build_polyglotfake_train_val(
        num_fake_sample = CFG['NUM_FAKE_SAMPLE'],
        val_ratio       = CFG['VAL_RATIO'],
        seed            = CFG['SEED']
    )

    # Train 캐시
    build_cache(train_df, CFG['CACHE_DIR'], prefix="train")

    # Val 캐시
    build_cache(val_df, CFG['CACHE_DIR'], prefix="val")

    # 최종 디스크 사용량
    total_size = 0
    for root, dirs, files in os.walk(CFG['CACHE_DIR']):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))

    print("\n" + "="*60)
    print("✅ 모든 캐시 생성 완료")
    print("="*60)
    print(f"💾 캐시 폴더: {os.path.abspath(CFG['CACHE_DIR'])}")
    print(f"📦 총 크기  : {total_size/1024**3:.2f} GB")
    print(f"\n다음 단계: python train_artifact_pgf_cached.py --model both")


if __name__ == "__main__":
    main()