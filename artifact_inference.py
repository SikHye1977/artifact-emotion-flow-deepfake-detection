"""
==============================================================================
[아티팩트 모델 재추론] X3D_m (비디오) + AASIST (오디오)
감정 모델과 동일한 1,000개 평가셋에서 아티팩트 확률 추출
==============================================================================

[목적]
기존 X3D + AASIST 확률적 OR 코드를
감정 모델과 동일한 평가셋(Real 500 + 학습 미사용 Fake 500)에서 재실행.

결과 CSV는 감정 모델의 sample_predictions.csv와 같은 샘플 순서로 생성하여
이후 4방향 확률적 OR 결합이 가능하도록 함.

[주의]
기존 99.8%는 평가셋이 불분명하고(random 1000) 학습 데이터와 겹쳤을
가능성이 높음. 감정 모델과 동일한 엄격 평가(Fake는 학습 미사용)에서는
수치가 다소 내려갈 수 있음. 이것이 진짜 일반화 성능.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import av
from torch.utils.data import Dataset, DataLoader

# torchvision 하위 호환
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist/models/AASIST.py 미발견")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리 유틸 (기존 코드와 동일)
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


def load_audio(video_path, target_sr=16000, max_length=64000):
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close()
            return None
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate
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
        if not frames:
            return None
        waveform = np.concatenate(frames, axis=-1)
        waveform = torch.from_numpy(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
            waveform = resampler(waveform)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            pad_size = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        return waveform.squeeze()
    except Exception:
        return None


def load_video(path):
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(frames) < 16:
            return None
        video = np.stack(frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class FakeAVCelebArtifactDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.base_dir  = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        rel_path   = row.iloc[-2].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])

        raw_video = load_video(video_path)
        if raw_video is None:
            return self.__getitem__((idx + 1) % len(self))

        video_tensor = self.transform(raw_video) if self.transform else raw_video

        audio_tensor = load_audio(video_path)
        if audio_tensor is None:
            return self.__getitem__((idx + 1) % len(self))

        # 라벨 구성 (기존 코드 방식 유지)
        is_video_fake = row['method'] != 'real'
        is_audio_fake = 'FakeAudio' in str(row['type'])
        final_label   = 1.0 if (is_video_fake or is_audio_fake) else 0.0

        return (video_tensor, audio_tensor,
                torch.tensor([final_label], dtype=torch.float32),
                row['method'], str(row['type']))


# ══════════════════════════════════════════════════════════════════════════════
# 3. 동일 평가셋 구성 (감정 모델과 맞춤)
# ══════════════════════════════════════════════════════════════════════════════
def build_same_eval_set(csv_path: str,
                        n_real: int = 500,
                        n_fake_eval: int = 500,
                        train_fake_n: int = 2000,
                        seed: int = 42) -> pd.DataFrame:
    """감정 모델(Multimodal_emtion_fusion.py)과 완전히 동일한 평가셋 재구성."""
    df = pd.read_csv(csv_path)
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)

    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)

    # Real: 전량(500개) 사용
    sampled_real = real_df.sample(
        n=min(n_real, len(real_df)), random_state=seed
    )

    # Fake: 학습에 썼던 2,000개를 제외하고 나머지에서 500개
    train_fake_idx = fake_df.sample(n=train_fake_n, random_state=seed).index
    unseen_fake_df = fake_df.drop(train_fake_idx).reset_index(drop=True)
    sampled_fake   = unseen_fake_df.sample(
        n=min(n_fake_eval, len(unseen_fake_df)), random_state=seed + 1
    )

    val_df = pd.concat([sampled_real, sampled_fake]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    return val_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    BASE_DIR       = "FakeAVCeleb_v1.2"
    CSV_PATH       = os.path.join(BASE_DIR, "meta_data.csv")
    X3D_WEIGHTS    = "x3d_model_best_final.pth"
    AASIST_WEIGHTS = "aasist_model_best_final.pth"
    AASIST_CONFIG  = "./aasist/config/AASIST.conf"
    OUTPUT_DIR     = "multimodal_report"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 체크포인트 확인 ────────────────────────────────────────────
    for p in [X3D_WEIGHTS, AASIST_WEIGHTS, AASIST_CONFIG]:
        if not os.path.exists(p):
            print(f"❌ 필수 파일 없음: {p}")
            sys.exit(1)

    # ── 감정 모델과 동일한 1,000개 평가셋 ─────────────────────────
    test_df = build_same_eval_set(
        CSV_PATH,
        n_real=500, n_fake_eval=500,
        train_fake_n=2000, seed=42
    )
    print(f"📊 평가셋: 총 {len(test_df)}개 "
          f"(Real={int((test_df['video_label']==0).sum())}, "
          f"Fake={int((test_df['video_label']==1).sum())})")
    print(f"   ※ 감정 모델과 동일 분할 (Fake는 100% 학습 미사용)")

    test_loader = DataLoader(
        FakeAVCelebArtifactDataset(test_df, BASE_DIR, video_transform),
        batch_size=1, shuffle=False, num_workers=4
    )

    # ── X3D 로드 ───────────────────────────────────────────────────
    print("🧠 X3D_m 로드 중...")
    x3d_model = x3d_m(pretrained=False)
    x3d_model.blocks[5].proj       = nn.Linear(2048, 1)
    x3d_model.blocks[5].activation = nn.Identity()
    x3d_model.load_state_dict(torch.load(X3D_WEIGHTS, map_location=device))
    x3d_model = x3d_model.to(device).eval()
    print("✅ X3D 로드 완료")

    # ── AASIST 로드 ────────────────────────────────────────────────
    print("🧠 AASIST 로드 중...")
    with open(AASIST_CONFIG, 'r') as f:
        config = json.load(f)
    aasist_model = AASISTModel(config['model_config'])
    aasist_model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
    aasist_model = aasist_model.to(device).eval()
    print("✅ AASIST 로드 완료")

    print("\n🚀 아티팩트 추론 시작\n")

    # ── 추론 루프 ──────────────────────────────────────────────────
    results = []
    t0 = time.time()

    with torch.no_grad():
        for i, (vids, waves, labels, methods, types) in enumerate(test_loader):
            vids, waves = vids.to(device), waves.to(device)

            with torch.amp.autocast('cuda'):
                # 비디오 아티팩트 확률
                v_logit = x3d_model(vids)
                v_prob  = torch.sigmoid(v_logit).item()

                # 오디오 아티팩트 확률
                _, a_out = aasist_model(waves)
                a_prob   = torch.softmax(a_out, dim=1)[0, 1].item()

            target  = labels.item()
            fused   = 1.0 - ((1.0 - v_prob) * (1.0 - a_prob))
            pred    = 1 if fused > 0.5 else 0
            correct = int(pred == int(target))

            # 모달리티 레이블 (기존 방식)
            is_v_fake = methods[0] != 'real'
            is_a_fake = 'FakeAudio' in str(types[0])
            if is_v_fake and is_a_fake:       modal = "FVFA"
            elif is_v_fake and not is_a_fake: modal = "FVRA"
            elif not is_v_fake and is_a_fake: modal = "RVFA"
            else:                              modal = "RVRA"

            results.append({
                '실제_레이블(0:Real 1:Fake)':  int(target),
                '조작_모달리티':                modal,
                '원본_기법':                    methods[0],
                '원본_type':                    str(types[0]),
                '비디오아티팩트_확률(%)':        round(v_prob * 100, 2),
                '오디오아티팩트_확률(%)':        round(a_prob * 100, 2),
                '아티팩트_OR_확률(%)':           round(fused  * 100, 2),
                '아티팩트_예측':                 pred,
                '정답여부':                      correct,
            })

            if (i + 1) % 50 == 0:
                acc = sum(r['정답여부'] for r in results) / len(results) * 100
                print(f"  [{i+1:4d}/{len(test_loader)}] 진행 중... "
                      f"현재 Acc: {acc:.2f}%")

    elapsed = time.time() - t0

    # ── 결과 저장 ──────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "artifact_predictions.csv")
    pd.DataFrame(results).to_csv(out_path, index=False, encoding='utf-8-sig')

    # ── 최종 지표 ──────────────────────────────────────────────────
    total   = len(results)
    correct = sum(r['정답여부'] for r in results)
    acc     = correct / total * 100

    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    labels  = np.array([r['실제_레이블(0:Real 1:Fake)'] for r in results])
    probs   = np.array([r['아티팩트_OR_확률(%)'] for r in results]) / 100.0
    preds   = (probs > 0.5).astype(int)

    auc  = roc_auc_score(labels, probs) * 100
    prec = precision_score(labels, preds, zero_division=0) * 100
    rec  = recall_score(labels, preds, zero_division=0) * 100
    f1   = f1_score(labels, preds, zero_division=0) * 100

    print("\n" + "=" * 60)
    print("🏆 아티팩트 모델 (X3D + AASIST OR) 최종 결과")
    print("=" * 60)
    print(f"  평가셋     : {total}개 (감정 모델과 동일)")
    print(f"  AUC        : {auc:.2f}%")
    print(f"  Accuracy   : {acc:.2f}%")
    print(f"  Precision  : {prec:.2f}%")
    print(f"  Recall     : {rec:.2f}%")
    print(f"  F1         : {f1:.2f}%")
    print(f"  소요 시간  : {elapsed:.1f}초")
    print(f"  저장       : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()