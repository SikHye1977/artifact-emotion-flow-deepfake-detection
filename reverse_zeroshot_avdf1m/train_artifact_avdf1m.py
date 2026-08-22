"""
==============================================================================
[AV-Deepfake1M 학습] 비디오/오디오 아티팩트 탐지
X3D_m + AASIST - Trained on AV-Deepfake1M

[목적]
기존 train_artifact_pgf.py를 AV-Deepfake1M용으로 변환.
두 모델(X3D, AASIST)을 같은 Dataset에서 동시에 학습.

[변경점] 데이터만 AV-Deepfake1M. 모델 구조 동일.

[실행 위치] ~/hsh/AIApplication/reverse_zeroshot_avdf1m/

[필요 파일]
- 같은 폴더: avdf1m_data.py
- 상위 폴더 (~/hsh/AIApplication/):
    * aasist/ 폴더
    * AV-Deepfake1M_RootFiles/ 데이터셋

[사용법]
  # X3D만 학습
  python train_artifact_avdf1m.py --model x3d
  
  # AASIST만 학습
  python train_artifact_avdf1m.py --model aasist
  
  # 둘 다 순차 학습
  python train_artifact_avdf1m.py --model both

[출력]
- x3d_model_avdf1m_best.pth
- aasist_model_avdf1m_best.pth
==============================================================================
"""

import sys
import os
import argparse
import functools
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import av
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# 경로 설정 (상위 폴더 import 가능하게)
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# torchvision 하위 호환
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

# PyTorch 2.6+ 보안 정책 우회
torch.load = functools.partial(torch.load, weights_only=False)

# 공통 데이터 로더
from avdf1m_data import build_avdf1m_train_val

# AASIST (상위 폴더)
try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist/models/AASIST.py 미발견")
    print(f"   {PARENT_DIR}/aasist/ 폴더 확인 필요")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리 (PGF 학습과 동일)
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


def load_video_for_x3d(path, max_frames=128):
    """긴 영상은 max_frames로 사전 다운샘플."""
    try:
        container = av.open(path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(all_frames) < 16:
            return None
        if len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
            all_frames = [all_frames[i] for i in indices]
        video = np.stack(all_frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


def load_audio_for_aasist(video_path, target_sr=16000, max_length=64000):
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


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset (X3D / AASIST 공통) — PGF 학습과 동일 패턴
# ══════════════════════════════════════════════════════════════════════════════
class AVDF1MArtifactDataset(Dataset):
    """
    mode='video': X3D용 영상 텐서 반환
    mode='audio': AASIST용 오디오 텐서 반환
    """
    def __init__(self, df: pd.DataFrame, mode: str = 'video'):
        assert mode in ['video', 'audio']
        self.df   = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']

        if self.mode == 'video':
            raw = load_video_for_x3d(video_path)
            if raw is None:
                return self.__getitem__((idx + 1) % len(self))
            tensor = video_transform(raw)
        else:  # audio
            tensor = load_audio_for_aasist(video_path)
            if tensor is None:
                return self.__getitem__((idx + 1) % len(self))

        label = row['video_label']
        return tensor, torch.tensor([label], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. X3D 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_x3d(train_df, val_df, device, cfg):
    print("\n" + "=" * 60)
    print("🚀 X3D_m 학습 시작 (AV-Deepfake1M)")
    print("=" * 60)

    train_loader = DataLoader(
        AVDF1MArtifactDataset(train_df, mode='video'),
        batch_size=cfg['BATCH_X3D'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        AVDF1MArtifactDataset(val_df, mode='video'),
        batch_size=cfg['BATCH_X3D'], num_workers=cfg['NUM_WORKERS'],
        pin_memory=True
    )

    model = x3d_m(pretrained=True)
    model.blocks[5].proj       = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity()
    model = model.to(device)

    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_w = torch.tensor([num_neg/num_pos], dtype=torch.float32).to(device)
    print(f"⚖️  BCE pos_weight: {pos_w.item():.4f} "
          f"(neg/pos = {num_neg}/{num_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['LR_X3D'],
                            weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_auc = 0.0
    history = []
    for epoch in range(cfg['EPOCHS']):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for i, (vids, labels) in enumerate(train_loader):
            vids, labels = vids.to(device), labels.squeeze(1).to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                logits = model(vids).squeeze(1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 10 == 0:
                print(f"  E{epoch:02d} [{i:3d}/{len(train_loader)}] Loss={loss.item():.4f}")

        # Validation
        model.eval()
        all_probs, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for vids, labels in val_loader:
                vids, labels = vids.to(device), labels.squeeze(1).to(device)
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    logits = model(vids).squeeze(1)
                    val_loss += criterion(logits, labels).item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader)
        preds = (np.array(all_probs) > 0.5).astype(int)
        acc = (preds == np.array(all_labels)).mean() * 100
        try: auc = roc_auc_score(all_labels, all_probs) * 100
        except: auc = 0.0

        print(f"\n🏁 E{epoch:02d} | Train={avg_train:.4f} | Val={avg_val:.4f} | "
              f"Acc={acc:.1f}% | AUC={auc:.1f}% | {time.time()-t0:.0f}s")

        history.append({
            'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val,
            'val_acc': acc, 'val_auc': auc,
        })
        pd.DataFrame(history).to_csv(cfg['X3D_LOG'], index=False)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), cfg['X3D_CKPT'])
            print(f"  ⭐ X3D Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ X3D 완료. Best AUC: {best_auc:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 4. AASIST 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_aasist(train_df, val_df, device, cfg):
    print("\n" + "=" * 60)
    print("🚀 AASIST 학습 시작 (AV-Deepfake1M)")
    print("=" * 60)

    train_loader = DataLoader(
        AVDF1MArtifactDataset(train_df, mode='audio'),
        batch_size=cfg['BATCH_AASIST'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        AVDF1MArtifactDataset(val_df, mode='audio'),
        batch_size=cfg['BATCH_AASIST'], num_workers=cfg['NUM_WORKERS'],
        pin_memory=True
    )

    with open(cfg['AASIST_CONFIG'], 'r') as f:
        config = json.load(f)
    model = AASISTModel(config['model_config']).to(device)

    # 클래스 가중치 (CrossEntropy용)
    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    weights = torch.tensor([1.0, num_neg/num_pos], dtype=torch.float32).to(device)
    print(f"⚖️  CE class weights: [1.0, {(num_neg/num_pos):.4f}] "
          f"(neg/pos = {num_neg}/{num_pos})")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['LR_AASIST'],
                            weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_auc = 0.0
    history = []
    for epoch in range(cfg['EPOCHS']):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for i, (waves, labels) in enumerate(train_loader):
            waves = waves.to(device)
            labels = labels.squeeze(1).long().to(device)  # CE는 long type
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                _, outputs = model(waves)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 10 == 0:
                print(f"  E{epoch:02d} [{i:3d}/{len(train_loader)}] Loss={loss.item():.4f}")

        # Validation
        model.eval()
        all_probs, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for waves, labels in val_loader:
                waves = waves.to(device)
                labels_long = labels.squeeze(1).long().to(device)
                with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                    _, outputs = model(waves)
                    val_loss += criterion(outputs, labels_long).item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.squeeze(1).numpy().tolist())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(val_loader)
        preds = (np.array(all_probs) > 0.5).astype(int)
        acc = (preds == np.array(all_labels)).mean() * 100
        try: auc = roc_auc_score(all_labels, all_probs) * 100
        except: auc = 0.0

        print(f"\n🏁 E{epoch:02d} | Train={avg_train:.4f} | Val={avg_val:.4f} | "
              f"Acc={acc:.1f}% | AUC={auc:.1f}% | {time.time()-t0:.0f}s")

        history.append({
            'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val,
            'val_acc': acc, 'val_auc': auc,
        })
        pd.DataFrame(history).to_csv(cfg['AASIST_LOG'], index=False)

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), cfg['AASIST_CKPT'])
            print(f"  ⭐ AASIST Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ AASIST 완료. Best AUC: {best_auc:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='both',
                        choices=['x3d', 'aasist', 'both'])
    args = parser.parse_args()

    CFG = dict(
        # AV-Deepfake1M 학습 가중치 출력
        X3D_CKPT        = "x3d_model_avdf1m_best.pth",
        AASIST_CKPT     = "aasist_model_avdf1m_best.pth",
        X3D_LOG         = "x3d_avdf1m_history.csv",
        AASIST_LOG      = "aasist_avdf1m_history.csv",

        # AASIST config (상위 폴더)
        AASIST_CONFIG   = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf"),

        # 학습 설정 (PGF 학습과 동일)
        EPOCHS          = 10,
        BATCH_X3D       = 4,
        BATCH_AASIST    = 16,
        LR_X3D          = 1e-4,
        LR_AASIST       = 5e-5,
        VAL_RATIO       = 0.1,
        NUM_WORKERS     = 3,

        # AVDF1M 학습 데이터 (Real 1000 + Fake 4000)
        N_REAL          = 1000,
        N_FAKE          = 4000,
        SEED            = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    print(f"📁 Parent dir : {PARENT_DIR}")

    # AASIST config 확인
    if not os.path.exists(CFG['AASIST_CONFIG']):
        print(f"❌ AASIST config 없음: {CFG['AASIST_CONFIG']}")
        sys.exit(1)

    # 데이터 분할
    train_df, val_df = build_avdf1m_train_val(
        n_real    = CFG['N_REAL'],
        n_fake    = CFG['N_FAKE'],
        val_ratio = CFG['VAL_RATIO'],
        seed      = CFG['SEED']
    )

    if args.model in ['x3d', 'both']:
        train_x3d(train_df, val_df, device, CFG)

    if args.model in ['aasist', 'both']:
        train_aasist(train_df, val_df, device, CFG)


if __name__ == "__main__":
    main()