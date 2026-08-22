"""
==============================================================================
[PolyGlotFake 학습] X3D_m + AASIST (아티팩트 탐지)

[실행 위치] ~/hsh/AIApplication/reverse_zero_shot/
[필요 파일]
- 같은 폴더: polyglotfake_data.py
- 상위 폴더: aasist/ 폴더 (config + models)

[출력 (현재 폴더)]
- x3d_model_pgf_best.pth
- aasist_model_pgf_best.pth

[사용법]
  python train_artifact_pgf.py --model x3d
  python train_artifact_pgf.py --model aasist
  python train_artifact_pgf.py --model both
==============================================================================
"""

import sys
import os
import argparse
import functools
import json
import time

# 경로 설정
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import av
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

torch.load = functools.partial(torch.load, weights_only=False)

from polyglotfake_data import build_polyglotfake_train_val

try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist 모듈 import 실패")
    print(f"   상위 폴더({PARENT_DIR})에 aasist/ 폴더가 있어야 합니다")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리
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
    try:
        container = av.open(path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(all_frames) < 16: return None
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
        return waveform.squeeze()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class PolyGlotFakeArtifactDataset(Dataset):
    def __init__(self, df, mode='video'):
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
            if raw is None: return self.__getitem__((idx + 1) % len(self))
            tensor = video_transform(raw)
        else:
            tensor = load_audio_for_aasist(video_path)
            if tensor is None: return self.__getitem__((idx + 1) % len(self))

        return tensor, torch.tensor([row['video_label']], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. X3D 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_x3d(train_df, val_df, device, cfg):
    print("\n" + "="*60)
    print("🚀 X3D_m 학습 시작 (PolyGlotFake)")
    print("="*60)

    train_loader = DataLoader(
        PolyGlotFakeArtifactDataset(train_df, mode='video'),
        batch_size=cfg['BATCH_X3D'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        PolyGlotFakeArtifactDataset(val_df, mode='video'),
        batch_size=cfg['BATCH_X3D'],
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )

    model = x3d_m(pretrained=True)
    model.blocks[5].proj       = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity()
    model = model.to(device)

    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_w = torch.tensor([num_neg/num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['LR_X3D'], weight_decay=1e-4)
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
            'val_acc': acc, 'val_auc': auc
        })
        os.makedirs(cfg['X3D_LOG_DIR'], exist_ok=True)
        pd.DataFrame(history).to_csv(
            os.path.join(cfg['X3D_LOG_DIR'], 'history.csv'), index=False
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), cfg['X3D_CKPT'])
            print(f"  ⭐ X3D Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ X3D 완료. Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(cfg['X3D_CKPT'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. AASIST 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_aasist(train_df, val_df, device, cfg):
    print("\n" + "="*60)
    print("🚀 AASIST 학습 시작 (PolyGlotFake)")
    print("="*60)

    train_loader = DataLoader(
        PolyGlotFakeArtifactDataset(train_df, mode='audio'),
        batch_size=cfg['BATCH_AASIST'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        PolyGlotFakeArtifactDataset(val_df, mode='audio'),
        batch_size=cfg['BATCH_AASIST'],
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )

    with open(cfg['AASIST_CONFIG'], 'r') as f:
        config = json.load(f)
    model = AASISTModel(config['model_config']).to(device)

    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    weights = torch.tensor([1.0, num_neg/num_pos], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['LR_AASIST'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_auc = 0.0
    history = []
    for epoch in range(cfg['EPOCHS']):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for i, (waves, labels) in enumerate(train_loader):
            waves = waves.to(device)
            labels_long = labels.squeeze(1).long().to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
                _, outputs = model(waves)
                loss = criterion(outputs, labels_long)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 10 == 0:
                print(f"  E{epoch:02d} [{i:3d}/{len(train_loader)}] Loss={loss.item():.4f}")

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
            'val_acc': acc, 'val_auc': auc
        })
        os.makedirs(cfg['AASIST_LOG_DIR'], exist_ok=True)
        pd.DataFrame(history).to_csv(
            os.path.join(cfg['AASIST_LOG_DIR'], 'history.csv'), index=False
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), cfg['AASIST_CKPT'])
            print(f"  ⭐ AASIST Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ AASIST 완료. Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(cfg['AASIST_CKPT'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='both',
                        choices=['x3d', 'aasist', 'both'])
    args = parser.parse_args()

    CFG = dict(
        # 출력 (현재 폴더)
        X3D_CKPT        = "x3d_model_pgf_best.pth",
        AASIST_CKPT     = "aasist_model_pgf_best.pth",
        X3D_LOG_DIR     = "x3d_logs_pgf",
        AASIST_LOG_DIR  = "aasist_logs_pgf",

        # 입력 (상위 폴더)
        AASIST_CONFIG   = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf"),

        EPOCHS          = 10,
        BATCH_X3D       = 4,
        BATCH_AASIST    = 16,
        LR_X3D          = 1e-4,
        LR_AASIST       = 5e-5,
        VAL_RATIO       = 0.1,
        NUM_WORKERS     = 3,
        NUM_FAKE_SAMPLE = 2000,
        SEED            = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")

    if not os.path.exists(CFG['AASIST_CONFIG']):
        print(f"❌ AASIST config 없음: {CFG['AASIST_CONFIG']}")
        sys.exit(1)

    train_df, val_df = build_polyglotfake_train_val(
        num_fake_sample=CFG['NUM_FAKE_SAMPLE'],
        val_ratio=CFG['VAL_RATIO'], seed=CFG['SEED']
    )

    if args.model in ['x3d', 'both']:
        train_x3d(train_df, val_df, device, CFG)

    if args.model in ['aasist', 'both']:
        train_aasist(train_df, val_df, device, CFG)


if __name__ == "__main__":
    main()