"""
==============================================================================
[캐시 기반 학습] X3D_m + AASIST (PolyGlotFake)
==============================================================================

[전제 조건]
build_artifact_cache.py를 먼저 실행해서 cache_pgf/ 폴더가 있어야 함.

[속도 비교]
                    기존 (디코딩)    캐시 사용
  X3D 1 epoch        ~6분            ~1분 ⚡
  AASIST 1 epoch     ~3분            ~30초 ⚡
  
  10 epoch 기준:
  X3D    : 60분 → 10분
  AASIST : 30분 → 5분
  합계   : 90분 → 15분 (6배 빠름)

[메모리 효율]
  - 영상 디코딩 없음 → RAM 사용량 90% 감소
  - VSCode 닫지 않아도 안정적

[사용법]
  python train_artifact_pgf_cached.py --model x3d
  python train_artifact_pgf_cached.py --model aasist
  python train_artifact_pgf_cached.py --model both
==============================================================================
"""

import sys
import os
import argparse
import functools
import json
import time

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
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

torch.load = functools.partial(torch.load, weights_only=False)

try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist 모듈 import 실패")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 캐시 기반 Dataset (디코딩 없음, 매우 빠름)
# ══════════════════════════════════════════════════════════════════════════════
class CachedArtifactDataset(Dataset):
    """
    사전 추출된 .pt 파일에서 텐서를 직접 로드.
    
    장점:
    - 디코딩 없음 → 워커 1개로도 빠름
    - 메모리 사용량 적음 → 워커 여러 개 안전
    """
    def __init__(self, manifest_df, mode='video'):
        assert mode in ['video', 'audio']
        self.df   = manifest_df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.mode == 'video':
            # fp16으로 저장됨 → 학습 시 fp32로 변환
            tensor = torch.load(row['video_cache'], map_location='cpu').to(torch.float32)
        else:
            tensor = torch.load(row['audio_cache'], map_location='cpu').to(torch.float32)

        return tensor, torch.tensor([row['video_label']], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. X3D 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_x3d(train_manifest, val_manifest, device, cfg):
    print("\n" + "="*60)
    print("🚀 X3D_m 학습 시작 (캐시 사용)")
    print("="*60)

    train_loader = DataLoader(
        CachedArtifactDataset(train_manifest, mode='video'),
        batch_size=cfg['BATCH_X3D'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        CachedArtifactDataset(val_manifest, mode='video'),
        batch_size=cfg['BATCH_X3D'],
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )

    model = x3d_m(pretrained=True)
    model.blocks[5].proj       = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity()
    model = model.to(device)

    num_neg = (train_manifest['video_label']==0).sum()
    num_pos = (train_manifest['video_label']==1).sum()
    pos_w = torch.tensor([num_neg/num_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['LR_X3D'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_auc = 0.0
    history = []
    log_dir = cfg['X3D_LOG_DIR']
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg['EPOCHS']):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for i, (vids, labels) in enumerate(train_loader):
            vids   = vids.to(device, non_blocking=True)
            labels = labels.squeeze(1).to(device, non_blocking=True)
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

            if i % 20 == 0:
                print(f"  E{epoch:02d} [{i:3d}/{len(train_loader)}] Loss={loss.item():.4f}")

        # Validation
        model.eval()
        all_probs, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for vids, labels in val_loader:
                vids   = vids.to(device, non_blocking=True)
                labels = labels.squeeze(1).to(device, non_blocking=True)
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

        elapsed = time.time() - t0
        print(f"\n🏁 E{epoch:02d} | Train={avg_train:.4f} | Val={avg_val:.4f} | "
              f"Acc={acc:.1f}% | AUC={auc:.1f}% | {elapsed:.0f}s")

        history.append({
            'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val,
            'val_acc': acc, 'val_auc': auc
        })
        pd.DataFrame(history).to_csv(
            os.path.join(log_dir, 'history.csv'), index=False
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), cfg['X3D_CKPT'])
            print(f"  ⭐ X3D Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ X3D 완료. Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(cfg['X3D_CKPT'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. AASIST 학습
# ══════════════════════════════════════════════════════════════════════════════
def train_aasist(train_manifest, val_manifest, device, cfg):
    print("\n" + "="*60)
    print("🚀 AASIST 학습 시작 (캐시 사용)")
    print("="*60)

    train_loader = DataLoader(
        CachedArtifactDataset(train_manifest, mode='audio'),
        batch_size=cfg['BATCH_AASIST'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        CachedArtifactDataset(val_manifest, mode='audio'),
        batch_size=cfg['BATCH_AASIST'],
        num_workers=cfg['NUM_WORKERS'], pin_memory=True
    )

    with open(cfg['AASIST_CONFIG'], 'r') as f:
        config = json.load(f)
    model = AASISTModel(config['model_config']).to(device)

    num_neg = (train_manifest['video_label']==0).sum()
    num_pos = (train_manifest['video_label']==1).sum()
    weights = torch.tensor([1.0, num_neg/num_pos], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['LR_AASIST'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_auc = 0.0
    history = []
    log_dir = cfg['AASIST_LOG_DIR']
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(cfg['EPOCHS']):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for i, (waves, labels) in enumerate(train_loader):
            waves       = waves.to(device, non_blocking=True)
            labels_long = labels.squeeze(1).long().to(device, non_blocking=True)
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

            if i % 20 == 0:
                print(f"  E{epoch:02d} [{i:3d}/{len(train_loader)}] Loss={loss.item():.4f}")

        # Validation
        model.eval()
        all_probs, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for waves, labels in val_loader:
                waves       = waves.to(device, non_blocking=True)
                labels_long = labels.squeeze(1).long().to(device, non_blocking=True)
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

        elapsed = time.time() - t0
        print(f"\n🏁 E{epoch:02d} | Train={avg_train:.4f} | Val={avg_val:.4f} | "
              f"Acc={acc:.1f}% | AUC={auc:.1f}% | {elapsed:.0f}s")

        history.append({
            'epoch': epoch, 'train_loss': avg_train, 'val_loss': avg_val,
            'val_acc': acc, 'val_auc': auc
        })
        pd.DataFrame(history).to_csv(
            os.path.join(log_dir, 'history.csv'), index=False
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), cfg['AASIST_CKPT'])
            print(f"  ⭐ AASIST Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ AASIST 완료. Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(cfg['AASIST_CKPT'])}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='both',
                        choices=['x3d', 'aasist', 'both'])
    args = parser.parse_args()

    CFG = dict(
        # 캐시 폴더
        CACHE_DIR       = "cache_pgf",

        # 출력 (현재 폴더)
        X3D_CKPT        = "x3d_model_pgf_best.pth",
        AASIST_CKPT     = "aasist_model_pgf_best.pth",
        X3D_LOG_DIR     = "x3d_logs_pgf",
        AASIST_LOG_DIR  = "aasist_logs_pgf",

        # 상위 폴더
        AASIST_CONFIG   = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf"),

        # 학습 설정
        EPOCHS          = 10,
        BATCH_X3D       = 4,
        BATCH_AASIST    = 16,
        LR_X3D          = 1e-4,
        LR_AASIST       = 5e-5,
        # 캐시 사용 시 워커 늘려도 메모리 부담 적음
        NUM_WORKERS     = 4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")

    # 캐시 manifest 로드
    train_manifest_path = os.path.join(CFG['CACHE_DIR'], 'manifest_train.csv')
    val_manifest_path   = os.path.join(CFG['CACHE_DIR'], 'manifest_val.csv')

    if not os.path.exists(train_manifest_path):
        print(f"❌ 캐시 manifest 없음: {train_manifest_path}")
        print(f"   먼저 실행: python build_artifact_cache.py")
        sys.exit(1)

    train_manifest = pd.read_csv(train_manifest_path)
    val_manifest   = pd.read_csv(val_manifest_path)

    print(f"📦 캐시 로드 완료")
    print(f"   Train: {len(train_manifest)}개 "
          f"(Real {(train_manifest['video_label']==0).sum()}, "
          f"Fake {(train_manifest['video_label']==1).sum()})")
    print(f"   Val  : {len(val_manifest)}개 "
          f"(Real {(val_manifest['video_label']==0).sum()}, "
          f"Fake {(val_manifest['video_label']==1).sum()})")

    if not os.path.exists(CFG['AASIST_CONFIG']):
        print(f"❌ AASIST config 없음: {CFG['AASIST_CONFIG']}")
        sys.exit(1)

    # 학습 실행
    if args.model in ['x3d', 'both']:
        train_x3d(train_manifest, val_manifest, device, CFG)

    if args.model in ['aasist', 'both']:
        train_aasist(train_manifest, val_manifest, device, CFG)


if __name__ == "__main__":
    main()