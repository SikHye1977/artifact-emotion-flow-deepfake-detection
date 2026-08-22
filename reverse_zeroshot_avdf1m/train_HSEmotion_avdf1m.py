"""
==============================================================================
[AV-Deepfake1M 학습] 감정 흐름 기반 딥페이크 탐지 (영상)
HSEmotion + GRU + Attention - Trained on AV-Deepfake1M

[목적]
기존 train_HSEmotion_pgf.py를 AV-Deepfake1M용으로 변환.
모델 구조 완전히 동일.

[변경점]
  1. 데이터 로더: PolyGlotFakeVideoDataset → AVDF1MVideoDataset
  2. 데이터 분할: build_polyglotfake_train_val → build_avdf1m_train_val
  3. 가중치 저장: emotion_flow_lite_avdf1m_best.pth
  4. 로그 폴더: emotion_lite_logs_avdf1m/

[실행 위치] ~/hsh/AIApplication/reverse_zeroshot_avdf1m/

[필요 파일]
- 같은 폴더: avdf1m_data.py
- 상위 폴더 (~/hsh/AIApplication/):
    * emotion_deepfake_detector_lite.py (또는 train_HSEmotion.py)
    * AV-Deepfake1M_RootFiles/ 데이터셋

[출력]
- emotion_flow_lite_avdf1m_best.pth
- emotion_lite_logs_avdf1m/history.csv
==============================================================================
"""

import sys
import os
import functools
import time

# ─ 상위 폴더를 import 경로에 추가 ──────────────────────────────────────────
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import av
from torchvision import transforms
from sklearn.metrics import roc_auc_score

torch.load = functools.partial(torch.load, weights_only=False)

# 같은 폴더의 데이터 유틸
from avdf1m_data import build_avdf1m_train_val

# 상위 폴더의 모델 클래스
try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    try:
        from train_HSEmotion import EmotionFlowDetectorLite
    except ImportError:
        print("❌ EmotionFlowDetectorLite 클래스 import 실패")
        print(f"   상위 폴더({PARENT_DIR})에서 다음 파일 중 하나가 필요:")
        print("   - emotion_deepfake_detector_lite.py")
        print("   - train_HSEmotion.py")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 영상 전처리 (PGF 학습과 동일)
# ══════════════════════════════════════════════════════════════════════════════
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


def extract_uniform_frames(video_path: str, num_frames: int = 16):
    """기존 코드와 동일한 프레임 추출."""
    try:
        container  = av.open(video_path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
    except Exception:
        return None
    if len(all_frames) < num_frames:
        return None
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    return torch.stack([frame_transform(all_frames[i]) for i in indices])


# ══════════════════════════════════════════════════════════════════════════════
# 2. AV-Deepfake1M용 Dataset
# ══════════════════════════════════════════════════════════════════════════════
class AVDF1MVideoDataset(Dataset):
    """
    AV-Deepfake1M 영상에서 16프레임을 추출.
    PolyGlotFakeVideoDataset과 입력/출력 형식 동일.
    """
    def __init__(self, df: pd.DataFrame, num_frames: int = 16):
        self.df         = df.reset_index(drop=True)
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']

        frames = extract_uniform_frames(video_path, self.num_frames)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))

        label = row['video_label']
        return frames, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 학습/검증 함수 (PGF 학습 코드 재사용)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (frames, labels) in enumerate(loader):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            loss      = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"  E{epoch:02d} [{i:3d}/{len(loader)}] Loss={loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            total_loss += criterion(logits, labels).item()
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(targets.tolist())

    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > 0.5).astype(int)
    acc = (preds == np.array(all_labels)).mean() * 100
    try: auc = roc_auc_score(all_labels, all_probs) * 100
    except: auc = 0.0
    return avg_loss, acc, auc


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        # 출력 경로 (현재 폴더 기준)
        CKPT_PATH       = "emotion_flow_lite_avdf1m_best.pth",
        LOG_DIR         = "emotion_lite_logs_avdf1m",

        # 학습 설정 (FakeAVCeleb/PolyGlotFake와 동일)
        NUM_FRAMES      = 16,
        BATCH_SIZE      = 8,
        EPOCHS          = 15,
        LR              = 1e-3,
        WEIGHT_DECAY    = 1e-4,
        GRU_HIDDEN      = 64,
        DROPOUT         = 0.3,
        MODEL_NAME      = 'enet_b0_8_best_afew',

        # AVDF1M 데이터 (Real 1000 + Fake 4000)
        N_REAL          = 1000,
        N_FAKE          = 4000,
        VAL_RATIO       = 0.1,
        NUM_WORKERS     = 3,
        SEED            = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    print(f"📁 Parent dir : {PARENT_DIR}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    # ── 데이터 분할 ─────────────────────────────────────────────────────
    train_df, val_df = build_avdf1m_train_val(
        n_real    = CFG['N_REAL'],
        n_fake    = CFG['N_FAKE'],
        val_ratio = CFG['VAL_RATIO'],
        seed      = CFG['SEED']
    )

    train_loader = DataLoader(
        AVDF1MVideoDataset(train_df, CFG['NUM_FRAMES']),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        AVDF1MVideoDataset(val_df, CFG['NUM_FRAMES']),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

    # ── 모델 ────────────────────────────────────────────────────────────
    model = EmotionFlowDetectorLite(
        model_name=CFG['MODEL_NAME'], num_frames=CFG['NUM_FRAMES'],
        gru_hidden=CFG['GRU_HIDDEN'], dropout=CFG['DROPOUT'], device='cpu'
    ).to(device)

    # pos_weight 자동 계산 (PGF/FAV 방식 그대로)
    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor(
        [num_neg / num_pos], dtype=torch.float32
    ).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f} "
          f"(neg/pos = {num_neg}/{num_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(model.parameters(), lr=CFG['LR'],
                            weight_decay=CFG['WEIGHT_DECAY'])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── 학습 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🚀 AV-Deepfake1M 학습 시작 (HSEmotion+GRU+Attention)")
    print("=" * 60)

    best_auc = 0.0
    history  = []
    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0
        print(f"\n🏁 E{epoch:02d} | Train={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | Acc={val_acc:.1f}% | "
              f"AUC={val_auc:.1f}% | {elapsed:.0f}s\n")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
        })
        pd.DataFrame(history).to_csv(
            os.path.join(CFG['LOG_DIR'], "history.csv"), index=False
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': CFG, 'epoch': epoch, 'val_auc': val_auc,
            }, CFG['CKPT_PATH'])
            print(f"  ⭐ Best 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(CFG['CKPT_PATH'])}")


if __name__ == "__main__":
    main()