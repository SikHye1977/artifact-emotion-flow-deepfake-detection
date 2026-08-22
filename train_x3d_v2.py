import sys
import os
import torch
import pandas as pd
import av
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. torchvision 호환성 패치
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

# --- 전처리 함수 ---
def rescale_video(x): return x / 255.0
def permute_to_tc(x): return x.permute(1, 0, 2, 3)
def permute_to_ct(x): return x.permute(1, 0, 2, 3)

# 2. Dataset 클래스
class FakeAVCelebDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def load_video(self, path):
        try:
            container = av.open(path)
            frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
            container.close()
            if len(frames) < 16: return None
            video = np.stack(frames)
            return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
        except Exception as e:
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.iloc[-1].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])
        
        video = self.load_video(video_path)
        if video is None: 
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform: video = self.transform(video)
        label = 0.0 if row['method'] == 'real' else 1.0
        return video, torch.tensor([label], dtype=torch.float32)

video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale_video),
    Lambda(permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224))
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = "FakeAVCeleb_v1.2"
    CSV_PATH = os.path.join(BASE_DIR, "meta_data.csv")
    CHECKPOINT_PATH = "x3d_checkpoint_final.pth"
    HISTORY_CSV = "train_history_final.csv"
    
    # 🌟 [요구사항 2] 예측 결과 저장용 폴더 생성
    LOG_DIR = "prediction_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # [1] 데이터 전처리 & 균형 맞추기
    # ---------------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    
    # 🌟 [핵심 수정] 비디오 모델의 적인 RVFA (진짜영상+가짜음성) 제외
    # (주의: meta_data.csv에 'type' 컬럼과 'RealVideo-FakeAudio' 값이 일치하는지 꼭 확인하세요!)
    if 'type' in df.columns:
        initial_len = len(df)
        df = df[df['type'] != 'RealVideo-FakeAudio']
        print(f"🧹 RVFA 데이터 {initial_len - len(df)}개 제거 완료 (순수 비디오 학습용)")

    real_df = df[df['method'] == 'real']
    fake_df = df[df['method'] != 'real']
    
    n_samples = min(len(real_df), len(fake_df))
    balanced_df = pd.concat([
        real_df.sample(n=n_samples, random_state=42),
        fake_df.sample(n=n_samples, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 Balanced Data: Total {len(balanced_df)} (Real {n_samples} : Fake {n_samples})")

    train_df, val_df = train_test_split(balanced_df, test_size=0.1, stratify=balanced_df['method'], random_state=42)
    train_loader = DataLoader(FakeAVCelebDataset(train_df, BASE_DIR, video_transform), batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(FakeAVCelebDataset(val_df, BASE_DIR, video_transform), batch_size=4, num_workers=4, pin_memory=True)

    # ---------------------------------------------------------
    # [2] 모델 설정 (Softmax 함정 해결)
    # ---------------------------------------------------------
    model = x3d_m(pretrained=True)
    model.blocks[5].proj = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity() # Softmax 무력화
    nn.init.xavier_normal_(model.blocks[5].proj.weight)
    nn.init.constant_(model.blocks[5].proj.bias, 0)
    model = model.to(device)
    
    # ---------------------------------------------------------
    # [3] 최적화 설정
    # ---------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_loss = float('inf')
    history = []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"♻️  Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if os.path.exists(HISTORY_CSV):
            history = pd.read_csv(HISTORY_CSV).to_dict('records')

    print(f"🚀 Training Started (Epoch {start_epoch}/20)")

    # ---------------------------------------------------------
    # [4] 본격적인 학습 루프
    # ---------------------------------------------------------
    for epoch in range(start_epoch, 20):
        model.train()
        epoch_start_time = time.time()
        
        train_preds_log = [] # 🌟 훈련 단계 예측 기록 리스트
        
        for i, (vids, labels) in enumerate(train_loader):
            step_start_time = time.time()
            vids, labels = vids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(vids)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 🌟 [기록] 현재 배치의 예측 결과를 저장
            with torch.no_grad():
                probs = torch.sigmoid(outputs.detach()).cpu().numpy().flatten()
                targets = labels.detach().cpu().numpy().flatten()
                for prob, target in zip(probs, targets):
                    pred_label = 1.0 if prob > 0.5 else 0.0
                    train_preds_log.append({
                        '데이터 라벨 (0:진짜, 1:가짜)': target,
                        '예측 확률 (%)': round(prob * 100, 2),
                        '모델의 판단': pred_label,
                        '정답 여부 (1:O, 0:X)': 1 if pred_label == target else 0
                    })
            
            if i % 10 == 0:
                raw_logits = np.round(outputs.detach().cpu().numpy().flatten()[:2], 3)
                print(f"E{epoch} S[{i}/{len(train_loader)}] Loss:{loss.item():.4f} | Logits:{raw_logits} | Time:{time.time()-step_start_time:.2f}s")

        # ---------------------------------------------------------
        # [5] 검증 단계
        # ---------------------------------------------------------
        model.eval()
        val_loss, correct, total = 0, 0, 0
        val_preds_log = [] # 🌟 검증 단계 예측 기록 리스트
        
        with torch.no_grad():
            for vids, labels in val_loader:
                vids, labels = vids.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(vids)
                    val_loss += criterion(outputs, labels).item()
                
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                targets = labels.cpu().numpy().flatten()
                
                # 🌟 [기록] 검증 배치의 예측 결과를 저장
                for prob, target in zip(probs, targets):
                    pred_label = 1.0 if prob > 0.5 else 0.0
                    val_preds_log.append({
                        '데이터 라벨 (0:진짜, 1:가짜)': target,
                        '예측 확률 (%)': round(prob * 100, 2),
                        '모델의 판단': pred_label,
                        '정답 여부 (1:O, 0:X)': 1 if pred_label == target else 0
                    })
                    
                    if pred_label == target: correct += 1
                    total += 1

        acc = (correct / total) * 100
        avg_val_loss = val_loss / len(val_loader)
        print(f"🏁 Epoch {epoch} Done | Val Acc: {acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        # ---------------------------------------------------------
        # [6] 에포크 결과 저장 (CSV 내보내기)
        # ---------------------------------------------------------
        # 🌟 훈련 및 검증 상세 로그를 csv로 저장 (엑셀에서 한글 안 깨지도록 utf-8-sig 사용)
        pd.DataFrame(train_preds_log).to_csv(os.path.join(LOG_DIR, f"epoch_{epoch}_train.csv"), index=False, encoding='utf-8-sig')
        pd.DataFrame(val_preds_log).to_csv(os.path.join(LOG_DIR, f"epoch_{epoch}_val.csv"), index=False, encoding='utf-8-sig')
        
        history.append({'epoch': epoch, 'val_loss': avg_val_loss, 'val_acc': acc})
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss
        }, CHECKPOINT_PATH)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "x3d_model_best_final.pth")
            print(f"⭐ Best Model Saved!")

        scheduler.step()

if __name__ == "__main__":
    main()