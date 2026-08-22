import sys
import os
import torch
import pandas as pd
import av
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. torchvision 호환성 패치 (최신 버전 대응)
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

# ---------------------------------------------------------
# [수정사항] Windows 멀티프로세싱을 위한 함수 정의 (Pickle 가능)
# ---------------------------------------------------------
def rescale_video(x):
    return x / 255.0

def permute_to_tc(x):
    return x.permute(1, 0, 2, 3)

def permute_to_ct(x):
    return x.permute(1, 0, 2, 3)

# ---------------------------------------------------------
# 2. Dataset 클래스
# ---------------------------------------------------------
class FakeAVCelebDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_video(self, path):
        try:
            container = av.open(path)
            frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
            container.close()
            if len(frames) < 16: return None
            video = np.stack(frames)
            # [T, H, W, C] -> [C, T, H, W]
            return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
        except:
            return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # CSV의 경로 형식을 로컬 폴더 구조에 맞게 수정
        rel_path = row.iloc[-1].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])
        
        video = self.load_video(video_path)
        if video is None: 
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform: 
            video = self.transform(video)
        
        label = 0.0 if row['method'] == 'real' else 1.0
        return video, torch.tensor([label], dtype=torch.float32)

# ---------------------------------------------------------
# 3. 비디오 전처리 (Lambda 대신 정의된 함수 사용)
# ---------------------------------------------------------
video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale_video),
    Lambda(permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224))
])

# ---------------------------------------------------------
# 4. 메인 학습 루프
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 경로 설정
    BASE_DIR = "FakeAVCeleb_v1.2"
    CSV_PATH = os.path.join(BASE_DIR, "meta_data.csv")

    if not os.path.exists(CSV_PATH):
        print(f"❌ 에러: {CSV_PATH} 파일을 찾을 수 없습니다.")
        return

    # 데이터 로드 및 분할
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['method'], random_state=42)

    train_ds = FakeAVCelebDataset(train_df, BASE_DIR, transform=video_transform)
    val_ds = FakeAVCelebDataset(val_df, BASE_DIR, transform=video_transform)

    # [수정] 속도 향상을 위해 num_workers=4, pin_memory=True 설정
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4, pin_memory=True)

    # 모델 설정
    model = x3d_m(pretrained=True)
    model.blocks[5].proj = nn.Linear(2048, 1)
    model = model.to(device)

    # [수정] RTX 3060 가속을 위한 Mixed Precision 설정
    scaler = torch.cuda.amp.GradScaler()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    epochs = 20
    best_val_loss = float('inf')

    print(f"🚀 학습 시작 (Train: {len(train_ds)}, Val: {len(val_ds)})")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (vids, labels) in enumerate(train_loader):
            vids, labels = vids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # [수정] FP16 가속 적용
            with torch.cuda.amp.autocast():
                outputs = model(vids)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # 검증 단계
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for vids, labels in val_loader:
                vids, labels = vids.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(vids)
                    val_loss += criterion(outputs, labels).item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = (correct / total) * 100
        print(f"✅ Epoch {epoch} 완료 | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")

        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "x3d_fakeavceleb_best.pth")
            print(f"⭐ Best 모델 저장 (Loss: {avg_val_loss:.4f})")

if __name__ == "__main__":
    main()