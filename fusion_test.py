import os
import json
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 💡 호환성을 위해 torch에서 직접 amp를 가져옵니다.
from torch import amp 
import timm 
from tqdm import tqdm
from PIL import Image

# ==========================================
# 1. 경로 및 하이퍼파라미터 설정
# ==========================================
TRAIN_PATH = r"C:\Users\dudgh\han\AIApplication\dfdc_train_part_00\dfdc_train_part_0"
TEST_PATH = r"C:\Users\dudgh\han\AIApplication\train_sample_videos"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 1 
MAX_FRAMES = 16 
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 오디오 전처리 (2D Mel-Spectrogram)
# ==========================================
def get_mel_spectrogram(v_path, sr=16000, n_mels=128):
    try:
        y, _ = librosa.load(v_path, sr=sr, duration=4.0)
        if len(y) == 0: return None
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        return mel_db
    except:
        return None

# ==========================================
# 3. 데이터셋 클래스
# ==========================================
class DFDCDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        meta_json_path = os.path.join(root_dir, "metadata.json")
        with open(meta_json_path, 'r') as f:
            meta_data = json.load(f)
            for vid_name, info in meta_data.items():
                v_path = os.path.join(root_dir, vid_name)
                if os.path.exists(v_path):
                    self.samples.append({'name': vid_name, 'path': v_path, 'label': 1 if info['label'] == 'FAKE' else 0})

        self.v_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cap = cv2.VideoCapture(s['path'])
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_f - 1, MAX_FRAMES).astype(int)
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.v_trans(Image.fromarray(frame)))
        cap.release()

        v_tensor = torch.stack(frames) if frames else torch.zeros(1, 3, 224, 224)
        mel = get_mel_spectrogram(s['path'])
        if mel is not None:
            a_tensor = torch.FloatTensor(mel).unsqueeze(0)
            a_tensor = F.interpolate(a_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        else:
            a_tensor = torch.zeros(1, 128, 128)

        return v_tensor, a_tensor, torch.tensor(s['label'], dtype=torch.float32), s['name']

# ==========================================
# 4. 모델 구조
# ==========================================
class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_net = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
        self.audio_net = timm.create_model('resnet18', pretrained=True, in_chans=1, num_classes=1)
        self.fusion = nn.Linear(2, 1)

    def forward(self, v, a):
        b, t, c, h, w = v.size()
        v = v.view(b * t, c, h, w)
        v_out = torch.sigmoid(self.visual_net(v)).view(b, t, 1)
        v_score = torch.mean(v_out, dim=1) 
        a_score = torch.sigmoid(self.audio_net(a))
        f_logit = self.fusion(torch.cat([v_score, a_score], dim=1))
        return f_logit, v_score, a_score

# ==========================================
# 5. 검증 및 상세 CSV 저장 함수
# ==========================================
def validate_and_save(model, loader, device, epoch):
    model.eval()
    results, correct, total = [], 0, 0
    with torch.no_grad():
        for v, a, l, names in tqdm(loader, desc=f"Epoch {epoch} Val"):
            v, a, l = v.to(device), a.to(device), l.to(device).unsqueeze(1)
            
            # 💡 최신 권장 방식: 'cuda' 문자열을 첫 번째 인자로 전달
            with amp.autocast('cuda'): 
                f_logit, v_s, a_s = model(v, a)
                f_s = torch.sigmoid(f_logit)
            
            correct += ((f_s > 0.5).float() == l).sum().item()
            total += l.size(0)
            for i in range(len(names)):
                results.append({
                    'video_name': names[i], 'vision_score': f"{v_s[i].item():.4f}",
                    'audio_score': f"{a_s[i].item():.4f}", 'fusion_score': f"{f_s[i].item():.4f}",
                    'prediction': 'FAKE' if f_s[i].item() > 0.5 else 'REAL',
                    'actual': 'FAKE' if l[i].item() == 1 else 'REAL'
                })
    df = pd.DataFrame(results)
    path = os.path.join(SAVE_DIR, f"val_results_epoch_{epoch}.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return (correct / total) * 100, path

# ==========================================
# 6. 메인 실행 루프
# ==========================================
def main():
    train_loader = DataLoader(DFDCDataset(TRAIN_PATH), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DFDCDataset(TEST_PATH), batch_size=BATCH_SIZE, shuffle=False)
    
    model = LateFusionModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss() 
    
    # 💡 GradScaler는 인자 없이 초기화 (버전 호환성)
    scaler = amp.GradScaler() 

    print(f"🚀 학습 시작 (안정화된 AMP 적용)")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for v, a, l, _ in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            v, a, l = v.to(DEVICE), a.to(DEVICE), l.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            
            # 💡 autocast에는 'cuda'를 명시적으로 전달
            with amp.autocast('cuda'):
                f_logit, _, _ = model(v, a)
                loss = criterion(f_logit, l)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        acc, path = validate_and_save(model, test_loader, DEVICE, epoch)
        print(f"✅ Epoch {epoch} 완료! 정확도: {acc:.2f}% | 결과: {path}")

if __name__ == "__main__":
    main()