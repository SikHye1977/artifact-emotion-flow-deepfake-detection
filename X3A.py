import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.models.x3d import create_x3d
from tqdm import tqdm

# 1. AASIST 모듈 경로 추가
sys.path.append(os.path.join(os.getcwd(), 'aasist'))

try:
    from models.AASIST import Model as AASISTModel
    print(">> AASIST source code loaded successfully.")
except ImportError:
    print(">> Error: Cannot find AASIST in 'aasist/' folder. Check your directory structure.")
    sys.exit(1)

# 2. 커스텀 데이터셋 클래스
class MultimodalDataset(Dataset):
    def __init__(self, csv_path, data_root, num_frames=16, audio_len=64600):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.num_frames = num_frames
        self.audio_len = audio_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 사용자 폴더 구조 반영: ./FakeAVCeleb_v1.2/FakeAVCeleb/...
        file_path = os.path.join(self.data_root, row['path'])
        label = 0 if row['category'] == 'A' else 1

        # 비디오 로드 (X3D 규격: C, T, H, W)
        try:
            v_frames, _, _ = io.read_video(file_path, pts_unit='sec', output_format="TCHW")
            if v_frames.shape[0] > self.num_frames:
                v_frames = v_frames[:self.num_frames]
            else:
                pad = self.num_frames - v_frames.shape[0]
                v_frames = torch.cat([v_frames, v_frames[-1:].repeat(pad, 1, 1, 1)])
            video_data = v_frames.permute(1, 0, 2, 3).float() / 255.0
        except:
            video_data = torch.zeros((3, self.num_frames, 224, 224))

        # 오디오 로드 (AASIST 규격: Raw Waveform)
        try:
            waveform, sr = torchaudio.load(file_path)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            waveform = waveform.mean(0)
            if waveform.shape[0] >= self.audio_len:
                audio_data = waveform[:self.audio_len]
            else:
                audio_data = torch.nn.functional.pad(waveform, (0, self.audio_len - waveform.shape[0]))
        except:
            audio_data = torch.zeros(self.audio_len)

        return video_data, audio_data, torch.tensor(label, dtype=torch.float32)

# 3. 통합 모델 정의
class X3D_AASIST_Fusion(nn.Module):
    def __init__(self, aasist_config):
        super(X3D_AASIST_Fusion, self).__init__()
        # 비디오 백본 (X3D)
        self.video_net = create_x3d(input_channel=3, input_clip_length=16, input_crop_size=224)
        self.video_net.blocks[-1].proj = nn.Identity() # Feature 추출 (2048-dim)
        
        # 오디오 백본 (AASIST)
        self.audio_net = AASISTModel(aasist_config) # 160-dim feature 반환 가정

        # 퓨전 분류기
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 160, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, v, a):
        v_feat = self.video_net(v)
        _, a_feat = self.audio_net(a)
        fused = torch.cat((v_feat, a_feat), dim=1)
        return self.classifier(fused)

# 4. 메인 학습 함수
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f">> Training on: {device}")

    # AASIST 기본 설정 (실제 모델 요구사항에 맞춰 수정 가능)
    aasist_config = {
        "nb_samp": 64600, "first_conv": 128, "in_channels": 1,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32], "pool_layers": [0, 1], "pool_size": [2, 2]
    }

    # 데이터 로더
    dataset = MultimodalDataset(args.csv_path, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 모델 초기화
    model = X3D_AASIST_Fusion(aasist_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # 학습 루프
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for v, a, labels in pbar:
            v, a, labels = v.to(device), a.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(v, a)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f">> Epoch {epoch+1} Avg Loss: {running_loss/len(loader):.4f}")
        
        # 모델 저장
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./meta_data.csv')
    parser.add_argument('--data_root', type=str, default='./FakeAVCeleb_v1.2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)