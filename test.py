import sys
import os
import torch
import pandas as pd
import av
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. torchvision 호환성 패치 (가장 먼저 실행)
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    ShortSideScale,
)
from torchvision.transforms import Compose, Lambda, Normalize, Resize

# 2. Dataset 클래스 정의
class FakeAVCelebDataset(Dataset):
    def __init__(self, csv_path, base_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_video(self, path):
        container = av.open(path)
        frames = []
        try:
            for frame in container.decode(video=0):
                frames.append(frame.to_rgb().to_ndarray())
        except Exception as e:
            return None
        finally:
            container.close()
        
        # (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
        return video

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 경로 생성 로직: 
        # CSV 마지막 열의 'FakeAVCeleb/' 부분을 'FakeAVCeleb_v1.2/'로 교체
        rel_path = row.iloc[-1].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])
        
        video = self.load_video(video_path)
        if video is None:
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform:
            video = self.transform(video)
            
        # Label: real 이면 0, 아니면 1
        label = 0 if row['method'] == 'real' else 1
        return video, label

# 3. 메인 확인 코드
def check_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 경로 설정
    BASE_DIR = "FakeAVCeleb_v1.2"
    CSV_PATH = "./FakeAVCeleb_v1.2/meta_data.csv"

    # 비디오 변환 설정 (X3D 표준)
    video_transform = Compose([
    UniformTemporalSubsample(16),  # 1) 16프레임 추출 -> (3, 16, H, W)
    Lambda(lambda x: x / 255.0),   # 2) 0~1 스케일링
    
    # --- 추가된 부분: Normalize를 위해 차원 변경 ---
    # (C, T, H, W) -> (T, C, H, W) 로 변경하여 채널(3)을 1번 인덱스로 보냄
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
    
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    
    # 다시 (T, C, H, W) -> (C, T, H, W) 로 복구 (X3D 모델 입력 규격)
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
    # ------------------------------------------
    
    ShortSideScale(size=256),
    Resize((224, 224))
    ])

    print("--- 1. 데이터셋 로드 시도 ---")
    dataset = FakeAVCelebDataset(CSV_PATH, BASE_DIR, transform=video_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 데이터 한 개 꺼내기
    video_tensor, label = next(iter(loader))
    print(f"비디오 텐서 모양: {video_tensor.shape}") # [1, 3, 16, 224, 224] 예상
    print(f"라벨: {label.item()}")

    print("\n--- 2. X3D 모델 로드 및 연산 시도 ---")
    model = x3d_m(pretrained=True)
    model.blocks[5].proj = torch.nn.Linear(2048, 1) # 레이어를 먼저 교체한 뒤
    model.to(device) # 모델 전체를 한꺼번에 GPU로 보냄

    with torch.no_grad():
        output = model(video_tensor.to(device))
        print(f"모델 출력 결과: {output}")
        print("\n✅ 모든 과정이 정상적으로 확인되었습니다!")

if __name__ == "__main__":
    check_pipeline()
