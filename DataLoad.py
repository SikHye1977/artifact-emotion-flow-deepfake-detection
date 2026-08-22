import sys
import os
import torch
import pandas as pd
import av
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

# --- 1. 전처리 함수 ---
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
            if arr.dtype == np.int16: arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32: arr = arr.astype(np.float32) / 2147483648.0
            else: arr = arr.astype(np.float32)
            if len(arr.shape) > 1 and arr.shape[0] > arr.shape[1]: arr = arr.T
            elif len(arr.shape) == 1: arr = arr[np.newaxis, :]
            frames.append(arr)
        container.close()
        if not frames: return None
        waveform = np.concatenate(frames, axis=-1)
        waveform = torch.from_numpy(waveform)
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
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
        if len(frames) < 16: return None
        video = np.stack(frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None

# 🌟 [완벽 해결] CSV의 마지막 컬럼(iloc[-1])을 사용해 경로 생성!
def get_actual_video_path(row, base_dir):
    dir_path = str(row.iloc[-1])  # <--- 문제의 원인 해결 (iloc[-2] -> iloc[-1])
    file_name = str(row['path'])
    
    if dir_path.startswith("FakeAVCeleb/"):
        dir_path = dir_path.replace("FakeAVCeleb/", f"{base_dir}/", 1)
    elif dir_path.startswith("FakeAVCeleb"):
        dir_path = dir_path.replace("FakeAVCeleb", base_dir, 1)
    else:
        dir_path = os.path.join(base_dir, dir_path)
        
    final_path = os.path.join(dir_path, file_name)
    return os.path.normpath(final_path)

# --- 2. 데이터셋 클래스 ---
class FakeAVCelebMultiModalDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        video_path = get_actual_video_path(row, self.base_dir)
        
        raw_video = load_video(video_path)
        if raw_video is None: 
            return self.__getitem__((idx + 1) % len(self))
        if self.transform: 
            video_tensor = self.transform(raw_video)
            
        audio_tensor = load_audio(video_path)
        if audio_tensor is None:
            return self.__getitem__((idx + 1) % len(self))
            
        is_video_fake = row['method'] != 'real'
        is_audio_fake = 'FakeAudio' in str(row['type'])
        final_label = 1.0 if (is_video_fake or is_audio_fake) else 0.0
        
        return video_tensor, audio_tensor, torch.tensor([final_label], dtype=torch.float32), row['method'], str(row['type'])

def check_file_exists(row, base_dir):
    video_path = get_actual_video_path(row, base_dir)
    return os.path.exists(video_path)

# --- 3. 메인 함수 (테스트 전용) ---
def main():
    BASE_DIR = "FakeAVCeleb_v1.2"
    CSV_PATH = os.path.join(BASE_DIR, "meta_data.csv")

    print("메타데이터를 불러오는 중...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ 에러: {CSV_PATH} 파일을 찾을 수 없습니다.")
        return

    print("파일 존재 여부 확인 및 무작위 추출 중...")
    sample_df = df.sample(n=100, random_state=42)
    valid_df = sample_df[sample_df.apply(lambda row: check_file_exists(row, BASE_DIR), axis=1)]
    
    if len(valid_df) < 10:
        print(f"⚠️ 유효한 파일이 {len(valid_df)}개 뿐입니다.")
        test_df = valid_df.reset_index(drop=True)
    else:
        test_df = valid_df.head(10).reset_index(drop=True)

    print(f"✅ 테스트 데이터 준비 완료: {len(test_df)}개 유효 파일 확보")

    if len(test_df) == 0:
        print("❌ 유효한 파일이 없습니다. 코드를 종료합니다.")
        return

    print("데이터셋 및 DataLoader 초기화 중...")
    test_dataset = FakeAVCelebMultiModalDataset(test_df, BASE_DIR, video_transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print("\n=== 데이터 로딩 및 텐서 분리 테스트 시작 ===")
    for batch_idx, (video, audio, label, method, fake_type) in enumerate(test_loader):
        print(f"\n[Batch {batch_idx + 1}]")
        
        print(f"▶ Video Tensor Shape : {video.shape} \t (Batch, C, F, H, W)")
        print(f"  - Video Type       : {video.dtype}")
        
        print(f"▶ Audio Tensor Shape : {audio.shape} \t (Batch, Sequence)")
        print(f"  - Audio Type       : {audio.dtype}")
        
        print(f"▶ Label (Fake=1, Real=0): {label.view(-1).tolist()}")
        print(f"  - Method              : {method}")
        print(f"  - Type                : {fake_type}")
        print("-" * 60)

    print("=== 테스트 종료 ===\n")

if __name__ == "__main__":
    main()