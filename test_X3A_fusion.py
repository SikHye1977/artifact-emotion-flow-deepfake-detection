import sys
import os
import torch
import pandas as pd
import av
import numpy as np
import time
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as F
    sys.modules["torchvision.transforms.functional_tensor"] = F

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ 에러: aasist/models/AASIST.py 파일을 찾을 수 없습니다.")
    sys.exit()

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
    except Exception as e:
        return None

def load_video(path):
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(frames) < 16: return None
        video = np.stack(frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception as e:
        return None

# --- 2. 데이터셋 클래스 ---
class FakeAVCelebMultiModalDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.iloc[-2].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])
        
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

# --- 3. 메인 함수 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = "FakeAVCeleb_v1.2"
    CSV_PATH = os.path.join(BASE_DIR, "meta_data.csv")
    
    X3D_WEIGHTS = "x3d_model_best_final.pth"
    AASIST_WEIGHTS = "aasist_model_best_final.pth"
    
    LOG_DIR = "x3a_fusion_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    test_df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    test_df['audio_label'] = test_df['type'].apply(lambda x: 1.0 if 'FakeAudio' in str(x) else 0.0)

    print(f"📊 테스트 데이터 로드 완료: Total {len(test_df)}")

    test_loader = DataLoader(FakeAVCelebMultiModalDataset(test_df, BASE_DIR, video_transform), batch_size=1, shuffle=False, num_workers=4)

    print("🧠 모델 세팅 중...")
    
    # X3D 초기화
    x3d_model = x3d_m(pretrained=False)
    x3d_model.blocks[5].proj = nn.Linear(2048, 1)
    x3d_model.blocks[5].activation = nn.Identity()
    if os.path.exists(X3D_WEIGHTS):
        x3d_model.load_state_dict(torch.load(X3D_WEIGHTS, map_location=device))
        print(f"✅ X3D 가중치 로드 완료")
    else:
        print(f"❌ X3D 가중치를 찾을 수 없습니다: {X3D_WEIGHTS}")
        sys.exit()
    x3d_model = x3d_model.to(device).eval()

    # AASIST 초기화
    try:
        with open('./aasist/config/AASIST.conf', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ 에러: aasist/config/AASIST.conf 파일을 찾을 수 없습니다.")
        sys.exit()

    aasist_model = AASISTModel(config['model_config'])
    if os.path.exists(AASIST_WEIGHTS):
        aasist_model.load_state_dict(torch.load(AASIST_WEIGHTS, map_location=device))
        print(f"✅ AASIST 가중치 로드 완료")
    else:
        print(f"❌ AASIST 가중치를 찾을 수 없습니다: {AASIST_WEIGHTS}")
        sys.exit()
    aasist_model = aasist_model.to(device).eval()

    print("🚀 X3A 멀티모달 테스트 시작!\n")
    
    results_log = []
    correct_count = 0
    total_count = 0
    
    with torch.no_grad():
        for i, (vids, waves, labels, methods, types) in enumerate(test_loader):
            vids, waves, labels = vids.to(device), waves.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                # 모델 추론
                v_logits = x3d_model(vids)
                v_prob = torch.sigmoid(v_logits).item() 
                
                _, a_outputs = aasist_model(waves)
                a_prob = torch.softmax(a_outputs, dim=1)[0, 1].item() 
                
                # 점수 융합
                fused_prob = 1.0 - ((1.0 - v_prob) * (1.0 - a_prob))
                
            target = labels.item()
            pred_label = 1.0 if fused_prob > 0.5 else 0.0
            is_correct = 1 if pred_label == target else 0
            
            if is_correct: correct_count += 1
            total_count += 1
            
            # 🌟 [수정된 부분] 어떤 모달리티가 조작되었는지 직관적인 한글로 판별
            video_method = methods[0]
            audio_type = types[0]
            
            is_video_fake = video_method != 'real'
            is_audio_fake = 'FakeAudio' in str(audio_type)
            
            if is_video_fake and is_audio_fake:
                manipulated_modal = "비디오 & 오디오 (FVFA)"
            elif is_video_fake and not is_audio_fake:
                manipulated_modal = "비디오 단독 (FVRA)"
            elif not is_video_fake and is_audio_fake:
                manipulated_modal = "오디오 단독 (RVFA)"
            else:
                manipulated_modal = "조작 없음 (RVRA)"
            
            # 기록 저장
            results_log.append({
                '조작된 모달리티': manipulated_modal,          # ⬅️ 새로 추가된 직관적인 컬럼!
                '원본 영상 기법 (참고용)': video_method,
                '데이터 라벨 (0:진짜, 1:가짜)': target,
                '비디오(X3D) 가짜 확률 (%)': round(v_prob * 100, 2),
                '오디오(AASIST) 가짜 확률 (%)': round(a_prob * 100, 2),
                '최종 융합 가짜 확률 (%)': round(fused_prob * 100, 2),
                '모델의 최종 예측': pred_label,
                '정답 여부 (1:O, 0:X)': is_correct
            })
            
            if (i+1) % 50 == 0:
                print(f"테스트 진행도: [{i+1}/{len(test_loader)}] | 현재 융합 정확도: {(correct_count/total_count)*100:.2f}%")

    final_accuracy = (correct_count / total_count) * 100
    print("\n=============================================")
    print(f"🏁 X3A 모델 테스트 완료!")
    print(f"🏆 최종 융합 정확도 (Accuracy): {final_accuracy:.2f}%")
    print("=============================================\n")

    csv_filename = os.path.join(LOG_DIR, "x3a_fusion_results.csv")
    pd.DataFrame(results_log).to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"💾 상세 예측 결과가 저장되었습니다: {csv_filename}")

if __name__ == "__main__":
    main()