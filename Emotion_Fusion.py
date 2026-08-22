# import sys
# import os
# import torch
# import torch.nn as nn
# import functools
# import pandas as pd
# import av
# import numpy as np
# import torchaudio
# from torch.utils.data import Dataset, DataLoader
# from hsemotion.facial_emotions import HSEmotionRecognizer
# from torch.nn.functional import cosine_similarity
# from tqdm import tqdm  # 🌟 로딩바 라이브러리 추가

# # 🌟 [보안 패치] PyTorch 2.6+ 보안 정책 우회
# torch.load = functools.partial(torch.load, weights_only=False)

# # --- 1. 오디오 감정 추출용 CRNN 모델 정의 ---
# class AudioEmotionCRNN(nn.Module):
#     def __init__(self, num_classes=8):
#         super(AudioEmotionCRNN, self).__init__()
#         self.mel_spec = torchaudio.transforms.MelSpectrogram(
#             sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
#         )
#         self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
#         )
#         self.gru = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
#         self.classifier = nn.Sequential(
#             nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
#         )

#     def forward(self, x):
#         x = self.mel_spec(x)
#         x = torch.clamp(x, min=1e-5)
#         x = self.amplitude_to_db(x)
#         x = self.cnn(x)
#         B, C, F, T = x.shape
#         x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
#         gru_out, hn = self.gru(x)
#         return self.classifier(hn.squeeze(0))

# # --- 2. 경로 생성 헬퍼 함수 ---
# def get_actual_video_path(row, base_dir):
#     dir_path = str(row.iloc[-1])
#     file_name = str(row['path'])
#     if dir_path.startswith("FakeAVCeleb"):
#         dir_path = dir_path.replace("FakeAVCeleb", base_dir, 1)
#     return os.path.normpath(os.path.join(dir_path, file_name))

# # --- 3. 멀티모달 데이터셋 클래스 (av 기반 통합 로드) ---
# class MultiModalEmotionDataset(Dataset):
#     def __init__(self, df, base_dir):
#         self.df = df
#         self.base_dir = base_dir

#     def __len__(self): return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         v_path = get_actual_video_path(row, self.base_dir)
        
#         try:
#             container = av.open(v_path)
#             # 비디오 프레임 추출
#             frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
            
#             # 오디오 추출
#             audio_frames = []
#             if container.streams.audio:
#                 for frame in container.decode(audio=0):
#                     arr = frame.to_ndarray().astype(np.float32)
#                     if arr.ndim > 1: arr = np.mean(arr, axis=0)
#                     audio_frames.append(arr)
#                 container.close()
                
#                 if not audio_frames: return self.__getitem__((idx + 1) % len(self))
#                 waveform = np.concatenate(audio_frames)
#                 waveform = torch.from_numpy(waveform).unsqueeze(0)
                
#                 if waveform.shape[1] > 64000:
#                     waveform = waveform[:, :64000]
#                 else:
#                     waveform = nn.functional.pad(waveform, (0, 64000 - waveform.shape[1]))
#             else:
#                 container.close()
#                 return self.__getitem__((idx + 1) % len(self))

#         except:
#             return self.__getitem__((idx + 1) % len(self))
        
#         is_fake = 1.0 if row['method'] != 'real' else 0.0
#         return frames, waveform.squeeze(0), is_fake, row['method'], str(row['type']), v_path

# # --- 4. 메인 분석 함수 ---
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     BASE_DIR = "FakeAVCeleb_v1.2"
#     AUDIO_WEIGHTS = "audio_emotion_crnn_best.pth"
    
#     print("\n[1/3] 🧠 모델 및 환경 준비 중...")
#     fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
    
#     audio_model = AudioEmotionCRNN(num_classes=8).to(device)
#     if os.path.exists(AUDIO_WEIGHTS):
#         audio_model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location=device))
#         print(f"   ✅ 오디오 가중치 로드 완료: {AUDIO_WEIGHTS}")
#     audio_model.eval()

#     print("[2/3] 📊 데이터셋 로드 중...")
#     df = pd.read_csv(os.path.join(BASE_DIR, "meta_data.csv"))
#     # 분석할 샘플 개수 설정
#     test_df = df.sample(n=50, random_state=42).reset_index(drop=True)
#     dataset = MultiModalEmotionDataset(test_df, BASE_DIR)

#     results_log = []
#     print(f"[3/3] 🚀 멀티모달 감정 유사도 분석 시작!")

#     # 🌟 tqdm 로딩바 적용
#     with torch.no_grad():
#         pbar = tqdm(range(len(dataset)), desc="딥페이크 분석 진행", unit="video", leave=True)
        
#         for i in pbar:
#             frames, waveform, label, method, fake_type, v_path = dataset[i]
#             waveform = waveform.to(device).unsqueeze(0)

#             # (1) 비디오 감정 분석
#             mid_idx = len(frames) // 2
#             v_emotion, v_scores = fer.predict_emotions(frames[mid_idx], logits=False)
#             v_prob_vec = torch.tensor(v_scores).to(device).unsqueeze(0)

#             # (2) 오디오 감정 분석
#             a_logits = audio_model(waveform)
#             a_prob_vec = torch.softmax(a_logits, dim=1)

#             # (3) 유사도 계산
#             sim_score = cosine_similarity(v_prob_vec, a_prob_vec).item()

#             results_log.append({
#                 'Video_Path': v_path,
#                 'Manipulation_Type': fake_type,
#                 'Method': method,
#                 'Is_Fake_Label': int(label),
#                 'Predicted_Video_Emotion': v_emotion,
#                 'Emotion_Similarity_Score': round(sim_score, 4)
#             })
            
#             # 로딩바 오른쪽에 현재 상태 표시 (선택 사항)
#             pbar.set_postfix(유사도=f"{sim_score:.4f}", 감정=v_emotion)

#     # 결과 저장
#     output_csv = "emotion_fusion_analysis.csv"
#     pd.DataFrame(results_log).to_csv(output_csv, index=False, encoding='utf-8-sig')
#     print(f"\n✨ 모든 분석이 완료되었습니다!")
#     print(f"📂 결과 저장 경로: {os.path.abspath(output_csv)}")

# if __name__ == "__main__":
#     main()

import sys
import os
import torch
import torch.nn as nn
import functools
import pandas as pd
import av
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from hsemotion.facial_emotions import HSEmotionRecognizer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

# 🌟 [1] PyTorch 2.6+ 보안 정책 우회 (HSEmotion 로딩용)
torch.load = functools.partial(torch.load, weights_only=False)

# --- [2] 오디오 감정 추출용 CRNN 모델 정의 ---
class AudioEmotionCRNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioEmotionCRNN, self).__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.mel_spec(x)
        x = torch.clamp(x, min=1e-5)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1) # CNN 입력을 위해 채널 차원 추가 (B, 1, 64, T)
        x = self.cnn(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)
        gru_out, hn = self.gru(x)
        return self.classifier(hn.squeeze(0))

# --- [3] 경로 생성 헬퍼 함수 ---
def get_actual_video_path(row, base_dir):
    dir_path = str(row.iloc[-1])
    file_name = str(row['path'])
    if dir_path.startswith("FakeAVCeleb"):
        dir_path = dir_path.replace("FakeAVCeleb", base_dir, 1)
    return os.path.normpath(os.path.join(dir_path, file_name))

# --- [4] 멀티모달 데이터셋 클래스 (16프레임 & 3초 오디오 고정) ---
class MultiModalEmotionDataset(Dataset):
    def __init__(self, df, base_dir, num_frames=16, audio_duration=3.0):
        self.df = df
        self.base_dir = base_dir
        self.num_frames = num_frames
        self.target_audio_len = int(16000 * audio_duration)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        v_path = get_actual_video_path(row, self.base_dir)
        
        try:
            container = av.open(v_path)
            all_video_frames = []
            audio_samples = []
            
            for frame in container.decode(video=0, audio=0):
                if isinstance(frame, av.VideoFrame):
                    all_video_frames.append(frame.to_rgb().to_ndarray())
                elif isinstance(frame, av.AudioFrame):
                    arr = frame.to_ndarray().astype(np.float32)
                    if arr.ndim > 1: arr = np.mean(arr, axis=0)
                    audio_samples.append(arr)
            container.close()
            
            if len(all_video_frames) < self.num_frames or len(audio_samples) == 0:
                return self.__getitem__((idx + 1) % len(self))
            
            indices = np.linspace(0, len(all_video_frames) - 1, self.num_frames, dtype=int)
            sampled_frames = [all_video_frames[i] for i in indices]
            
            waveform = np.concatenate(audio_samples)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            if waveform.shape[1] > self.target_audio_len:
                waveform = waveform[:, :self.target_audio_len]
            else:
                waveform = nn.functional.pad(waveform, (0, self.target_audio_len - waveform.shape[1]))
            
            is_fake = 1.0 if row['method'] != 'real' else 0.0
            return sampled_frames, waveform.squeeze(0), is_fake, row['method'], str(row['type']), v_path

        except Exception:
            return self.__getitem__((idx + 1) % len(self))

# --- [5] 메인 분석 함수 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = "FakeAVCeleb_v1.2"
    AUDIO_WEIGHTS = "audio_emotion_crnn_best.pth"
    
    print("\n[1/3] 🧠 모델 및 환경 준비 중...")
    fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
    
    audio_model = AudioEmotionCRNN(num_classes=8).to(device)
    if os.path.exists(AUDIO_WEIGHTS):
        audio_model.load_state_dict(torch.load(AUDIO_WEIGHTS, map_location=device))
        print(f"   ✅ 오디오 가중치 로드 완료")
    audio_model.eval()

    print("[2/3] 📊 데이터 샘플링 중 (Real 50 : Fake 50)...")
    df = pd.read_csv(os.path.join(BASE_DIR, "meta_data.csv"))
    
    # 🌟 진짜와 가짜를 분리하여 각각 50개씩 추출
    real_df = df[df['method'] == 'real']
    fake_df = df[df['method'] != 'real']
    
    if len(real_df) < 50 or len(fake_df) < 50:
        print(f"⚠️ 경고: 데이터가 부족합니다. (Real: {len(real_df)}, Fake: {len(fake_df)})")
        n_sample = min(len(real_df), len(fake_df), 50)
    else:
        n_sample = 50

    sampled_real = real_df.sample(n=n_sample, random_state=42)
    sampled_fake = fake_df.sample(n=n_sample, random_state=42)
    
    # 두 데이터를 합치고 무작위로 섞음
    test_df = pd.concat([sampled_real, sampled_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = MultiModalEmotionDataset(test_df, BASE_DIR)

    results_log = []
    print(f"[3/3] 🚀 분석 시작 (총 {len(test_df)}개 샘플)...")

    with torch.no_grad():
        pbar = tqdm(range(len(dataset)), desc="감정 유사도 분석", unit="video")
        
        for i in pbar:
            frames, waveform, label, method, fake_type, v_path = dataset[i]
            waveform = waveform.to(device).unsqueeze(0)

            # (1) 비디오 감정 분석 (16프레임 평균)
            video_probs_list = []
            for f in frames:
                _, v_scores = fer.predict_emotions(f, logits=False)
                video_probs_list.append(v_scores)
            
            avg_v_prob = np.mean(video_probs_list, axis=0)
            v_prob_vec = torch.tensor(avg_v_prob).to(device).unsqueeze(0)
            v_emotion = fer.idx_to_class[np.argmax(avg_v_prob)]

            # (2) 오디오 감정 분석
            a_logits = audio_model(waveform)
            a_prob_vec = torch.softmax(a_logits, dim=1)

            # (3) 코사인 유사도 계산
            sim_score = cosine_similarity(v_prob_vec, a_prob_vec).item()

            results_log.append({
                'Video_Path': v_path,
                'Manipulation_Type': fake_type,
                'Method': method,
                'Is_Fake_Label': int(label),
                'Predicted_Video_Emotion': v_emotion,
                'Emotion_Similarity_Score': round(sim_score, 4)
            })
            
            pbar.set_postfix(유사도=f"{sim_score:.4f}", 라벨="Fake" if label == 1.0 else "Real")

    # 결과 저장
    output_csv = "emotion_fusion_balanced_results.csv"
    pd.DataFrame(results_log).to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✨ 분석 완료! 결과 파일: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    main()