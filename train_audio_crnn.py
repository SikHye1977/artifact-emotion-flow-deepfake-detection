import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# --- 1. 초경량 오디오 감정 CRNN 모델 ---
class AudioEmotionCRNN(nn.Module):
    def __init__(self, num_classes=8):
        super(AudioEmotionCRNN, self).__init__()
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gru = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 🌟 [NaN 방지 패치 1] 완전한 무음(0) 때문에 BatchNorm 분산이 0이 되는 것을 방지
        # 정답에 영향을 주지 않는 초미세 백색 소음(White Noise)을 학습 중에만 깔아줍니다.
        if self.training:
            x = x + torch.randn_like(x) * 1e-6
            
        # 1. 멜-스펙트로그램 추출
        x = self.mel_spec(x)
        
        # 🌟 [NaN 방지 패치 2] autocast(float16)가 0으로 뭉개지 않는 안전한 하한선 설정
        x = torch.clamp(x, min=1e-5)
        
        x = self.amplitude_to_db(x) 

        # 2. CNN 통과
        x = self.cnn(x) 

        # 3. Shape 변경 (CNN -> GRU)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(B, T, C * F)                

        # 4. GRU 통과 및 최종 판별
        gru_out, hn = self.gru(x) 
        final_feature = hn.squeeze(0) 
        out = self.classifier(final_feature) 
        
        return out

# --- 2. RAVDESS 데이터셋 클래스 ---
class RAVDESSDataset(Dataset):
    def __init__(self, file_paths, target_sr=16000, duration=3.0):
        self.file_paths = file_paths
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration) 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        filename = os.path.basename(file_path)
        emotion_str = filename.split('-')[2]
        label = int(emotion_str) - 1 

        waveform, sr = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        current_length = waveform.shape[1]
        if current_length > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif current_length < self.target_length:
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform, label

# --- 3. 학습 루프 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 학습 장치: {device}")

    DATA_DIR = './archive' 
    CHECKPOINT_PATH = 'audio_emotion_crnn_best.pth'

    all_files = glob.glob(os.path.join(DATA_DIR, 'Actor_*', '*.wav'))
    print(f"📁 총 {len(all_files)}개의 음성 파일을 찾았습니다.")

    if len(all_files) == 0:
        print("❌ 에러: DATA_DIR 경로를 찾을 수 없거나 파일이 없습니다.")
        return

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(RAVDESSDataset(train_files), batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(RAVDESSDataset(val_files), batch_size=32, num_workers=4)

    model = AudioEmotionCRNN(num_classes=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')

    print("🚀 오디오 감정(CRNN) 전문가 훈련 시작! (Epoch 1~30)")

    for epoch in range(30):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        start_time = time.time()
        
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(waveforms)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100

        print(f"E{epoch:02d} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | Time: {time.time()-start_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ⭐ 최저 검증 손실 갱신! 모델 저장 완료 -> {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()