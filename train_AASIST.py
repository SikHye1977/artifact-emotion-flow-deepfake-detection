import sys
import os
import torch
import pandas as pd
import av
import numpy as np
import time
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchaudio

# [수정 1] 폴더 구조에 맞게 import 경로 변경
try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ 에러: aasist/models/AASIST.py 파일을 찾을 수 없습니다.")
    sys.exit()

# --- 1. 오디오 전처리 함수 ---
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
            
        return waveform.squeeze() # (64000,)

    except Exception as e:
        return None

# --- 2. Dataset 클래스 (오디오 전용) ---
class FakeAudioDataset(Dataset):
    def __init__(self, df, base_dir):
        self.df = df
        self.base_dir = base_dir

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.iloc[-2].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])
        
        waveform = load_audio(video_path)
        if waveform is None: 
            return self.__getitem__((idx + 1) % len(self))
            
        label = int(row['audio_label'])
        return waveform, torch.tensor(label, dtype=torch.long)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = "FakeAVCeleb_v1.2"
    CSV_PATH = os.path.join(BASE_DIR, "meta_data.csv")
    CHECKPOINT_PATH = "aasist_checkpoint_final.pth"
    HISTORY_CSV = "aasist_train_history.csv"
    LOG_DIR = "aasist_prediction_logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df['audio_label'] = df['type'].apply(lambda x: 1.0 if 'FakeAudio' in str(x) else 0.0)

    real_df = df[df['audio_label'] == 0.0]
    fake_df = df[df['audio_label'] == 1.0]
    
    n_samples = min(len(real_df), len(fake_df))
    balanced_df = pd.concat([
        real_df.sample(n=n_samples, random_state=42),
        fake_df.sample(n=n_samples, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 오디오 데이터 균형 완료: Total {len(balanced_df)} (Real {n_samples} : Fake {n_samples})")

    train_df, val_df = train_test_split(balanced_df, test_size=0.1, stratify=balanced_df['audio_label'], random_state=42)
    train_loader = DataLoader(FakeAudioDataset(train_df, BASE_DIR), batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(FakeAudioDataset(val_df, BASE_DIR), batch_size=8, num_workers=4, pin_memory=True)

    try:
        with open('./aasist/config/AASIST.conf', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ 에러: aasist/config/AASIST.conf 파일을 찾을 수 없습니다.")
        sys.exit()

    model = AASISTModel(config['model_config'])
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0
    best_val_loss = float('inf')
    history = []

    # 🌟 [조건 5 추가] 이어하기(Checkpoint) 로딩 부분
    if os.path.exists(CHECKPOINT_PATH):
        print(f"♻️ 이어하기: 이전 체크포인트를 불러옵니다 ({CHECKPOINT_PATH})")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if os.path.exists(HISTORY_CSV):
            history = pd.read_csv(HISTORY_CSV).to_dict('records')

    print(f"🚀 AASIST Training Started (Epoch {start_epoch}/20)")

    for epoch in range(start_epoch, 20):
        model.train()
        train_preds_log = [] 
        
        for i, (waves, labels) in enumerate(train_loader):
            step_start_time = time.time()
            waves, labels = waves.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                _, outputs = model(waves)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                probs = torch.softmax(outputs.detach(), dim=1).cpu().numpy()
                targets = labels.cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for prob, pred, target in zip(probs, preds, targets):
                    fake_prob = prob[1]
                    train_preds_log.append({
                        '데이터 라벨 (0:진짜, 1:가짜)': target,
                        '예측 확률 (%)': round(float(fake_prob) * 100, 2),
                        '모델의 판단': pred,
                        '정답 여부 (1:O, 0:X)': 1 if pred == target else 0
                    })
            
            if i % 10 == 0:
                print(f"E{epoch} S[{i}/{len(train_loader)}] Loss:{loss.item():.4f} | Time:{time.time()-step_start_time:.2f}s")

        model.eval()
        val_loss, correct, total = 0, 0, 0
        val_preds_log = [] 
        
        with torch.no_grad():
            for waves, labels in val_loader:
                waves, labels = waves.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    _, outputs = model(waves)
                    val_loss += criterion(outputs, labels).item()
                
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                targets = labels.cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                for prob, pred, target in zip(probs, preds, targets):
                    fake_prob = prob[1]
                    val_preds_log.append({
                        '데이터 라벨 (0:진짜, 1:가짜)': target,
                        '예측 확률 (%)': round(float(fake_prob) * 100, 2),
                        '모델의 판단': pred,
                        '정답 여부 (1:O, 0:X)': 1 if pred == target else 0
                    })
                    if pred == target: correct += 1
                    total += 1

        acc = (correct / total) * 100
        avg_val_loss = val_loss / len(val_loader)
        print(f"🏁 Epoch {epoch} Done | Val Acc: {acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        pd.DataFrame(train_preds_log).to_csv(os.path.join(LOG_DIR, f"epoch_{epoch}_train.csv"), index=False, encoding='utf-8-sig')
        pd.DataFrame(val_preds_log).to_csv(os.path.join(LOG_DIR, f"epoch_{epoch}_val.csv"), index=False, encoding='utf-8-sig')
        
        history.append({'epoch': epoch, 'val_loss': avg_val_loss, 'val_acc': acc})
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
        
        # 🌟 [조건 5 추가] 매 에포크마다 현재 상태(체크포인트) 저장
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'best_val_loss': best_val_loss
        }, CHECKPOINT_PATH)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "aasist_model_best_final.pth")
            print(f"⭐ Best AASIST Model Saved!")

        scheduler.step()

if __name__ == "__main__":
    main()