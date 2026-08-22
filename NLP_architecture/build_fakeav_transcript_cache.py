"""
build_fakeav_transcript_cache.py
─────────────────────────────────────────────────────────────────────
FakeAVCeleb 전체 21,544개 → Whisper transcript 캐시 생성

출력: NLP_architecture/fakeav_transcript_cache.json
형식: { "RealVideo-RealAudio/African/men/id00076/00109.mp4": {
          "text": "...",
          "language": "en",
          "words": [{"word":"...", "start":0.0, "end":0.3}, ...]
        }, ... }

예상 시간: 21,544 × 1.3초 ≈ 7.8시간
중단 후 재개 지원
"""

import json, os, tempfile, subprocess, time, random
import numpy as np
import whisper, av
import scipy.io.wavfile as wavfile
import pandas as pd
from tqdm import tqdm

SEED = 42
random.seed(SEED)

# ── 경로 ─────────────────────────────────────────────────────────
BASE     = os.path.expanduser("~/hsh/AIApplication")
FAV_ROOT = os.path.join(BASE, "FakeAVCeleb/dataset/FakeAVCeleb_v1.2")
META_CSV = os.path.join(FAV_ROOT, "meta_data.csv")
OUT_DIR  = os.path.join(BASE, "NLP_architecture")
CACHE_PATH = os.path.join(OUT_DIR, "fakeav_transcript_cache.json")
SAVE_EVERY = 500

# ── 메타데이터 로드 & 경로 구성 ──────────────────────────────────
df = pd.read_csv(META_CSV)
print(f"전체 메타데이터: {len(df)}개")

samples = []
missing = 0
for _, row in df.iterrows():
    video_path = os.path.join(
        FAV_ROOT, row['type'], row['race'],
        row['gender'], row['source'], row['path']
    )
    # file_key: 캐시 딕셔너리 키 (type/race/gender/source/filename)
    file_key = os.path.join(
        row['type'], row['race'],
        row['gender'], row['source'], row['path']
    )
    if os.path.exists(video_path):
        samples.append({
            "path":     video_path,
            "file_key": file_key,
            "type":     row['type'],
            "method":   row['method'],
            "label":    0 if row['type'] == 'RealVideo-RealAudio' else 1
        })
    else:
        missing += 1

print(f"실제 존재: {len(samples)}개 / 누락: {missing}개")
from collections import Counter
type_dist = Counter(s['type'] for s in samples)
for t, n in type_dist.items():
    label = 0 if t == 'RealVideo-RealAudio' else 1
    print(f"  {t}: {n}개 (label={label})")

# ── 오디오 추출 ───────────────────────────────────────────────────
def extract_audio_16k(video_path, out_wav):
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close(); return False
        sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        container.close()
        if not frames: return False
        wav  = np.concatenate(frames)
        peak = np.abs(wav).max()
        if peak > 0: wav = wav / peak * 0.95
        wavfile.write(out_wav+".tmp.wav", sr, (wav*32767).astype(np.int16))
        subprocess.run(["ffmpeg","-y","-i",out_wav+".tmp.wav",
                        "-ar","16000","-ac","1",out_wav],
                       capture_output=True, timeout=30)
        os.remove(out_wav+".tmp.wav")
        return True
    except Exception:
        return False

def whisper_to_entry(result):
    words = [{"word": w["word"], "start": w["start"], "end": w["end"]}
             for seg in result["segments"]
             for w in seg.get("words", [])]
    return {
        "text":     result["text"].strip(),
        "language": result["language"],
        "words":    words
    }

# ── 기존 캐시 로드 (중단 후 재개) ────────────────────────────────
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"\n기존 캐시 {len(cache)}개 발견 → 이어서 진행")
else:
    cache = {}

todo = [s for s in samples if s["file_key"] not in cache]
print(f"남은 작업: {len(todo)}개\n")

# ── Whisper 로드 ──────────────────────────────────────────────────
print("Whisper medium 로드 중...")
model = whisper.load_model("medium")
print("로드 완료\n")

# ── 메인 루프 ─────────────────────────────────────────────────────
err_count = 0
start_time = time.time()

for idx, s in enumerate(tqdm(todo, desc="FakeAV Transcribing")):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    ok = extract_audio_16k(s["path"], tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        err_count += 1
        continue

    try:
        result = model.transcribe(tmp_wav, word_timestamps=True,
                                  task="transcribe", verbose=False)
        cache[s["file_key"]] = whisper_to_entry(result)
    except Exception:
        err_count += 1
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    if (idx+1) % SAVE_EVERY == 0:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        elapsed = time.time() - start_time
        per_clip = elapsed / (idx+1)
        remain   = per_clip * (len(todo)-idx-1)
        print(f"\n[{idx+1}/{len(todo)}] 저장 | "
              f"경과 {elapsed/3600:.1f}h | "
              f"남은 예상 {remain/3600:.1f}h | "
              f"오류 {err_count}개")

# 최종 저장
with open(CACHE_PATH, "w") as f:
    json.dump(cache, f, ensure_ascii=False)

elapsed = time.time() - start_time
print(f"\n✅ 완료: {len(cache)}개 저장 | 오류 {err_count}개 | 총 {elapsed/3600:.1f}h")
print(f"저장 위치: {CACHE_PATH}")

# ── 언어 분포 확인 ────────────────────────────────────────────────
langs = Counter(v['language'] for v in cache.values())
empty = sum(1 for v in cache.values() if not v['text'].strip())
print(f"\n언어 분포: {dict(langs.most_common(5))}")
print(f"빈 transcript: {empty}개")
