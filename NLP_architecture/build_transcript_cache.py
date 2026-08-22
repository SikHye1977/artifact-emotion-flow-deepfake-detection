"""
build_transcript_cache.py
─────────────────────────────────────────────────────────────────────
AVDF1M val 영상 → Whisper transcript 캐시 JSON 생성

출력: NLP_architecture/transcript_cache.json
형식: { "id03704/NKP8TxPYkCs/00025/real.mp4": {
          "text": "...",
          "language": "en",
          "words": [{"word":"We","start":0.0,"end":0.46}, ...]
        }, ... }

샘플링:
  fake (audio_modified + both_modified) : real 수에 맞춰 균형
  real + visual_modified                : real 전체 사용
  → 총 ~28,470개
"""

import json, os, tempfile, subprocess, time, random
import numpy as np
import whisper, av
import scipy.io.wavfile as wavfile
from tqdm import tqdm

# ── 경로 ─────────────────────────────────────────────────────────
BASE        = os.path.expanduser("~/hsh/AIApplication")
META_PATH   = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
AVDF1M_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = os.path.join(OUT_DIR, "transcript_cache.json")
SEED        = 42

random.seed(SEED)

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

# ── Whisper 결과 → 저장 포맷 ─────────────────────────────────────
def whisper_to_entry(result):
    words = [{"word": w["word"], "start": w["start"], "end": w["end"]}
             for seg in result["segments"]
             for w in seg.get("words", [])]
    return {"text": result["text"].strip(),
            "language": result["language"],
            "words": words}

# ── 메타 로드 & 샘플링 ────────────────────────────────────────────
print("메타데이터 로드 중...")
with open(META_PATH) as f:
    meta = json.load(f)

real_samples   = [m for m in meta if m["modify_type"] == "real"]
visual_samples = [m for m in meta if m["modify_type"] == "visual_modified"]
audio_samples  = [m for m in meta if m["modify_type"] == "audio_modified"]
both_samples   = [m for m in meta if m["modify_type"] == "both_modified"]

n_real = len(real_samples)   # 14,235

# fake: audio+both 합쳐서 real 수만큼 균형 샘플링
fake_pool = audio_samples + both_samples
random.shuffle(fake_pool)
fake_samples = fake_pool[:n_real]

# real side: real 전체 + visual (real 수만큼)
random.shuffle(visual_samples)
real_side = real_samples + visual_samples[:n_real]

selected = fake_samples + real_side
random.shuffle(selected)

print(f"선택된 샘플 수 : {len(selected)}")
print(f"  fake         : {len(fake_samples)}")
print(f"  real+visual  : {len(real_side)}")

# ── 기존 캐시 로드 (중단 후 재개 지원) ───────────────────────────
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"기존 캐시 {len(cache)}개 발견 → 이어서 진행")
else:
    cache = {}

todo = [m for m in selected if m["file"] not in cache]
print(f"남은 작업: {len(todo)}개\n")

# ── Whisper 로드 ──────────────────────────────────────────────────
print("Whisper medium 로드 중...")
model = whisper.load_model("medium")
print("로드 완료\n")

# ── 메인 루프 ─────────────────────────────────────────────────────
SAVE_EVERY  = 500    # 500개마다 중간 저장
err_count   = 0
start_time  = time.time()

for idx, entry in enumerate(tqdm(todo, desc="Transcribing")):
    video_path = os.path.join(AVDF1M_ROOT, entry["file"])
    if not os.path.exists(video_path):
        err_count += 1
        continue

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    ok = extract_audio_16k(video_path, tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        err_count += 1
        continue

    try:
        result = model.transcribe(tmp_wav, word_timestamps=True,
                                  task="transcribe", verbose=False)
        cache[entry["file"]] = whisper_to_entry(result)
    except Exception as e:
        err_count += 1
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    # 중간 저장
    if (idx + 1) % SAVE_EVERY == 0:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        elapsed = time.time() - start_time
        per_clip = elapsed / (idx + 1)
        remain  = per_clip * (len(todo) - idx - 1)
        print(f"\n[{idx+1}/{len(todo)}] 저장 완료 | "
              f"경과 {elapsed/3600:.1f}h | "
              f"남은 예상 {remain/3600:.1f}h | "
              f"오류 {err_count}개")

# 최종 저장
with open(CACHE_PATH, "w") as f:
    json.dump(cache, f, ensure_ascii=False)

elapsed = time.time() - start_time
print(f"\n✅ 완료: {len(cache)}개 저장 | 오류 {err_count}개 | 총 {elapsed/3600:.1f}h")
print(f"저장 위치: {CACHE_PATH}")
