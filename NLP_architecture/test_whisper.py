"""
Step 1: Whisper ASR 파이프라인 검증
- AVDF1M val 샘플로 word-level timestamp 추출 확인
- fake_segments와 교차 분석
- 저장 위치: ~/hsh/AIApplication/NLP_architecture/
"""

import json, os, tempfile, subprocess
import numpy as np
import whisper
import av
import scipy.io.wavfile as wavfile
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)

# ── 경로 설정 ────────────────────────────────────────────────────
BASE      = os.path.expanduser("~/hsh/AIApplication")
META_PATH = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
AVDF1M_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")

# ── Whisper 로드 ─────────────────────────────────────────────────
print("Whisper medium 로드 중...")
model = whisper.load_model("medium")
print("로드 완료\n")

# ── 메타데이터 로드 ──────────────────────────────────────────────
with open(META_PATH) as f:
    meta = json.load(f)

# 각 modify_type에서 1개씩 샘플 추출
TYPES = ["real", "audio_modified", "visual_modified", "both_modified"]
samples = {}
for t in TYPES:
    s = next((m for m in meta if m["modify_type"] == t), None)
    if s:
        samples[t] = s
        print(f"[{t}] {s['file']}")

# ── 오디오 추출 함수 ─────────────────────────────────────────────
def extract_audio_16k(video_path, out_wav):
    """영상 → 16kHz mono wav"""
    container = av.open(video_path)
    if not container.streams.audio:
        container.close()
        return False
    sr = container.streams.audio[0].rate
    frames = []
    for frame in container.decode(audio=0):
        arr = frame.to_ndarray().astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        frames.append(arr)
    container.close()
    if not frames:
        return False
    wav = np.concatenate(frames)
    # 클리핑 방지
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.95
    wav_int16 = (wav * 32767).astype(np.int16)
    tmp = out_wav + ".tmp.wav"
    wavfile.write(tmp, sr, wav_int16)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-ar", "16000", "-ac", "1", out_wav],
        capture_output=True
    )
    os.remove(tmp)
    return True

# ── Whisper 추론 함수 ────────────────────────────────────────────
def run_whisper(meta_entry, label):
    video_path = os.path.join(AVDF1M_ROOT, meta_entry["file"])
    print(f"\n{'='*60}")
    print(f"[{label}]")
    print(f"  파일: {meta_entry['file']}")
    print(f"  audio_model: {meta_entry.get('audio_model', 'N/A')}")
    print(f"  fake_segments: {meta_entry.get('fake_segments', [])}")
    print(f"  audio_fake_segments: {meta_entry.get('audio_fake_segments', [])}")

    if not os.path.exists(video_path):
        print(f"  ❌ 파일 없음: {video_path}")
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    ok = extract_audio_16k(video_path, tmp_wav)
    if not ok:
        print("  ❌ 오디오 추출 실패")
        os.remove(tmp_wav)
        return None

    result = model.transcribe(
        tmp_wav,
        word_timestamps=True,
        task="transcribe",
        verbose=False
    )
    os.remove(tmp_wav)

    print(f"\n  ✅ 감지 언어: {result['language']}")
    print(f"  📝 transcript: {result['text'].strip()}")

    # 단어 타임스탬프 추출
    words = [w for seg in result["segments"] for w in seg.get("words", [])]
    print(f"  단어 수: {len(words)}")
    if words:
        print("  단어 타임스탬프 (처음 10개):")
        for w in words[:10]:
            print(f"    [{w['start']:5.2f}s ~ {w['end']:5.2f}s]  '{w['word']}'")

    # ── fake_segments와 단어 교차 분석 ──────────────────────────
    audio_segs = meta_entry.get("audio_fake_segments", [])
    if audio_segs and words:
        print(f"\n  🔍 audio_fake_segments 구간 내 단어:")
        found = False
        for seg in audio_segs:
            s, e = seg[0], seg[1]
            overlap = [w for w in words if w["start"] < e and w["end"] > s]
            if overlap:
                found = True
                print(f"    구간 [{s:.2f}s ~ {e:.2f}s]:")
                for w in overlap:
                    print(f"      [{w['start']:.2f}~{w['end']:.2f}s] '{w['word']}'")
        if not found:
            print("    (해당 구간에 단어 없음 — 무음/짧은 구간일 수 있음)")

    return result

# ── 실행 ─────────────────────────────────────────────────────────
results = {}
for label, entry in samples.items():
    results[label] = run_whisper(entry, label)

# ── 요약 ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("[ 요약 ]")
for label, r in results.items():
    if r:
        words = [w for seg in r["segments"] for w in seg.get("words", [])]
        print(f"  {label:20s} | 언어: {r['language']:5s} | 단어수: {len(words):4d} | {r['text'].strip()[:60]}")
    else:
        print(f"  {label:20s} | ❌ 실패")
