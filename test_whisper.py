import json
import whisper
import av
import numpy as np
import tempfile, os, subprocess

# ── 모델 로드 ──────────────────────────────────────────
model = whisper.load_model("medium")   # 첫 실행 시 자동 다운로드 (~1.5GB)

# ── AVDF1M 샘플 1개 경로 가져오기 ──────────────────────
META_PATH = "AV-Deepfake1M_RootFiles/val_metadata.json"
AVDF1M_ROOT = "AV-Deepfake1M_RootFiles/"   # 실제 영상 루트로 수정

with open(META_PATH) as f:
    meta = json.load(f)

# 'real' 샘플 1개, 'audio_modified' 샘플 1개 골라서 비교
real_sample   = next(m for m in meta if m['modify_type'] == 'real')
fake_sample   = next(m for m in meta if m['modify_type'] == 'audio_modified')

def extract_audio_to_wav(video_path, out_path):
    """av로 오디오 추출 → 16kHz mono wav (Whisper용)"""
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
    # ffmpeg로 16kHz 변환 (whisper는 16kHz 기대)
    tmp_raw = out_path + ".raw.wav"
    wav_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
    import scipy.io.wavfile as wavfile
    wavfile.write(tmp_raw, sr, wav_int16)
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_raw, "-ar", "16000", "-ac", "1", out_path],
        capture_output=True
    )
    os.remove(tmp_raw)
    return True

def run_whisper(video_path, label):
    print(f"\n{'='*50}")
    print(f"[{label}] {video_path}")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    ok = extract_audio_to_wav(video_path, tmp_wav)
    if not ok:
        print("  ❌ 오디오 추출 실패")
        return

    # word-level timestamp 포함 transcribe
    result = model.transcribe(
        tmp_wav,
        word_timestamps=True,   # ← 핵심: 단어별 타임스탬프
        task="transcribe",      # translate 아닌 원어 그대로
        verbose=False
    )
    os.remove(tmp_wav)

    print(f"  언어 감지: {result['language']}")
    print(f"  전체 transcript: {result['text'][:200]}")
    print(f"  세그먼트 수: {len(result['segments'])}")
    
    # 단어별 타임스탬프 출력 (처음 10개)
    words = []
    for seg in result['segments']:
        for w in seg.get('words', []):
            words.append(w)
    print(f"  단어 타임스탬프 (처음 10개):")
    for w in words[:10]:
        print(f"    [{w['start']:.2f}s ~ {w['end']:.2f}s]  '{w['word']}'")
    
    return result

# ── 실행 ───────────────────────────────────────────────
# 실제 영상 경로는 메타데이터 'file' 필드 기준으로 수정 필요
# 예: AVDF1M_ROOT + real_sample['file']
video_real = os.path.join(AVDF1M_ROOT, real_sample['file'])
video_fake = os.path.join(AVDF1M_ROOT, fake_sample['file'])

r1 = run_whisper(video_real, "REAL")
r2 = run_whisper(video_fake, "AUDIO_MODIFIED")

# fake_segments와 whisper word timestamp 교차 확인
if r2:
    print(f"\n[조작 구간] fake_segments: {fake_sample['audio_fake_segments']}")
    words = [w for seg in r2['segments'] for w in seg.get('words', [])]
    print("[해당 구간에 걸친 단어들]:")
    for seg_range in fake_sample['audio_fake_segments']:
        start, end = seg_range
        overlap = [w for w in words if w['start'] < end and w['end'] > start]
        for w in overlap:
            print(f"  [{w['start']:.2f}~{w['end']:.2f}s] '{w['word']}'")