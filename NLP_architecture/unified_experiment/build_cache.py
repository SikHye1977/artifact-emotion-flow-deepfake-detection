"""
build_cache.py  (서브셋 버전)
AVDF1M + PGF 균형 서브셋만 캐시 (word별 probability, duration 포함)

사용법:
  python3 build_cache.py avdf1m
  python3 build_cache.py pgf

서브셋 구성 (seed 고정):
  AVDF1M: 학습 클래스당 5000 + 평가 클래스당 500
          modify_type 4종 → 학습 20000, 평가 2000 (중복 제거)
  PGF:    real 전체(766) + fake 균형 샘플
          학습/평가 분리
"""
import os, sys, json, time, tempfile, subprocess, random
import numpy as np
import whisper, av
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

SEED = 42
random.seed(SEED)
SAVE_EVERY = 300

# 서브셋 크기
AVDF1M_TRAIN_PER_TYPE = 2000   # modify_type당 학습 샘플
AVDF1M_EVAL_PER_TYPE  = 500    # modify_type당 평가 샘플
PGF_FAKE_TRAIN        = 3000   # fake 학습
PGF_FAKE_EVAL         = 1500   # fake 평가

def extract_audio_16k(video_path, out_wav):
    try:
        c = av.open(video_path)
        if not c.streams.audio: c.close(); return False
        sr = c.streams.audio[0].rate
        frames = []
        for fr in c.decode(audio=0):
            arr = fr.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        c.close()
        if not frames: return False
        wav = np.concatenate(frames)
        peak = np.abs(wav).max()
        if peak > 0: wav = wav / peak * 0.95
        wavfile.write(out_wav+".tmp.wav", sr, (wav*32767).astype(np.int16))
        subprocess.run(["ffmpeg","-y","-i",out_wav+".tmp.wav",
                        "-ar","16000","-ac","1",out_wav],
                       capture_output=True, timeout=30)
        os.remove(out_wav+".tmp.wav")
        return True
    except: return False

def whisper_entry(result):
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "word": w["word"], "start": w["start"], "end": w["end"],
                "prob": w.get("probability", 1.0),
                "dur": round(w["end"] - w["start"], 3),
            })
    return {"text": result["text"].strip(),
            "language": result["language"], "words": words}

mode = sys.argv[1] if len(sys.argv) > 1 else "avdf1m"

# ── AVDF1M 서브셋 ─────────────────────────────────────────────────
if mode == "avdf1m":
    CACHE_PATH = AVDF1M_CACHE
    SPLIT_PATH = os.path.join(RESULTS_DIR, "avdf1m_split.json")
    with open(AVDF1M_VAL_META) as f:
        meta = json.load(f)

    # modify_type별 그룹
    by_type = defaultdict(list)
    for m in meta:
        path = os.path.join(AVDF1M_VAL_ROOT, m["file"])
        if not os.path.exists(path): continue
        by_type[m["modify_type"]].append(m)

    train_keys, eval_keys = [], []
    samples = []
    split_info = {"train": [], "eval": []}

    for mtype, items in by_type.items():
        random.shuffle(items)
        n_train = AVDF1M_TRAIN_PER_TYPE
        n_eval  = AVDF1M_EVAL_PER_TYPE
        train_items = items[:n_train]
        eval_items  = items[n_train:n_train+n_eval]
        for m in train_items:
            samples.append({"key": m["file"],
                            "path": os.path.join(AVDF1M_VAL_ROOT, m["file"])})
            split_info["train"].append(m["file"])
        for m in eval_items:
            samples.append({"key": m["file"],
                            "path": os.path.join(AVDF1M_VAL_ROOT, m["file"])})
            split_info["eval"].append(m["file"])
        print(f"  {mtype}: train {len(train_items)} + eval {len(eval_items)}")

    with open(SPLIT_PATH,"w") as f:
        json.dump(split_info, f)
    print(f"AVDF1M 서브셋: {len(samples)}개 (split 저장: {SPLIT_PATH})")

# ── PGF 서브셋 ────────────────────────────────────────────────────
elif mode == "pgf":
    CACHE_PATH = PGF_CACHE
    SPLIT_PATH = os.path.join(RESULTS_DIR, "pgf_split.json")
    with open(os.path.join(PGF_JSON_DIR,"fake_Json_file/all_fake_video.json")) as f:
        fake = json.load(f)["video"]
    with open(os.path.join(PGF_JSON_DIR,"real_json_file/all_real_video.json")) as f:
        real = json.load(f)["videos"]

    # real 전체 (766개)
    real_items = []
    for v in real:
        path = os.path.join(PGF_ROOT, f'real/{v["lang"]}', v["filename"])
        if os.path.exists(path):
            real_items.append({"key": f'real/{v["filename"]}', "path": path,
                               "lang": v["lang"]})

    # fake 샘플 (target_lang 균형)
    fake_by_lang = defaultdict(list)
    for v in fake:
        path = os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', v["filename"])
        if os.path.exists(path):
            fake_by_lang[v["target_lang"]].append(
                {"key": f'fake/{v["filename"]}', "path": path,
                 "tgt": v["target_lang"]})

    n_total_fake = PGF_FAKE_TRAIN + PGF_FAKE_EVAL
    per_lang = n_total_fake // len(fake_by_lang)
    fake_items = []
    for lang, items in fake_by_lang.items():
        random.shuffle(items)
        fake_items.extend(items[:per_lang])
    random.shuffle(fake_items)

    # train/eval 분리
    random.shuffle(real_items)
    n_real_eval = len(real_items) // 2
    real_eval  = real_items[:n_real_eval]
    real_train = real_items[n_real_eval:]
    fake_train = fake_items[:PGF_FAKE_TRAIN]
    fake_eval  = fake_items[PGF_FAKE_TRAIN:PGF_FAKE_TRAIN+PGF_FAKE_EVAL]

    samples = []
    split_info = {"train": [], "eval": []}
    for it in real_train + fake_train:
        samples.append(it); split_info["train"].append(it["key"])
    for it in real_eval + fake_eval:
        samples.append(it); split_info["eval"].append(it["key"])

    with open(SPLIT_PATH,"w") as f:
        json.dump(split_info, f)
    print(f"PGF 서브셋: {len(samples)}개")
    print(f"  train: real {len(real_train)} + fake {len(fake_train)}")
    print(f"  eval:  real {len(real_eval)} + fake {len(fake_eval)}")
    print(f"  (split 저장: {SPLIT_PATH})")
else:
    print("사용법: python3 build_cache.py [avdf1m|pgf]")
    sys.exit(1)

# ── 캐시 (재개) ───────────────────────────────────────────────────
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f"기존 캐시: {len(cache)}개")
else:
    cache = {}

todo = [s for s in samples if s["key"] not in cache]
print(f"남은 작업: {len(todo)}개\n")

if todo:
    print("Whisper medium 로드...")
    model = whisper.load_model("medium")
    print("완료\n")

    err = 0
    t0 = time.time()
    for idx, s in enumerate(tqdm(todo, desc=f"{mode}")):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        if not extract_audio_16k(s["path"], tmp):
            if os.path.exists(tmp): os.remove(tmp)
            err += 1; continue
        try:
            result = model.transcribe(tmp, word_timestamps=True,
                                      task="transcribe", verbose=False)
            cache[s["key"]] = whisper_entry(result)
        except: err += 1
        finally:
            if os.path.exists(tmp): os.remove(tmp)
        if (idx+1) % SAVE_EVERY == 0:
            with open(CACHE_PATH,"w") as f:
                json.dump(cache, f, ensure_ascii=False)
            el = time.time()-t0
            eta = el/(idx+1)*(len(todo)-idx-1)
            print(f"\n[{idx+1}/{len(todo)}] {el/60:.0f}분 | 남은 {eta/60:.0f}분 | 오류 {err}")

    with open(CACHE_PATH,"w") as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"\n✅ 완료: {len(cache)}개 | 오류 {err}개")
else:
    print("이미 모든 샘플 캐시됨")
print(f"   저장: {CACHE_PATH}")
