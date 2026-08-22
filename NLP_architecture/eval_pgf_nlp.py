"""
eval_pgf_nlp.py
─────────────────────────────────────────────────────────────────────
PolyGlotFake NLP 브랜치 평가
탐지 전략: Whisper 언어 감지 vs 원본 화자 언어 불일치 → fake 판정

출력:
  pgf_nlp_results.json   클립별 결과
  pgf_nlp_report.txt     최종 성능 리포트
"""

import json, os, tempfile, subprocess, time
import numpy as np
import whisper, av
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, f1_score,
                              accuracy_score, confusion_matrix,
                              classification_report)

# ── 경로 ─────────────────────────────────────────────────────────
BASE     = os.path.expanduser("~/hsh/AIApplication")
PGF_ROOT = os.path.join(BASE, "PolyGlotFake")
JSON_DIR = os.path.join(PGF_ROOT, "json_file")
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

RESULT_PATH = os.path.join(OUT_DIR, "pgf_nlp_results.json")
REPORT_PATH = os.path.join(OUT_DIR, "pgf_nlp_report.txt")

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

# ── 메타 로드 ─────────────────────────────────────────────────────
print("메타데이터 로드...")
with open(os.path.join(JSON_DIR, "fake_Json_file/all_fake_video.json")) as f:
    all_fake = json.load(f)
with open(os.path.join(JSON_DIR, "real_json_file/all_real_video.json")) as f:
    all_real = json.load(f)

# ── 샘플 목록 구성 ────────────────────────────────────────────────
samples = []

# fake 전체 (target=en 포함)
for v in all_fake["video"]:
    video_path = os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', v["filename"])
    if not os.path.exists(video_path):
        continue
    samples.append({
        "path":      video_path,
        "label":     1,                    # fake
        "raw_lang":  v["raw_lang"],        # 원본 화자 언어
        "tgt_lang":  v["target_lang"],     # 조작된 언어
        "filename":  v["filename"],
        "tts":       v["tts_technique"],
        "sync":      v["sync_tech"],
    })

# real 전체
for v in all_real["videos"]:
    lang       = v["lang"]
    video_path = os.path.join(PGF_ROOT, f"real/{lang}", v["filename"])
    if not os.path.exists(video_path):
        continue
    samples.append({
        "path":      video_path,
        "label":     0,                    # real
        "raw_lang":  lang,
        "tgt_lang":  lang,                 # real은 원본=타겟
        "filename":  v["filename"],
        "tts":       None,
        "sync":      None,
    })

print(f"fake: {sum(1 for s in samples if s['label']==1):,}개")
print(f"real: {sum(1 for s in samples if s['label']==0):,}개")
print(f"전체: {len(samples):,}개\n")

# ── 기존 결과 로드 (중단 후 재개) ────────────────────────────────
if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH) as f:
        results = json.load(f)
    done = {r["filename"] for r in results}
    print(f"기존 결과 {len(results)}개 발견 → 이어서 진행")
else:
    results = []
    done    = set()

todo = [s for s in samples if s["filename"] not in done]
print(f"남은 작업: {len(todo)}개\n")

# ── Whisper 로드 ──────────────────────────────────────────────────
print("Whisper medium 로드...")
model = whisper.load_model("medium")
print("로드 완료\n")

# ── 탐지 로직 ─────────────────────────────────────────────────────
def detect(whisper_lang: str, raw_lang: str) -> dict:
    """
    Whisper 감지 언어 vs 원본 화자 언어 비교
    
    불일치 → fake (score=0.92)
    일치   → real (score=0.08)
    
    단, Whisper는 짧은 클립에서 오인식 가능 → 확률로 표현
    """
    mismatch = (whisper_lang.lower() != raw_lang.lower())
    score_t  = 0.92 if mismatch else 0.08
    return {
        "whisper_lang": whisper_lang,
        "mismatch":     mismatch,
        "score_t":      score_t,
        "pred":         1 if mismatch else 0
    }

# ── 메인 루프 ─────────────────────────────────────────────────────
SAVE_EVERY = 200
err_count  = 0
start_time = time.time()

for idx, s in enumerate(tqdm(todo, desc="Evaluating PGF")):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name

    ok = extract_audio_16k(s["path"], tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        err_count += 1
        continue

    try:
        # 언어 감지만 하면 되므로 30초 이내 오디오로 충분
        wresult      = model.transcribe(tmp_wav, task="transcribe", verbose=False)
        whisper_lang = wresult["language"]
        transcript   = wresult["text"].strip()
    except Exception:
        err_count += 1
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        continue
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    det = detect(whisper_lang, s["raw_lang"])

    results.append({
        "filename":     s["filename"],
        "label":        s["label"],
        "raw_lang":     s["raw_lang"],
        "tgt_lang":     s["tgt_lang"],
        "whisper_lang": det["whisper_lang"],
        "mismatch":     det["mismatch"],
        "score_t":      det["score_t"],
        "pred":         det["pred"],
        "tts":          s["tts"],
        "sync":         s["sync"],
        "transcript":   transcript[:100],
    })

    if (idx + 1) % SAVE_EVERY == 0:
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - start_time
        per_clip = elapsed / (idx + 1)
        remain   = per_clip * (len(todo) - idx - 1)
        print(f"\n[{idx+1}/{len(todo)}] 저장 | "
              f"경과 {elapsed/3600:.1f}h | 남은 예상 {remain/3600:.1f}h")

# 최종 저장
with open(RESULT_PATH, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ── 성능 평가 ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("성능 평가")
print("="*60)

labels = [r["label"] for r in results]
preds  = [r["pred"]  for r in results]
scores = [r["score_t"] for r in results]

overall_acc = accuracy_score(labels, preds)
overall_f1  = f1_score(labels, preds, zero_division=0)
overall_auc = roc_auc_score(labels, scores)
cm          = confusion_matrix(labels, preds)

report_lines = []
report_lines.append("="*60)
report_lines.append("PolyGlotFake NLP Branch 평가 리포트")
report_lines.append("탐지 전략: Whisper 언어 감지 vs 원본 화자 언어 불일치")
report_lines.append("="*60)
report_lines.append(f"전체 샘플  : {len(results):,}개")
report_lines.append(f"  fake     : {sum(1 for r in results if r['label']==1):,}개")
report_lines.append(f"  real     : {sum(1 for r in results if r['label']==0):,}개")
report_lines.append(f"오류       : {err_count}개")
report_lines.append("")
report_lines.append(f"[전체 성능]")
report_lines.append(f"  AUC      : {overall_auc:.4f}")
report_lines.append(f"  F1       : {overall_f1:.4f}")
report_lines.append(f"  ACC      : {overall_acc:.4f}")
report_lines.append(f"  Confusion Matrix:")
report_lines.append(f"             pred=real  pred=fake")
report_lines.append(f"  label=real   {cm[0][0]:6d}     {cm[0][1]:6d}")
report_lines.append(f"  label=fake   {cm[1][0]:6d}     {cm[1][1]:6d}")

# target_lang별 성능
report_lines.append("")
report_lines.append("[target_lang별 탐지율]")
from collections import defaultdict
by_tgt = defaultdict(lambda: {"labels":[], "preds":[]})
for r in results:
    if r["label"] == 1:   # fake만
        by_tgt[r["tgt_lang"]]["labels"].append(r["label"])
        by_tgt[r["tgt_lang"]]["preds"].append(r["pred"])

for lang in sorted(by_tgt.keys()):
    d   = by_tgt[lang]
    acc = accuracy_score(d["labels"], d["preds"])
    n   = len(d["labels"])
    report_lines.append(f"  to_{lang}: {acc*100:.1f}%  ({n}개)")

# raw_lang별 성능
report_lines.append("")
report_lines.append("[raw_lang별 탐지율 (fake만)]")
by_raw = defaultdict(lambda: {"labels":[], "preds":[]})
for r in results:
    if r["label"] == 1:
        by_raw[r["raw_lang"]]["labels"].append(r["label"])
        by_raw[r["raw_lang"]]["preds"].append(r["pred"])

for lang in sorted(by_raw.keys()):
    d   = by_raw[lang]
    acc = accuracy_score(d["labels"], d["preds"])
    n   = len(d["labels"])
    report_lines.append(f"  raw={lang}: {acc*100:.1f}%  ({n}개)")

# Whisper 언어 감지 정확도 (real 기준)
report_lines.append("")
report_lines.append("[Whisper 언어 감지 정확도 (real 기준)]")
by_real_lang = defaultdict(lambda: {"total":0, "correct":0})
for r in results:
    if r["label"] == 0:
        lang = r["raw_lang"]
        by_real_lang[lang]["total"] += 1
        if r["whisper_lang"] == lang:
            by_real_lang[lang]["correct"] += 1

for lang in sorted(by_real_lang.keys()):
    d   = by_real_lang[lang]
    acc = d["correct"] / d["total"] if d["total"] > 0 else 0
    report_lines.append(f"  {lang}: {acc*100:.1f}%  ({d['correct']}/{d['total']})")

report_text = "\n".join(report_lines)
print(report_text)

with open(REPORT_PATH, "w") as f:
    f.write(report_text)

print(f"\n✅ 저장 완료")
print(f"  결과: {RESULT_PATH}")
print(f"  리포트: {REPORT_PATH}")
