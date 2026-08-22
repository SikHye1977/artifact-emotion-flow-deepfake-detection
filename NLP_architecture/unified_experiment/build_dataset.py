"""
build_dataset.py
캐시 → 통합 NLP 학습용 토큰 데이터셋 (token_ids, prob, dur, labels)

사용법:
  python3 build_dataset.py avdf1m
  python3 build_dataset.py pgf

출력 CSV 컬럼:
  key, token_ids, token_probs, token_durs, token_labels, clip_label, split
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

random.seed(42)
mode = sys.argv[1] if len(sys.argv) > 1 else "avdf1m"
tokenizer = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])

def overlaps(ws, we, segs):
    return any(ws < e and we > s for s,e in segs)

def align_tokens(words, segs):
    """word + prob + dur → subword token으로 확장, label은 조작 구간 겹침"""
    token_ids, token_probs, token_durs, token_labels = [], [], [], []
    for w in words:
        sub = tokenizer.encode(w["word"], add_special_tokens=False)
        if not sub: continue
        is_fake = 1 if (segs and overlaps(w["start"], w["end"], segs)) else 0
        for tid in sub:
            token_ids.append(tid)
            token_probs.append(round(w["prob"], 4))
            token_durs.append(round(w["dur"], 3))
            token_labels.append(is_fake)
    return token_ids, token_probs, token_durs, token_labels

# ── AVDF1M ────────────────────────────────────────────────────────
if mode == "avdf1m":
    CACHE = AVDF1M_CACHE
    with open(CACHE) as f: cache = json.load(f)
    with open(os.path.join(RESULTS_DIR,"avdf1m_split.json")) as f: split = json.load(f)
    with open(AVDF1M_VAL_META) as f: meta = json.load(f)
    meta_map = {m["file"]: m for m in meta}
    OUT = os.path.join(RESULTS_DIR, "avdf1m_dataset.csv")

    rows = []
    for split_name in ["train","eval"]:
        for key in split[split_name]:
            if key not in cache: continue
            m = meta_map.get(key)
            if not m: continue
            entry = cache[key]
            if not entry["words"]: continue

            # clip label: 모든 조작 = fake
            clip_label = 0 if m["modify_type"]=="real" else 1
            # token label: audio 조작 구간만 (audio/both)
            segs = m.get("audio_fake_segments",[]) \
                   if m["modify_type"] in ("audio_modified","both_modified") else []

            tid, tp, td, tl = align_tokens(entry["words"], segs)
            if not tid: continue
            rows.append({
                "key": key, "modify_type": m["modify_type"],
                "token_ids": json.dumps(tid),
                "token_probs": json.dumps(tp),
                "token_durs": json.dumps(td),
                "token_labels": json.dumps(tl),
                "clip_label": clip_label,
                "split": split_name,
            })

# ── PGF ───────────────────────────────────────────────────────────
elif mode == "pgf":
    CACHE = PGF_CACHE
    with open(CACHE) as f: cache = json.load(f)
    with open(os.path.join(RESULTS_DIR,"pgf_split.json")) as f: split = json.load(f)
    OUT = os.path.join(RESULTS_DIR, "pgf_dataset.csv")

    rows = []
    for split_name in ["train","eval"]:
        for key in split[split_name]:
            if key not in cache: continue
            entry = cache[key]
            if not entry["words"]: continue

            clip_label = 0 if key.startswith("real/") else 1
            # PGF는 token-level 조작 위치 없음 → 전체 fake면 모든 토큰 label=clip_label
            # 단 token label은 clip_label로 채움 (전체 치환이므로)
            segs = []  # token 위치 정보 없음
            tid, tp, td, tl = align_tokens(entry["words"], segs)
            if not tid: continue
            # PGF: fake면 모든 토큰을 fake로 (전체 조작)
            tl = [clip_label]*len(tid)
            rows.append({
                "key": key, "modify_type": "real" if clip_label==0 else "lang_swap",
                "token_ids": json.dumps(tid),
                "token_probs": json.dumps(tp),
                "token_durs": json.dumps(td),
                "token_labels": json.dumps(tl),
                "clip_label": clip_label,
                "split": split_name,
            })
else:
    print("사용법: python3 build_dataset.py [avdf1m|pgf]")
    sys.exit(1)

df = pd.DataFrame(rows)
print(f"[{mode}] 총 {len(df)}개")
print(f"  split: {dict(df['split'].value_counts())}")
print(f"  clip_label: {dict(df['clip_label'].value_counts())}")
print(f"  modify_type: {dict(df['modify_type'].value_counts())}")

# 클래스 균형 (train만)
train_df = df[df["split"]=="train"]
eval_df  = df[df["split"]=="eval"]
fake_t = train_df[train_df["clip_label"]==1]
real_t = train_df[train_df["clip_label"]==0]
n = min(len(fake_t), len(real_t))
train_bal = pd.concat([fake_t.sample(n, random_state=42),
                       real_t.sample(n, random_state=42)])
print(f"\n  train 균형: fake={n}, real={n} → {len(train_bal)}개")

final = pd.concat([train_bal, eval_df]).reset_index(drop=True)
final.to_csv(OUT, index=False)
print(f"\n✅ 저장: {OUT}")
print(f"   train {len(train_bal)} + eval {len(eval_df)} = {len(final)}")
