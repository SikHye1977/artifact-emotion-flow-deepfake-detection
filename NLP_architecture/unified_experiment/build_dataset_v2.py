"""
build_dataset_v2.py
메타 신호 4종(prob, dur, gap_prev, gap_next) 포함 데이터셋

사용법:
  python3 build_dataset_v2.py avdf1m
  python3 build_dataset_v2.py pgf
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

random.seed(42)
mode = sys.argv[1] if len(sys.argv) > 1 else "avdf1m"
tokenizer = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])

def overlaps(ws, we, segs):
    return any(ws < e and we > s for s,e in segs)

def build_word_meta(words):
    """각 word에 gap_prev, gap_next 추가"""
    n = len(words)
    metas = []
    for i, w in enumerate(words):
        prob = w["prob"]; dur = w["dur"]
        gap_prev = abs(prob - words[i-1]["prob"]) if i>0 else 0.0
        gap_next = abs(prob - words[i+1]["prob"]) if i<n-1 else 0.0
        metas.append([prob, dur, gap_prev, gap_next])
    return metas

def align(words, segs, force_label=None):
    """word → subword 토큰 + 메타 4종 + label"""
    word_metas = build_word_meta(words)
    ids, metas, labels = [], [], []
    for w, m in zip(words, word_metas):
        sub = tokenizer.encode(w["word"], add_special_tokens=False)
        if not sub: continue
        if force_label is not None:
            lab = force_label
        else:
            lab = 1 if (segs and overlaps(w["start"], w["end"], segs)) else 0
        for tid in sub:
            ids.append(tid); metas.append(m); labels.append(lab)
    return ids, metas, labels

if mode == "avdf1m":
    CACHE = AVDF1M_CACHE
    with open(CACHE) as f: cache = json.load(f)
    with open(os.path.join(RESULTS_DIR,"avdf1m_split.json")) as f: split = json.load(f)
    with open(AVDF1M_VAL_META) as f: meta = json.load(f)
    mmap = {m["file"]: m for m in meta}
    OUT = os.path.join(RESULTS_DIR, "avdf1m_dataset_v2.csv")
    rows = []
    for sp in ["train","eval"]:
        for key in split[sp]:
            if key not in cache: continue
            m = mmap.get(key)
            if not m: continue
            e = cache[key]
            if not e["words"]: continue
            clip = 0 if m["modify_type"]=="real" else 1
            segs = m.get("audio_fake_segments",[]) if m["modify_type"] in ("audio_modified","both_modified") else []
            ids, metas, labels = align(e["words"], segs)
            if not ids: continue
            rows.append({"key":key,"modify_type":m["modify_type"],
                "token_ids":json.dumps(ids),
                "token_metas":json.dumps(metas),
                "token_labels":json.dumps(labels),
                "clip_label":clip,"split":sp})

elif mode == "pgf":
    CACHE = PGF_CACHE
    with open(CACHE) as f: cache = json.load(f)
    with open(os.path.join(RESULTS_DIR,"pgf_split.json")) as f: split = json.load(f)
    OUT = os.path.join(RESULTS_DIR, "pgf_dataset_v2.csv")
    rows = []
    for sp in ["train","eval"]:
        for key in split[sp]:
            if key not in cache: continue
            e = cache[key]
            if not e["words"]: continue
            clip = 0 if key.startswith("real/") else 1
            ids, metas, labels = align(e["words"], [], force_label=clip)
            if not ids: continue
            rows.append({"key":key,"modify_type":"real" if clip==0 else "lang_swap",
                "token_ids":json.dumps(ids),
                "token_metas":json.dumps(metas),
                "token_labels":json.dumps(labels),
                "clip_label":clip,"split":sp})
else:
    print("사용법: python3 build_dataset_v2.py [avdf1m|pgf]"); sys.exit(1)

df = pd.DataFrame(rows)
train_df = df[df["split"]=="train"]; eval_df = df[df["split"]=="eval"]
fake_t = train_df[train_df["clip_label"]==1]; real_t = train_df[train_df["clip_label"]==0]
n = min(len(fake_t), len(real_t))
train_bal = pd.concat([fake_t.sample(n,random_state=42), real_t.sample(n,random_state=42)])
final = pd.concat([train_bal, eval_df]).reset_index(drop=True)
final.to_csv(OUT, index=False)
print(f"[{mode}] train {len(train_bal)} (fake={n},real={n}) + eval {len(eval_df)} = {len(final)}")
print(f"✅ 저장: {OUT}")
