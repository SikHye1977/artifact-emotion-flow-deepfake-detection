"""
build_token_dataset.py
─────────────────────────────────────────────────────────────────────
transcript_cache + val_metadata → token-level 레이블 CSV

출력 형식 (한 행 = 한 클립):
  file, modify_type, tokens (JSON), token_labels (JSON), text, clip_label

token_labels: 각 토큰이 조작 구간에 속하면 1, 아니면 0
clip_label:   audio_modified/both_modified=1, real/visual_modified=0
"""

import json, os, random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

SEED = 42
random.seed(SEED)

BASE      = os.path.expanduser("~/hsh/AIApplication")
META_PATH = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
CACHE     = os.path.join(BASE, "NLP_architecture/transcript_cache.json")
OUT_DIR   = os.path.join(BASE, "NLP_architecture")

print("로드 중...")
with open(META_PATH) as f:
    meta = json.load(f)
with open(CACHE) as f:
    cache = json.load(f)

meta_dict = {m["file"]: m for m in meta}
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# ── 레코드 생성 ───────────────────────────────────────────────────
records = []
skipped = {"no_cache":0, "no_words":0, "non_en":0, "empty":0}

for file_key, entry in tqdm(cache.items(), desc="레이블 생성"):
    m = meta_dict.get(file_key)
    if not m: skipped["no_cache"]+=1; continue

    text  = entry.get("text","").strip()
    words = entry.get("words",[])
    lang  = entry.get("language","")

    if not text:  skipped["empty"]+=1;    continue
    if not words: skipped["no_words"]+=1; continue
    if lang != "en": skipped["non_en"]+=1; continue

    modify_type = m["modify_type"]
    fake_segs   = m.get("audio_fake_segments",[])

    # clip 레이블
    clip_label = 1 if modify_type in ("audio_modified","both_modified") else 0

    # ── 토크나이징 + 단어-토큰 매핑 ─────────────────────────────
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        add_special_tokens=True,
        return_offsets_mapping=True
    )
    token_ids   = enc["input_ids"][0].tolist()
    offsets     = enc["offset_mapping"][0].tolist()   # (char_start, char_end)

    # 각 단어의 char 범위를 텍스트에서 찾아서 토큰 레이블 생성
    # Whisper words → char offset 매핑
    # 단어를 순서대로 텍스트에서 찾음
    word_char_spans = []
    search_pos = 0
    for w in words:
        word_str = w["word"]
        # 앞뒤 공백 제거 후 검색
        stripped = word_str.strip()
        if not stripped:
            word_char_spans.append(None)
            continue
        pos = text.find(stripped, search_pos)
        if pos == -1:
            word_char_spans.append(None)
        else:
            word_char_spans.append((pos, pos+len(stripped)))
            search_pos = pos + len(stripped)

    # 각 단어가 fake_segs에 속하는지 판별
    def in_fake_seg(w_dict):
        if not fake_segs: return False
        ws, we = w_dict["start"], w_dict["end"]
        return any(ws < fe and we > fs          # 겹침 조건
                   for fs, fe in fake_segs)

    # 토큰별 레이블: char offset → 단어 fake 여부
    token_labels = []
    special_ids  = set(tokenizer.all_special_ids)

    for tok_id, (cs, ce) in zip(token_ids, offsets):
        if tok_id in special_ids or (cs==0 and ce==0):
            token_labels.append(-1)   # special token → ignore
            continue

        # 이 토큰의 char 범위가 어느 단어에 속하는지
        tok_label = 0
        for w_dict, span in zip(words, word_char_spans):
            if span is None: continue
            wcs, wce = span
            # 토큰과 단어 char 범위가 겹치면
            if cs < wce and ce > wcs:
                if in_fake_seg(w_dict):
                    tok_label = 1
                break
        token_labels.append(tok_label)

    # 유효 토큰 수 (special 제외)
    valid_tokens = [l for l in token_labels if l != -1]
    if not valid_tokens: skipped["empty"]+=1; continue

    fake_token_ratio = sum(valid_tokens) / len(valid_tokens)

    records.append({
        "file":             file_key,
        "modify_type":      modify_type,
        "clip_label":       clip_label,
        "text":             text,
        "token_ids":        json.dumps(token_ids),
        "token_labels":     json.dumps(token_labels),
        "n_tokens":         len(valid_tokens),
        "n_fake_tokens":    sum(valid_tokens),
        "fake_token_ratio": round(fake_token_ratio, 4),
    })

print(f"\n유효 레코드: {len(records):,}개")
print(f"스킵: {skipped}")

df = pd.DataFrame(records)
print(f"\n[clip_label 분포]")
print(df["clip_label"].value_counts().to_string())
print(f"\n[fake token 통계 (audio_modified/both_modified만)]")
fake_df = df[df["clip_label"]==1]
print(f"  n_fake_tokens=0 (포착 실패): {(fake_df['n_fake_tokens']==0).sum()}개")
print(f"  n_fake_tokens>0 (포착 성공): {(fake_df['n_fake_tokens']>0).sum()}개")
print(f"  mean fake tokens: {fake_df['n_fake_tokens'].mean():.2f}")

# ── 클래스 균형 + train/val 분리 ─────────────────────────────────
fake_df = df[df["clip_label"]==1].sample(frac=1, random_state=SEED)
real_df = df[df["clip_label"]==0].sample(frac=1, random_state=SEED)
n_min   = min(len(fake_df), len(real_df))
df_bal  = pd.concat([fake_df.iloc[:n_min],
                     real_df.iloc[:n_min]]).sample(frac=1, random_state=SEED)

val_size = int(len(df_bal)*0.2)
df_val   = df_bal.iloc[:val_size]
df_train = df_bal.iloc[val_size:]

train_path = os.path.join(OUT_DIR, "token_dataset_train.csv")
val_path   = os.path.join(OUT_DIR, "token_dataset_val.csv")
df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path,   index=False)

print(f"\n균형 조정: {len(df_bal):,}개 (fake={n_min}, real={n_min})")
print(f"train: {len(df_train):,}개  val: {len(df_val):,}개")
print(f"\n✅ 저장 완료")
print(f"  {train_path}")
print(f"  {val_path}")
