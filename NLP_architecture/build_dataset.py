"""
build_dataset.py
─────────────────────────────────────────────────────────────────────
transcript_cache.json + val_metadata.json → train/val CSV

레이블 정의:
  fake (1) : audio_modified, both_modified  (오디오 조작 있음)
  real (0) : real, visual_modified          (오디오 정상)

출력:
  NLP_architecture/dataset_train.csv
  NLP_architecture/dataset_val.csv
"""

import json, os, random
import pandas as pd
from collections import Counter

SEED = 42
random.seed(SEED)

BASE       = os.path.expanduser("~/hsh/AIApplication")
META_PATH  = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "transcript_cache.json")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── 로드 ─────────────────────────────────────────────────────────
print("로드 중...")
with open(META_PATH) as f:
    meta = json.load(f)
with open(CACHE_PATH) as f:
    cache = json.load(f)

meta_dict = {m["file"]: m for m in meta}

# ── 레코드 생성 ───────────────────────────────────────────────────
records = []
skipped = {"no_cache": 0, "empty_text": 0, "non_english": 0}

for file_key, entry in cache.items():
    # 빈 transcript 제거
    if not entry["text"].strip():
        skipped["empty_text"] += 1
        continue

    # 영어만 사용 (non-en은 AVDF1M에서 Whisper 오인식 가능성)
    if entry["language"] != "en":
        skipped["non_english"] += 1
        continue

    # 메타 매핑
    meta_entry = meta_dict.get(file_key)
    if meta_entry is None:
        skipped["no_cache"] += 1
        continue

    modify_type = meta_entry["modify_type"]

    # 레이블
    if modify_type in ("audio_modified", "both_modified"):
        label = 1   # fake
    elif modify_type in ("real", "visual_modified"):
        label = 0   # real
    else:
        continue    # 혹시 모를 미분류 제외

    records.append({
        "file":        file_key,
        "modify_type": modify_type,
        "label":       label,
        "text":        entry["text"],
        "language":    entry["language"],
        "n_words":     len(entry["words"])
    })

print(f"\n유효 레코드: {len(records):,}개")
print(f"스킵: {skipped}")

# ── 클래스 분포 확인 ──────────────────────────────────────────────
df = pd.DataFrame(records)
print("\n[modify_type 분포]")
print(df["modify_type"].value_counts().to_string())
print(f"\n[레이블 분포]")
print(df["label"].value_counts().to_string())

# ── 클래스 균형 맞추기 (언더샘플링) ──────────────────────────────
fake_df = df[df["label"] == 1]
real_df = df[df["label"] == 0]
n_min   = min(len(fake_df), len(real_df))

fake_df = fake_df.sample(n=n_min, random_state=SEED)
real_df = real_df.sample(n=n_min, random_state=SEED)
df_bal  = pd.concat([fake_df, real_df]).sample(frac=1, random_state=SEED)

print(f"\n균형 조정 후: {len(df_bal):,}개 (fake={n_min}, real={n_min})")

# ── train / val 분리 (8:2) ────────────────────────────────────────
val_size   = int(len(df_bal) * 0.2)
df_val     = df_bal.iloc[:val_size]
df_train   = df_bal.iloc[val_size:]

print(f"train: {len(df_train):,}개  |  val: {len(df_val):,}개")

# ── 저장 ─────────────────────────────────────────────────────────
train_path = os.path.join(OUT_DIR, "dataset_train.csv")
val_path   = os.path.join(OUT_DIR, "dataset_val.csv")

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path,   index=False)

print(f"\n✅ 저장 완료")
print(f"  {train_path}")
print(f"  {val_path}")

# ── 샘플 미리보기 ─────────────────────────────────────────────────
print("\n[train 샘플 5개]")
for _, row in df_train.head(5).iterrows():
    print(f"  label={row['label']} ({row['modify_type']:<20}) | {row['text'][:60]}")
