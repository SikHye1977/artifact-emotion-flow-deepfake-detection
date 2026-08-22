"""
build_fakeav_dataset.py
─────────────────────────────────────────────────────────────────────
fakeav_transcript_cache.json + meta_data.csv
→ NLP 학습용 train/val CSV

레이블:
  fake (1): FakeVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-FakeAudio
  real (0): RealVideo-RealAudio

필터:
  - language != 'en' 제외 (Whisper 오인식)
  - 빈 transcript 제외
  - 클래스 균형 맞춤 (언더샘플링)
"""

import json, os, random
import pandas as pd
from collections import Counter

SEED = 42
random.seed(SEED)

BASE       = os.path.expanduser("~/hsh/AIApplication")
FAV_ROOT   = os.path.join(BASE, "FakeAVCeleb/dataset/FakeAVCeleb_v1.2")
META_CSV   = os.path.join(FAV_ROOT, "meta_data.csv")
CACHE_PATH = os.path.join(BASE, "NLP_architecture/fakeav_transcript_cache.json")
OUT_DIR    = os.path.join(BASE, "NLP_architecture")

print("로드 중...")
df_meta = pd.read_csv(META_CSV)
with open(CACHE_PATH) as f:
    cache = json.load(f)

# file_key 생성 함수
def make_key(row):
    return os.path.join(row['type'], row['race'],
                        row['gender'], row['source'], row['path'])

# ── 레코드 생성 ───────────────────────────────────────────────────
records = []
skipped = {"no_cache": 0, "empty": 0, "non_en": 0}

for _, row in df_meta.iterrows():
    file_key = make_key(row)
    entry    = cache.get(file_key)
    if entry is None:
        skipped["no_cache"] += 1; continue

    text = entry.get("text", "").strip()
    lang = entry.get("language", "")

    if not text:
        skipped["empty"] += 1; continue
    if lang != "en":
        skipped["non_en"] += 1; continue

    label = 0 if row['type'] == 'RealVideo-RealAudio' else 1

    records.append({
        "file_key":   file_key,
        "type":       row['type'],
        "method":     row['method'],
        "label":      label,
        "text":       text,
        "language":   lang,
        "n_words":    len(entry.get("words", [])),
    })

print(f"\n유효 레코드: {len(records):,}개")
print(f"스킵: {skipped}")

df = pd.DataFrame(records)
print(f"\n[type 분포]")
print(df['type'].value_counts().to_string())
print(f"\n[label 분포]")
print(df['label'].value_counts().to_string())
print(f"\n[method 분포]")
print(df['method'].value_counts().to_string())

# ── 클래스 균형 맞추기 ────────────────────────────────────────────
fake_df = df[df['label']==1].sample(frac=1, random_state=SEED)
real_df = df[df['label']==0].sample(frac=1, random_state=SEED)
n_min   = min(len(fake_df), len(real_df))
df_bal  = pd.concat([fake_df.iloc[:n_min],
                     real_df.iloc[:n_min]]).sample(frac=1, random_state=SEED)

print(f"\n균형 조정 후: {len(df_bal):,}개 (fake={n_min}, real={n_min})")

# ── train / val 분리 (8:2) ────────────────────────────────────────
val_size = int(len(df_bal) * 0.2)
df_val   = df_bal.iloc[:val_size]
df_train = df_bal.iloc[val_size:]

train_path = os.path.join(OUT_DIR, "fakeav_dataset_train.csv")
val_path   = os.path.join(OUT_DIR, "fakeav_dataset_val.csv")
df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path,   index=False)

print(f"train: {len(df_train):,}개  val: {len(df_val):,}개")
print(f"\n[train 샘플 5개]")
for _, row in df_train.head(5).iterrows():
    print(f"  label={row['label']} ({row['type']:<25}) | {row['text'][:60]}")

print(f"\n✅ 저장 완료")
print(f"  {train_path}")
print(f"  {val_path}")
