"""
빠른 진단 스크립트.
메타데이터 첫 10개 + predictions 첫 5개의 file path 형식을 출력하여
매칭이 왜 실패했는지 확인합니다.
"""
import json
import pandas as pd
import sys

META_PATH = 'AV-Deepfake1M_RootFiles/val_metadata.json'
PRED_PATH = 'avdf1m_zeroshot_report/predictions.csv'

print("=" * 70)
print("1. 메타데이터 첫 10개 file path")
print("=" * 70)

with open(META_PATH, 'r') as f:
    meta = json.load(f)

for i, entry in enumerate(meta[:10]):
    print(f"\n[{i}] file = {entry.get('file', 'N/A')}")
    print(f"    modify_type = {entry.get('modify_type', 'N/A')}")
    print(f"    audio_model = {entry.get('audio_model', 'N/A')}")
    print(f"    split = {entry.get('split', 'N/A')}")
    fs = entry.get('fake_segments', [])
    if fs:
        print(f"    fake_segments[0] = {fs[0] if fs else 'none'}")

# modify_type별 분포
print("\n" + "=" * 70)
print("2. modify_type 분포")
print("=" * 70)
mt_counts = {}
for e in meta:
    mt = e.get('modify_type', 'unknown')
    mt_counts[mt] = mt_counts.get(mt, 0) + 1
for k, v in sorted(mt_counts.items()):
    print(f"  {k}: {v}")

# audio_model별 분포
print("\n" + "=" * 70)
print("3. audio_model 분포 (None 제외)")
print("=" * 70)
am_counts = {}
for e in meta:
    am = e.get('audio_model')
    if am:
        am_counts[am] = am_counts.get(am, 0) + 1
for k, v in sorted(am_counts.items()):
    print(f"  {k}: {v}")

# split별 분포
print("\n" + "=" * 70)
print("4. split 분포")
print("=" * 70)
sp_counts = {}
for e in meta:
    sp = e.get('split', 'unknown')
    sp_counts[sp] = sp_counts.get(sp, 0) + 1
for k, v in sorted(sp_counts.items()):
    print(f"  {k}: {v}")

# Predictions의 식별자 형식
print("\n" + "=" * 70)
print("5. Predictions 첫 5개 식별자")
print("=" * 70)
df = pd.read_csv(PRED_PATH, encoding='utf-8-sig')
print(f"컬럼: {list(df.columns)}")
for i, row in df.head(5).iterrows():
    print(f"\n[{i}] speaker={row.get('speaker')}, "
          f"youtube_id={row.get('youtube_id')}, "
          f"seq_id={row.get('seq_id')}, "
          f"fake_type={row.get('fake_type')}")

# 메타데이터에서 fake_video_real_audio 패턴 검색
print("\n" + "=" * 70)
print("6. 메타데이터에서 비슷한 file path 찾기")
print("=" * 70)

# predictions의 첫 행 speaker로 메타데이터 검색
sample_speaker = df.iloc[0]['speaker']
sample_youtube = df.iloc[0]['youtube_id']
print(f"\n검색어: speaker={sample_speaker}, youtube={sample_youtube}")

found = []
for e in meta:
    fp = e.get('file', '')
    if sample_speaker in fp or sample_youtube in fp:
        found.append(fp)
        if len(found) >= 5:
            break

if found:
    print(f"매칭된 메타데이터 file paths ({len(found)}개):")
    for f in found:
        print(f"  {f}")
else:
    print("⚠️  메타데이터에 해당 speaker/youtube_id 없음")
    
    # 첫 메타 file의 구성요소 분석
    sample_meta_file = meta[0].get('file', '')
    print(f"\n메타데이터 file 예시: {sample_meta_file}")
    print(f"분해: {sample_meta_file.split('/')}")
