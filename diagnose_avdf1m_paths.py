"""
AVDF1M 영상 파일 구조 진단.
메타데이터의 'file' 필드와 실제 영상 위치를 매칭해서 올바른 root 경로를 찾는다.
"""
import os
import json
import glob

AVDF1M_ROOT = os.path.expanduser("~/hsh/AIApplication/AV-Deepfake1M_RootFiles")
META = os.path.join(AVDF1M_ROOT, "val_metadata.json")

print("=" * 70)
print("1. AVDF1M_RootFiles 내부 구조")
print("=" * 70)
for item in sorted(os.listdir(AVDF1M_ROOT)):
    full = os.path.join(AVDF1M_ROOT, item)
    if os.path.isdir(full):
        n_sub = len(os.listdir(full))
        print(f"  📁 {item}/  ({n_sub}개 하위 항목)")
    else:
        size_mb = os.path.getsize(full) / (1024*1024)
        print(f"  📄 {item}  ({size_mb:.1f} MB)")

print("\n" + "=" * 70)
print("2. 메타데이터에서 첫 3개 file path 확인")
print("=" * 70)
with open(META, 'r') as f:
    meta = json.load(f)

samples = meta[:3]
for i, e in enumerate(samples):
    print(f"\n  [{i}] file = {e['file']}")
    print(f"      modify_type = {e['modify_type']}")

print("\n" + "=" * 70)
print("3. 실제 영상 파일 위치 검색")
print("=" * 70)

# 첫 메타 file의 마지막 파일명만 추출
sample_file = samples[0]['file']
filename = os.path.basename(sample_file)  # ex: fake_video_real_audio.mp4
speaker = sample_file.split('/')[0]       # ex: id02432
print(f"\n검색 대상 파일명: {filename}")
print(f"검색 대상 speaker: {speaker}")

# AVDF1M_RootFiles 안에서 재귀 검색
print("\n검색 결과 (최대 5개):")
found = []
for root, dirs, files in os.walk(AVDF1M_ROOT):
    if filename in files:
        found.append(os.path.join(root, filename))
        if len(found) >= 5:
            break

if not found:
    # 그 다음 후보: speaker 폴더만 찾아봄
    print(f"  ⚠️  '{filename}' 직접 검색 실패")
    print(f"\n  '{speaker}' 폴더 검색 중...")
    for root, dirs, files in os.walk(AVDF1M_ROOT):
        if speaker in dirs:
            speaker_dir = os.path.join(root, speaker)
            print(f"  ✅ 발견: {speaker_dir}")
            # 그 안의 구조 보기
            for sub in sorted(os.listdir(speaker_dir))[:3]:
                sub_path = os.path.join(speaker_dir, sub)
                if os.path.isdir(sub_path):
                    sub_items = os.listdir(sub_path)[:5]
                    print(f"     📁 {sub}/  → {sub_items}")
            break
else:
    for f in found:
        print(f"  ✅ {f}")
    
    # 메타 file과 비교해서 root 추출
    print("\n" + "=" * 70)
    print("4. 올바른 video_root 추정")
    print("=" * 70)
    real_path = found[0]
    meta_path = samples[0]['file']
    if real_path.endswith(meta_path):
        root = real_path[:-len(meta_path)].rstrip('/')
        print(f"\n  🎯 정답: AVDF1M_VIDEO = '{root}'")

print("\n" + "=" * 70)
print("5. AVDF1M_RootFiles 하위 폴더 트리 (2단계)")
print("=" * 70)
for item in sorted(os.listdir(AVDF1M_ROOT)):
    full = os.path.join(AVDF1M_ROOT, item)
    if os.path.isdir(full):
        print(f"\n  📁 {item}/")
        try:
            subs = sorted(os.listdir(full))[:5]
            for s in subs:
                s_path = os.path.join(full, s)
                if os.path.isdir(s_path):
                    print(f"     📁 {s}/")
                else:
                    print(f"     📄 {s}")
            if len(os.listdir(full)) > 5:
                print(f"     ... (총 {len(os.listdir(full))}개)")
        except PermissionError:
            print(f"     (권한 없음)")