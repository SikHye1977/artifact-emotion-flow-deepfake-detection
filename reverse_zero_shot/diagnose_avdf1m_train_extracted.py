"""
extracted_train/train/ 에 실제로 추출된 영상이 얼마나 있는지 진단.

train_metadata.json: 746,180 entries (가능한 모든 파일)
extracted_train/train/: 실제로는 일부만 추출됐을 가능성

[목표]
1. 추출된 speaker 폴더 수 확인
2. 각 speaker별 영상 파일 수 추정
3. modify_type별 분포 확인
4. 결론: 현재 추출분으로 학습 가능한지 판단
"""
import os
import json
import glob

AVDF1M_ROOT = os.path.expanduser("~/hsh/AIApplication/AV-Deepfake1M_RootFiles")
TRAIN_VIDEO = os.path.join(AVDF1M_ROOT, "extracted_train/train")
TRAIN_META  = os.path.join(AVDF1M_ROOT, "train_metadata.json")

print("=" * 70)
print("1. extracted_train/train/ 폴더 스캔")
print("=" * 70)

if not os.path.isdir(TRAIN_VIDEO):
    print(f"❌ {TRAIN_VIDEO} 폴더 자체가 없음")
    exit(1)

speaker_dirs = [d for d in os.listdir(TRAIN_VIDEO)
                if os.path.isdir(os.path.join(TRAIN_VIDEO, d)) and d.startswith('id')]
print(f"  추출된 speaker 폴더: {len(speaker_dirs)}개")
print(f"  처음 5개: {sorted(speaker_dirs)[:5]}")
print(f"  마지막 5개: {sorted(speaker_dirs)[-5:]}")

print("\n" + "=" * 70)
print("2. 추출된 영상 파일 수 카운트 (처음 5개 speaker 기준)")
print("=" * 70)

total_videos = 0
modify_type_counts = {'real': 0, 'fake_video_real_audio': 0,
                       'real_video_fake_audio': 0, 'fake_video_fake_audio': 0}

# 빠른 추정: 처음 5개 speaker만 자세히 보고, 나머지는 .mp4 카운트만
for sp in sorted(speaker_dirs)[:5]:
    sp_path = os.path.join(TRAIN_VIDEO, sp)
    mp4_files = glob.glob(os.path.join(sp_path, '**/*.mp4'), recursive=True)
    print(f"\n  📁 {sp}: {len(mp4_files)}개 영상")
    for mp4 in mp4_files[:3]:
        rel = os.path.relpath(mp4, sp_path)
        print(f"     {rel}")
    for mp4 in mp4_files:
        fname = os.path.basename(mp4)
        if fname == 'real.mp4':
            modify_type_counts['real'] += 1
        elif fname == 'fake_video_real_audio.mp4':
            modify_type_counts['fake_video_real_audio'] += 1
        elif fname == 'real_video_fake_audio.mp4':
            modify_type_counts['real_video_fake_audio'] += 1
        elif fname == 'fake_video_fake_audio.mp4':
            modify_type_counts['fake_video_fake_audio'] += 1
        total_videos += 1

print("\n" + "=" * 70)
print("3. 전체 추출 영상 수 (빠른 카운트)")
print("=" * 70)

# 더 빠른 방법: find 명령 결과를 흉내
import subprocess
try:
    result = subprocess.run(
        ['find', TRAIN_VIDEO, '-name', '*.mp4', '-type', 'f'],
        capture_output=True, text=True, timeout=120
    )
    all_mp4s = result.stdout.strip().split('\n') if result.stdout.strip() else []
    print(f"  전체 .mp4 파일 수: {len(all_mp4s)}개")
    
    # modify_type별 카운트
    type_count = {
        'real.mp4': 0,
        'fake_video_real_audio.mp4': 0,
        'real_video_fake_audio.mp4': 0,
        'fake_video_fake_audio.mp4': 0,
    }
    for path in all_mp4s:
        fname = os.path.basename(path)
        if fname in type_count:
            type_count[fname] += 1
    
    print(f"\n  파일명별 분포:")
    for k, v in type_count.items():
        print(f"     {k}: {v}")
        
except subprocess.TimeoutExpired:
    print("  ⚠️ find 명령 타임아웃 (120초 초과). 너무 많은 파일.")
except Exception as e:
    print(f"  ⚠️ 카운트 실패: {e}")

print("\n" + "=" * 70)
print("4. 메타데이터와 매칭 — 학습 가능한 샘플 수")
print("=" * 70)

# 메타데이터 로드
with open(TRAIN_META, 'r') as f:
    meta = json.load(f)

# 추출된 speaker set
extracted_speakers = set(speaker_dirs)
print(f"  추출된 speaker set 크기: {len(extracted_speakers)}")

# 메타 중 추출된 speaker에 속한 엔트리만 필터
filtered = [e for e in meta if e['file'].split('/')[0] in extracted_speakers]
print(f"  메타 중 추출된 speaker 엔트리: {len(filtered)} / {len(meta)}")

if len(filtered) > 0:
    # modify_type 분포
    from collections import Counter
    mt_counter = Counter(e['modify_type'] for e in filtered)
    print(f"\n  추출 가능한 modify_type 분포:")
    for k, v in mt_counter.most_common():
        print(f"     {k}: {v}")
    
    # 실제 파일 존재 확인 (10개 샘플)
    n_exists = 0; n_missing = 0
    import random
    random.seed(42)
    samples = random.sample(filtered, min(50, len(filtered)))
    missing_examples = []
    for e in samples:
        full_path = os.path.join(TRAIN_VIDEO, e['file'])
        if os.path.exists(full_path):
            n_exists += 1
        else:
            n_missing += 1
            if len(missing_examples) < 3:
                missing_examples.append(full_path)
    
    print(f"\n  50개 랜덤 샘플 검증:")
    print(f"     ✅ 존재: {n_exists}")
    print(f"     ❌ 누락: {n_missing}")
    if missing_examples:
        print(f"     누락 예시:")
        for ex in missing_examples:
            print(f"       {ex}")
    
    print(f"\n  📊 추출률 추정: {100*n_exists/len(samples):.1f}%")
    estimated_available = int(len(filtered) * n_exists / len(samples))
    print(f"  📊 실제 사용 가능한 학습 샘플 추정: ~{estimated_available}")
    
    # 학습 10000개 가능한가?
    print("\n" + "=" * 70)
    print("5. 결론")
    print("=" * 70)
    n_real = mt_counter.get('real', 0) * n_exists // len(samples)
    n_fake = sum(mt_counter.get(k, 0) for k in 
                 ['visual_modified', 'audio_modified', 'both_modified']) * n_exists // len(samples)
    print(f"  추정 Real: ~{n_real}, Fake: ~{n_fake}")
    if n_real >= 2000 and n_fake >= 8000:
        print(f"  ✅ 학습 가능 (목표 Real 2000 + Fake 8000 확보)")
    elif n_real >= 500 and n_fake >= 2000:
        print(f"  ⚠️  중간 규모(2k+8k)는 불가, 가벼운 설정(500+2000)으로 가능")
    else:
        print(f"  ❌ 학습 데이터 부족. 더 압축 해제 필요")