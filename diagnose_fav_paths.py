"""
FakeAVCeleb 메타데이터와 실제 영상 파일 위치 진단.
meta_data.csv의 컬럼과 영상 path를 실제 디스크와 매칭한다.
"""
import os
import pandas as pd

# 가능한 FAV 위치 후보
CANDIDATES = [
    os.path.expanduser("~/hsh/AIApplication/FakeAVCeleb_v1.2"),
    os.path.expanduser("~/hsh/FakeAVCeleb_v1.2"),
    os.path.expanduser("~/FakeAVCeleb_v1.2"),
    "FakeAVCeleb_v1.2",
    os.path.expanduser("~/hsh/AIApplication/FakeAVCeleb"),
    os.path.expanduser("~/hsh/FakeAVCeleb"),
]

print("=" * 70)
print("1. FAV 폴더 위치 검색")
print("=" * 70)
fav_root = None
for c in CANDIDATES:
    if os.path.exists(c):
        print(f"  ✅ 발견: {c}")
        if os.path.exists(os.path.join(c, "meta_data.csv")):
            fav_root = c
            print(f"     → meta_data.csv 존재")
            break

if fav_root is None:
    print("  ❌ FAV 폴더를 못 찾음. find로 검색:")
    import subprocess
    try:
        result = subprocess.run(
            ['find', os.path.expanduser('~'), '-maxdepth', '4',
             '-name', 'meta_data.csv', '-type', 'f'],
            capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
    except Exception as e:
        print(f"  find 실패: {e}")
    exit(1)

print(f"\n사용할 FAV root: {fav_root}")

print("\n" + "=" * 70)
print("2. meta_data.csv 컬럼 구조")
print("=" * 70)
df = pd.read_csv(os.path.join(fav_root, 'meta_data.csv'))
print(f"  Shape: {df.shape}")
print(f"  컬럼: {list(df.columns)}")
print(f"\n  첫 3행:")
print(df.head(3).to_string())

print("\n" + "=" * 70)
print("3. FAV root 하위 폴더 구조 (2단계)")
print("=" * 70)
for item in sorted(os.listdir(fav_root))[:10]:
    full = os.path.join(fav_root, item)
    if os.path.isdir(full):
        n_sub = len(os.listdir(full))
        print(f"  📁 {item}/  ({n_sub} items)")
        # 1단계 더
        try:
            for sub in sorted(os.listdir(full))[:3]:
                sub_full = os.path.join(full, sub)
                if os.path.isdir(sub_full):
                    print(f"     📁 {sub}/")
                else:
                    print(f"     📄 {sub}")
        except: pass
    else:
        print(f"  📄 {item}")

print("\n" + "=" * 70)
print("4. 실제 mp4 파일 1개 찾기")
print("=" * 70)
import subprocess
try:
    r = subprocess.run(['find', fav_root, '-name', '*.mp4', '-type', 'f'],
                       capture_output=True, text=True, timeout=60)
    mp4s = r.stdout.strip().split('\n')[:5]
    print(f"  발견한 mp4 (처음 5개):")
    for m in mp4s:
        print(f"    {m}")
except Exception as e:
    print(f"  find 실패: {e}")

print("\n" + "=" * 70)
print("5. 메타 첫 행과 실제 파일 매칭 시도")
print("=" * 70)
first = df.iloc[0]
print(f"\n  첫 행 전체:")
for col in df.columns:
    val = first[col]
    print(f"    {col:<20} = {val}")

# path 컬럼이 있으면 그걸 기준으로 매칭 시도
if 'path' in df.columns:
    fname = first['path']
    print(f"\n  '{fname}' 검색 중...")
    try:
        r = subprocess.run(['find', fav_root, '-name', fname, '-type', 'f'],
                           capture_output=True, text=True, timeout=60)
        matches = r.stdout.strip().split('\n')
        if matches and matches[0]:
            print(f"  ✅ 발견:")
            for m in matches[:3]:
                print(f"    {m}")
            print(f"\n  📝 이 경로에서 fav_root('{fav_root}') 차감:")
            for m in matches[:1]:
                rel = os.path.relpath(m, fav_root)
                print(f"    상대 경로: {rel}")
                print(f"    \n    상대 경로 분해:")
                parts = rel.split('/')
                for i, p in enumerate(parts):
                    print(f"      [{i}] = '{p}'")
        else:
            print(f"  ❌ '{fname}' 못 찾음")
    except Exception as e:
        print(f"  검색 실패: {e}")