"""
extract_sync_cache.py
공식 SyncNet 파이프라인으로 clip별 (conf, dist) 추출 → 캐시

사용법:
  python3 extract_sync_cache.py avdf1m
  python3 extract_sync_cache.py pgf

기존 unified_experiment의 split을 사용해 동일 샘플 추출.
단, 시간 단축 위해 split에서 train 600 + eval 600만.
"""
import os, sys, json, time, random, shutil, subprocess
import numpy as np

BASE = os.path.expanduser("~/hsh/AIApplication")
SYNC_DIR = os.path.join(BASE, "syncnet_python")
RESULTS = os.path.join(BASE, "NLP_architecture/unified_experiment/results")
WORK = "/tmp/synccache"
random.seed(42)

AVDF_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
AVDF_META = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
PGF_ROOT  = os.path.join(BASE, "PolyGlotFake")

N_TRAIN = 600
N_EVAL  = 600

os.makedirs(WORK, exist_ok=True)

def get_sync(video_path, ref):
    for sub in ["pyavi","pycrop","pyframes","pywork"]:
        p = os.path.join(WORK, sub, ref)
        if os.path.exists(p): shutil.rmtree(p, ignore_errors=True)
    try:
        r = subprocess.run(["python3","run_pipeline.py",
            "--videofile",video_path,"--reference",ref,"--data_dir",WORK],
            capture_output=True, timeout=90, cwd=SYNC_DIR)
        if r.returncode != 0: return None
        r2 = subprocess.run(["python3","run_syncnet.py",
            "--reference",ref,"--data_dir",WORK],
            capture_output=True, timeout=90, cwd=SYNC_DIR)
        out = (r2.stdout + r2.stderr).decode()
        conf, dist = None, None
        for line in out.split("\n"):
            if "Confidence" in line:
                try: conf = float(line.split(":")[-1].strip())
                except: pass
            if "Min dist" in line:
                try: dist = float(line.split(":")[-1].strip())
                except: pass
        if conf is None: return None
        return (conf, dist)
    except: return None

mode = sys.argv[1] if len(sys.argv) > 1 else "avdf1m"
CACHE_PATH = os.path.join(RESULTS, f"sync_cache_{mode}.json")

# ── 샘플 구성 (기존 split 사용) ───────────────────────────────────
if mode == "avdf1m":
    with open(os.path.join(RESULTS,"avdf1m_split.json")) as f: split = json.load(f)
    def path_of(key): return os.path.join(AVDF_ROOT, key)
elif mode == "pgf":
    with open(os.path.join(RESULTS,"pgf_split.json")) as f: split = json.load(f)
    with open(os.path.join(PGF_ROOT,"json_file/fake_Json_file/all_fake_video.json")) as f:
        fake = {v["filename"]:v for v in json.load(f)["video"]}
    with open(os.path.join(PGF_ROOT,"json_file/real_json_file/all_real_video.json")) as f:
        real = {v["filename"]:v for v in json.load(f)["videos"]}
    def path_of(key):
        fn = key.split("/",1)[1]
        if key.startswith("real/"):
            v = real.get(fn)
            return os.path.join(PGF_ROOT, f'real/{v["lang"]}', fn) if v else None
        else:
            v = fake.get(fn)
            return os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', fn) if v else None
else:
    print("사용법: python3 extract_sync_cache.py [avdf1m|pgf]"); sys.exit(1)

# train 600 + eval 600
train_keys = split["train"][:N_TRAIN]
eval_keys  = split["eval"][:N_EVAL]
all_keys = [(k,"train") for k in train_keys] + [(k,"eval") for k in eval_keys]
print(f"[{mode}] 추출 대상: train {len(train_keys)} + eval {len(eval_keys)} = {len(all_keys)}")

# 재개
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f: cache = json.load(f)
    print(f"기존 캐시: {len(cache)}개")
else:
    cache = {}

todo = [(k,s) for k,s in all_keys if k not in cache]
print(f"남은: {len(todo)}개\n")

err = 0
t0 = time.time()
for i, (key, sp) in enumerate(todo):
    p = path_of(key)
    if not p or not os.path.exists(p):
        err += 1; continue
    res = get_sync(p, "synccur")
    if res is None:
        err += 1
        continue
    cache[key] = {"conf": res[0], "dist": res[1], "split": sp}
    if (i+1) % 50 == 0:
        with open(CACHE_PATH,"w") as f: json.dump(cache,f)
        el = time.time()-t0
        eta = el/(i+1)*(len(todo)-i-1)
        print(f"[{i+1}/{len(todo)}] {el/60:.0f}분 | 남은 {eta/60:.0f}분 | 실패 {err}")

with open(CACHE_PATH,"w") as f: json.dump(cache,f)
valid = sum(1 for v in cache.values() if v["conf"] is not None)
print(f"\n✅ [{mode}] 완료: {len(cache)}개 (유효 {valid}, 실패 {err})")
print(f"   저장: {CACHE_PATH}")
