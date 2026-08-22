"""
verify_sync_signal.py
AVDF1M / PGF fake vs real sync 신호 검증
"""
import os, sys, json, subprocess, shutil, random
import numpy as np

BASE = os.path.expanduser("~/hsh/AIApplication")
SYNC_DIR = os.path.join(BASE, "syncnet_python")
WORK = "/tmp/syncverify"
random.seed(42)

def get_sync(video_path, ref):
    """run_pipeline + run_syncnet → (confidence, min_dist) or None"""
    # 작업 폴더 정리
    for sub in ["pyavi","pycrop","pyframes","pywork","pytmp"]:
        p = os.path.join(WORK, sub, ref)
        if os.path.exists(p): shutil.rmtree(p, ignore_errors=True)
    try:
        r = subprocess.run(["python3","run_pipeline.py",
            "--videofile",video_path,"--reference",ref,"--data_dir",WORK],
            capture_output=True, timeout=90, cwd=SYNC_DIR)
        if r.returncode != 0:
            return None
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
    except Exception as e:
        return None

# ── AVDF1M ────────────────────────────────────────────────────────
print("="*60)
print("AVDF1M sync 검증")
print("="*60)
AVDF_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
with open(os.path.join(BASE,"AV-Deepfake1M_RootFiles/val_metadata.json")) as f:
    meta = json.load(f)

real_s = [m for m in meta if m["modify_type"]=="real"]
# both_modified: 입+소리 둘 다 조작 (sync 가장 어긋날 것)
both_s = [m for m in meta if m["modify_type"]=="both_modified"]
# audio_modified: 소리만 (입은 원본)
audio_s = [m for m in meta if m["modify_type"]=="audio_modified"]
random.shuffle(real_s); random.shuffle(both_s); random.shuffle(audio_s)

def eval_group(items, name, n=15):
    confs, dists = [], []
    cnt = 0
    for m in items:
        if cnt >= n: break
        path = os.path.join(AVDF_ROOT, m["file"])
        if not os.path.exists(path): continue
        res = get_sync(path, "avdf_test")
        if res is None: continue
        confs.append(res[0]); dists.append(res[1]); cnt += 1
    if confs:
        print(f"  {name:<16} n={len(confs):2d}  "
              f"conf={np.mean(confs):.3f}  dist={np.mean(dists):.3f}")
    return confs, dists

r_c, r_d = eval_group(real_s,  "real")
a_c, a_d = eval_group(audio_s, "audio_modified")
b_c, b_d = eval_group(both_s,  "both_modified")

print()
if r_c and b_c:
    print(f"  real conf {np.mean(r_c):.3f} vs both conf {np.mean(b_c):.3f} "
          f"→ 차이 {np.mean(r_c)-np.mean(b_c):+.3f}")
    print(f"  (real이 더 높으면 sync 신호 유효)")

# ── PGF ───────────────────────────────────────────────────────────
print()
print("="*60)
print("PGF sync 검증")
print("="*60)
PGF_ROOT = os.path.join(BASE, "PolyGlotFake")
with open(os.path.join(PGF_ROOT,"json_file/fake_Json_file/all_fake_video.json")) as f:
    pgf_fake = json.load(f)["video"]
with open(os.path.join(PGF_ROOT,"json_file/real_json_file/all_real_video.json")) as f:
    pgf_real = json.load(f)["videos"]
random.shuffle(pgf_fake); random.shuffle(pgf_real)

def eval_pgf(items, name, is_fake, n=15):
    confs, dists = [], []
    cnt = 0
    for v in items:
        if cnt >= n: break
        if is_fake:
            path = os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', v["filename"])
        else:
            path = os.path.join(PGF_ROOT, f'real/{v["lang"]}', v["filename"])
        if not os.path.exists(path): continue
        res = get_sync(path, "pgf_test")
        if res is None: continue
        confs.append(res[0]); dists.append(res[1]); cnt += 1
    if confs:
        print(f"  {name:<16} n={len(confs):2d}  "
              f"conf={np.mean(confs):.3f}  dist={np.mean(dists):.3f}")
    return confs, dists

pr_c, pr_d = eval_pgf(pgf_real, "real",      False)
pf_c, pf_d = eval_pgf(pgf_fake, "lang_swap", True)

print()
if pr_c and pf_c:
    print(f"  real conf {np.mean(pr_c):.3f} vs fake conf {np.mean(pf_c):.3f} "
          f"→ 차이 {np.mean(pr_c)-np.mean(pf_c):+.3f}")
    print(f"  (real이 더 높으면 sync 신호 유효)")

print()
print("="*60)
print("결론: real conf > fake conf 이면 SyncNet 브랜치 가치 있음")
print("="*60)
