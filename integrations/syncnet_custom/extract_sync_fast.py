"""
extract_sync_fast.py
모델 1회 로드 + 간소화 크롭으로 빠른 sync 추출

사용법:
  python3 extract_sync_fast.py pgf
  python3 extract_sync_fast.py avdf1m
"""
import os, sys, json, time, random, shutil, subprocess, glob, math
import numpy as np
import torch
import cv2
import python_speech_features
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SyncNetModel import S
from detectors import S3FD

BASE = os.path.expanduser("~/hsh/AIApplication")
SYNC_DIR = os.path.join(BASE, "syncnet_python")
RESULTS = os.path.join(BASE, "NLP_architecture/unified_experiment/results")
SYNC_MODEL = os.path.join(SYNC_DIR, "data/syncnet_v2.model")
AVDF_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
PGF_ROOT  = os.path.join(BASE, "PolyGlotFake")
WORK = "/tmp/syncfast"
random.seed(42)
os.makedirs(WORK, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = int(os.environ.get("N_TRAIN", 600)); N_EVAL = int(os.environ.get("N_EVAL", 600))
BATCH = 20
VSHIFT = 15

print("모델 로드 중...")
# SyncNet
net = S(num_layers_in_fc_layers=1024).to(DEVICE)
ckpt = torch.load(SYNC_MODEL, map_location=DEVICE)
net.load_state_dict(ckpt)
net.eval()
# S3FD
s3fd = S3FD(device=DEVICE)
print("로드 완료")

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(
            feat1[[i],:].repeat(win_size,1), feat2p[i:i+win_size,:]))
    return dists

def extract_one(video_path):
    """크롭→sync. (conf, dist) or None"""
    ref = os.path.join(WORK, "cur")
    if os.path.exists(ref): shutil.rmtree(ref, ignore_errors=True)
    os.makedirs(ref)
    try:
        # 25fps 프레임 + 16k 오디오
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",video_path,
            "-threads","1","-r","25","-f","image2",
            os.path.join(ref,"%06d.jpg")], check=True, timeout=60)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",video_path,
            "-async","1","-ac","1","-vn","-acodec","pcm_s16le","-ar","16000",
            os.path.join(ref,"audio.wav")], check=True, timeout=60)

        flist = sorted(glob.glob(os.path.join(ref,"*.jpg")))
        if len(flist) < 10: return None
        frames = [cv2.imread(f) for f in flist]

        # 첫 프레임에서 얼굴 검출 → 고정 박스 크롭
        det = s3fd.detect_faces(frames[0], conf_th=0.9, scales=[0.25])
        if len(det)==0:
            det = s3fd.detect_faces(frames[0], conf_th=0.5, scales=[0.5])
        if len(det)==0: return None
        x1,y1,x2,y2 = det[0][:4]
        cx,cy = (x1+x2)/2, (y1+y2)/2
        bs = max(x2-x1, y2-y1)/2
        bsi = int(bs*1.4)
        cropped = []
        for f in frames:
            h,w = f.shape[:2]
            # 패딩 후 크롭
            pad = cv2.copyMakeBorder(f, bsi,bsi,bsi,bsi, cv2.BORDER_CONSTANT)
            mx, my = int(cx)+bsi, int(cy)+bsi
            face = pad[my-bsi:my+bsi, mx-bsi:mx+bsi]
            face = cv2.resize(face, (224,224))
            cropped.append(face)

        im = np.stack(cropped, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0,3,4,1,2))
        imtv = torch.from_numpy(im.astype(float)).float()

        sr, audio = wavfile.read(os.path.join(ref,"audio.wav"))
        mfcc = zip(*python_speech_features.mfcc(audio, sr))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.from_numpy(cc.astype(float)).float()

        min_len = min(len(frames), math.floor(len(audio)/640))
        lastframe = min_len-5
        if lastframe < 1: return None

        im_feat, cc_feat = [], []
        with torch.no_grad():
            for i in range(0, lastframe, BATCH):
                imb = [imtv[:,:,vf:vf+5,:,:] for vf in range(i,min(lastframe,i+BATCH))]
                im_out = net.forward_lip(torch.cat(imb,0).to(DEVICE))
                im_feat.append(im_out.data.cpu())
                ccb = [cct[:,:,:,vf*4:vf*4+20] for vf in range(i,min(lastframe,i+BATCH))]
                cc_out = net.forward_aud(torch.cat(ccb,0).to(DEVICE))
                cc_feat.append(cc_out.data.cpu())
        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        dists = calc_pdist(im_feat, cc_feat, vshift=VSHIFT)
        mdist = torch.mean(torch.stack(dists,1),1)
        minval, minidx = torch.min(mdist,0)
        conf = float(torch.median(mdist) - minval)
        dist = float(minval)
        return (conf, dist)
    except Exception as e:
        return None

mode = sys.argv[1] if len(sys.argv)>1 else "pgf"
CACHE_PATH = os.path.join(RESULTS, f"sync_cache_{mode}.json")

if mode=="avdf1m":
    with open(os.path.join(RESULTS,"avdf1m_split.json")) as f: split=json.load(f)
    def path_of(k): return os.path.join(AVDF_ROOT, k)
else:
    with open(os.path.join(RESULTS,"pgf_split.json")) as f: split=json.load(f)
    with open(os.path.join(PGF_ROOT,"json_file/fake_Json_file/all_fake_video.json")) as f:
        fake={v["filename"]:v for v in json.load(f)["video"]}
    with open(os.path.join(PGF_ROOT,"json_file/real_json_file/all_real_video.json")) as f:
        real={v["filename"]:v for v in json.load(f)["videos"]}
    def path_of(k):
        fn=k.split("/",1)[1]
        if k.startswith("real/"):
            v=real.get(fn); return os.path.join(PGF_ROOT,f'real/{v["lang"]}',fn) if v else None
        v=fake.get(fn); return os.path.join(PGF_ROOT,f'fake/to_{v["target_lang"]}',fn) if v else None

keys = [(k,"train") for k in split["train"][:N_TRAIN]] + \
       [(k,"eval") for k in split["eval"][:N_EVAL]]
print(f"[{mode}] 대상 {len(keys)}개")

cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f: cache=json.load(f)
todo = [(k,s) for k,s in keys if k not in cache]
print(f"남은 {len(todo)}개")

err=0; t0=time.time()
for i,(k,sp) in enumerate(todo):
    p=path_of(k)
    if not p or not os.path.exists(p): err+=1; continue
    r=extract_one(p)
    if r is None: err+=1; continue
    cache[k]={"conf":r[0],"dist":r[1],"split":sp}
    if (i+1)%50==0:
        with open(CACHE_PATH,"w") as f: json.dump(cache,f)
        el=time.time()-t0; eta=el/(i+1)*(len(todo)-i-1)
        print(f"[{i+1}/{len(todo)}] {el/60:.1f}분 | 남은 {eta/60:.1f}분 | 실패 {err} | clip당 {el/(i+1):.1f}s", flush=True)

with open(CACHE_PATH,"w") as f: json.dump(cache,f)
valid=sum(1 for v in cache.values() if v.get("conf") is not None)
print(f"\n✅ [{mode}] {len(cache)}개 (유효 {valid}, 실패 {err})")
