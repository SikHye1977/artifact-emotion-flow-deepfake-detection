"""
extract_sync_balanced.py
AVDF1M sync를 modify_type 균형으로 추출
기존 캐시(visual_modified)는 유지, 부족한 type만 추가
"""
import os, sys, json, time, random, shutil, subprocess, glob, math
import numpy as np
import torch, cv2, python_speech_features
from scipy.io import wavfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from SyncNetModel import S
from detectors import S3FD

BASE = os.path.expanduser("~/hsh/AIApplication")
SYNC_DIR = os.path.join(BASE, "syncnet_python")
RESULTS = os.path.join(BASE, "NLP_architecture/unified_experiment/results")
SYNC_MODEL = os.path.join(SYNC_DIR, "data/syncnet_v2.model")
AVDF_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
AVDF_META = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
WORK = "/tmp/syncbal"
random.seed(42)
os.makedirs(WORK, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, VSHIFT = 20, 15

# 각 modify_type별 목표 (train+eval split에서)
PER_TYPE_TRAIN = 200   # type당 train 200
PER_TYPE_EVAL  = 150   # type당 eval 150

print("모델 로드 중...")
net = S(num_layers_in_fc_layers=1024).to(DEVICE)
net.load_state_dict(torch.load(SYNC_MODEL, map_location=DEVICE))
net.eval()
s3fd = S3FD(device=DEVICE)
print("로드 완료")

def calc_pdist(f1, f2, vshift=10):
    ws = vshift*2+1
    f2p = torch.nn.functional.pad(f2,(0,0,vshift,vshift))
    return [torch.nn.functional.pairwise_distance(
        f1[[i],:].repeat(ws,1), f2p[i:i+ws,:]) for i in range(len(f1))]

def extract_one(video_path):
    ref = os.path.join(WORK,"cur")
    if os.path.exists(ref): shutil.rmtree(ref,ignore_errors=True)
    os.makedirs(ref)
    try:
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",video_path,
            "-threads","1","-r","25","-f","image2",os.path.join(ref,"%06d.jpg")],
            check=True, timeout=60)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",video_path,
            "-async","1","-ac","1","-vn","-acodec","pcm_s16le","-ar","16000",
            os.path.join(ref,"audio.wav")], check=True, timeout=60)
        flist = sorted(glob.glob(os.path.join(ref,"*.jpg")))
        if len(flist)<10: return None
        frames=[cv2.imread(f) for f in flist]
        det = s3fd.detect_faces(frames[0], conf_th=0.9, scales=[0.25])
        if len(det)==0: det = s3fd.detect_faces(frames[0], conf_th=0.5, scales=[0.5])
        if len(det)==0: return None
        x1,y1,x2,y2 = det[0][:4]
        cx,cy = (x1+x2)/2,(y1+y2)/2
        bsi = int(max(x2-x1,y2-y1)/2*1.4)
        cropped=[]
        for f in frames:
            pad=cv2.copyMakeBorder(f,bsi,bsi,bsi,bsi,cv2.BORDER_CONSTANT)
            mx,my=int(cx)+bsi,int(cy)+bsi
            face=cv2.resize(pad[my-bsi:my+bsi,mx-bsi:mx+bsi],(224,224))
            cropped.append(face)
        im=np.stack(cropped,axis=3); im=np.expand_dims(im,0); im=np.transpose(im,(0,3,4,1,2))
        imtv=torch.from_numpy(im.astype(float)).float()
        sr,audio=wavfile.read(os.path.join(ref,"audio.wav"))
        mfcc=zip(*python_speech_features.mfcc(audio,sr))
        mfcc=np.stack([np.array(i) for i in mfcc])
        cc=np.expand_dims(np.expand_dims(mfcc,0),0)
        cct=torch.from_numpy(cc.astype(float)).float()
        ml=min(len(frames),math.floor(len(audio)/640)); lf=ml-5
        if lf<1: return None
        imf,ccf=[],[]
        with torch.no_grad():
            for i in range(0,lf,BATCH):
                ib=[imtv[:,:,vf:vf+5,:,:] for vf in range(i,min(lf,i+BATCH))]
                imf.append(net.forward_lip(torch.cat(ib,0).to(DEVICE)).data.cpu())
                cb=[cct[:,:,:,vf*4:vf*4+20] for vf in range(i,min(lf,i+BATCH))]
                ccf.append(net.forward_aud(torch.cat(cb,0).to(DEVICE)).data.cpu())
        imf=torch.cat(imf,0); ccf=torch.cat(ccf,0)
        d=calc_pdist(imf,ccf,vshift=VSHIFT)
        md=torch.mean(torch.stack(d,1),1)
        mv,_=torch.min(md,0)
        return (float(torch.median(md)-mv), float(mv))
    except: return None

# split + meta
with open(os.path.join(RESULTS,"avdf1m_split.json")) as f: split=json.load(f)
with open(AVDF_META) as f: meta=json.load(f)
mmap={m["file"]:m for m in meta}

CACHE_PATH=os.path.join(RESULTS,"sync_cache_avdf1m.json")
cache={}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f: cache=json.load(f)
print(f"기존 캐시 {len(cache)}개")

# modify_type별 균형 선정
def pick_balanced(keys, per_type):
    by_type={}
    for k in keys:
        if k not in mmap: continue
        t=mmap[k]["modify_type"]
        by_type.setdefault(t,[]).append(k)
    sel=[]
    for t,ks in by_type.items():
        random.shuffle(ks)
        sel+=[(k,t) for k in ks[:per_type]]
    return sel

targets=[]
for k,t in pick_balanced(split["train"], PER_TYPE_TRAIN):
    targets.append((k,"train"))
for k,t in pick_balanced(split["eval"], PER_TYPE_EVAL):
    targets.append((k,"eval"))

todo=[(k,sp) for k,sp in targets if k not in cache]
print(f"균형 추출 대상 {len(targets)}개 중 남은 {len(todo)}개")
cur=Counter(mmap[k]["modify_type"] for k in cache if k in mmap)
print(f"현재 캐시 분포: {dict(cur)}")

err=0; t0=time.time()
for i,(k,sp) in enumerate(todo):
    p=os.path.join(AVDF_ROOT,k)
    if not os.path.exists(p): err+=1; continue
    r=extract_one(p)
    if r is None: err+=1; continue
    cache[k]={"conf":r[0],"dist":r[1],"split":sp}
    if (i+1)%50==0:
        with open(CACHE_PATH,"w") as f: json.dump(cache,f)
        el=time.time()-t0
        print(f"[{i+1}/{len(todo)}] {el/60:.1f}분 | 실패 {err} | clip당 {el/(i+1):.1f}s", flush=True)

with open(CACHE_PATH,"w") as f: json.dump(cache,f)
final=Counter(mmap[k]["modify_type"] for k in cache if k in mmap)
print(f"\n✅ 완료: {len(cache)}개 | 분포 {dict(final)}")
