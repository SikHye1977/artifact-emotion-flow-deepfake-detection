"""
eval_matrix_v2.py
v2 NLP(메타 신호 4종) 모델로 4조합 평가 매트릭스 갱신
"""
import os, sys, json, functools
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import av
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *
from model_v2 import UnifiedNLPModelV2

torch.load = functools.partial(torch.load, weights_only=False)

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = _Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

BASE_APP = os.path.expanduser("~/hsh/AIApplication")
sys.path.insert(0, BASE_APP)
from aasist.models.AASIST import Model as AASISTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
ML = NLP_CFG["max_len"]
N_META = 4

NLP_V2_AVDF1M = os.path.join(RESULTS_DIR, "nlp_v2_avdf1m_best.pth")
NLP_V2_PGF    = os.path.join(RESULTS_DIR, "nlp_v2_pgf_best.pth")

def rescale(x): return x/255.0
def to_tc(x): return x.permute(1,0,2,3)
def to_ct(x): return x.permute(1,0,2,3)
x3d_tf = Compose([UniformTemporalSubsample(16), Lambda(rescale),
    Lambda(to_tc), Normalize([0.45]*3,[0.225]*3),
    Lambda(to_ct), ShortSideScale(256), Resize((224,224))])

def load_video(path):
    try:
        c=av.open(path); fr=[f.to_rgb().to_ndarray() for f in c.decode(video=0)]; c.close()
        if len(fr)<16: return None
        if len(fr)>128:
            idx=np.linspace(0,len(fr)-1,128,dtype=int); fr=[fr[i] for i in idx]
        return torch.from_numpy(np.stack(fr)).permute(3,0,1,2).float()
    except: return None

def load_audio(path, sr=16000, ml=64000):
    try:
        c=av.open(path)
        if not c.streams.audio: c.close(); return None
        osr=c.streams.audio[0].rate; frames=[]
        for f in c.decode(audio=0):
            a=f.to_ndarray().astype(np.float32)
            if a.ndim>1: a=a.mean(axis=0)
            frames.append(a)
        c.close()
        if not frames: return None
        w=torch.from_numpy(np.concatenate(frames)).unsqueeze(0)
        if osr!=sr: w=torchaudio.transforms.Resample(osr,sr)(w)
        if w.shape[1]>ml: w=w[:,:ml]
        else: w=torch.nn.functional.pad(w,(0,ml-w.shape[1]))
        return w.squeeze()
    except: return None

def load_x3d(p):
    m=x3d_m(pretrained=False)
    m.blocks[5].proj=nn.Linear(2048,1); m.blocks[5].activation=nn.Identity()
    ck=torch.load(p,map_location=DEVICE); m.load_state_dict(ck.get("state_dict",ck))
    return m.to(DEVICE).eval()

def load_aasist(p):
    with open(AASIST_CFG) as f: cfg=json.load(f)
    m=AASISTModel(cfg["model_config"])
    ck=torch.load(p,map_location=DEVICE); m.load_state_dict(ck.get("state_dict",ck))
    return m.to(DEVICE).eval()

def load_nlp_v2(p):
    ck=torch.load(p,map_location=DEVICE)
    m=UnifiedNLPModelV2(n_meta=N_META); m.load_state_dict(ck["state_dict"])
    return m.to(DEVICE).eval()

@torch.no_grad()
def infer_x3d(m,p):
    v=load_video(p)
    if v is None: return None
    return torch.sigmoid(m(x3d_tf(v).unsqueeze(0).to(DEVICE))).item()

@torch.no_grad()
def infer_aasist(m,p):
    a=load_audio(p)
    if a is None: return None
    _,out=m(a.unsqueeze(0).to(DEVICE))
    return torch.softmax(out,dim=1)[0,1].item()

def build_word_meta(words):
    n=len(words); metas=[]
    for i,w in enumerate(words):
        prob=w["prob"]; dur=w["dur"]
        gp=abs(prob-words[i-1]["prob"]) if i>0 else 0.0
        gn=abs(prob-words[i+1]["prob"]) if i<n-1 else 0.0
        metas.append([prob,dur,gp,gn])
    return metas

@torch.no_grad()
def infer_nlp_v2(m, entry):
    if not entry["words"]: return 0.5
    word_metas = build_word_meta(entry["words"])
    ids, metas = [], []
    for w, mt in zip(entry["words"], word_metas):
        sub=tokenizer.encode(w["word"], add_special_tokens=False)
        for tid in sub:
            ids.append(tid); metas.append(mt)
    ids=ids[:ML]; metas=metas[:ML]
    L=len(ids)
    if L==0: return 0.5
    pad=ML-L
    input_ids=torch.tensor([ids+[1]*pad]).to(DEVICE)
    mask=torch.tensor([[1]*L+[0]*pad]).to(DEVICE)
    mf=torch.tensor([metas+[[0.0]*N_META]*pad],dtype=torch.float).to(DEVICE)
    logits=m(input_ids,mask,mf)
    probs=torch.sigmoid(logits[0,:L])
    k=min(NLP_CFG["topk"],L)
    return probs.topk(k).values.mean().item()

def prob_or3(a,b,c): return 1-(1-a)*(1-b)*(1-c)
tokenizer = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])

def get_eval_samples(dataset):
    if dataset=="avdf1m":
        with open(os.path.join(RESULTS_DIR,"avdf1m_split.json")) as f: split=json.load(f)
        with open(AVDF1M_CACHE) as f: cache=json.load(f)
        with open(AVDF1M_VAL_META) as f: meta=json.load(f)
        mmap={m["file"]:m for m in meta}
        samples=[]
        for key in split["eval"]:
            if key not in cache: continue
            m=mmap.get(key)
            if not m: continue
            samples.append({"path":os.path.join(AVDF1M_VAL_ROOT,key),
                "cache":cache[key],"label":0 if m["modify_type"]=="real" else 1,
                "type":m["modify_type"]})
        return samples
    else:
        with open(os.path.join(RESULTS_DIR,"pgf_split.json")) as f: split=json.load(f)
        with open(PGF_CACHE) as f: cache=json.load(f)
        with open(os.path.join(PGF_JSON_DIR,"fake_Json_file/all_fake_video.json")) as f:
            fake={v["filename"]:v for v in json.load(f)["video"]}
        with open(os.path.join(PGF_JSON_DIR,"real_json_file/all_real_video.json")) as f:
            real={v["filename"]:v for v in json.load(f)["videos"]}
        samples=[]
        for key in split["eval"]:
            if key not in cache: continue
            fn=key.split("/",1)[1]
            if key.startswith("real/"):
                v=real.get(fn)
                if not v: continue
                path=os.path.join(PGF_ROOT,f'real/{v["lang"]}',fn); label=0
            else:
                v=fake.get(fn)
                if not v: continue
                path=os.path.join(PGF_ROOT,f'fake/to_{v["target_lang"]}',fn); label=1
            if not os.path.exists(path): continue
            samples.append({"path":path,"cache":cache[key],
                "label":label,"type":"real" if label==0 else "fake"})
        return samples

def run(train_domain, eval_dataset):
    tag=f"{train_domain}학습 → {eval_dataset}평가"
    result_path=os.path.join(RESULTS_DIR, f"eval_v2_{train_domain}_to_{eval_dataset}.json")
    x3d_ckpt    = X3D_AVDF1M    if train_domain=="avdf1m" else X3D_PGF
    aasist_ckpt = AASIST_AVDF1M if train_domain=="avdf1m" else AASIST_PGF
    nlp_ckpt    = NLP_V2_AVDF1M if train_domain=="avdf1m" else NLP_V2_PGF
    print(f"\n{'='*60}\n[{tag}] 로드 중...")
    x3d=load_x3d(x3d_ckpt); aasist=load_aasist(aasist_ckpt); nlp=load_nlp_v2(nlp_ckpt)
    samples=get_eval_samples(eval_dataset)
    print(f"  평가 샘플: {len(samples)}개")
    if os.path.exists(result_path):
        with open(result_path) as f: results=json.load(f)
        done={r["path"] for r in results}
        todo=[s for s in samples if s["path"] not in done]
    else:
        results,todo=[],samples
    for i,s in enumerate(tqdm(todo, desc=tag)):
        sv=infer_x3d(x3d,s["path"]); sa=infer_aasist(aasist,s["path"])
        st=infer_nlp_v2(nlp,s["cache"])
        if sv is None or sa is None: continue
        results.append({"path":s["path"],"label":s["label"],"type":s["type"],
            "sv":round(sv,4),"sa":round(sa,4),"st":round(st,4),
            "final":round(prob_or3(sv,sa,st),4)})
        if (i+1)%200==0:
            with open(result_path,"w") as f: json.dump(results,f)
    with open(result_path,"w") as f: json.dump(results,f)
    return tag, results

def report(tag, results):
    labels=np.array([r["label"] for r in results])
    sv=np.array([r["sv"] for r in results]); sa=np.array([r["sa"] for r in results])
    st=np.array([r["st"] for r in results]); fin=np.array([r["final"] for r in results])
    def met(name,p):
        pr=(p>0.5).astype(int)
        print(f"  {name:<18} AUC={roc_auc_score(labels,p)*100:6.2f}%  "
              f"F1={f1_score(labels,pr,zero_division=0)*100:6.2f}%  "
              f"ACC={accuracy_score(labels,pr)*100:6.2f}%")
        return roc_auc_score(labels,p)*100
    print(f"\n[{tag}]")
    av=met("X3D",sv); aa=met("AASIST",sa); an=met("NLP-v2",st); af=met("3-way OR",fin)
    return {"tag":tag,"x3d":av,"aasist":aa,"nlp":an,"3way":af}

combos=[("avdf1m","avdf1m"),("avdf1m","pgf"),("pgf","pgf"),("pgf","avdf1m")]
summary=[]
for td,ed in combos:
    tag,res=run(td,ed)
    summary.append(report(tag,res))

print(f"\n{'='*60}\n평가 매트릭스 v2 요약 (AUC %)\n{'='*60}")
print(f"  {'조합':<28} {'X3D':>7} {'AASIST':>7} {'NLP':>7} {'3way':>7}")
for s in summary:
    print(f"  {s['tag']:<28} {s['x3d']:>6.2f} {s['aasist']:>6.2f} {s['nlp']:>6.2f} {s['3way']:>6.2f}")
with open(os.path.join(RESULTS_DIR,"matrix_summary_v2.json"),"w") as f:
    json.dump(summary,f,indent=2,ensure_ascii=False)
print(f"\n✅ 완료: matrix_summary_v2.json")
