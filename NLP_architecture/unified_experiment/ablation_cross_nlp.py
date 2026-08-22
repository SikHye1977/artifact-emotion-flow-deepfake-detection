"""
ablation_cross_nlp.py
Cross-dataset에서 NLP branch 유무 ablation
비교: V+A (2-way OR) vs V+A+NLP (3-way OR)
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

def load_scores(train, evalset):
    path = os.path.join(RESULTS_DIR, f"eval_v2_{train}_to_{evalset}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def metrics(y, s):
    y = np.array(y); s = np.array(s)
    auc = roc_auc_score(y, s)*100 if len(set(y))>1 else float('nan')
    pred = (s>0.5).astype(int)
    f1 = f1_score(y, pred, average='macro')*100
    acc = accuracy_score(y, pred)*100
    return auc, f1, acc

def OR(*scores):
    p = np.ones(len(scores[0]))
    for s in scores:
        p = p*(1-np.array(s))
    return 1-p

settings = [
    ("avdf1m","avdf1m","In-domain"),
    ("pgf","pgf","In-domain"),
    ("avdf1m","pgf","Cross"),
    ("pgf","avdf1m","Cross"),
]

print(f"{'Setting':<20}{'Type':<11}{'V+A AUC':>9}{'+NLP AUC':>10}{'dAUC':>8}{'V+A F1':>9}{'+NLP F1':>9}{'dF1':>8}")
print("-"*84)
for tr, ev, typ in settings:
    d = load_scores(tr, ev)
    if d is None:
        print(f"{tr}->{ev:<12}{typ:<11}  (캐시 없음)")
        continue
    y  = [x["label"] for x in d]
    sv = [x["sv"] for x in d]
    sa = [x["sa"] for x in d]
    st = [x["st"] for x in d]
    va  = OR(sv, sa)
    van = OR(sv, sa, st)
    a1,f1a,_ = metrics(y, va)
    a2,f2a,_ = metrics(y, van)
    name = f"{tr}->{ev}"
    print(f"{name:<20}{typ:<11}{a1:>9.2f}{a2:>10.2f}{a2-a1:>+8.2f}{f1a:>9.2f}{f2a:>9.2f}{f2a-f1a:>+8.2f}")

print()
print("dAUC/dF1 = NLP 추가 효과 (V+A+NLP 빼기 V+A)")
print("양수 = NLP가 도움, 음수 = NLP가 해로움")
