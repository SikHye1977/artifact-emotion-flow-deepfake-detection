"""통합 NLP 실험 공통 설정"""
import os

BASE        = os.path.expanduser("~/hsh/AIApplication")
EXP_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

AVDF1M_VAL_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
AVDF1M_VAL_META = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
PGF_ROOT        = os.path.join(BASE, "PolyGlotFake")
PGF_JSON_DIR    = os.path.join(PGF_ROOT, "json_file")

X3D_AVDF1M    = os.path.join(BASE, "reverse_zeroshot_avdf1m/x3d_model_avdf1m_best.pth")
X3D_PGF       = os.path.join(BASE, "reverse_zero_shot/x3d_model_pgf_best.pth")
AASIST_AVDF1M = os.path.join(BASE, "NLP_architecture/avdf1m_experiment/results/aasist_avdf1m_best.pth")
AASIST_PGF    = os.path.join(BASE, "reverse_zero_shot/aasist_model_pgf_best.pth")
AASIST_CFG    = os.path.join(BASE, "aasist/config/AASIST.conf")

AVDF1M_CACHE = os.path.join(RESULTS_DIR, "avdf1m_cache_conf.json")
PGF_CACHE    = os.path.join(RESULTS_DIR, "pgf_cache_conf.json")

NLP_AVDF1M_CKPT = os.path.join(RESULTS_DIR, "nlp_unified_avdf1m_best.pth")
NLP_PGF_CKPT    = os.path.join(RESULTS_DIR, "nlp_unified_pgf_best.pth")

NLP_CFG = {
    "model_name": "xlm-roberta-base",
    "max_len": 128, "batch_size": 16, "epochs": 15,
    "lr": 2e-5, "warmup_ratio": 0.1, "topk": 3, "seed": 42,
}
