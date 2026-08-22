"""
configs.py
공통 경로 및 하이퍼파라미터 중앙 관리
"""
import os

# ── 루트 경로 ─────────────────────────────────────────────────────
BASE        = os.path.expanduser("~/hsh/AIApplication")
EXP_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 데이터셋 경로 ─────────────────────────────────────────────────
AVDF1M_TRAIN_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_train/train")
AVDF1M_VAL_ROOT   = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")
AVDF1M_TRAIN_META = os.path.join(BASE, "AV-Deepfake1M_RootFiles/train_metadata.json")
AVDF1M_VAL_META   = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
PGF_ROOT          = os.path.join(BASE, "PolyGlotFake")
PGF_JSON_DIR      = os.path.join(PGF_ROOT, "json_file")
FAKEAV_ROOT       = os.path.join(BASE, "FakeAVCeleb/dataset/FakeAVCeleb_v1.2")

# ── 사전학습 가중치 (FakeAV) ──────────────────────────────────────
FAKEAV_X3D_CKPT    = os.path.join(BASE, "x3d_model_best_final.pth")
FAKEAV_AASIST_CKPT = os.path.join(BASE, "aasist_model_best_final.pth")
AASIST_CFG         = os.path.join(BASE, "aasist/config/AASIST.conf")

# ── AVDF1M 학습 결과 저장 경로 ────────────────────────────────────
X3D_SAVE_PATH    = os.path.join(RESULTS_DIR, "x3d_avdf1m_best.pth")
AASIST_SAVE_PATH = os.path.join(RESULTS_DIR, "aasist_avdf1m_best.pth")
NLP_SAVE_PATH    = os.path.join(RESULTS_DIR, "nlp_avdf1m_best.pth")

# ── transcript 캐시 ───────────────────────────────────────────────
AVDF1M_VAL_CACHE   = os.path.join(BASE, "NLP_architecture/transcript_cache.json")
AVDF1M_TRAIN_CACHE = os.path.join(RESULTS_DIR, "avdf1m_train_transcript_cache.json")

# ── 평가 결과 저장 경로 ───────────────────────────────────────────
PGF_RESULT_FAKEAV  = os.path.join(RESULTS_DIR, "pgf_zeroshot_fakeav.json")
PGF_RESULT_AVDF1M  = os.path.join(RESULTS_DIR, "pgf_zeroshot_avdf1m.json")

# ── 하이퍼파라미터 ────────────────────────────────────────────────
X3D_CFG = {
    "batch_size":   8,
    "epochs":       5,
    "lr":           1e-4,
    "warmup_ratio": 0.1,
    "n_per_class":  10000,   # 클래스당 최대 샘플 수
    "val_ratio":    0.2,
    "seed":         42,
}

AASIST_CFG_TRAIN = {
    "batch_size":   16,
    "epochs":       5,
    "lr":           1e-4,
    "warmup_ratio": 0.1,
    "n_per_class":  10000,
    "val_ratio":    0.2,
    "max_len":      64000,   # 4초 @ 16kHz
    "seed":         42,
}

NLP_CFG = {
    "model_name":   "xlm-roberta-base",
    "max_len":      128,
    "batch_size":   32,
    "epochs":       5,
    "lr":           2e-5,
    "warmup_ratio": 0.1,
    "topk":         3,
    "seed":         42,
}
