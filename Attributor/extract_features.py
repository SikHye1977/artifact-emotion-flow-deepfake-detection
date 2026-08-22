"""
Step 1. X3D / AASIST feature extraction for Attribution Module

지원 데이터셋:
1. FakeAVCeleb
2. AV-Deepfake1M
3. PolyGlotFake

실행 예시:

FakeAVCeleb balanced:
python extract_features.py \
  --dataset fakeavceleb \
  --samples-per-method 100 \
  --clips 4 \
  --output features/fakeavceleb_balanced_100pm.pt

AVDF1M:
python extract_features.py \
  --dataset avdf1m \
  --max-samples 1000 \
  --clips 4 \
  --output features/avdf1m_n1000.pt

PolyGlotFake balanced:
python extract_features.py \
  --dataset pgf \
  --samples-per-tts 100 \
  --clips 4 \
  --output features/pgf_balanced_100ptts.pt
"""

import os
import sys
import json
import argparse
import functools
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import av
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from tqdm import tqdm


# ---------------------------------------------------------------------
# PyTorch 2.5 weights_only 이슈 회피
# ---------------------------------------------------------------------

torch.load = functools.partial(torch.load, weights_only=False)


# ---------------------------------------------------------------------
# torchvision / pytorchvideo 호환 이슈 대응
# ---------------------------------------------------------------------

try:
    import torchvision.transforms.functional_tensor
except Exception:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except Exception:
        pass

from torchvision.transforms import Compose, Lambda, Normalize, Resize
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from pytorchvideo.models.hub import x3d_m


# ---------------------------------------------------------------------
# 기본 경로
# ---------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

X3D_CKPT = PROJECT_ROOT / "x3d_model_best_final.pth"
AASIST_CKPT = PROJECT_ROOT / "aasist_model_best_final.pth"
AASIST_CONF = PROJECT_ROOT / "aasist" / "config" / "AASIST.conf"

FEATURE_DIR = THIS_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------

BINARY_MAP = {
    "real": 0,
    "fake": 1,
}

MODALITY_MAP = {
    "real": 0,
    "visual_modified": 1,
    "audio_modified": 2,
    "both_modified": 3,
    "unknown": -1,
}

FAV_TECH_MAP = {
    "real": 0,
    "faceswap": 1,
    "fsgan": 2,
    "wav2lip": 3,
    "rtvc": 4,
    "faceswap-wav2lip": 5,
    "fsgan-wav2lip": 6,
    "unknown": -1,
}

AVDF_AUDIO_TECH_MAP = {
    "none": 0,
    "vits": 1,
    "vits_word": 2,
    "yourtts": 3,
    "yourtts_word": 4,
    "unknown": -1,
}

AVDF_VISUAL_TECH_MAP = {
    "none": 0,
    "TalkLip": 1,
    "unknown": -1,
}

PGF_AUDIO_TECH_MAP = {
    "none": 0,
    "Bark": 1,
    "MicroTts": 2,
    "Tacotron": 3,
    "Vall": 4,
    "Xtts": 5,
    "unknown": -1,
}

PGF_VISUAL_TECH_MAP = {
    "none": 0,
    "PGF_sync": 1,
    "unknown": -1,
}


# ---------------------------------------------------------------------
# FakeAVCeleb dataframe
# ---------------------------------------------------------------------

def find_fakeavceleb_root() -> Path:
    candidates = [
        PROJECT_ROOT / "FakeAVCeleb_v1.2",
        PROJECT_ROOT.parent / "FakeAVCeleb_v1.2",
        PROJECT_ROOT.parent.parent / "FakeAVCeleb_v1.2",
        Path.cwd() / "FakeAVCeleb_v1.2",
        Path.cwd().parent / "FakeAVCeleb_v1.2",
        Path.cwd().parent.parent / "FakeAVCeleb_v1.2",
    ]

    print("[INFO] searching FakeAVCeleb root candidates...", flush=True)

    for c in candidates:
        print(f"  - {c}", flush=True)
        if (c / "meta_data.csv").exists():
            print(f"[INFO] found FakeAVCeleb root: {c}", flush=True)
            return c

    raise FileNotFoundError("FakeAVCeleb_v1.2/meta_data.csv를 찾지 못했습니다.")


def build_fakeavceleb_path(row: pd.Series, base_dir: Path) -> Optional[str]:
    filename = str(row["path"]).strip()
    candidate_paths = []

    if "Unnamed: 9" in row.index and pd.notna(row["Unnamed: 9"]):
        rel = str(row["Unnamed: 9"]).strip()
        rel_replaced = rel.replace("FakeAVCeleb", base_dir.name, 1)

        candidate_paths.append(base_dir.parent / rel_replaced)

        if rel.startswith("FakeAVCeleb/"):
            rel_inside = rel.replace("FakeAVCeleb/", "", 1)
            candidate_paths.append(base_dir / rel_inside)

        candidate_paths.append(base_dir.parent / rel_replaced / filename)

        if rel.startswith("FakeAVCeleb/"):
            rel_inside = rel.replace("FakeAVCeleb/", "", 1)
            candidate_paths.append(base_dir / rel_inside / filename)

        candidate_paths.append(base_dir.parent / rel / filename)
        candidate_paths.append(base_dir / rel / filename)

    try:
        video_type = str(row["type"]).strip()
        race = str(row["race"]).strip()
        gender = str(row["gender"]).strip()
        source = str(row["source"]).strip()
        candidate_paths.append(base_dir / video_type / race / gender / source / filename)
    except Exception:
        pass

    try:
        video_type = str(row["type"]).strip()
        race = str(row["race"]).strip()
        gender = str(row["gender"]).strip()
        target1 = str(row["target1"]).strip()

        if target1 != "-":
            candidate_paths.append(base_dir / video_type / race / gender / target1 / filename)
    except Exception:
        pass

    candidate_paths.append(base_dir / filename)
    candidate_paths.append(base_dir.parent / filename)

    seen = set()
    for p in candidate_paths:
        p = Path(p)
        key = str(p)

        if key in seen:
            continue
        seen.add(key)

        if p.exists() and p.is_file():
            return str(p)

    return None


def fakeavceleb_method_to_modality(method: str) -> str:
    method = str(method).strip().lower()

    if method == "real":
        return "real"

    if method in ["faceswap", "fsgan", "wav2lip"]:
        return "visual_modified"

    if method == "rtvc":
        return "audio_modified"

    if method in ["faceswap-wav2lip", "fsgan-wav2lip"]:
        return "both_modified"

    return "unknown"


def build_fakeavceleb_dataframe(
    max_samples: Optional[int] = None,
    samples_per_method: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    base_dir = find_fakeavceleb_root()
    meta_path = base_dir / "meta_data.csv"

    print(f"[INFO] FakeAVCeleb root: {base_dir}", flush=True)
    print(f"[INFO] Loading metadata: {meta_path}", flush=True)

    df = pd.read_csv(meta_path)

    print(f"[INFO] metadata rows: {len(df)}", flush=True)
    print(f"[INFO] columns: {list(df.columns)}", flush=True)

    required_cols = ["method", "path", "type", "race", "gender", "source"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"FakeAVCeleb meta_data.csv에 '{col}' 컬럼이 없습니다.")

    df["method"] = df["method"].astype(str).str.strip().str.lower()

    print("[INFO] method distribution:", flush=True)
    print(df["method"].value_counts(dropna=False), flush=True)

    if samples_per_method is not None and samples_per_method > 0:
        print(f"[INFO] balanced sampling: {samples_per_method} samples per method", flush=True)

        sampled_parts = []
        for method_name, g in df.groupby("method"):
            n = min(samples_per_method, len(g))
            sampled = g.sample(n=n, random_state=seed)
            sampled_parts.append(sampled)
            print(f"  - {method_name}: {n}/{len(g)}", flush=True)

        df = pd.concat(sampled_parts, axis=0)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        print("[INFO] balanced sampled method distribution:", flush=True)
        print(df["method"].value_counts(dropna=False), flush=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rows = []
    missing = 0
    checked = 0

    if max_samples is None:
        pbar_total = len(df)
        check_limit = len(df)
    else:
        if samples_per_method is not None and samples_per_method > 0:
            pbar_total = len(df)
            check_limit = len(df)
        else:
            pbar_total = min(len(df), max_samples * 50)
            check_limit = min(len(df), max_samples * 50)

    pbar = tqdm(total=pbar_total, desc="build fakeavceleb df", dynamic_ncols=True)

    first_missing_debug_printed = False

    for idx, row in df.iterrows():
        checked += 1

        method = str(row["method"]).strip().lower()
        binary = "real" if method == "real" else "fake"
        modality = fakeavceleb_method_to_modality(method)
        technique = method if method in FAV_TECH_MAP else "unknown"

        video_path = build_fakeavceleb_path(row, base_dir)

        if video_path is None:
            missing += 1

            if not first_missing_debug_printed:
                first_missing_debug_printed = True
                print("[DEBUG] first missing FakeAVCeleb row:", flush=True)
                print(row, flush=True)
        else:
            rows.append({
                "sample_id": f"fakeavceleb_{idx}",
                "dataset": "fakeavceleb",
                "video_path": video_path,
                "binary_label": BINARY_MAP[binary],
                "modality_label": MODALITY_MAP.get(modality, -1),
                "fav_tech_label": FAV_TECH_MAP.get(technique, -1),
                "audio_tech_label": -1,
                "visual_tech_label": -1,
                "method": method,
                "modify_type": "",
                "audio_model": "",
            })

        if checked <= pbar_total:
            pbar.update(1)

        if checked % 100 == 0:
            pbar.set_postfix({
                "valid": len(rows),
                "missing": missing,
                "checked": checked,
            })

        if max_samples is not None and len(rows) >= max_samples:
            break

        if max_samples is not None and checked >= check_limit:
            break

    pbar.close()

    print(f"[INFO] checked rows: {checked}", flush=True)
    print(f"[INFO] valid videos: {len(rows)}", flush=True)
    print(f"[INFO] missing videos: {missing}", flush=True)

    if len(rows) == 0:
        raise RuntimeError("FakeAVCeleb에서 유효한 비디오 경로를 하나도 찾지 못했습니다.")

    out = pd.DataFrame(rows)

    if max_samples is not None and len(out) > max_samples:
        out = out.iloc[:max_samples].reset_index(drop=True)

    print("[INFO] built dataframe method distribution:", flush=True)
    print(out["method"].value_counts(dropna=False), flush=True)

    print("[INFO] built dataframe modality distribution:", flush=True)
    print(out["modality_label"].value_counts(dropna=False), flush=True)

    print("[INFO] built dataframe technique distribution:", flush=True)
    print(out["fav_tech_label"].value_counts(dropna=False), flush=True)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# AVDF1M dataframe
# ---------------------------------------------------------------------

def normalize_avdf_modify_type(modify_type: str) -> str:
    modify_type = str(modify_type)

    if modify_type == "real":
        return "real"
    if modify_type == "visual_modified":
        return "visual_modified"
    if modify_type == "audio_modified":
        return "audio_modified"
    if modify_type == "both_modified":
        return "both_modified"

    if modify_type == "fake_video_only":
        return "visual_modified"
    if modify_type == "fake_audio_only":
        return "audio_modified"
    if modify_type == "fake_both":
        return "both_modified"

    return "unknown"


def build_avdf1m_dataframe(
    max_samples: Optional[int] = None,
    seed: int = 42,
    avdf1m_root: Optional[str] = None,
) -> pd.DataFrame:
    meta_path = PROJECT_ROOT / "AV-Deepfake1M_RootFiles" / "val_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"AVDF1M metadata not found: {meta_path}")

    print(f"[INFO] Loading AVDF1M metadata: {meta_path}", flush=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if isinstance(meta, dict):
        entries = list(meta.values())
    elif isinstance(meta, list):
        entries = meta
    else:
        raise ValueError("val_metadata.json 형식을 해석할 수 없습니다.")

    print(f"[INFO] AVDF1M metadata entries: {len(entries)}", flush=True)

    root_candidates = []

    if avdf1m_root is not None:
        root_candidates.append(Path(avdf1m_root).expanduser())

    root_candidates.extend([
        PROJECT_ROOT / "AV-Deepfake1M",
        PROJECT_ROOT / "AV-Deepfake1M_RootFiles",
        PROJECT_ROOT / "AV-Deepfake1M_RootFiles" / "val",
        PROJECT_ROOT / "AV-Deepfake1M_RootFiles" / "videos",
        PROJECT_ROOT / "AV-Deepfake1M_RootFiles" / "data",
        PROJECT_ROOT,
        PROJECT_ROOT.parent / "AV-Deepfake1M",
        PROJECT_ROOT.parent / "AV-Deepfake1M_RootFiles",
        PROJECT_ROOT.parent / "datasets" / "AV-Deepfake1M",
        PROJECT_ROOT.parent.parent / "AV-Deepfake1M",
        PROJECT_ROOT.parent.parent / "datasets" / "AV-Deepfake1M",
    ])

    deduped = []
    seen = set()
    for r in root_candidates:
        r = Path(r)
        key = str(r)
        if key not in seen:
            deduped.append(r)
            seen.add(key)

    root_candidates = deduped

    extracted_test_dir = PROJECT_ROOT / "AV-Deepfake1M_RootFiles" / "extracted_test" / "test"

    print("[INFO] AVDF1M root candidates:", flush=True)
    for r in root_candidates:
        print(f"  - {r}", flush=True)

    print(f"[INFO] AVDF1M extracted test dir: {extracted_test_dir}", flush=True)
    print(f"[INFO] extracted test dir exists: {extracted_test_dir.exists()}", flush=True)

    if extracted_test_dir.exists():
        n_mp4 = len(list(extracted_test_dir.glob("*.mp4")))
        print(f"[INFO] extracted test mp4 count: {n_mp4}", flush=True)

    rows = []
    missing = 0
    checked = 0

    rng = np.random.default_rng(seed)
    indices = np.arange(len(entries))
    rng.shuffle(indices)

    if max_samples is None:
        pbar_total = len(indices)
        check_limit = len(indices)
    else:
        pbar_total = min(len(indices), max_samples * 50)
        check_limit = min(len(indices), max_samples * 50)

    pbar = tqdm(total=pbar_total, desc="build avdf1m df", dynamic_ncols=True)

    first_missing_debug_printed = False
    first_valid_debug_printed = False

    for meta_idx in indices:
        checked += 1
        item = entries[int(meta_idx)]

        file_rel = item.get("file", None)
        if file_rel is None:
            missing += 1
            continue

        modify_type = str(item.get("modify_type", "unknown"))
        modality = normalize_avdf_modify_type(modify_type)
        binary = "real" if modality == "real" else "fake"

        audio_model = item.get("audio_model", None)
        if audio_model is None or modality in ["real", "visual_modified"]:
            audio_tech = "none"
        else:
            audio_tech = str(audio_model)

        if modality in ["visual_modified", "both_modified"]:
            visual_tech = "TalkLip"
        else:
            visual_tech = "none"

        video_path = None
        checked_paths = []

        for root in root_candidates:
            p = root / file_rel
            checked_paths.append(p)

            if p.exists() and p.is_file():
                video_path = str(p)
                break

        if video_path is None:
            extracted_name = f"{int(meta_idx):06d}.mp4"
            p = extracted_test_dir / extracted_name
            checked_paths.append(p)

            if p.exists() and p.is_file():
                video_path = str(p)

        if video_path is None:
            missing += 1

            if not first_missing_debug_printed:
                first_missing_debug_printed = True
                print("[DEBUG] first missing AVDF1M item", flush=True)
                print("meta_idx:", int(meta_idx), flush=True)
                print("file_rel:", file_rel, flush=True)
                print("modify_type:", modify_type, flush=True)
                print("checked paths:", flush=True)
                for cp in checked_paths[:20]:
                    print(
                        "  -",
                        cp,
                        "exists:",
                        cp.exists(),
                        "is_file:",
                        cp.is_file(),
                        flush=True,
                    )
        else:
            if not first_valid_debug_printed:
                first_valid_debug_printed = True
                print("[DEBUG] first valid AVDF1M item", flush=True)
                print("meta_idx:", int(meta_idx), flush=True)
                print("file_rel:", file_rel, flush=True)
                print("video_path:", video_path, flush=True)
                print("modify_type:", modify_type, flush=True)

            rows.append({
                "sample_id": f"avdf1m_{meta_idx}",
                "dataset": "avdf1m",
                "video_path": video_path,
                "binary_label": BINARY_MAP[binary],
                "modality_label": MODALITY_MAP.get(modality, -1),
                "fav_tech_label": -1,
                "audio_tech_label": AVDF_AUDIO_TECH_MAP.get(audio_tech, -1),
                "visual_tech_label": AVDF_VISUAL_TECH_MAP.get(visual_tech, -1),
                "method": "",
                "modify_type": modify_type,
                "audio_model": "" if audio_model is None else str(audio_model),
            })

        if checked <= pbar_total:
            pbar.update(1)

        if checked % 100 == 0:
            pbar.set_postfix({
                "valid": len(rows),
                "missing": missing,
                "checked": checked,
            })

        if max_samples is not None and len(rows) >= max_samples:
            break

        if max_samples is not None and checked >= check_limit:
            break

    pbar.close()

    print(f"[INFO] checked entries: {checked}", flush=True)
    print(f"[INFO] valid videos: {len(rows)}", flush=True)
    print(f"[INFO] missing videos: {missing}", flush=True)

    if len(rows) == 0:
        raise RuntimeError("AVDF1M에서 유효한 비디오 경로를 찾지 못했습니다.")

    out = pd.DataFrame(rows)

    print("[INFO] AVDF1M modify_type distribution:", flush=True)
    print(out["modify_type"].value_counts(dropna=False), flush=True)

    print("[INFO] AVDF1M modality distribution:", flush=True)
    print(out["modality_label"].value_counts(dropna=False), flush=True)

    print("[INFO] AVDF1M audio_model distribution:", flush=True)
    print(out["audio_model"].value_counts(dropna=False), flush=True)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# PolyGlotFake dataframe
# ---------------------------------------------------------------------

def find_pgf_root() -> Path:
    candidates = [
        PROJECT_ROOT / "PolyGlotFake",
        PROJECT_ROOT.parent / "PolyGlotFake",
        PROJECT_ROOT.parent.parent / "PolyGlotFake",
        Path.cwd() / "PolyGlotFake",
        Path.cwd().parent / "PolyGlotFake",
    ]

    print("[INFO] searching PolyGlotFake root candidates...", flush=True)

    for c in candidates:
        print(f"  - {c}", flush=True)
        if c.exists() and (c / "fake").exists() and (c / "real").exists():
            print(f"[INFO] found PolyGlotFake root: {c}", flush=True)
            return c

    raise FileNotFoundError("PolyGlotFake root를 찾지 못했습니다.")


def parse_pgf_key(key: str):
    """
    key 예:
    - fake/ru_61_to_en_MicroTts.mp4
    - real/en_31.mp4

    return:
    is_fake, src_lang, target_lang, tts_name
    """
    key = str(key).strip()
    filename = Path(key).name

    if key.startswith("real/"):
        stem = Path(filename).stem
        lang = stem.split("_")[0]
        return False, lang, lang, "none"

    if key.startswith("fake/"):
        stem = Path(filename).stem
        parts = stem.split("_")

        # ru_61_to_en_MicroTts
        if len(parts) >= 5 and "to" in parts:
            src_lang = parts[0]
            to_idx = parts.index("to")
            target_lang = parts[to_idx + 1]
            tts_name = parts[-1]
        else:
            src_lang = "unknown"
            target_lang = "unknown"
            tts_name = "unknown"

        return True, src_lang, target_lang, tts_name

    return False, "unknown", "unknown", "unknown"


def build_pgf_path(key: str, pgf_root: Path) -> Optional[str]:
    """
    metadata:
    - fake/ru_61_to_en_MicroTts.mp4
    - real/en_31.mp4

    actual:
    - PolyGlotFake/fake/to_en/ru_61_to_en_MicroTts.mp4
    - PolyGlotFake/real/en/en_31.mp4
    """
    key = str(key).strip()
    filename = Path(key).name

    is_fake, src_lang, target_lang, _ = parse_pgf_key(key)

    candidate_paths = []

    if is_fake:
        candidate_paths.append(pgf_root / "fake" / f"to_{target_lang}" / filename)
        candidate_paths.append(pgf_root / key)
        candidate_paths.append(pgf_root / "fake" / filename)
    else:
        candidate_paths.append(pgf_root / "real" / src_lang / filename)
        candidate_paths.append(pgf_root / key)
        candidate_paths.append(pgf_root / "real" / filename)

    seen = set()
    for p in candidate_paths:
        p = Path(p)
        s = str(p)
        if s in seen:
            continue
        seen.add(s)

        if p.exists() and p.is_file():
            return str(p)

    return None


def build_pgf_dataframe(
    max_samples: Optional[int] = None,
    samples_per_tts: Optional[int] = None,
    seed: int = 42,
    pgf_metadata: Optional[str] = None,
) -> pd.DataFrame:
    pgf_root = find_pgf_root()

    if pgf_metadata is None:
        meta_path = (
            PROJECT_ROOT
            / "NLP_architecture"
            / "unified_experiment"
            / "results"
            / "pgf_dataset.csv"
        )
    else:
        meta_path = Path(pgf_metadata).expanduser()

    if not meta_path.exists():
        raise FileNotFoundError(f"PGF metadata not found: {meta_path}")

    print(f"[INFO] PGF root: {pgf_root}", flush=True)
    print(f"[INFO] Loading PGF metadata: {meta_path}", flush=True)

    df = pd.read_csv(meta_path)

    print(f"[INFO] PGF rows: {len(df)}", flush=True)
    print(f"[INFO] PGF columns: {list(df.columns)}", flush=True)

    if "key" not in df.columns:
        raise ValueError("PGF metadata에 'key' 컬럼이 없습니다.")
    if "clip_label" not in df.columns:
        raise ValueError("PGF metadata에 'clip_label' 컬럼이 없습니다.")

    print("[INFO] PGF clip_label distribution:", flush=True)
    print(df["clip_label"].value_counts(dropna=False), flush=True)

    if "split" in df.columns:
        print("[INFO] PGF split distribution:", flush=True)
        print(df["split"].value_counts(dropna=False), flush=True)

    parsed = df["key"].apply(parse_pgf_key)
    df["is_fake_parsed"] = parsed.apply(lambda x: x[0])
    df["src_lang"] = parsed.apply(lambda x: x[1])
    df["target_lang"] = parsed.apply(lambda x: x[2])
    df["tts_name"] = parsed.apply(lambda x: x[3])

    print("[INFO] PGF tts distribution:", flush=True)
    print(df["tts_name"].value_counts(dropna=False), flush=True)

    if samples_per_tts is not None and samples_per_tts > 0:
        print(f"[INFO] PGF balanced sampling: {samples_per_tts} samples per tts/real group", flush=True)

        sampled_parts = []

        real_df = df[df["clip_label"].astype(int) == 0]
        if len(real_df) > 0:
            n = min(samples_per_tts, len(real_df))
            sampled_parts.append(real_df.sample(n=n, random_state=seed))
            print(f"  - real: {n}/{len(real_df)}", flush=True)

        fake_df = df[df["clip_label"].astype(int) == 1]
        for tts_name, g in fake_df.groupby("tts_name"):
            n = min(samples_per_tts, len(g))
            sampled_parts.append(g.sample(n=n, random_state=seed))
            print(f"  - {tts_name}: {n}/{len(g)}", flush=True)

        df = pd.concat(sampled_parts, axis=0)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rows = []
    missing = 0
    checked = 0

    if max_samples is None:
        pbar_total = len(df)
        check_limit = len(df)
    else:
        if samples_per_tts is not None and samples_per_tts > 0:
            pbar_total = len(df)
            check_limit = len(df)
        else:
            pbar_total = min(len(df), max_samples * 20)
            check_limit = min(len(df), max_samples * 20)

    pbar = tqdm(total=pbar_total, desc="build pgf df", dynamic_ncols=True)

    first_missing_debug_printed = False
    first_valid_debug_printed = False

    for idx, row in df.iterrows():
        checked += 1

        key = str(row["key"]).strip()
        clip_label = int(row["clip_label"])

        is_fake, src_lang, target_lang, tts_name = parse_pgf_key(key)

        video_path = build_pgf_path(key, pgf_root)

        if video_path is None:
            missing += 1

            if not first_missing_debug_printed:
                first_missing_debug_printed = True
                print("[DEBUG] first missing PGF row:", flush=True)
                print(row, flush=True)
                print("expected key:", key, flush=True)
                print("parsed:", is_fake, src_lang, target_lang, tts_name, flush=True)
        else:
            if not first_valid_debug_printed:
                first_valid_debug_printed = True
                print("[DEBUG] first valid PGF row:", flush=True)
                print("key:", key, flush=True)
                print("video_path:", video_path, flush=True)
                print("parsed:", is_fake, src_lang, target_lang, tts_name, flush=True)

            binary = "fake" if clip_label == 1 else "real"

            if clip_label == 0:
                modality = "real"
                audio_tech = "none"
                visual_tech = "none"
                modify_type = "real"
            else:
                modality = "both_modified"
                audio_tech = tts_name if tts_name in PGF_AUDIO_TECH_MAP else "unknown"
                visual_tech = "PGF_sync"
                modify_type = str(row.get("modify_type", "lang_swap"))

            rows.append({
                "sample_id": f"pgf_{idx}",
                "dataset": "pgf",
                "video_path": video_path,

                "binary_label": BINARY_MAP[binary],
                "modality_label": MODALITY_MAP.get(modality, -1),
                "fav_tech_label": -1,

                "audio_tech_label": PGF_AUDIO_TECH_MAP.get(audio_tech, -1),
                "visual_tech_label": PGF_VISUAL_TECH_MAP.get(visual_tech, -1),

                "method": "",
                "modify_type": modify_type,
                "audio_model": audio_tech,

                "pgf_key": key,
                "src_lang": src_lang,
                "target_lang": target_lang,
                "tts_name": tts_name,
                "split": str(row.get("split", "")),
            })

        if checked <= pbar_total:
            pbar.update(1)

        if checked % 100 == 0:
            pbar.set_postfix({
                "valid": len(rows),
                "missing": missing,
                "checked": checked,
            })

        if max_samples is not None and len(rows) >= max_samples:
            break

        if max_samples is not None and checked >= check_limit:
            break

    pbar.close()

    print(f"[INFO] checked rows: {checked}", flush=True)
    print(f"[INFO] valid videos: {len(rows)}", flush=True)
    print(f"[INFO] missing videos: {missing}", flush=True)

    if len(rows) == 0:
        raise RuntimeError("PGF에서 유효한 비디오 경로를 하나도 찾지 못했습니다.")

    out = pd.DataFrame(rows)

    if max_samples is not None and len(out) > max_samples:
        out = out.iloc[:max_samples].reset_index(drop=True)

    print("[INFO] PGF built binary distribution:", flush=True)
    print(out["binary_label"].value_counts(dropna=False), flush=True)

    print("[INFO] PGF built modality distribution:", flush=True)
    print(out["modality_label"].value_counts(dropna=False), flush=True)

    print("[INFO] PGF built audio tech distribution:", flush=True)
    print(out["audio_tech_label"].value_counts(dropna=False), flush=True)

    print("[INFO] PGF built visual tech distribution:", flush=True)
    print(out["visual_tech_label"].value_counts(dropna=False), flush=True)

    print("[INFO] PGF built audio_model distribution:", flush=True)
    print(out["audio_model"].value_counts(dropna=False), flush=True)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Video / Audio loading
# ---------------------------------------------------------------------

def load_video_frames(path: str, max_frames: int = 128) -> Optional[torch.Tensor]:
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
    except Exception as e:
        print(f"[WARN] video decode failed: {path} / {e}", flush=True)
        return None

    if len(frames) < 16:
        print(f"[WARN] too few frames: {path} / frames={len(frames)}", flush=True)
        return None

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in idx]

    video = np.stack(frames)
    video = torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    return video


def load_audio_waveform(video_path: str, target_sr: int = 16000) -> Optional[torch.Tensor]:
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close()
            print(f"[WARN] no audio stream: {video_path}", flush=True)
            return None

        sr = container.streams.audio[0].rate
        frames = []

        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()

            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            else:
                arr = arr.astype(np.float32)

            if arr.ndim > 1 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            elif arr.ndim == 1:
                arr = arr[np.newaxis, :]

            frames.append(arr)

        container.close()

        if not frames:
            print(f"[WARN] empty audio frames: {video_path}", flush=True)
            return None

        wav = torch.from_numpy(np.concatenate(frames, axis=-1))

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

        return wav.squeeze(0)

    except Exception as e:
        print(f"[WARN] audio decode failed: {video_path} / {e}", flush=True)
        return None


def split_video_into_clips(
    video: torch.Tensor,
    num_clips: int,
    frames_per_clip: int = 16,
) -> List[torch.Tensor]:
    _, T, _, _ = video.shape

    if T < frames_per_clip:
        return []

    clips = []

    if num_clips <= 1:
        idx = torch.linspace(0, T - 1, frames_per_clip).long()
        clips.append(video[:, idx])
        return clips

    centers = np.linspace(0, T - 1, num_clips)

    for c in centers:
        half = frames_per_clip // 2
        start = int(round(c)) - half
        end = start + frames_per_clip

        if start < 0:
            start = 0
            end = frames_per_clip

        if end > T:
            end = T
            start = T - frames_per_clip

        clip = video[:, start:end]
        if clip.shape[1] == frames_per_clip:
            clips.append(clip)

    return clips


def split_audio_into_chunks(
    wav: torch.Tensor,
    num_chunks: int,
    chunk_size: int = 64000,
) -> List[torch.Tensor]:
    N = wav.numel()

    if num_chunks <= 1:
        if N >= chunk_size:
            return [wav[:chunk_size]]
        return [F.pad(wav, (0, chunk_size - N))]

    chunks = []
    centers = np.linspace(0, max(N - 1, 1), num_chunks)

    for c in centers:
        half = chunk_size // 2
        start = int(round(c)) - half
        end = start + chunk_size

        if start < 0:
            start = 0
            end = chunk_size

        if end > N:
            end = N
            start = max(0, N - chunk_size)

        chunk = wav[start:end]

        if chunk.numel() < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - chunk.numel()))

        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def build_x3d_transform():
    return Compose([
        UniformTemporalSubsample(16),
        Lambda(lambda x: x / 255.0),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ShortSideScale(size=256),
        Resize((224, 224)),
    ])


class X3DWithEmbedding(nn.Module):
    def __init__(self, ckpt_path: Path):
        super().__init__()

        self.model = x3d_m(pretrained=False)
        self.model.blocks[5].proj = nn.Linear(2048, 1)
        self.model.blocks[5].activation = nn.Identity()

        print(f"[INFO] loading X3D checkpoint: {ckpt_path}", flush=True)
        state = torch.load(str(ckpt_path), map_location="cpu")
        state = state.get("model_state_dict", state)
        self.model.load_state_dict(state, strict=True)

        self._last_embedding = None

        try:
            self.model.blocks[5].dropout.register_forward_hook(self._hook_embedding)
            print("[INFO] X3D embedding hook registered: blocks[5].dropout", flush=True)
        except Exception as e:
            print(f"[WARN] X3D embedding hook registration failed: {e}", flush=True)

    def _hook_embedding(self, module, inp, out):
        self._last_embedding = out

    def forward(self, x):
        self._last_embedding = None
        logit = self.model(x)

        if logit.ndim > 2:
            logit = logit.flatten(1)

        if self._last_embedding is not None:
            emb = self._last_embedding.flatten(1)
        else:
            emb = torch.empty((x.size(0), 0), device=x.device)

        return emb, logit


def load_aasist_model(ckpt_path: Path, conf_path: Path, device: torch.device):
    sys.path.insert(0, str(PROJECT_ROOT))

    from aasist.models.AASIST import Model as AASISTModel

    print(f"[INFO] loading AASIST config: {conf_path}", flush=True)
    with open(conf_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = AASISTModel(config["model_config"])

    print(f"[INFO] loading AASIST checkpoint: {ckpt_path}", flush=True)
    state = torch.load(str(ckpt_path), map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_x3d_clip_features(
    model: X3DWithEmbedding,
    transform,
    clips: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_list = []
    logit_list = []
    prob_list = []

    for clip in clips:
        x = transform(clip)
        x = x.unsqueeze(0).to(device)

        emb, logit = model(x)
        prob = torch.sigmoid(logit)

        z_list.append(emb.squeeze(0).detach().cpu())
        logit_list.append(logit.squeeze(0).detach().cpu())
        prob_list.append(prob.squeeze(0).detach().cpu())

    return (
        torch.stack(z_list, dim=0),
        torch.stack(logit_list, dim=0),
        torch.stack(prob_list, dim=0),
    )


@torch.no_grad()
def extract_aasist_chunk_features(
    model,
    chunks: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_list = []
    logit_list = []
    prob_list = []

    for chunk in chunks:
        x = chunk.unsqueeze(0).to(device)

        out = model(x)

        if isinstance(out, tuple):
            hidden, logits = out
        else:
            hidden = None
            logits = out

        if logits.ndim > 2:
            logits = logits.flatten(1)

        prob = torch.softmax(logits, dim=1)[:, 1:2]

        if hidden is None:
            emb = logits
        else:
            if isinstance(hidden, (list, tuple)):
                hidden = hidden[0]
            emb = hidden
            if emb.ndim > 2:
                emb = emb.flatten(1)

        z_list.append(emb.squeeze(0).detach().cpu())
        logit_list.append(logits.squeeze(0).detach().cpu())
        prob_list.append(prob.squeeze(0).detach().cpu())

    return (
        torch.stack(z_list, dim=0),
        torch.stack(logit_list, dim=0),
        torch.stack(prob_list, dim=0),
    )


def process_one_sample(
    row: pd.Series,
    x3d_model: X3DWithEmbedding,
    x3d_transform,
    aasist_model,
    device: torch.device,
    num_clips: int,
) -> Optional[Dict[str, Any]]:
    video_path = row["video_path"]

    video = load_video_frames(video_path)
    if video is None:
        return None

    wav = load_audio_waveform(video_path)
    if wav is None:
        return None

    v_clips = split_video_into_clips(video, num_clips=num_clips, frames_per_clip=16)
    a_chunks = split_audio_into_chunks(wav, num_chunks=num_clips, chunk_size=64000)

    if len(v_clips) == 0 or len(a_chunks) == 0:
        return None

    K = min(len(v_clips), len(a_chunks))
    v_clips = v_clips[:K]
    a_chunks = a_chunks[:K]

    try:
        z_v, logit_v, p_v = extract_x3d_clip_features(
            x3d_model,
            x3d_transform,
            v_clips,
            device,
        )

        z_a, logits_a, p_a = extract_aasist_chunk_features(
            aasist_model,
            a_chunks,
            device,
        )

    except Exception as e:
        print(f"[WARN] feature extraction failed: {video_path} / {e}", flush=True)
        return None

    item = {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "video_path": video_path,

        "binary_label": int(row["binary_label"]),
        "modality_label": int(row["modality_label"]),
        "fav_tech_label": int(row["fav_tech_label"]),
        "audio_tech_label": int(row["audio_tech_label"]),
        "visual_tech_label": int(row["visual_tech_label"]),

        "method": str(row.get("method", "")),
        "modify_type": str(row.get("modify_type", "")),
        "audio_model": str(row.get("audio_model", "")),

        "z_v": z_v,
        "logit_v": logit_v,
        "p_v": p_v,

        "z_a": z_a,
        "logits_a": logits_a,
        "p_a": p_a,
    }

    # optional metadata
    for k in [
        "pgf_key",
        "src_lang",
        "target_lang",
        "tts_name",
        "split",
    ]:
        if k in row.index:
            item[k] = str(row.get(k, ""))

    return item


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fakeavceleb", "avdf1m", "pgf"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--samples-per-method",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--samples-per-tts",
        type=int,
        default=None,
        help="PGF real 및 TTS별 균형 샘플 수.",
    )
    parser.add_argument(
        "--avdf1m-root",
        type=str,
        default=None,
        help="AVDF1M 원본 영상 root 경로. extracted_test/test 방식이면 생략 가능.",
    )
    parser.add_argument(
        "--pgf-metadata",
        type=str,
        default=None,
        help="PGF metadata csv path. 기본값은 unified_experiment/results/pgf_dataset.csv",
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device = {device}", flush=True)
    print(f"[INFO] project root = {PROJECT_ROOT}", flush=True)
    print(f"[INFO] current dir = {Path.cwd()}", flush=True)
    print(f"[INFO] dataset = {args.dataset}", flush=True)
    print(f"[INFO] clips = {args.clips}", flush=True)
    print(f"[INFO] max_samples = {args.max_samples}", flush=True)
    print(f"[INFO] samples_per_method = {args.samples_per_method}", flush=True)
    print(f"[INFO] samples_per_tts = {args.samples_per_tts}", flush=True)
    print(f"[INFO] avdf1m_root = {args.avdf1m_root}", flush=True)
    print(f"[INFO] pgf_metadata = {args.pgf_metadata}", flush=True)

    if not X3D_CKPT.exists():
        raise FileNotFoundError(f"X3D checkpoint not found: {X3D_CKPT}")

    if not AASIST_CKPT.exists():
        raise FileNotFoundError(f"AASIST checkpoint not found: {AASIST_CKPT}")

    if not AASIST_CONF.exists():
        raise FileNotFoundError(f"AASIST config not found: {AASIST_CONF}")

    print("[INFO] building dataframe...", flush=True)

    if args.dataset == "fakeavceleb":
        df = build_fakeavceleb_dataframe(
            max_samples=args.max_samples,
            samples_per_method=args.samples_per_method,
            seed=args.seed,
        )

    elif args.dataset == "avdf1m":
        df = build_avdf1m_dataframe(
            max_samples=args.max_samples,
            seed=args.seed,
            avdf1m_root=args.avdf1m_root,
        )

    elif args.dataset == "pgf":
        df = build_pgf_dataframe(
            max_samples=args.max_samples,
            samples_per_tts=args.samples_per_tts,
            seed=args.seed,
            pgf_metadata=args.pgf_metadata,
        )

    else:
        raise ValueError(args.dataset)

    print(f"[INFO] samples = {len(df)}", flush=True)
    print(df.head(), flush=True)

    print("[INFO] label distribution:", flush=True)

    if "method" in df.columns:
        print("[method]", flush=True)
        print(df["method"].value_counts(dropna=False), flush=True)

    if "modify_type" in df.columns:
        print("[modify_type]", flush=True)
        print(df["modify_type"].value_counts(dropna=False), flush=True)

    if "modality_label" in df.columns:
        print("[modality_label]", flush=True)
        print(df["modality_label"].value_counts(dropna=False), flush=True)

    if "fav_tech_label" in df.columns:
        print("[fav_tech_label]", flush=True)
        print(df["fav_tech_label"].value_counts(dropna=False), flush=True)

    if "audio_tech_label" in df.columns:
        print("[audio_tech_label]", flush=True)
        print(df["audio_tech_label"].value_counts(dropna=False), flush=True)

    if "visual_tech_label" in df.columns:
        print("[visual_tech_label]", flush=True)
        print(df["visual_tech_label"].value_counts(dropna=False), flush=True)

    print("[INFO] loading X3D...", flush=True)
    x3d_model = X3DWithEmbedding(X3D_CKPT).to(device)
    x3d_model.eval()
    x3d_transform = build_x3d_transform()

    print("[INFO] loading AASIST...", flush=True)
    aasist_model = load_aasist_model(AASIST_CKPT, AASIST_CONF, device)

    features = []
    failed = 0

    print("[INFO] starting feature extraction...", flush=True)

    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        desc="extract",
        dynamic_ncols=True,
    )

    for _, row in pbar:
        item = process_one_sample(
            row=row,
            x3d_model=x3d_model,
            x3d_transform=x3d_transform,
            aasist_model=aasist_model,
            device=device,
            num_clips=args.clips,
        )

        if item is None:
            failed += 1
        else:
            features.append(item)

        if len(features) > 0:
            last = features[-1]
            pbar.set_postfix({
                "ok": len(features),
                "fail": failed,
                "last_pv": f"{float(last['p_v'].mean()):.3f}",
                "last_pa": f"{float(last['p_a'].mean()):.3f}",
            })
        else:
            pbar.set_postfix({
                "ok": len(features),
                "fail": failed,
            })

    if args.output is None:
        if args.dataset == "fakeavceleb" and args.samples_per_method is not None:
            out_path = FEATURE_DIR / f"{args.dataset}_balanced_{args.samples_per_method}pm.pt"
        elif args.dataset == "pgf" and args.samples_per_tts is not None:
            out_path = FEATURE_DIR / f"{args.dataset}_balanced_{args.samples_per_tts}ptts.pt"
        elif args.max_samples is not None:
            out_path = FEATURE_DIR / f"{args.dataset}_features_n{args.max_samples}.pt"
        else:
            out_path = FEATURE_DIR / f"{args.dataset}_features.pt"
    else:
        out_path = Path(args.output)

    payload = {
        "dataset": args.dataset,
        "num_clips": args.clips,
        "num_samples": len(features),
        "failed": failed,
        "binary_map": BINARY_MAP,
        "modality_map": MODALITY_MAP,
        "fav_tech_map": FAV_TECH_MAP,
        "avdf_audio_tech_map": AVDF_AUDIO_TECH_MAP,
        "avdf_visual_tech_map": AVDF_VISUAL_TECH_MAP,
        "pgf_audio_tech_map": PGF_AUDIO_TECH_MAP,
        "pgf_visual_tech_map": PGF_VISUAL_TECH_MAP,
        "features": features,
    }

    print(f"[INFO] saving features to: {out_path}", flush=True)
    torch.save(payload, out_path)

    print("=" * 80, flush=True)
    print("[DONE] feature extraction complete", flush=True)
    print(f"[SAVE] {out_path}", flush=True)
    print(f"[OK]   extracted: {len(features)}", flush=True)
    print(f"[FAIL] failed:    {failed}", flush=True)

    if len(features) > 0:
        ex = features[0]

        print("-" * 80, flush=True)
        print("[CHECK] first sample", flush=True)
        print("sample_id:", ex["sample_id"], flush=True)
        print("dataset:", ex["dataset"], flush=True)
        print("binary_label:", ex["binary_label"], flush=True)
        print("modality_label:", ex["modality_label"], flush=True)
        print("fav_tech_label:", ex["fav_tech_label"], flush=True)
        print("audio_tech_label:", ex["audio_tech_label"], flush=True)
        print("visual_tech_label:", ex["visual_tech_label"], flush=True)
        print("method:", ex["method"], flush=True)
        print("modify_type:", ex["modify_type"], flush=True)
        print("audio_model:", ex["audio_model"], flush=True)

        if "pgf_key" in ex:
            print("pgf_key:", ex["pgf_key"], flush=True)
            print("src_lang:", ex["src_lang"], flush=True)
            print("target_lang:", ex["target_lang"], flush=True)
            print("tts_name:", ex["tts_name"], flush=True)
            print("split:", ex["split"], flush=True)

        print("z_v:", tuple(ex["z_v"].shape), flush=True)
        print("logit_v:", tuple(ex["logit_v"].shape), flush=True)
        print("p_v:", tuple(ex["p_v"].shape), flush=True)
        print("z_a:", tuple(ex["z_a"].shape), flush=True)
        print("logits_a:", tuple(ex["logits_a"].shape), flush=True)
        print("p_a:", tuple(ex["p_a"].shape), flush=True)


if __name__ == "__main__":
    main()