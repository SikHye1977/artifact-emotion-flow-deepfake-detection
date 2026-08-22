# 딥페이크 탐지 연구 인수인계 문서

**작성일**: 2026-08-22
**연구 주제**: Hierarchical Score-Level Fusion of Artifact and Emotion Flow Signals for Cross-Dataset Deepfake Detection
**논문 상태**: ICIP 2022 형식 v7 작성 완료, Reviewer response 준비 중
**목적**: 새 PC로 이관하여 남은 작업 (X3D AVDF1M 학습 + Phase 3 평가 3개 시나리오) 완료

---

## 📑 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [하드웨어 및 환경](#2-하드웨어-및-환경)
3. [디렉토리 구조](#3-디렉토리-구조)
4. [데이터셋 상세](#4-데이터셋-상세)
5. [모델 아키텍처](#5-모델-아키텍처)
6. [학습 설정 (공통 하이퍼파라미터)](#6-학습-설정-공통-하이퍼파라미터)
7. [진행 상황: 9-시나리오 매트릭스](#7-진행-상황-9-시나리오-매트릭스)
8. [가중치 파일 인벤토리](#8-가중치-파일-인벤토리)
9. [학습 실행 매뉴얼](#9-학습-실행-매뉴얼)
10. [평가 실행 매뉴얼](#10-평가-실행-매뉴얼)
11. [알려진 이슈 및 해결](#11-알려진-이슈-및-해결)
12. [논문 작성 방향](#12-논문-작성-방향)
13. [Reviewer Response 진행 상황](#13-reviewer-response-진행-상황)
14. [남은 작업 To-Do](#14-남은-작업-to-do)
15. [파일 이관 체크리스트](#15-파일-이관-체크리스트)

---

## 1. 프로젝트 개요

### 핵심 아이디어

Cross-dataset 딥페이크 탐지를 위한 **hierarchical score-level fusion**. 두 축의 신호를 결합:

- **Artifact 신호** (합성 파이프라인 의존적): X3D (video) + AASIST (audio)
- **Emotion Flow 신호** (합성 파이프라인 비의존적): HSEmotion (video) + CRNN (audio)

계층적 융합 구조:
```
Score_artifact = OR(X3D, AASIST)
Score_emotion  = OR(HSEmotion, CRNN)
Score_final    = OR(Score_artifact, Score_emotion)
```

### 최종 목표: 9-시나리오 Cross-Dataset Matrix

3개 데이터셋 (FAV, PGF, AVDF1M) × 3개 평가셋 = 9 시나리오 매트릭스 완성.

| 학습\평가 | FAV | PGF | AVDF1M |
|---|---|---|---|
| **FAV** | F→F ✅ | F→P ✅ | F→A ✅ |
| **PGF** | P→F ✅ | P→P ✅ | P→A ✅ |
| **AVDF1M** | A→F ⏳ | A→P ⏳ | A→A ⏳ |

**남은 작업**: AVDF1M으로 학습한 모델 4개(X3D 미완료) + Phase 3 평가 3개 시나리오.

### 파라미터/연산량

- **파라미터**: 8.79M
- **연산량**: 12.36 GFLOPs
- **모바일 배포 가능** (핵심 논문 셀링 포인트)

---

## 2. 하드웨어 및 환경

### 원본 PC 사양

- **CPU**: Intel i7-11700
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA RTX 3060 12GB
- **OS**: Ubuntu 24 (WSL2 아님, 네이티브)
- **사용자명**: `uxm2`

### Conda 환경

- **환경명**: `deepfake`
- **Python**: 3.10
- **주요 패키지**:
  - PyTorch 2.5.1 + CUDA 12.1
  - torchvision (functional_tensor deprecation 처리 필요, §11 참고)
  - torchaudio
  - timm (HSEmotion backbone용)
  - hsemotion (`pip install hsemotion`)
  - librosa, soundfile (audio 처리)
  - scikit-learn (평가 metric)
  - pandas, numpy, matplotlib
  - tqdm, pyyaml

### 환경 이관 방법

**원본 PC에서**:
```bash
conda env export --name deepfake > deepfake_env.yml
# 또는 requirements만 필요하면:
pip freeze > requirements.txt
```

**새 PC에서**:
```bash
# 방법 1: conda env
conda env create -f deepfake_env.yml
conda activate deepfake

# 방법 2: 처음부터 재구축
conda create -n deepfake python=3.10 -y
conda activate deepfake
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm hsemotion librosa soundfile scikit-learn pandas numpy matplotlib tqdm pyyaml
pip install pytorchvideo  # X3D용
```

### GPU 확인

```bash
nvidia-smi  # RTX 3060 12GB 인식 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True, NVIDIA GeForce RTX 3060
```

---

## 3. 디렉토리 구조

### 작업 루트

```
~/hsh/AIApplication/
├── FakeAVCeleb_v1.2/                         # 데이터셋 1 (FAV)
│   ├── RealVideo-RealAudio/
│   ├── FakeVideo-RealAudio/
│   ├── RealVideo-FakeAudio/
│   ├── FakeVideo-FakeAudio/
│   └── meta_data.csv                          # 21,566 entries, 10 cols
│
├── PolyGlotFake_dataset/                     # 데이터셋 2 (PGF, 경로 확인 필요)
│   └── ... (real/fake 하위 구조)
│
├── AV-Deepfake1M_RootFiles/                  # 데이터셋 3 (AVDF1M)
│   ├── extracted_train/train/{speaker}/{youtube_id}/{seq:05d}/{filename}.mp4
│   ├── extracted_val/val/{speaker}/{youtube_id}/{seq:05d}/{filename}.mp4
│   ├── train_metadata.json                   # 238MB, 746,180 entries
│   └── val_metadata.json                     # 18MB, 57,340 entries
│
├── reverse_zero_shot/                        # PGF/AVDF1M 가중치 폴더
│   ├── polyglotfake_data.py                  # PGF 데이터 로더
│   ├── x3d_model_pgf_best.pth
│   ├── aasist_model_pgf_best.pth
│   ├── emotion_flow_lite_pgf_v2_best.pth
│   ├── audio_flow_deepfake_pgf_v2_best.pth
│   ├── aasist_model_avdf1m_best.pth          # AVDF1M 학습 완료 (AUC 99.87%)
│   ├── audio_flow_deepfake_avdf1m_v2_best.pth  # CRNN AVDF1M (AUC 50.72% 실패)
│   └── emotion_flow_lite_avdf1m_v2_best.pth  # HSEmotion AVDF1M 완료 (결과 확인 필요)
│
├── aasist/                                   # AASIST 저장소
│   └── models/AASIST.py
│
├── x3d_model_best_final.pth                  # FAV X3D 가중치
├── aasist_model_best_final.pth               # FAV AASIST 가중치
├── emotion_flow_lite_best.pth                # FAV HSEmotion v1 가중치
├── audio_flow_deepfake_best.pth              # FAV CRNN v1 가중치
├── audio_emotion_crnn_best.pth               # RAVDESS 사전학습 (CRNN 초기화용)
│
├── train_x3d.py, train_x3d_pgf.py, train_x3d_avdf1m.py
├── train_aasist.py, train_aasist_pgf.py, train_aasist_avdf1m.py
├── train_HSEmotion.py, train_HSEmotion_pgf.py, train_HSEmotion_avdf1m.py
├── train_CRNN.py, train_CRNN_pgf.py, train_CRNN_avdf1m.py
│
├── evaluate_*.py                             # 평가 스크립트들
├── avdf1m_train_data.py                      # AVDF1M 데이터 모듈
│
└── emotion_deepfake_detector_lite.py         # 모델 클래스 정의
    audio_emotion_deepfake_detector.py        # CRNN 모델 클래스
```

### `~/.hsemotion/`

HSEmotion 사전학습 가중치 자동 다운로드 위치:
```
~/.hsemotion/enet_b0_8_best_afew.pt
```
새 PC에서 첫 실행 시 자동으로 다운로드됨.

---

## 4. 데이터셋 상세

### 4.1 FakeAVCeleb (FAV)

- **위치**: `~/hsh/AIApplication/FakeAVCeleb_v1.2/`
- **메타**: `meta_data.csv` (21,566 entries)
- **컬럼**: `source`, `target1`, `target2`, `method`, `category`, `type`, `race`, `gender`, `path`, `Unnamed: 9`
- **하위 구조**: 4개 type × 5개 race × 2개 gender × N개 id × 여러 clip
- **합성 기법**: `real`, `wav2lip`, `fsgan`, `fsgan-wav2lip`, `faceswap`, `faceswap-wav2lip`, `rtvc`

**경로 빌드 로직** (중요):
```python
# 예: 'Unnamed: 9' = "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076"
#     'path'       = "00109.mp4"
rel_dir = row['Unnamed: 9'].replace('FakeAVCeleb', 'FakeAVCeleb_v1.2')
video_path = os.path.join(base_parent, rel_dir, row['path'])
# → <PROJECT_ROOT>/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4
```

### 4.2 PolyGlotFake (PGF)

- **위치**: `~/hsh/AIApplication/PolyGlotFake_dataset/` (경로 확인 필요)
- **데이터 로더**: `reverse_zero_shot/polyglotfake_data.py`의 `build_polyglotfake_dataframe()`
- **합성 기법**:
  - **TTS**: `Bark`, `MicroTts`, `Tacotron`, `Vall-E`, `Xtts`
  - **Sync**: `Wav2Lip`, `video_retalking`
- **언어**: 다국어 (English, Chinese, ...)

### 4.3 AV-Deepfake1M (AVDF1M)

- **위치**: `~/hsh/AIApplication/AV-Deepfake1M_RootFiles/`
- **학습 영상**: `extracted_train/train/{speaker}/{youtube_id}/{seq:05d}/{filename}.mp4`
- **검증 영상**: `extracted_val/val/{speaker}/{youtube_id}/{seq:05d}/{filename}.mp4`
- **학습 메타**: `train_metadata.json` (746,180 entries)
- **검증 메타**: `val_metadata.json` (57,340 entries)
- **실제 추출**: 141 speakers, 58,636 mp4 (~98% 추출률)
- **특징**: LLM 기반 word-level micro-manipulation (특정 단어만 교체)
- **seq_id 주의**: 5자리 zero-pad 필요 (`f"{seq:05d}"`)

**데이터 모듈**: `avdf1m_train_data.py`의 `build_avdf1m_train_val_split()`
- 141 speaker 필터링 (실제 mp4 추출된 speaker만)
- speaker-level train/val 분할 (leakage 방지)

---

## 5. 모델 아키텍처

| 모델 | Modality | 역할 | Backbone | 파라미터 |
|---|---|---|---|---|
| **X3D-M** | Video | Visual artifact | X3D_m (Kinetics-400 pretrained) | ~3.8M |
| **AASIST** | Audio | Audio artifact | AASIST (ASVspoof2019 pretrained) | ~0.3M |
| **HSEmotion** | Video | Visual emotion flow | EfficientNet-B0 (AffectNet 8-class) + GRU + Attention | ~4.5M |
| **CRNN** | Audio | Audio emotion flow | AudioEmotionCRNN (RAVDESS 8-class) + GRU + Attention | ~0.2M |
| **합계** | - | - | - | **~8.79M** |

### 5.1 X3D
- **입력**: 16 frames, 224×224 RGB
- **출력**: 2-class logit (real/fake)
- **Loss**: BCEWithLogitsLoss(fake_logit)

### 5.2 AASIST
- **입력**: raw audio waveform (64,600 samples ≈ 4초 @16kHz)
- **출력 형태**: `(embedding, logits)` 튜플 반환
- **BCE용 fake logit**: `logits[:,1] - logits[:,0]`

```python
_, logits = aasist_model(audio)
fake_logit = logits[:, 1] - logits[:, 0]  # BCE에 이걸 넣음
```

### 5.3 HSEmotion (Emotion Flow — Visual)
- **파일**: `emotion_deepfake_detector_lite.py`
- **클래스**: `EmotionFlowDetectorLite`
- **입력**: 16 frames, 224×224 RGB
- **처리**:
  1. HSEmotion backbone (EfficientNet-B0)로 각 frame → 8-class emotion softmax
  2. `(B, T=16, C=8)` 시퀀스를 GRU에 입력
  3. Attention pooling으로 시퀀스 요약
  4. FC → 2-class logit
- **v2 특징**: Backbone 마지막 2 블록 unfreeze (`3,155,740` trainable params)

### 5.4 CRNN (Emotion Flow — Audio)
- **파일**: `audio_emotion_deepfake_detector.py`
- **클래스**: `AudioEmotionFlowDetector`
- **입력**: 16 audio segments × 3.0초 @16kHz
- **처리**:
  1. RAVDESS 사전학습 CRNN backbone으로 각 segment → 8-class emotion embedding
  2. `(B, T=16, C=8)` 시퀀스를 GRU에 입력
  3. Attention pooling
  4. FC → 2-class logit
- **v2 특징**: Audio backbone GRU + classifier unfreeze (`255,304` trainable params)
- **사전학습 가중치**: `audio_emotion_crnn_best.pth` (RAVDESS)
- **segment 함수**: `extract_audio_segments(path, num_segments=16, target_sr=16000, segment_duration=3.0)`

---

## 6. 학습 설정 (공통 하이퍼파라미터)

### 모든 데이터셋 공통

| 항목 | 값 |
|---|---|
| **데이터 구성** | Real 2000 + Fake 8000 (1:4 ratio) |
| **분할** | 랜덤 90/10 (train/val), seed=42 |
| **Epoch** | 20 |
| **Batch size** | 8 (train), 1 (eval) |
| **Num workers** | 2 (train), 0 (eval) |
| **Optimizer** | AdamW |
| **Learning rate** | 5e-4 (기본), 1e-4 (CRNN) |
| **Weight decay** | 1e-4 |
| **Loss** | `BCEWithLogitsLoss(pos_weight=0.25)` |
| **Early stop** | patience=5 (val AUC 기준) |
| **Best 저장 기준** | validation AUC |

### `pos_weight=0.25` 계산

Real:Fake = 2000:8000 = 1:4 → `pos_weight = num_neg / num_pos = 2000/8000 = 0.25`
- BCE는 양성(fake) 샘플이 많으면 gradient가 편향됨
- pos_weight로 양성 gradient 스케일링 → 클래스 균형 보정

### 모델별 특이사항

| 모델 | 특이사항 |
|---|---|
| X3D | Kinetics pretrained, 전체 fine-tune |
| AASIST | ASVspoof pretrained, 전체 fine-tune, `_, logits = model(audio)` 튜플 반환 |
| HSEmotion | Backbone 마지막 2 블록 unfreeze, GRU+Attention은 처음부터 학습 |
| CRNN | RAVDESS pretrained backbone, GRU+classifier만 unfreeze, lr=1e-4 (더 낮음) |

### Checkpoint 구조

모든 학습 코드는 다음 형식으로 저장:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'cfg': config_dict,
    'val_auc': best_val_auc,
    'val_f1': best_val_f1,
    'epoch': best_epoch,
}
torch.save(checkpoint, save_path)
```

### PyTorch 2.5 호환 처리 (모든 스크립트 상단)

```python
import functools
import torch
torch.load = functools.partial(torch.load, weights_only=False)

# torchvision.transforms.functional_tensor deprecation
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as functional
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = functional
```

---

## 7. 진행 상황: 9-시나리오 매트릭스

### Score_final F1 (%) 매트릭스

| 학습\평가 | FAV | PGF | AVDF1M |
|---|---|---|---|
| **FAV** | 87.85 ✅ | 91.10 ✅ | 75.05 ✅ |
| **PGF** | 77.27 ✅ | 94.12 ✅ | 72.73 ✅ |
| **AVDF1M** | ⏳ Phase 3 | ⏳ Phase 3 | ⏳ Phase 3 |

### AVDF1M 학습 진행

| 모델 | 상태 | Best AUC | 비고 |
|---|---|---|---|
| **AASIST** | ✅ 완료 | 99.87% | Epoch 1 달성, epoch 6 collapse → early stop |
| **CRNN** | ⚠️ 실패 | 50.72% | 2회 재시도 모두 random 수준 → 실험적 발견 (§12 참고) |
| **HSEmotion** | ✅ 완료 | 확인 필요 | (트랜스크립트에서 결과 확인) |
| **X3D** | ⏳ **미실행** | - | 새 PC에서 실행 예정 (~2시간) |

### Phase 3 평가 미완료

- **A→F**: AVDF1M 학습 모델로 FAV 평가 (스크립트 미작성)
- **A→P**: AVDF1M 학습 모델로 PGF 평가 (스크립트 미작성)
- **A→A**: AVDF1M in-domain (스크립트 미작성)

---

## 8. 가중치 파일 인벤토리

### FAV 학습 (`~/hsh/AIApplication/`)

| 파일 | 모델 | 크기 | 생성일 | 버전 |
|---|---|---|---|---|
| `x3d_model_best_final.pth` | X3D | ~15MB | - | 최신 |
| `aasist_model_best_final.pth` | AASIST | ~1.3MB | - | 최신 |
| `emotion_flow_lite_best.pth` | HSEmotion | 17.2MB | 4/24 14:53 | **v1** (v2 없음) |
| `audio_flow_deepfake_best.pth` | CRNN | 2.1MB | 4/24 16:25 | **v1** (v2 없음) |

### PGF 학습 (`~/hsh/AIApplication/reverse_zero_shot/`)

| 파일 | 모델 | 버전 |
|---|---|---|
| `x3d_model_pgf_best.pth` | X3D | v1 |
| `aasist_model_pgf_best.pth` | AASIST | v1 |
| `emotion_flow_lite_pgf_v2_best.pth` | HSEmotion | **v2** ✅ |
| `audio_flow_deepfake_pgf_v2_best.pth` | CRNN | **v2** ✅ |

### AVDF1M 학습 (`~/hsh/AIApplication/reverse_zero_shot/`)

| 파일 | 모델 | AUC | 상태 |
|---|---|---|---|
| `aasist_model_avdf1m_best.pth` | AASIST | 99.87% | ✅ 성공 |
| `audio_flow_deepfake_avdf1m_v2_best.pth` | CRNN | 50.72% | ❌ 실패 |
| `emotion_flow_lite_avdf1m_v2_best.pth` | HSEmotion | ? | ✅ 완료 (결과 확인) |
| `x3d_model_avdf1m_best.pth` | X3D | - | ⏳ 미학습 |

### 보조 가중치

| 파일 | 용도 |
|---|---|
| `audio_emotion_crnn_best.pth` | RAVDESS 사전학습 (CRNN 초기화용) |
| `~/.hsemotion/enet_b0_8_best_afew.pt` | HSEmotion backbone 자동 다운로드 |

---

## 9. 학습 실행 매뉴얼

### 9.1 X3D AVDF1M 학습 (남은 유일한 학습)

```bash
cd ~/hsh/AIApplication
python -u train_x3d_avdf1m.py 2>&1 | tee x3d_avdf1m_log.txt
```

**설정 확인 사항** (스크립트 상단):
- Real 2000 + Fake 8000
- Epoch 20, batch 8
- AdamW lr=5e-4, pos_weight=0.25
- 저장 경로: `reverse_zero_shot/x3d_model_avdf1m_best.pth`

**예상 소요**: ~2시간 (RTX 3060)
**VRAM 사용**: 약 6-8GB

### 9.2 다른 학습 스크립트 (참고용, 이미 완료됨)

```bash
# FAV 학습 (v1 재현이 필요할 때)
python -u train_x3d.py
python -u train_aasist.py
python -u train_HSEmotion.py
python -u train_CRNN.py

# PGF 학습
python -u train_x3d_pgf.py
python -u train_aasist_pgf.py
python -u train_HSEmotion_pgf.py
python -u train_CRNN_pgf.py

# AVDF1M 학습
python -u train_aasist_avdf1m.py     # 완료
python -u train_CRNN_avdf1m.py       # 실패 (재시도 무의미)
python -u train_HSEmotion_avdf1m.py  # 완료
python -u train_x3d_avdf1m.py        # ⏳ 실행 예정
```

### 9.3 학습 중 모니터링

**터미널 1 (학습)**:
```bash
python -u train_x3d_avdf1m.py 2>&1 | tee x3d_avdf1m_log.txt
```
`-u` 옵션 필수 (unbuffered, tee와 함께 쓰면 출력 지연 발생하므로).

**터미널 2 (GPU/CPU 모니터)**:
```bash
watch -n 2 nvidia-smi
```

**터미널 3 (진행률 확인)**:
```bash
tail -f x3d_avdf1m_log.txt
```

### 9.4 학습 완료 후 확인

```bash
# 가중치 파일 생성 확인
ls -la reverse_zero_shot/x3d_model_avdf1m_best.pth

# 로그에서 best epoch, best AUC 확인
grep -E "best_auc|Best" x3d_avdf1m_log.txt | tail -20
```

---

## 10. 평가 실행 매뉴얼

### 10.1 완료된 평가 스크립트

| 스크립트 | 시나리오 | 상태 |
|---|---|---|
| `evaluate_pgf_to_avdf1m.py` | P→A | ✅ (F1 72.73%) |
| `evaluate_fav_breakdown.py` | F→F, F→P (method별) | ✅ |
| (다른 시나리오 스크립트) | F→F, F→P, F→A, P→F, P→P | ✅ (완료) |

### 10.2 Phase 3 - 필요한 평가 스크립트 (작성 필요)

각 스크립트는 `evaluate_pgf_to_avdf1m.py`를 템플릿으로 사용해 작성:

**A→F 평가**:
```python
# WEIGHTS를 AVDF1M 학습 가중치로 변경
WEIGHTS = {
    'x3d':    'reverse_zero_shot/x3d_model_avdf1m_best.pth',
    'aasist': 'reverse_zero_shot/aasist_model_avdf1m_best.pth',
    'hsemo':  'reverse_zero_shot/emotion_flow_lite_avdf1m_v2_best.pth',
    'crnn':   'reverse_zero_shot/audio_flow_deepfake_avdf1m_v2_best.pth',
}
# 평가 데이터: FAV Real 500 + Fake 500
# 나머지 로직은 F→F 평가 코드 참조
```

**A→P 평가**:
- WEIGHTS: 위와 동일 (AVDF1M 가중치)
- 평가 데이터: PGF Real 500 + Fake 500

**A→A 평가**:
- WEIGHTS: 위와 동일
- 평가 데이터: AVDF1M validation set 1000샘플 (같은 seed=42)

### 10.3 공통 평가 설정

- **Batch size**: 1 (num_workers=0)
- **비디오 프레임**: 16 frames
- **오디오 segments**: 16 × 3.0초
- **평가 metric**: F1, Recall, AUC, Accuracy (Score_final 기준)

### 10.4 평가 시 CRNN 처리 방안

CRNN AVDF1M 학습이 실패했으므로 A→F/A→P/A→A 평가에서 CRNN을 어떻게 다룰지 결정 필요:

**옵션 A**: 실패 가중치 그대로 사용 (예상 F1 50%)
- 장점: 매트릭스 일관성
- 단점: 결과 왜곡

**옵션 B**: PGF 가중치를 fixed로 사용
- 장점: emotion 신호 유지
- 단점: 매트릭스 정의 모호

**옵션 C**: 3-way fusion (X3D + AASIST + HSEmotion)
- 장점: 실패 모델 제외
- 단점: 매트릭스가 다른 시나리오와 다름

**권장**: 옵션 B (PGF CRNN을 auxiliary로 사용) + Discussion에서 이유 명시
→ 이는 논문 Discussion §5.3 Limitations의 새로운 발견으로 서술 가능

---

## 11. 알려진 이슈 및 해결

### 11.1 PyTorch 2.5 호환 이슈

**증상**: `weights_only=True` 기본값으로 인해 checkpoint 로드 실패
**해결**: 모든 스크립트 상단에 아래 삽입
```python
import functools
import torch
torch.load = functools.partial(torch.load, weights_only=False)
```

### 11.2 `torchvision.transforms.functional_tensor` deprecation

**증상**: 최신 torchvision에서 해당 모듈 삭제됨
**해결**:
```python
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as functional
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = functional
```

### 11.3 `polyglotfake_data` ModuleNotFoundError

**증상**: `polyglotfake_data.py`가 `reverse_zero_shot/` 폴더에 있어 import 실패
**해결**: 스크립트 상단에서 자동 경로 추가
```python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for sub in ['reverse_zero_shot', '.', '..']:
    p = os.path.abspath(os.path.join(CURRENT_DIR, sub))
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
```

### 11.4 `python | tee` 출력 지연

**증상**: `python script.py | tee log.txt`로 실행 시 출력이 화면에 안 나옴
**원인**: stdout block buffering
**해결**: `python -u` 옵션 사용
```bash
python -u script.py 2>&1 | tee log.txt
# 또는
PYTHONUNBUFFERED=1 python script.py 2>&1 | tee log.txt
```

### 11.5 FAV 경로 매칭 (매우 주의)

**증상**: 메타데이터의 path와 실제 파일 위치가 다름
**정확한 매칭**:
```python
# 메타 컬럼:
#   'Unnamed: 9' = "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076"
#   'path'       = "00109.mp4"

base_dir = "<PROJECT_ROOT>/FakeAVCeleb_v1.2"
base_parent = os.path.dirname(base_dir)  # "<PROJECT_ROOT>"
base_name = os.path.basename(base_dir)   # "FakeAVCeleb_v1.2"

rel_dir = row['Unnamed: 9'].replace('FakeAVCeleb', base_name)
video_path = os.path.join(base_parent, rel_dir, row['path'])
# → <PROJECT_ROOT>/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4
```

### 11.6 AVDF1M seq_id 5자리 zero-pad

**증상**: 메타에서 seq_id가 정수인데 실제 폴더명은 5자리 zero-pad
**해결**: `f"{seq:05d}"` 필수

### 11.7 AASIST 모델 출력 형태

**주의**: AASIST는 `(embedding, logits)` 튜플 반환
```python
_, logits = aasist_model(audio)  # unpack 필수
fake_logit = logits[:, 1] - logits[:, 0]  # BCE에 사용
```

### 11.8 CRNN AVDF1M 학습 실패 (실험적 발견)

**증상**: 2회 시도 (PGF pretrain 제거, lr 1e-4 등) 모두 AUC ~50% (random)
**원인**: AVDF1M의 word-level micro-manipulation은 emotion 기반 CRNN이 포착하기 어려움
**의미**: 논문의 새로운 발견 → §12 Discussion에 서술

---

## 12. 논문 작성 방향

### 12.1 논문 정보

- **제목**: Hierarchical Score-Level Fusion of Artifact and Emotion Flow Signals for Cross-Dataset Deepfake Detection
- **저자**: 한성훈, 김영환, 홍건우, 김주영, 김상균 (명지대)
- **형식**: ICIP 2022 template, 4 pages
- **최신 버전**: v7 (emotion-focus + backbone justification)
- **참고 트랜스크립트**: `/mnt/transcripts/2026-05-27-12-28-58-deepfake-paper-icip-v7-emotion-focus.txt`

### 12.2 핵심 결과 (논문 리포트값)

| 시나리오 | F1 | AUC |
|---|---|---|
| FAV in-domain | 87.85% | 99.81% |
| PGF zero-shot (baseline artifact-only) | 7.97% | - |
| **PGF zero-shot (Ours full)** | **91.10%** | 71.92% |
| **개선폭** | **+83.13%p** | - |

### 12.3 Discussion 5가지 논점 (v7)

1. **Emotion flow의 일반화 이유**: 시간적 일관성은 합성 기법 무관
2. **Emotion backbone 선택 정당화**: HSEmotion/CRNN의 8-class structured output이 필수 (opaque embedding backbone은 부적합)
3. **Hierarchical fusion의 의의**: Flat OR과 수학적 동치지만 decision attribution 가능 (PGF fake의 87%에서 emotion override)
4. **CRNN의 의도치 않은 transfer**: 영어 학습이지만 영어에서 가장 어려움 (언어 독립적 prosodic 학습)
5. **Limitations**: FP 증가, Vall-E 어려움, 데이터셋 2개만 평가

### 12.4 Future Work (원 논문)

- Multilingual emotion fine-tuning
- Adversarial robustness evaluation

### 12.5 새 발견을 반영한 확장 Future Work

**A. 데이터셋 확장**:
- ✅ AV-Deepfake1M 통합 (진행 중, 이 인수인계 이후 완료 예정)
- 9-시나리오 매트릭스 완성

**B. 구조 개선** (AVDF1M CRNN 실패 반영):
- CRNN의 한계 확인: word-level micro-fake에 emotion 기반 audio 부적합
- 대안: audio branch에 NLP/transcript 신호 추가 (ASR + 텍스트 일관성)

**C. 평가 강화**:
- Adversarial robustness (원 논문 유지)
- Calibration (ECE/Brier)

### 12.6 v8/v9 확장 방향

AVDF1M 결과가 매트릭스에 추가되면:
- **Table 1 확장**: 2×2 → 3×3 matrix
- **Discussion 추가**: AVDF1M의 word-level fake가 emotion 신호를 무력화하는 새로운 발견 → 논문의 "emotion flow" 가설의 boundary condition 제시
- **Limitations 갱신**: "Emotion flow는 clip-level 합성에 유효하지만 word-level micro-fake에 대해서는 audio-side transcript 신호가 필요"

---

## 13. Reviewer Response 진행 상황

### 13.1 받은 리뷰 (요약)

**Minor**:
1. Artifact 정의 부족 → Introduction 보완 필요
2. Method overview 폰트 작음 → Figure 2 재작업

**Major**:
3. Emotion flow ablation (mean/GRU/attention) 필요
4. Recall 우선 정당화 근거 부족
5. AVDF1M 1000샘플 이유 + repeated sampling
6. 성공/실패 사례 정성 시각화

### 13.2 답변 문서

- **파일**: `Response_to_Reviewers.md` (완료, 미래형으로 작성됨)
- **상태**: 답변 초안 완성, 실제 실험/수정은 미진행
- **핵심 영어 본문**: Comment 1의 artifact 정의 단락만 영어로 작성됨

### 13.3 답변 완성을 위한 남은 작업

| 우선순위 | 작업 | 예상 소요 |
|---|---|---|
| ⭐⭐⭐ | Comment 3: Ablation 4-variant 학습 (mean/MLP/GRU/GRU+Attn) | 2-3일 |
| ⭐⭐⭐ | Comment 5: 5-seed repeated sampling | 5-6시간 |
| ⭐⭐⭐ | Comment 6: Emotion trajectory + attention 시각화 | 1-2일 |
| ⭐⭐ | Comment 6: Temporal consistency metric (TV/std/entropy) | 반나절 |
| ⭐⭐ | Comment 4: 인용 4편 확인 (Khalid, Verdoliva, Chesney, Gorwa) | 1시간 |
| ⭐ | Comment 1: Introduction 문단 삽입 | 30분 |
| ⭐ | Comment 2: Figure 2 폰트 재작업 | 1시간 |

---

## 14. 남은 작업 To-Do

### Phase 3: AVDF1M 매트릭스 완성

- [ ] **X3D AVDF1M 학습** (2시간): `python -u train_x3d_avdf1m.py`
- [ ] HSEmotion 학습 결과 재확인 (best AUC/F1 로그 확인)
- [ ] **A→A 평가 스크립트 작성** + 실행 (~30분)
- [ ] **A→F 평가 스크립트 작성** + 실행 (~30분)
- [ ] **A→P 평가 스크립트 작성** + 실행 (~30분)
- [ ] CRNN 처리 방안 결정 (옵션 A/B/C 중)
- [ ] 9-시나리오 매트릭스 최종 정리

### Reviewer Response 대응

- [ ] Ablation 4-variant 학습 + 평가
- [ ] 5-seed repeated sampling (기존 스크립트에 seed 루프 추가)
- [ ] 성공/실패 케이스 선정 + trajectory plot 코드 작성
- [ ] Attention heatmap 시각화
- [ ] Temporal consistency metric 계산
- [ ] Response 문서에 실측 수치 반영
- [ ] 논문 v8 작성 (수정 반영)

### 논문 확장

- [ ] v7 → v8 (AVDF1M 결과 포함)
- [ ] Table 1: 3-데이터셋 매트릭스로 확장
- [ ] Discussion: AVDF1M CRNN 실패 관련 새 발견 추가
- [ ] Future Work: 확장된 4가지 방향으로 갱신

---

## 15. 파일 이관 체크리스트

### 이관 필수 파일

**A. 코드 (~/hsh/AIApplication/)**
- [ ] `train_x3d.py`, `train_x3d_pgf.py`, `train_x3d_avdf1m.py`
- [ ] `train_aasist.py`, `train_aasist_pgf.py`, `train_aasist_avdf1m.py`
- [ ] `train_HSEmotion.py`, `train_HSEmotion_pgf.py`, `train_HSEmotion_avdf1m.py`
- [ ] `train_CRNN.py`, `train_CRNN_pgf.py`, `train_CRNN_avdf1m.py`
- [ ] `evaluate_*.py` (모든 평가 스크립트)
- [ ] `avdf1m_train_data.py`
- [ ] `emotion_deepfake_detector_lite.py`
- [ ] `audio_emotion_deepfake_detector.py`
- [ ] `reverse_zero_shot/polyglotfake_data.py`
- [ ] `aasist/` 폴더 전체 (AASIST 모델 코드)

**B. 가중치 (모두 이관 필수)**
- [ ] `x3d_model_best_final.pth` (FAV)
- [ ] `aasist_model_best_final.pth` (FAV)
- [ ] `emotion_flow_lite_best.pth` (FAV v1)
- [ ] `audio_flow_deepfake_best.pth` (FAV v1)
- [ ] `audio_emotion_crnn_best.pth` (RAVDESS 사전학습)
- [ ] `reverse_zero_shot/*.pth` (PGF/AVDF1M 가중치 전부)

**C. 데이터셋 (매우 큼, 별도 저장매체 권장)**
- [ ] `FakeAVCeleb_v1.2/` (~수십 GB)
- [ ] `PolyGlotFake_dataset/`
- [ ] `AV-Deepfake1M_RootFiles/` (~수백 GB, extracted_train + extracted_val)

**D. 문서**
- [ ] 이 문서 (`HANDOFF_인수인계.md`)
- [ ] `Response_to_Reviewers.md`
- [ ] `실험환경_세팅.md` (이전 세션에서 작성됨, 있다면)
- [ ] 논문 LaTeX 파일 (v7)
- [ ] `/mnt/transcripts/` 세션 로그들 (참고용)

**E. 로그 파일 (참고용)**
- [ ] `*_log.txt` 학습 로그 파일들

### 이관 방법 제안

**옵션 1: 외장 SSD로 물리 복사** (권장, 데이터셋 큼)
```bash
# 원본 PC에서
rsync -av --progress ~/hsh/AIApplication/ /mnt/external_ssd/AIApplication/

# 새 PC에서
rsync -av --progress /mnt/external_ssd/AIApplication/ ~/hsh/AIApplication/
```

**옵션 2: 코드+가중치만 GitHub, 데이터셋만 외장**
- `.pth` 파일은 Git LFS 또는 별도 저장
- 데이터셋은 외장 SSD

**옵션 3: 네트워크 전송 (대규모 데이터는 비추)**
```bash
# 원본 PC
tar czf - AIApplication | ssh new_pc "tar xzf - -C /home/user/hsh/"
```

### 이관 후 검증

1. **환경 재구축**:
```bash
conda env create -f deepfake_env.yml
conda activate deepfake
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

2. **경로 재설정 확인**:
- 새 PC의 사용자명이 `uxm2`가 아니라면 코드 내 하드코딩 경로 확인
- `~/hsh/AIApplication/` 구조 그대로 유지 권장

3. **테스트 실행** (10분):
```bash
# 이미 완료된 평가 스크립트를 다시 돌려 재현성 확인
cd ~/hsh/AIApplication
python -u evaluate_pgf_to_avdf1m.py  # F1 72.73% 재현되면 이관 성공
```

4. **HSEmotion 자동 다운로드 확인**:
- 첫 실행 시 `~/.hsemotion/enet_b0_8_best_afew.pt` 자동 다운로드
- 인터넷 연결 필요

---

## 16. 참고 자료 (트랜스크립트)

새 PC로 이관 시 아래 트랜스크립트도 함께 복사하면 세부 진행 이력 참조 가능:

- `/mnt/transcripts/2026-05-27-07-01-40-deepfake-paper-v4-rewrite.txt` — 논문 v4 초기 작성
- `/mnt/transcripts/2026-05-27-11-56-34-deepfake-paper-icip-revision.txt` — ICIP 형식 변환
- `/mnt/transcripts/2026-05-27-12-20-40-deepfake-paper-icip-v6-avdf1m-synthesis.txt` — v6
- `/mnt/transcripts/2026-05-27-12-28-58-deepfake-paper-icip-v7-emotion-focus.txt` — **v7 최신 (핵심 참조)**
- `/mnt/transcripts/2026-06-10-00-57-06-9scenario-deepfake-avdf1m-training.txt` — AVDF1M 학습 세션

---

## 17. 빠른 시작 (새 PC에서 첫날)

```bash
# 1. 환경 활성화
conda activate deepfake

# 2. 작업 폴더 이동
cd ~/hsh/AIApplication

# 3. GPU/환경 확인
nvidia-smi
python -c "import torch, torchvision, timm, hsemotion; print('OK')"

# 4. 재현성 테스트 (10분)
python -u evaluate_pgf_to_avdf1m.py  # F1 72.73% 나오는지 확인

# 5. X3D AVDF1M 학습 시작 (2시간)
python -u train_x3d_avdf1m.py 2>&1 | tee x3d_avdf1m_log.txt

# 6. 학습 중 별도 터미널에서:
watch -n 2 nvidia-smi
tail -f x3d_avdf1m_log.txt

# 7. 학습 완료 후 Phase 3 평가 스크립트 작성 시작
```

---

## 문의 및 참고

- **원 작성 PC 사용자**: uxm2@uxm2-desktop
- **작업 시작 시점**: 2026년 (초기 v1 학습: 4월 24일)
- **논문 제출 목표**: ICIP 2027 (또는 국내 학회)
- **핵심 이슈 로그**: 트랜스크립트 저널 (`/mnt/transcripts/journal.txt`)

**이 문서에 없는 세부 사항이 필요하면**:
1. 트랜스크립트 참조 (특히 v7 논문 세션)
2. `reverse_zero_shot/polyglotfake_data.py` 내부 주석 확인
3. 각 학습 스크립트 상단 config 확인

