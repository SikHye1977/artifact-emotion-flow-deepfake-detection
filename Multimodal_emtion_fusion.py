"""
==============================================================================
[멀티모달 앙상블] 감정 흐름 기반 딥페이크 탐지 (Late Fusion)
Multimodal Deepfake Detection via Probabilistic OR
==============================================================================

[핵심 아이디어]
두 개의 독립적으로 학습된 모델을 확률적 OR로 결합:

  P(fake) = 1 - (1 - P_video) × (1 - P_audio)

즉, 두 모델 중 하나라도 "가짜"라고 판단하면 최종 확률이 높아짐.

[왜 확률적 OR인가?]
FakeAVCeleb의 딥페이크 유형:
  - FaceSwap + real audio     : 영상만 가짜 (영상 모델이 포착)
  - real face + wav2lip audio : 오디오만 가짜 (오디오 모델이 포착)
  - FaceSwap + wav2lip        : 둘 다 가짜 (양쪽 모두 포착)

단순 평균(Late Fusion)은 "둘 다 조금씩 이상함"을 잡는 반면,
확률적 OR은 "한쪽만 명백히 이상해도" 가짜로 판단 가능.
→ 멀티모달 딥페이크 대응에 더 적합.

[사용 모델]
  1. EmotionFlowDetectorLite (영상) — emotion_flow_lite_best.pth
  2. AudioEmotionFlowDetector (오디오) — audio_flow_deepfake_best.pth

  두 모델 모두 완전 동결, 추가 학습 없음. 순수 추론만 수행.

[평가 프로세스]
  동일한 FakeAVCeleb Val 셋에 대해:
    - 영상 모델 확률 추출
    - 오디오 모델 확률 추출
    - 확률적 OR 결합 → 최종 확률
    - AUC, Accuracy, Confusion Matrix 계산
    - 각 모델 vs 앙상블 비교 리포트
"""

import os
import sys
import functools
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report
)

# PyTorch 2.6+ 보안 정책 우회
torch.load = functools.partial(torch.load, weights_only=False)

# ── 기존 모델 파일에서 클래스 임포트 ────────────────────────────────────────
# 두 모델 파일이 같은 디렉토리에 있다고 가정
from train_HSEmotion import EmotionFlowDetectorLite, extract_uniform_frames
from train_CRNN import AudioEmotionFlowDetector, extract_audio_segments

# ══════════════════════════════════════════════════════════════════════════════
# 1. 멀티모달 Dataset — 영상 프레임 + 오디오 세그먼트 동시 반환
# ══════════════════════════════════════════════════════════════════════════════
class MultimodalFakeAVCelebDataset(Dataset):
    """
    하나의 영상에서 영상 프레임과 오디오 세그먼트를 동시에 추출.
    실패 시 다음 샘플로 fallback.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str,
        num_frames: int = 16,
        num_segments: int = 16,
        segment_duration: float = 3.0,
        target_sr: int = 16000
    ):
        self.df               = df.reset_index(drop=True)
        self.base_dir         = base_dir
        self.num_frames       = num_frames
        self.num_segments     = num_segments
        self.segment_duration = segment_duration
        self.target_sr        = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        rel_path   = row.iloc[-2].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])

        # 영상 프레임
        frames = extract_uniform_frames(video_path, self.num_frames)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))

        # 오디오 세그먼트
        segments = extract_audio_segments(
            video_path,
            num_segments     = self.num_segments,
            target_sr        = self.target_sr,
            segment_duration = self.segment_duration
        )
        if segments is None:
            return self.__getitem__((idx + 1) % len(self))

        label = 1.0 if row['method'] != 'real' else 0.0
        return frames, segments, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 체크포인트 로드 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_video_model(ckpt_path: str, device: torch.device):
    """학습된 영상 모델(EmotionFlowDetectorLite) 로드."""
    print(f"📹 영상 모델 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('cfg', {})
    state_dict = ckpt.get('model_state_dict', ckpt)

    # 체크포인트 cfg가 없거나 불완전한 경우,
    # state_dict에서 실제 구조를 역추적해서 정확한 값 사용
    if 'gru.weight_hh_l0' in state_dict:
        # weight_hh_l0 shape: (3*hidden, hidden) — GRU는 3배수
        gru_hidden_from_ckpt = state_dict['gru.weight_hh_l0'].shape[1]
        print(f"   └ 체크포인트에서 GRU hidden 감지: {gru_hidden_from_ckpt}")
    else:
        gru_hidden_from_ckpt = None

    model = EmotionFlowDetectorLite(
        model_name  = cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames  = cfg.get('NUM_FRAMES', 16),
        gru_hidden  = gru_hidden_from_ckpt or cfg.get('GRU_HIDDEN', 64),
        dropout     = cfg.get('DROPOUT', 0.3),
        device      = 'cpu'
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 완전 동결 확인
    for p in model.parameters():
        p.requires_grad = False

    return model, cfg


def load_audio_model(ckpt_path: str, pretrained_path: str, device: torch.device):
    """학습된 오디오 모델(AudioEmotionFlowDetector) 로드."""
    print(f"🎵 오디오 모델 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('cfg', {})
    state_dict = ckpt.get('model_state_dict', ckpt)

    # 체크포인트에서 GRU hidden 역추적
    if 'gru.weight_hh_l0' in state_dict:
        gru_hidden_from_ckpt = state_dict['gru.weight_hh_l0'].shape[1]
        print(f"   └ 체크포인트에서 GRU hidden 감지: {gru_hidden_from_ckpt}")
    else:
        gru_hidden_from_ckpt = None

    model = AudioEmotionFlowDetector(
        pretrained_path = pretrained_path,
        num_segments    = cfg.get('NUM_SEGMENTS', 16),
        gru_hidden      = gru_hidden_from_ckpt or cfg.get('GRU_HIDDEN', 128),
        dropout         = cfg.get('DROPOUT', 0.4)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model, cfg


# ══════════════════════════════════════════════════════════════════════════════
# 3. 확률적 OR Fusion
# ══════════════════════════════════════════════════════════════════════════════
def probabilistic_or(p_video: np.ndarray, p_audio: np.ndarray) -> np.ndarray:
    """
    P(fake) = 1 - (1 - P_video) × (1 - P_audio)

    - 둘 다 0 → 0         (둘 다 진짜로 판단)
    - 한쪽 1 → 1          (한쪽이라도 가짜면 가짜)
    - 0.5, 0.5 → 0.75     (양쪽 애매해도 합쳐지면 가짜 쪽으로 기움)
    - 0.8, 0.2 → 0.84     (강한 신호가 약한 신호에 덜 희석됨)
    """
    return 1.0 - (1.0 - p_video) * (1.0 - p_audio)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 추론 루프
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_multimodal(
    video_model,
    audio_model,
    loader: DataLoader,
    device: torch.device
):
    """
    전체 Val 셋에 대해 영상/오디오/앙상블 확률을 모두 계산하고 지표 반환.
    """
    video_model.eval()
    audio_model.eval()

    all_p_video = []
    all_p_audio = []
    all_labels  = []

    total = len(loader)
    print(f"\n🔎 추론 시작 (총 {total} 배치)")

    for i, (frames, segments, labels) in enumerate(loader):
        frames   = frames.to(device)
        segments = segments.to(device)
        labels   = labels.numpy()

        # 영상 모델
        v_logit, _ = video_model(frames)
        p_video    = torch.sigmoid(v_logit).cpu().numpy()

        # 오디오 모델
        a_logit, _ = audio_model(segments)
        p_audio    = torch.sigmoid(a_logit).cpu().numpy()

        all_p_video.extend(p_video.tolist())
        all_p_audio.extend(p_audio.tolist())
        all_labels.extend(labels.tolist())

        if i % 10 == 0:
            print(f"  [{i:3d}/{total}] 처리 중...")

    return (np.array(all_p_video),
            np.array(all_p_audio),
            np.array(all_labels))


# ══════════════════════════════════════════════════════════════════════════════
# 5. 지표 계산 & 리포트
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    preds = (probs > threshold).astype(int)
    labels_int = labels.astype(int)

    try:
        auc = roc_auc_score(labels, probs) * 100
    except Exception:
        auc = 0.0

    acc = accuracy_score(labels_int, preds) * 100
    cm  = confusion_matrix(labels_int, preds, labels=[0, 1])

    # TN FP
    # FN TP
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return dict(
        auc=auc, acc=acc, precision=precision, recall=recall, f1=f1,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )


def print_report(name: str, m: dict):
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  📊 {name}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  AUC       : {m['auc']:.2f}%")
    print(f"  Accuracy  : {m['acc']:.2f}%")
    print(f"  Precision : {m['precision']:.2f}%  (Fake 예측의 정확도)")
    print(f"  Recall    : {m['recall']:.2f}%  (실제 Fake 검출률)")
    print(f"  F1        : {m['f1']:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"              예측: Real  예측: Fake")
    print(f"    실제 Real    {m['tn']:5d}      {m['fp']:5d}")
    print(f"    실제 Fake    {m['fn']:5d}      {m['tp']:5d}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        BASE_DIR        = "FakeAVCeleb_v1.2",
        CSV_PATH        = "FakeAVCeleb_v1.2/meta_data.csv",

        # 학습 완료된 체크포인트
        VIDEO_CKPT      = "emotion_flow_lite_best.pth",
        AUDIO_CKPT      = "audio_flow_deepfake_best.pth",
        AUDIO_PRETRAINED= "audio_emotion_crnn_best.pth",

        # 1:1 균형 대규모 평가 (Real 500 + Fake 500 = 1,000개)
        EVAL_SIZE_PER_CLASS = 500,  # Real/Fake 각 500개
        BATCH_SIZE          = 4,    # 영상 + 오디오 동시 처리라 메모리 줄임
        NUM_WORKERS         = 4,

        # 학습 시 사용했던 Fake 개수 (이를 제외하고 평가 Fake 추출)
        TRAIN_FAKE_N        = 2000,
        TRAIN_SEED          = 42,

        # 오디오 파라미터
        NUM_SEGMENTS    = 16,
        SEGMENT_DURATION= 3.0,
        TARGET_SR       = 16000,

        # 리포트 저장 경로
        REPORT_DIR      = "multimodal_report",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(CFG['REPORT_DIR'], exist_ok=True)

    # 체크포인트 확인
    for path in [CFG['VIDEO_CKPT'], CFG['AUDIO_CKPT'], CFG['AUDIO_PRETRAINED']]:
        if not os.path.exists(path):
            print(f"❌ 필수 파일 없음: {path}")
            sys.exit(1)

    # ── 데이터 준비: 1:1 균형 대규모 평가 ──────────────────────────────
    df = pd.read_csv(CFG['CSV_PATH'])
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)

    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)

    print(f"📊 원본 데이터: Real={len(real_df)}, Fake={len(fake_df)}")

    # Real: 전량(500) 사용 — FakeAVCeleb 특성상 Real이 적어 학습에도 전량 썼음
    #                        → Real 성능은 약간 과대평가 가능성 있음
    n_per_class  = min(CFG['EVAL_SIZE_PER_CLASS'], len(real_df))
    sampled_real = real_df.sample(n=n_per_class, random_state=CFG['TRAIN_SEED'])

    # Fake: 학습에 썼던 2,000개를 먼저 제외하고, 남은 것 중에서 500개 추출
    #       → 평가 Fake는 100% 학습 미사용 → 진짜 일반화 성능 측정
    train_fake_idx = fake_df.sample(
        n=CFG['TRAIN_FAKE_N'], random_state=CFG['TRAIN_SEED']
    ).index
    unseen_fake_df = fake_df.drop(train_fake_idx).reset_index(drop=True)
    print(f"   학습 미사용 Fake: {len(unseen_fake_df)}개")

    sampled_fake = unseen_fake_df.sample(
        n=min(CFG['EVAL_SIZE_PER_CLASS'], len(unseen_fake_df)),
        random_state=CFG['TRAIN_SEED'] + 1
    )

    val_df = pd.concat([sampled_real, sampled_fake]).sample(
        frac=1, random_state=CFG['TRAIN_SEED']
    ).reset_index(drop=True)

    print(f"📂 평가 데이터: 총 {len(val_df)}개 "
          f"(Real {len(sampled_real)} : Fake {len(sampled_fake)})")
    print(f"   ✓ Fake: 100% 학습 미사용 (엄격 평가)")
    print(f"   ⚠ Real: 학습과 겹침 (FakeAVCeleb 특성상 불가피)")

    val_loader = DataLoader(
        MultimodalFakeAVCelebDataset(
            val_df, CFG['BASE_DIR'],
            num_frames       = 16,
            num_segments     = CFG['NUM_SEGMENTS'],
            segment_duration = CFG['SEGMENT_DURATION'],
            target_sr        = CFG['TARGET_SR']
        ),
        batch_size  = CFG['BATCH_SIZE'],
        shuffle     = False,
        num_workers = CFG['NUM_WORKERS'],
        pin_memory  = True
    )

    # ── 모델 로드 (둘 다 동결) ───────────────────────────────────────────
    video_model, v_cfg = load_video_model(CFG['VIDEO_CKPT'], device)
    audio_model, a_cfg = load_audio_model(
        CFG['AUDIO_CKPT'], CFG['AUDIO_PRETRAINED'], device
    )
    print("✅ 두 모델 모두 로드 및 동결 완료")

    # ── 추론 실행 ─────────────────────────────────────────────────────────
    t0 = time.time()
    p_video, p_audio, labels = evaluate_multimodal(
        video_model, audio_model, val_loader, device
    )
    elapsed = time.time() - t0
    print(f"\n⏱  전체 추론 시간: {elapsed:.1f}초")

    # ── 확률적 OR 결합 ────────────────────────────────────────────────────
    p_fusion = probabilistic_or(p_video, p_audio)

    # ── 지표 계산 ──────────────────────────────────────────────────────────
    m_video  = compute_metrics(p_video,  labels)
    m_audio  = compute_metrics(p_audio,  labels)
    m_fusion = compute_metrics(p_fusion, labels)

    # ── 리포트 출력 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎯 멀티모달 앙상블 평가 결과")
    print("=" * 60)

    print_report("영상 모델 (HSEmotion + GRU + Attention)", m_video)
    print_report("오디오 모델 (CRNN + GRU + Attention)",    m_audio)
    print_report("🌟 확률적 OR Fusion",                     m_fusion)

    # ── 개선폭 요약 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈 앙상블 개선폭")
    print("=" * 60)
    print(f"  영상 단독  AUC : {m_video['auc']:.2f}%")
    print(f"  오디오 단독 AUC: {m_audio['auc']:.2f}%")
    print(f"  Fusion     AUC : {m_fusion['auc']:.2f}%  "
          f"(+{m_fusion['auc'] - max(m_video['auc'], m_audio['auc']):+.2f}%p)")
    print(f"\n  영상 단독  Acc : {m_video['acc']:.2f}%")
    print(f"  오디오 단독 Acc: {m_audio['acc']:.2f}%")
    print(f"  Fusion     Acc : {m_fusion['acc']:.2f}%  "
          f"(+{m_fusion['acc'] - max(m_video['acc'], m_audio['acc']):+.2f}%p)")

    # ── 샘플별 상세 CSV 저장 ─────────────────────────────────────────────
    detail_df = pd.DataFrame({
        '실제_레이블(0:Real 1:Fake)': labels.astype(int),
        '영상_Fake확률(%)':  np.round(p_video  * 100, 2),
        '오디오_Fake확률(%)': np.round(p_audio * 100, 2),
        'Fusion_Fake확률(%)': np.round(p_fusion * 100, 2),
        '영상_예측':   (p_video  > 0.5).astype(int),
        '오디오_예측': (p_audio  > 0.5).astype(int),
        'Fusion_예측': (p_fusion > 0.5).astype(int),
    })
    detail_path = os.path.join(CFG['REPORT_DIR'], "sample_predictions.csv")
    detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 샘플별 예측 저장: {detail_path}")

    # ── 지표 요약 CSV ────────────────────────────────────────────────────
    summary_df = pd.DataFrame([
        dict(model='영상 (HSEmotion)',  **m_video),
        dict(model='오디오 (CRNN)',     **m_audio),
        dict(model='Fusion (OR)',       **m_fusion),
    ])
    summary_path = os.path.join(CFG['REPORT_DIR'], "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"💾 지표 요약 저장: {summary_path}")

    # ── 케이스 분석 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 케이스 분석 (앙상블이 기여한 경우)")
    print("=" * 60)

    preds_video  = (p_video  > 0.5).astype(int)
    preds_audio  = (p_audio  > 0.5).astype(int)
    preds_fusion = (p_fusion > 0.5).astype(int)
    labels_int   = labels.astype(int)

    # 단독 모델이 맞춘 경우
    video_correct  = (preds_video  == labels_int)
    audio_correct  = (preds_audio  == labels_int)
    fusion_correct = (preds_fusion == labels_int)

    only_video  = int(( video_correct & ~audio_correct).sum())
    only_audio  = int((~video_correct &  audio_correct).sum())
    both_right  = int(( video_correct &  audio_correct).sum())
    both_wrong  = int((~video_correct & ~audio_correct).sum())
    fusion_save = int((~video_correct & ~audio_correct & fusion_correct).sum())

    print(f"  영상만 맞춘 경우    : {only_video:4d}개")
    print(f"  오디오만 맞춘 경우  : {only_audio:4d}개")
    print(f"  둘 다 맞춘 경우     : {both_right:4d}개")
    print(f"  둘 다 틀린 경우     : {both_wrong:4d}개")
    print(f"  ↳ Fusion이 살려낸 경우: {fusion_save:4d}개")

    print("\n" + "=" * 60)
    print(f"✅ 완료! Fusion Best AUC: {m_fusion['auc']:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()