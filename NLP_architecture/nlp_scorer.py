"""
nlp_scorer.py  v3
─────────────────────────────────────────────────────────────────────
v3 변경사항:
  - Segment-level NLP 도입
  - 모드 A: fake_segments 알 때  → 구간/전체 ppl 비율로 score
  - 모드 B: fake_segments 모를 때 → 슬라이딩 윈도우 이상 탐지
  - PolyGlotFake 언어 불일치 탐지 유지
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re

MODEL_NAME          = "xlm-roberta-base"
LANG_MISMATCH_SCORE = 0.92
EXPECTED_LANG       = "en"
_PUNCT_RE           = re.compile(r"^[\W\d_]+$")


class NLPScorer:

    def __init__(self, device: str = None, model_name: str = MODEL_NAME):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[NLPScorer] 모델 로드 중: {model_name} → {device}")
        self.tokenizer   = AutoTokenizer.from_pretrained(model_name)
        self.model       = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self._special_ids = set(self.tokenizer.all_special_ids)
        print(f"[NLPScorer] 로드 완료")

    # ── Public API ────────────────────────────────────────────────

    def score(self,
              transcript:    str,
              detected_lang: str   = "en",
              word_timestamps: list = None,   # Whisper segments → words
              fake_segments:  list  = None    # [[start, end], ...] 초 단위
              ) -> dict:
        """
        Parameters
        ----------
        transcript       : Whisper 전체 텍스트
        detected_lang    : Whisper result['language']
        word_timestamps  : [{'word': str, 'start': float, 'end': float}, ...]
        fake_segments    : [[s, e], ...]  알면 모드 A, None이면 모드 B

        Returns
        -------
        dict {
            'score_t'        : float [0,1]
            'mode'           : 'lang_mismatch' | 'segment' | 'sliding' | 'global'
            'ppl_fake_region': float | None   (모드 A)
            'ppl_background' : float | None   (모드 A)
            'ppl_median'     : float | None   (모드 B / global)
            'anomaly_ratio'  : float | None
            'lang_mismatch'  : bool
        }
        """
        if not transcript or not transcript.strip():
            return self._empty_result()

        # ── 언어 불일치 (PolyGlotFake) ───────────────────────────
        if detected_lang.lower() != EXPECTED_LANG:
            return {
                'score_t': LANG_MISMATCH_SCORE, 'mode': 'lang_mismatch',
                'ppl_fake_region': None, 'ppl_background': None,
                'ppl_median': None, 'anomaly_ratio': None, 'lang_mismatch': True
            }

        # ── 토큰별 ppl 계산 (공통) ────────────────────────────────
        token_ppls, token_meta = self._compute_token_ppls_with_meta(transcript)
        if not token_ppls:
            return self._empty_result()

        # ── 모드 A: fake_segments 알 때 ──────────────────────────
        if fake_segments and word_timestamps:
            return self._score_segment_mode(
                token_ppls, token_meta, word_timestamps, fake_segments)

        # ── 모드 B: fake_segments 모를 때 → 슬라이딩 윈도우 ──────
        if word_timestamps:
            return self._score_sliding_mode(token_ppls, token_meta, word_timestamps)

        # ── 폴백: word_timestamps 없을 때 global ppl ─────────────
        return self._score_global(token_ppls)

    # ── 모드 A: 구간 ppl vs 배경 ppl ────────────────────────────

    def _score_segment_mode(self, token_ppls, token_meta,
                             word_timestamps, fake_segments) -> dict:
        """
        조작 구간에 속하는 토큰 ppl과 나머지 토큰 ppl을 분리
        score = sigmoid(log(ppl_fake / ppl_background) * α)
        """
        fake_set, bg_set = [], []
        for ppl, (start, end) in zip(token_ppls, token_meta):
            in_fake = any(s <= start < e or s < end <= e
                          for s, e in fake_segments)
            (fake_set if in_fake else bg_set).append(np.log(ppl + 1e-8))

        # 조작 구간에 토큰이 없으면 global로 폴백
        if not fake_set:
            return self._score_global(token_ppls)

        log_ppl_fake = float(np.median(fake_set))
        log_ppl_bg   = float(np.median(bg_set)) if bg_set else log_ppl_fake

        # 비율: 조작 구간이 배경보다 얼마나 이상한가
        ratio    = log_ppl_fake - log_ppl_bg            # 양수 = fake 구간이 더 이상
        score_t  = self._sigmoid(ratio * 1.5)           # α=1.5

        return {
            'score_t':         round(score_t, 4),
            'mode':            'segment',
            'ppl_fake_region': round(float(np.exp(log_ppl_fake)), 2),
            'ppl_background':  round(float(np.exp(log_ppl_bg)), 2),
            'ppl_median':      None,
            'anomaly_ratio':   None,
            'lang_mismatch':   False
        }

    # ── 모드 B: 슬라이딩 윈도우 ─────────────────────────────────

    def _score_sliding_mode(self, token_ppls, token_meta,
                             word_timestamps, window_sec=1.0) -> dict:
        """
        word_timestamps를 1초 윈도우로 슬라이딩
        각 윈도우의 log-ppl 평균 계산 → 최대 윈도우 score를 반환
        """
        if not token_meta:
            return self._score_global(token_ppls)

        times    = np.array([t[0] for t in token_meta])
        log_ppls = np.log(np.array(token_ppls) + 1e-8)
        duration = times[-1] - times[0] + 1e-8

        if duration < window_sec:
            return self._score_global(token_ppls)

        window_scores = []
        t = times[0]
        while t + window_sec <= times[-1] + 1e-6:
            mask = (times >= t) & (times < t + window_sec)
            if mask.sum() >= 2:
                window_scores.append(float(np.mean(log_ppls[mask])))
            t += window_sec / 2    # 50% overlap

        if not window_scores:
            return self._score_global(token_ppls)

        global_median = float(np.median(log_ppls))
        max_window    = float(max(window_scores))
        ratio         = max_window - global_median
        score_t       = self._sigmoid(ratio * 1.2)

        return {
            'score_t':         round(score_t, 4),
            'mode':            'sliding',
            'ppl_fake_region': round(float(np.exp(max_window)), 2),
            'ppl_background':  round(float(np.exp(global_median)), 2),
            'ppl_median':      round(float(np.exp(global_median)), 2),
            'anomaly_ratio':   self._anomaly_ratio(log_ppls),
            'lang_mismatch':   False
        }

    # ── 폴백: global ppl ─────────────────────────────────────────

    def _score_global(self, token_ppls) -> dict:
        log_ppls      = np.log(np.array(token_ppls) + 1e-8)
        ppl_median    = float(np.median(log_ppls))
        anomaly_ratio = self._anomaly_ratio(log_ppls)
        score_t       = self._sigmoid((ppl_median - 2.0) * 1.2 + anomaly_ratio * 4.0)
        return {
            'score_t':         round(score_t, 4),
            'mode':            'global',
            'ppl_fake_region': None,
            'ppl_background':  None,
            'ppl_median':      round(float(np.exp(ppl_median)), 2),
            'anomaly_ratio':   self._anomaly_ratio(log_ppls),
            'lang_mismatch':   False
        }

    # ── 공통 유틸 ────────────────────────────────────────────────

    def _compute_token_ppls_with_meta(self, text: str):
        """
        각 토큰의 (ppl, char_start_approx) 반환
        word_timestamps와 매핑하기 위해 토큰 순서 인덱스도 보존
        """
        enc            = self.tokenizer(text, return_tensors="pt",
                                         truncation=True, max_length=512,
                                         add_special_tokens=True)
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        L              = input_ids.shape[1]

        # 토큰→문자 오프셋 매핑
        enc_with_offset = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            add_special_tokens=True, return_offsets_mapping=True
        )
        offsets = enc_with_offset["offset_mapping"][0].tolist()

        token_ppls, token_meta = [], []

        with torch.no_grad():
            for i in range(1, L - 1):
                tok_id  = input_ids[0, i].item()
                if tok_id in self._special_ids:
                    continue
                tok_str = self.tokenizer.convert_ids_to_tokens(tok_id)
                clean   = tok_str.lstrip("▁").strip()
                if not clean or _PUNCT_RE.match(clean):
                    continue

                masked_ids       = input_ids.clone()
                masked_ids[0, i] = self.tokenizer.mask_token_id
                logits           = self.model(masked_ids,
                                              attention_mask=attention_mask).logits
                log_p            = torch.log_softmax(logits[0, i], dim=-1)[tok_id].item()
                ppl              = float(np.exp(-log_p))

                # 문자 오프셋 저장 (word_timestamps 매핑에 사용)
                char_start = offsets[i][0] if i < len(offsets) else 0
                char_end   = offsets[i][1] if i < len(offsets) else 0
                token_ppls.append(ppl)
                token_meta.append((char_start, char_end))

        return token_ppls, token_meta

    def _anomaly_ratio(self, log_ppls: np.ndarray, z_thresh=2.0) -> float:
        if len(log_ppls) < 3: return 0.0
        mean = log_ppls.mean(); std = log_ppls.std() + 1e-8
        return round(float(np.mean(log_ppls > mean + z_thresh * std)), 4)

    def _sigmoid(self, x): return float(1 / (1 + np.exp(-x)))

    def _empty_result(self):
        return {'score_t': 0.5, 'mode': 'empty', 'ppl_fake_region': None,
                'ppl_background': None, 'ppl_median': None,
                'anomaly_ratio': None, 'lang_mismatch': False}


# ══════════════════════════════════════════════════════════════════
# 단독 실행 테스트   python nlp_scorer.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os, tempfile, subprocess
    import whisper, av
    import scipy.io.wavfile as wavfile

    BASE        = os.path.expanduser("~/hsh/AIApplication")
    META_PATH   = os.path.join(BASE, "AV-Deepfake1M_RootFiles/val_metadata.json")
    AVDF1M_ROOT = os.path.join(BASE, "AV-Deepfake1M_RootFiles/extracted_val/val")

    scorer        = NLPScorer()
    whisper_model = whisper.load_model("medium")

    def extract_audio_16k(video_path, out_wav):
        container = av.open(video_path)
        if not container.streams.audio:
            container.close(); return False
        sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        container.close()
        if not frames: return False
        wav = np.concatenate(frames)
        peak = np.abs(wav).max()
        if peak > 0: wav = wav / peak * 0.95
        wavfile.write(out_wav + ".tmp.wav", sr, (wav * 32767).astype(np.int16))
        subprocess.run(["ffmpeg","-y","-i",out_wav+".tmp.wav",
                        "-ar","16000","-ac","1",out_wav], capture_output=True)
        os.remove(out_wav + ".tmp.wav")
        return True

    def whisper_to_word_timestamps(wresult):
        words = []
        for seg in wresult["segments"]:
            for w in seg.get("words", []):
                words.append({'word': w['word'], 'start': w['start'], 'end': w['end']})
        return words

    with open(META_PATH) as f:
        meta = json.load(f)

    TYPES   = ["real", "audio_modified", "visual_modified", "both_modified"]
    samples = {t: next((m for m in meta if m["modify_type"] == t), None)
               for t in TYPES}

    print(f"\n{'='*78}")
    print(f"{'TYPE':<20} {'MODE':<10} {'PPL_fake':>9} {'PPL_bg':>8} {'SCORE_T':>8}")
    print(f"{'='*78}")

    for label, entry in samples.items():
        if entry is None: continue
        video_path = os.path.join(AVDF1M_ROOT, entry["file"])
        if not os.path.exists(video_path):
            print(f"  {label:<20} 파일 없음"); continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        if not extract_audio_16k(video_path, tmp_wav):
            print(f"  {label:<20} 오디오 추출 실패"); continue

        wresult    = whisper_model.transcribe(tmp_wav, word_timestamps=True,
                                              task="transcribe", verbose=False)
        os.remove(tmp_wav)

        transcript  = wresult["text"].strip()
        lang        = wresult["language"]
        word_ts     = whisper_to_word_timestamps(wresult)
        fake_segs   = entry.get("audio_fake_segments", [])

        # 모드 A (평가용): fake_segments 제공
        r_A = scorer.score(transcript, lang, word_ts, fake_segs if fake_segs else None)
        # 모드 B (추론용): fake_segments 미제공
        r_B = scorer.score(transcript, lang, word_ts, None)

        def fmt(v): return f"{v:.2f}" if v is not None else "  N/A"

        print(f"\n  {label}")
        print(f"    audio_fake_segs : {fake_segs}")
        print(f"    transcript      : {transcript[:70]}")
        print(f"    [모드A-segment]  ppl_fake={fmt(r_A['ppl_fake_region'])}  "
              f"ppl_bg={fmt(r_A['ppl_background'])}  score_t={r_A['score_t']:.4f}")
        print(f"    [모드B-sliding]  ppl_fake={fmt(r_B['ppl_fake_region'])}  "
              f"ppl_bg={fmt(r_B['ppl_background'])}  score_t={r_B['score_t']:.4f}")
