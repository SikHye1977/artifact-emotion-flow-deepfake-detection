"""
model_v2.py
메타 신호 확장 통합 NLP 모델

token feature:
  XLM-RoBERTa hidden (768)
  + prob (confidence)
  + dur (duration)
  + gap_prev (인접 confidence 급변) ⭐ 신규
  + gap_next (다음 단어와의 급변)   ⭐ 신규
  → 5개 메타 신호
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class UnifiedNLPModelV2(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3, n_meta=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768
        self.n_meta = n_meta

        # 메타 신호 전용 인코더 (작은 MLP로 비선형성 부여)
        self.meta_encoder = nn.Sequential(
            nn.Linear(n_meta, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        # 텍스트 + 메타 결합
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask, meta_feats):
        """
        meta_feats: (B, L, n_meta)  [prob, dur, gap_prev, gap_next]
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state           # (B, L, 768)

        meta = self.meta_encoder(meta_feats)     # (B, L, 64)
        feat = torch.cat([hidden, meta], dim=-1) # (B, L, 832)
        feat = self.feat_proj(feat)              # (B, L, 768)
        token_logits = self.token_head(feat).squeeze(-1)  # (B, L)
        return token_logits
