"""
model_v3.py
메타 신호 6종 통합 NLP 모델
  token-level (4): prob, dur, gap_prev, gap_next
  clip-level  (2): sync_conf, sync_dist (모든 토큰에 broadcast)
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class UnifiedNLPModelV3(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3, n_meta=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.n_meta = n_meta
        self.meta_encoder = nn.Sequential(
            nn.Linear(n_meta, 64), nn.GELU(),
            nn.Linear(64, 64),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.token_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
    def forward(self, input_ids, attention_mask, meta_feats):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        meta = self.meta_encoder(meta_feats)
        feat = torch.cat([hidden, meta], dim=-1)
        feat = self.feat_proj(feat)
        return self.token_head(feat).squeeze(-1)
