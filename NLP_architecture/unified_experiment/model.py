"""
model.py
통합 NLP 모델: XLM-RoBERTa 임베딩 + ASR confidence + duration

token feature:
  XLM-RoBERTa hidden (768) || prob (1) || dur (1)
  → projection → token head → token fake 확률
  → top-k mean → clip score
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class UnifiedNLPModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768

        # confidence + duration을 hidden에 결합
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask, token_probs, token_durs):
        """
        input_ids:    (B, L)
        attention_mask:(B, L)
        token_probs:  (B, L)  ASR confidence
        token_durs:   (B, L)  word duration
        """
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        hidden = out.last_hidden_state               # (B, L, 768)

        # confidence, duration을 추가 feature로 concat
        probs = token_probs.unsqueeze(-1)            # (B, L, 1)
        durs  = token_durs.unsqueeze(-1)             # (B, L, 1)
        feat  = torch.cat([hidden, probs, durs], dim=-1)  # (B, L, 770)

        feat  = self.feat_proj(feat)                 # (B, L, 768)
        token_logits = self.token_head(feat).squeeze(-1)  # (B, L)
        return token_logits
