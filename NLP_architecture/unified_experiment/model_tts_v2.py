"""
model_tts_v2.py
NLP 기반 TTS 기법 분류 (텍스트 임베딩 + confidence 통계 융합)
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class TTSClassifierV2(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", n_class=5, n_stat=22, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(dropout),
        )
        self.stat_encoder = nn.Sequential(
            nn.Linear(n_stat, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128+128, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_class),
        )

    def forward(self, input_ids, attention_mask, stat_feats):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        text_pool = (out.last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1)
        text_feat = self.text_proj(text_pool)
        stat_feat = self.stat_encoder(stat_feats)
        fused = torch.cat([text_feat, stat_feat], dim=-1)
        return self.classifier(fused)
