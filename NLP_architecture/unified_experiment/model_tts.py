"""
model_tts.py
TTS 기법 5분류 NLP 모델
  토큰 임베딩 + word별 메타(prob,dur,gap_prev,gap_next)
  → clip-level 5-class
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class TTSClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", n_class=5, n_meta=4, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.meta_encoder = nn.Sequential(
            nn.Linear(n_meta, 64), nn.GELU(), nn.Linear(64, 64),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        # clip-level 분류 (mean+max pooling)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )
    def forward(self, input_ids, attention_mask, meta_feats):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                       # (B,L,768)
        m = self.meta_encoder(meta_feats)               # (B,L,64)
        feat = self.proj(torch.cat([h,m],dim=-1))       # (B,L,768)
        # masked pooling
        mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (feat*mask).sum(1)/mask.sum(1).clamp(min=1)
        max_pool = (feat.masked_fill(mask==0,-1e9)).max(1)[0]
        clip = torch.cat([mean_pool, max_pool], dim=-1) # (B,1536)
        return self.classifier(clip)                    # (B,n_class)
