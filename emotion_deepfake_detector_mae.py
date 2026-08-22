"""
==============================================================================
[MAE-DFER 기반] 감정 흐름 딥페이크 탐지 모델
Emotion Flow Deepfake Detector with MAE-DFER backbone
==============================================================================

[아키텍처]

  영상 프레임 (B, T=16, 3, 160, 160)
        │
        ▼ permute → (B, 3, 16, 160, 160)
  ┌────────────────────────────────────────────────┐
  │  MAE-DFER Backbone (LGI-Former, Frozen)        │
  │  - VoxCeleb2 self-supervised pretrained        │
  │  - 84.88M params, hidden=512, 16 blocks        │
  │  - Region size: (2, 5, 10) → 8 regions         │
  │                                                │
  │  patch_embed: (B, 800, 512)                    │
  │   ↓ region partition: (B, 8, 100, 512)         │
  │   ↓ + region tokens: (B, 8, 101, 512)          │
  │   ↓ blocks (LGI-Former)                        │
  │   ↓ region tokens 추출: (B, 8, 512)            │
  │   ↓ reshape (nt, nh, nw): (B, 4, 2, 1, 512)    │
  │   ↓ 공간 평균: (B, 4, 512) ⭐ 시계열 feature    │
  └────────────────────────────────────────────────┘
        │
        ▼ (B, 4, 512)
  ┌─────────────────────────────────────────┐
  │ Bottleneck (feat_reduce)                │
  │  512 → 128 (Linear/BN/ReLU/Dropout)     │
  │  → (B, 4, 128)                          │
  └─────────────────────────────────────────┘
        │
        ▼ (B, 4, 128)
  ┌─────────────────────────────────────────┐
  │  LayerNorm (128)                        │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  GRU (1층, hidden=64)                   │
  │  → (B, 4, 64)                           │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  Temporal Attention                     │
  │  → context (B, 64), weights (B, 4)      │
  └─────────────────────────────────────────┘
        │
        ▼
  ┌──────────────────────┐
  │  Classifier          │
  │  64 → 64 → 1         │
  └──────────────────────┘
        │
        ▼
   Raw Logit (B,)
   ※ Sigmoid는 외부에서 적용 (BCEWithLogitsLoss)

[학습 전략]
- MAE-DFER backbone: 기본 동결 (84.88M params 보존)
- 옵션: unfreeze_last_blocks > 0 시 마지막 N개 블록 fine-tune
- 시계열 feature 추출은 항상 backbone에서 직접

[입력 사양]
- 해상도: 160x160 (MAE-DFER 사양)
- 프레임: 16개 균등 샘플링
- Tubelet: 시간 2 frames씩 묶음 → 8 tubelets
- LGI region: 시간 4개 × 공간 (2,1) = 8 regions
==============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ══════════════════════════════════════════════════════════════════════════════
# Temporal Attention (HSEmotion과 동일)
# ══════════════════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w = nn.Linear(hidden_size, 1)

    def forward(self, gru_out: torch.Tensor):
        scores  = self.w(gru_out)                     # (B, T, 1)
        weights = F.softmax(scores, dim=1)            # (B, T, 1)
        context = (weights * gru_out).sum(dim=1)      # (B, H)
        return context, weights.squeeze(-1)           # (B, H), (B, T)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 모델
# ══════════════════════════════════════════════════════════════════════════════
class EmotionFlowDetectorMAE(nn.Module):
    """
    MAE-DFER 백본 기반 감정 흐름 딥페이크 탐지 모델.
    
    Args:
        mae_dfer_path: MAE-DFER 코드 폴더 경로 (modeling_finetune.py 위치)
        pretrained_ckpt: VoxCeleb2 사전학습 가중치 (.pth) 경로
        gru_hidden: GRU hidden size (기본 64)
        dropout: Dropout 비율
        unfreeze_last_blocks: 마지막 N 블록을 학습 가능하게 (기본 0=동결)
    """
    def __init__(
        self,
        mae_dfer_path: str,
        pretrained_ckpt: str,
        gru_hidden: int = 64,
        dropout: float = 0.3,
        unfreeze_last_blocks: int = 0,
    ):
        super().__init__()
        
        self.gru_hidden = gru_hidden
        self.unfreeze_last_blocks = unfreeze_last_blocks
        
        # ── MAE-DFER 모듈 import ─────────────────────────────────
        if mae_dfer_path not in sys.path:
            sys.path.insert(0, mae_dfer_path)
        
        try:
            from modeling_finetune import vit_base_dim512_no_depth_patch16_160
        except ImportError as e:
            raise ImportError(
                f"MAE-DFER modeling_finetune import 실패. "
                f"mae_dfer_path 확인: {mae_dfer_path}\n에러: {e}"
            )
        
        # ── 백본 생성 ────────────────────────────────────────────
        print(f"🧠 MAE-DFER 백본 생성 (LGI-Former)")
        self.backbone = vit_base_dim512_no_depth_patch16_160(
            num_classes=7,                  # 임시 (head 사용 안 함)
            all_frames=16,
            tubelet_size=2,
            attn_type='local_global',       # ⭐ LGI-Former
            depth=16,
            lg_region_size=(2, 5, 10),      # ⭐ 사전학습과 일치
        )
        
        # ── 사전학습 가중치 로드 ─────────────────────────────────
        print(f"📦 사전학습 가중치 로드: {pretrained_ckpt}")
        if not os.path.exists(pretrained_ckpt):
            raise FileNotFoundError(f"가중치 없음: {pretrained_ckpt}")
        
        ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        
        # encoder. prefix 제거
        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_k = k[len('encoder.'):]
                encoder_sd[new_k] = v
        
        missing, unexpected = self.backbone.load_state_dict(encoder_sd, strict=False)
        # 누락은 head/fc_norm만 OK, 예상밖은 norm.weight/bias만 OK
        non_head_missing = [k for k in missing if 'head' not in k and 'fc_norm' not in k]
        if non_head_missing:
            print(f"  ⚠️  본질적 누락: {non_head_missing}")
        else:
            print(f"  ✅ 본질적 가중치 100% 로드")
        
        # ── 백본 동결 ─────────────────────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if unfreeze_last_blocks > 0:
            self._unfreeze_last_blocks(unfreeze_last_blocks)
        else:
            print(f"  ✅ Backbone 완전 동결")
        
        # 백본의 LGI 설정 저장 (forward 시 사용)
        self.lg_region_size = self.backbone.lg_region_size
        self.lg_num_region_size = self.backbone.lg_num_region_size
        
        # 시간 축 region 개수 (시계열 차원)
        self.num_temporal_regions = self.lg_num_region_size[0]  # nt = 4
        
        # ── Bottleneck (512 → 128) ─────────────────────────────
        self.feat_reduce = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # ── 입력 정규화 ──────────────────────────────────────
        self.input_norm = nn.LayerNorm(128)
        
        # ── GRU ───────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        
        # ── Attention + 분류기 ──────────────────────────────────
        self.attention = TemporalAttention(gru_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
        # ── 학습 가능 파라미터 카운트 ───────────────────────────
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 모델 통계:")
        print(f"   총 파라미터: {total:,}")
        print(f"   학습 가능   : {trainable:,} ({trainable/total*100:.1f}%)")
    
    def _unfreeze_last_blocks(self, n_blocks: int):
        """MAE-DFER 마지막 N개 블록 unfreeze."""
        if not hasattr(self.backbone, 'blocks'):
            print(f"  ⚠️  blocks 속성 없음, unfreeze 건너뜀")
            return
        
        num_blocks = len(self.backbone.blocks)
        unfreeze_from = max(0, num_blocks - n_blocks)
        
        unfrozen_count = 0
        for i in range(unfreeze_from, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
        
        # norm 레이어도 unfreeze (있으면)
        for attr_name in ['fc_norm', 'norm']:
            if hasattr(self.backbone, attr_name):
                module = getattr(self.backbone, attr_name)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
                        unfrozen_count += param.numel()
        
        print(f"🔓 Backbone 마지막 {n_blocks} blocks unfreeze")
        print(f"   학습 가능 backbone params: {unfrozen_count:,}")
    
    def _extract_temporal_features(self, x: torch.Tensor):
        """
        MAE-DFER 백본에서 시계열 feature 추출.
        
        Input:  x (B, 3, T=16, H=160, W=160)
        Output: temporal_features (B, num_temporal_regions=4, 512)
        """
        B = x.size(0)
        
        # patch_embed
        x = self.backbone.patch_embed(x)            # (B, 800, 512)
        
        # position embedding
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.backbone.pos_drop(x)
        
        # local_global region partition
        nt, t = self.lg_num_region_size[0], self.lg_region_size[0]
        nh, h = self.lg_num_region_size[1], self.lg_region_size[1]
        nw, w = self.lg_num_region_size[2], self.lg_region_size[2]
        
        x = rearrange(x, 'b (nt t nh h nw w) c -> b (nt nh nw) (t h w) c',
                      nt=nt, nh=nh, nw=nw, t=t, h=h, w=w)
        
        # region tokens 추가
        region_tokens = repeat(self.backbone.lg_region_tokens, 'n c -> b n 1 c', b=B)
        x = torch.cat([region_tokens, x], dim=2)
        
        # blocks 통과
        x = rearrange(x, 'b n s c -> (b n) s c')
        for blk in self.backbone.blocks:
            x = blk(x, B)
        
        x = rearrange(x, '(b n) s c -> b n s c', b=B)
        # x shape: (B, n_regions=8, 1+thw, 512)
        
        # region tokens만 추출 (각 region의 대표 토큰)
        region_only = x[:, :, 0]                    # (B, 8, 512)
        
        # 시간/공간 분리
        region_reshaped = rearrange(region_only, 'b (nt nh nw) c -> b nt nh nw c',
                                     nt=nt, nh=nh, nw=nw)
        # (B, 4, 2, 1, 512)
        
        # 공간 평균 → 시계열 feature
        temporal_features = region_reshaped.mean(dim=[2, 3])  # (B, 4, 512)
        
        return temporal_features
    
    def forward(self, x: torch.Tensor):
        """
        Input: 
          x (B, T=16, 3, H=160, W=160) — Dataset에서 일반적인 형태
          또는
          x (B, 3, T=16, H=160, W=160) — MAE-DFER 직접 입력 형태
        
        Output:
          logit (B,)
          attn_weights (B, num_temporal_regions=4)
        """
        # Input shape 자동 감지 및 변환
        if x.dim() == 5 and x.shape[1] == 16 and x.shape[2] == 3:
            # (B, T, C, H, W) → (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        # 아니면 이미 (B, C, T, H, W)
        
        B = x.size(0)
        
        # ── Backbone에서 시계열 feature 추출 ─────────────────
        if self.unfreeze_last_blocks > 0:
            # 일부 backbone 학습 → gradient 흐름
            temporal_feat = self._extract_temporal_features(x)
        else:
            # 완전 동결 → no_grad 컨텍스트
            with torch.no_grad():
                temporal_feat = self._extract_temporal_features(x)
            temporal_feat = temporal_feat.detach()
        
        # temporal_feat: (B, 4, 512)
        T = temporal_feat.size(1)
        
        # ── Bottleneck (512 → 128) ─────────────────────────
        # BatchNorm1d는 (B*T, 512)에서 처리
        feat_flat = temporal_feat.reshape(B * T, 512)
        reduced = self.feat_reduce(feat_flat)            # (B*T, 128)
        reduced = reduced.view(B, T, 128)                # (B, 4, 128)
        
        # ── LayerNorm + GRU + Attention ────────────────────
        normed = self.input_norm(reduced)
        gru_out, _ = self.gru(normed)                    # (B, 4, 64)
        context, attn_w = self.attention(gru_out)        # (B, 64), (B, 4)
        
        # ── 분류 ────────────────────────────────────────────
        logit = self.classifier(context)                 # (B, 1)
        
        return logit.squeeze(1), attn_w


# ══════════════════════════════════════════════════════════════════════════════
# Self-test (직접 실행 시)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*70)
    print("EmotionFlowDetectorMAE Self-test")
    print("="*70)
    
    # 경로는 사용자가 자신의 환경에 맞게 수정
    MAE_DFER_PATH = os.path.expanduser("~/hsh/AIApplication/mae_dfer")
    PRETRAINED_CKPT = os.path.join(MAE_DFER_PATH, "saved/pretrained/checkpoint-49.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # 모델 생성
    model = EmotionFlowDetectorMAE(
        mae_dfer_path=MAE_DFER_PATH,
        pretrained_ckpt=PRETRAINED_CKPT,
        gru_hidden=64,
        dropout=0.3,
        unfreeze_last_blocks=0,  # 일단 동결
    ).to(device)
    
    model.eval()
    
    # 더미 입력 (Dataset 일반 형태: B, T, C, H, W)
    print("\n[Test 1] (B, T, C, H, W) 입력")
    dummy = torch.randn(2, 16, 3, 160, 160).to(device)
    with torch.no_grad():
        logit, attn = model(dummy)
    print(f"  입력: {dummy.shape}")
    print(f"  logit: {logit.shape}")
    print(f"  attn: {attn.shape}")
    print(f"  prob: {torch.sigmoid(logit).cpu().numpy()}")
    
    # 다른 입력 형태
    print("\n[Test 2] (B, C, T, H, W) 입력")
    dummy2 = torch.randn(2, 3, 16, 160, 160).to(device)
    with torch.no_grad():
        logit2, attn2 = model(dummy2)
    print(f"  입력: {dummy2.shape}")
    print(f"  logit: {logit2.shape}")
    
    # Backward 테스트 (gradient 흐름)
    print("\n[Test 3] Backward (학습 가능 여부)")
    model.train()
    dummy3 = torch.randn(2, 16, 3, 160, 160).to(device)
    target = torch.tensor([0.0, 1.0]).to(device)
    
    logit, _ = model(dummy3)
    loss = F.binary_cross_entropy_with_logits(logit, target)
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  gradient 있는 파라미터: {grad_count}")
    
    print(f"\n{'='*70}")
    print("✅ Self-test 통과!")
    print(f"{'='*70}")