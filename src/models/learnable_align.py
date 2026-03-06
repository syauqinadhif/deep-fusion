"""
Learnable Alignment module for DeepFusion.
FIXED VERSION — menghapus Python chunked-loop dan menggantinya dengan
torch.nn.functional.scaled_dot_product_attention (Flash Attention-compatible)
yang sepenuhnya berjalan di GPU tanpa loop Python.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class LearnableAlignment(nn.Module):
    """
    Learnable Alignment menggunakan Cross-Attention untuk DeepFusion.

    FIX UTAMA:
      - Chunked Python loop dihapus → diganti torch.nn.functional.scaled_dot_product_attention
        yang mendukung Flash Attention otomatis di PyTorch >= 2.0 (RTX 4080 ✓)
      - Satu operasi matrix penuh → GPU fully utilized

    Args:
        lidar_channels:     Jumlah channel LiDAR feature
        image_channels:     Jumlah channel image feature
        hidden_dim:         Hidden dim untuk attention
        num_heads:          Jumlah attention head
        num_layers:         Jumlah cross-attention layer
        dropout:            Dropout probability
        compression_factor: Faktor kompresi spasial sebelum attention
    """

    def __init__(
        self,
        lidar_channels:     int   = 256,
        image_channels:     int   = 256,
        hidden_dim:         int   = 256,
        num_heads:          int   = 8,
        num_layers:         int   = 1,
        dropout:            float = 0.1,
        compression_factor: int   = 4,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim harus habis dibagi num_heads"

        self.hidden_dim         = hidden_dim
        self.num_heads          = num_heads
        self.num_layers         = num_layers
        self.compression_factor = compression_factor
        self.head_dim           = hidden_dim // num_heads

        # Kompresi spasial (mengurangi N_q dan N_k agar attention cepat)
        self.compress_lidar = nn.Sequential(
            nn.Conv2d(lidar_channels, hidden_dim,
                      kernel_size=3, stride=compression_factor, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.compress_image = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim,
                      kernel_size=3, stride=compression_factor, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Q, K, V projections — satu Linear per layer (efisien)
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.out_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Layer norms dan FFN per layer
        self.norm1s = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norm2s = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffns   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.attn_dropout = dropout

        # Output projection kembali ke lidar_channels
        self.output_proj = nn.Conv2d(hidden_dim, lidar_channels, kernel_size=1)
        self.skip_proj   = nn.Conv2d(lidar_channels, lidar_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            lidar_features: (B, C_lidar, H, W)
            image_features: (B, C_image, H, W)

        Returns:
            aligned_features: (B, C_lidar, H, W)  — sama shape dengan input
            None  (attention map tidak dikembalikan untuk efisiensi)
        """
        B, C_l, H_orig, W_orig = lidar_features.shape

        lidar_skip = lidar_features   # untuk residual di akhir

        # 1. Kompresi spasial
        lidar_c = self.compress_lidar(lidar_features)   # (B, D, H/cf, W/cf)
        image_c = self.compress_image(image_features)   # (B, D, H/cf, W/cf)

        _, D, H_c, W_c = lidar_c.shape
        N = H_c * W_c   # panjang sequence setelah kompresi

        # 2. Reshape ke sequence: (B, N, D)
        lidar_seq = lidar_c.flatten(2).transpose(1, 2).contiguous()   # (B, N, D)
        image_seq = image_c.flatten(2).transpose(1, 2).contiguous()   # (B, N, D)

        # 3. Cross-attention layers
        for i in range(self.num_layers):
            # ── QKV projection ──────────────────────────────────────────────
            Q = self.q_projs[i](lidar_seq)   # (B, N, D)
            K = self.k_projs[i](image_seq)   # (B, N, D)
            V = self.v_projs[i](image_seq)   # (B, N, D)

            # ── Reshape ke multi-head format: (B, H, N, d) ─────────────────
            Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

            # ── FIX UTAMA: satu operasi GPU, bukan Python loop ──────────────
            # scaled_dot_product_attention otomatis pakai Flash Attention
            # kalau tersedia (PyTorch >= 2.0, CUDA, causal=False)
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.attn_dropout if self.training else 0.0
            )   # (B, H, N, d)

            # ── Reshape balik ke (B, N, D) ──────────────────────────────────
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
            attn_out = self.out_projs[i](attn_out)

            # ── Residual + LayerNorm ─────────────────────────────────────────
            lidar_seq = self.norm1s[i](lidar_seq + attn_out)

            # ── FFN + Residual + LayerNorm ───────────────────────────────────
            lidar_seq = self.norm2s[i](lidar_seq + self.ffns[i](lidar_seq))

        # 4. Reshape sequence kembali ke 2-D spatial
        out_c = lidar_seq.transpose(1, 2).view(B, D, H_c, W_c)   # (B, D, H_c, W_c)

        # 5. Upsample kembali ke resolusi asli
        out = F.interpolate(out_c, size=(H_orig, W_orig), mode='bilinear', align_corners=False)

        # 6. Project ke lidar_channels + skip connection
        out = self.output_proj(out) + self.skip_proj(lidar_skip)

        return out, None


class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding."""

    def __init__(self, d_model: int, max_res: int = 512):
        super().__init__()
        pe = torch.zeros(d_model, max_res, max_res)

        pos   = torch.arange(max_res, dtype=torch.float32)
        dterm = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                          * (-math.log(10000.0) / d_model))

        # X dimension (broadcast over H)
        pe[0::2, :, :] = (torch.sin(pos[None, :] * dterm[:, None])
                          .unsqueeze(1).repeat(1, max_res, 1))
        pe[1::2, :, :] = (torch.cos(pos[None, :] * dterm[:, None])
                          .unsqueeze(1).repeat(1, max_res, 1))

        # Add Y dimension
        pe[0::2, :, :] += (torch.sin(pos[:, None] * dterm[:, None])
                           .unsqueeze(2).repeat(1, 1, max_res))
        pe[1::2, :, :] += (torch.cos(pos[:, None] * dterm[:, None])
                           .unsqueeze(2).repeat(1, 1, max_res))

        self.register_buffer('pe', pe)   # (C, max_res, max_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return self.pe[:, :H, :W].unsqueeze(0)   # (1, C, H, W) — broadcast


class SpatialLearnableAlignment(nn.Module):
    """LearnableAlignment dengan 2-D positional encoding."""

    def __init__(
        self,
        lidar_channels:     int   = 256,
        image_channels:     int   = 256,
        hidden_dim:         int   = 256,
        num_heads:          int   = 8,
        num_layers:         int   = 1,
        dropout:            float = 0.1,
        max_spatial_resolution: int = 512,
        compression_factor: int   = 4,
    ):
        super().__init__()
        self.pos_enc   = PositionalEncoding2D(lidar_channels, max_spatial_resolution)
        self.pos_enc_i = PositionalEncoding2D(image_channels, max_spatial_resolution)
        self.alignment = LearnableAlignment(
            lidar_channels, image_channels, hidden_dim,
            num_heads, num_layers, dropout, compression_factor
        )

    def forward(self, lidar_features, image_features, return_attention=False):
        lidar_in = lidar_features + self.pos_enc(lidar_features)
        image_in = image_features + self.pos_enc_i(image_features)
        return self.alignment(lidar_in, image_in, return_attention)


# ── quick sanity-check ───────────────────────────────────────────────────────
if __name__ == '__main__':
    B, C, H, W = 2, 256, 252, 252
    lidar = torch.randn(B, C, H, W, device='cuda' if torch.cuda.is_available() else 'cpu')
    image = torch.randn(B, C, H, W, device=lidar.device)

    model = LearnableAlignment(
        lidar_channels=C, image_channels=C, hidden_dim=256,
        num_heads=8, num_layers=1, compression_factor=4
    ).to(lidar.device)

    out, _ = model(lidar, image)
    assert out.shape == lidar.shape, f"Shape mismatch: {out.shape} vs {lidar.shape}"
    print(f"✓ Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")