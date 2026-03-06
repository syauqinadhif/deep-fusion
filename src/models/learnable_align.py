"""
Learnable Alignment module for DeepFusion.
Uses memory-efficient cross-attention with spatial reduction.

This version uses spatial compression before attention to fit within GPU memory constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MemoryEfficientMultiHeadAttention(nn.Module):
    """
    Memory-Efficient Multi-Head Attention mechanism.

    Uses chunked processing to reduce memory footprint for large spatial dimensions.

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
        chunk_size: Size of chunks for processing (reduces memory)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        chunk_size: int = 256
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = chunk_size

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Remove dropout on attention weights to save memory
        # self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with chunked processing for memory efficiency.

        Args:
            query: (B, N_q, C) query tensor
            key: (B, N_k, C) key tensor
            value: (B, N_k, C) value tensor
            key_padding_mask: (B, N_k) mask for padding
            need_weights: Whether to return attention weights

        Returns:
            output: (B, N_q, C) output tensor
            attn_weights: Attention weights (optional)
        """
        B, N_q, C = query.shape
        N_k = key.shape[1]

        # Project Q, K, V
        Q = self.q_proj(query)  # (B, N_q, C)
        K = self.k_proj(key)    # (B, N_k, C)
        V = self.v_proj(value)  # (B, N_k, C)

        # Reshape for multi-head
        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, D)
        K = K.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, D)
        V = V.reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, D)

        # Compute output using chunked processing to save memory
        outputs = []

        # Process queries in chunks
        for q_start in range(0, N_q, self.chunk_size):
            q_end = min(q_start + self.chunk_size, N_q)
            Q_chunk = Q[:, :, q_start:q_end, :]  # (B, H, chunk_size, D)

            # Compute attention scores for this chunk
            attn_scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * self.scale  # (B, H, chunk_size, N_k)

            # Apply key padding mask if provided
            if key_padding_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )

            # Softmax to get attention weights (no dropout to save memory)
            attn_weights = F.softmax(attn_scores, dim=-1)
            # attn_weights = self.dropout(attn_weights)  # REMOVED to save memory

            # Apply attention to values
            output_chunk = torch.matmul(attn_weights, V)  # (B, H, chunk_size, D)
            outputs.append(output_chunk)

        # Concatenate chunks
        output = torch.cat(outputs, dim=2)  # (B, H, N_q, D)

        # Reshape back
        output = output.transpose(1, 2).contiguous().reshape(B, N_q, C)  # (B, N_q, C)

        # Output projection
        output = self.out_proj(output)

        if need_weights:
            return output, None
        return output, None


class LearnableAlignment(nn.Module):
    """
    Memory-Efficient Learnable Alignment using Cross-Attention for DeepFusion.

    This version uses spatial compression before attention to reduce memory usage.

    IMPORTANT: This version maintains spatial dimension consistency throughout the
    forward pass to ensure compatibility with detection head targets.

    Args:
        lidar_channels: Number of LiDAR feature channels
        image_channels: Number of image feature channels
        hidden_dim: Hidden dimension for the alignment module
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout probability
        chunk_size: Size of chunks for memory-efficient processing
        compression_factor: Factor to reduce spatial dimensions (e.g., 4 = 1/4 resolution)
    """

    def __init__(
        self,
        lidar_channels: int = 256,
        image_channels: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        chunk_size: int = 256,
        compression_factor: int = 4
    ):
        super().__init__()

        self.lidar_channels = lidar_channels
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.compression_factor = compression_factor

        # Spatial compression before attention (reduces memory significantly)
        self.compress = nn.Sequential(
            nn.Conv2d(lidar_channels, hidden_dim, kernel_size=3, stride=compression_factor, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.compress_image = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=compression_factor, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MemoryEfficientMultiHeadAttention(hidden_dim, num_heads, dropout, chunk_size=chunk_size),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),  # Reduced from 4 to save memory
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout)
                )
            })
            for _ in range(num_layers)
        ])

        # Projection to lidar channels
        self.output_proj = nn.Conv2d(hidden_dim, lidar_channels, kernel_size=1)

        # Skip connection projection
        self.skip_proj = nn.Conv2d(lidar_channels, lidar_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Learnable Alignment with spatial compression.

        This version:
        1. Compresses spatial dimensions before attention
        2. Processes attention at lower resolution
        3. Decompresses back to original resolution

        Args:
            lidar_features: (B, C_lidar, H, W) LiDAR features
            image_features: (B, C_image, H, W) image features
            return_attention: Whether to return attention weights

        Returns:
            aligned_features: (B, C_lidar, H, W) aligned features
            attention_weights: Attention weights (optional)
        """
        assert lidar_features.dim() == 4, "Only 2D feature maps (B, C, H, W) are supported"
        assert image_features.dim() == 4, "Only 2D feature maps (B, C, H, W) are supported"

        B, C_l, H_orig, W_orig = lidar_features.shape

        # Store original for skip connection
        lidar_orig = lidar_features

        # Compress spatial dimensions
        lidar_compressed = self.compress(lidar_features)  # (B, hidden_dim, H/c, W/c)
        image_compressed = self.compress_image(image_features)  # (B, hidden_dim, H/c, W/c)

        # Get compressed spatial dimensions
        _, _, H_comp, W_comp = lidar_compressed.shape

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        lidar_seq = lidar_compressed.reshape(B, -1, H_comp * W_comp).transpose(1, 2)  # (B, N, hidden_dim)
        image_seq = image_compressed.reshape(B, -1, H_comp * W_comp).transpose(1, 2)  # (B, N, hidden_dim)

        # Apply dropout
        lidar_seq = self.dropout(lidar_seq)
        image_seq = self.dropout(image_seq)

        # Apply cross-attention layers
        for layer in self.cross_attn_layers:
            # Cross-attention with residual
            attn_out, _ = layer['attn'](lidar_seq, image_seq, image_seq, need_weights=return_attention)
            lidar_seq = lidar_seq + self.dropout(attn_out)
            lidar_seq = layer['norm1'](lidar_seq)

            # FFN with residual
            ffn_out = layer['ffn'](lidar_seq)
            lidar_seq = lidar_seq + ffn_out
            lidar_seq = layer['norm2'](lidar_seq)

        # Reshape back to 2D: (B, N, hidden_dim) -> (B, hidden_dim, H_comp, W_comp)
        output_compressed = lidar_seq.transpose(1, 2).reshape(B, -1, H_comp, W_comp)

        # Decompress back to original resolution using bilinear interpolation (exact size)
        # This ensures we get exactly H_orig x W_orig
        output = F.interpolate(
            output_compressed,
            size=(H_orig, W_orig),
            mode='bilinear',
            align_corners=False
        )

        # Project to lidar channels
        output = self.output_proj(output)

        # Add skip connection from original lidar features
        output = output + self.skip_proj(lidar_orig)

        if return_attention:
            return output, None
        return output, None


class SpatialLearnableAlignment(nn.Module):
    """
    Spatial Learnable Alignment with position-aware cross-attention.

    This version includes positional encodings to better handle spatial relationships.
    """

    def __init__(
        self,
        lidar_channels: int = 256,
        image_channels: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_spatial_resolution: int = 512,
        chunk_size: int = 256,
        compression_factor: int = 4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(hidden_dim, max_spatial_resolution)

        # Core alignment module
        self.alignment = LearnableAlignment(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            chunk_size=chunk_size,
            compression_factor=compression_factor
        )

    def forward(
        self,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with positional encoding.
        """
        # Add positional encoding
        lidar_pos = self.pos_encoder(lidar_features)
        image_pos = self.pos_encoder(image_features)

        # Add positional encoding to features
        lidar_encoded = lidar_features + lidar_pos
        image_encoded = image_features + image_pos

        # Apply alignment
        aligned, attn = self.alignment(
            lidar_encoded,
            image_encoded,
            return_attention=return_attention
        )

        return aligned, attn


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for spatial features.

    Args:
        d_model: Dimension of the model
        max_res: Maximum spatial resolution
    """

    def __init__(self, d_model: int, max_res: int = 512):
        super().__init__()

        # Create positional encoding
        pe = torch.zeros(max_res, max_res, d_model)

        # Generate position indices
        position = torch.arange(max_res).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Compute positional encoding
        pe[:, :, 0::2] = torch.sin(position * div_term).transpose(0, 1).unsqueeze(2).repeat(1, max_res, 1)
        pe[:, :, 1::2] = torch.cos(position * div_term).transpose(0, 1).unsqueeze(2).repeat(1, max_res, 1)

        # Add y dimension
        pe[:, :, 0::2] += torch.sin(position * div_term).unsqueeze(1).repeat(1, max_res, 1)[:, :, 0::2]
        pe[:, :, 1::2] += torch.cos(position * div_term).unsqueeze(1).repeat(1, max_res, 1)[:, :, 1::2]

        # Register as buffer
        self.register_buffer('pe', pe.permute(2, 0, 1))  # (C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            (B, C, H, W) positional encoding (cropped to match input size)
        """
        B, C, H, W = x.shape
        return self.pe[:, :H, :W].unsqueeze(0).repeat(B, 1, 1, 1)


if __name__ == "__main__":
    # Test the learnable alignment module
    batch_size = 2
    lidar_channels = 256
    image_channels = 256
    height = 250  # Large spatial dimension
    width = 250

    # Create dummy features
    lidar_features = torch.randn(batch_size, lidar_channels, height, width)
    image_features = torch.randn(batch_size, image_channels, height, width)

    # Test basic learnable alignment with compression
    print("Testing LearnableAlignment with compression...")
    model = LearnableAlignment(
        lidar_channels=lidar_channels,
        image_channels=image_channels,
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
        chunk_size=256,
        compression_factor=4  # Compress to 1/4 resolution
    )

    aligned_features, _ = model(lidar_features, image_features, return_attention=False)

    print(f"LiDAR features shape: {lidar_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Aligned features shape: {aligned_features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify shape preservation
    assert aligned_features.shape == lidar_features.shape, "Output shape must match input shape!"
    print("✓ Shape preservation verified")

    # Test with larger compression for even more memory efficiency
    print("\nTesting with compression factor 8...")
    model_more_comp = LearnableAlignment(
        lidar_channels=lidar_channels,
        image_channels=image_channels,
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
        chunk_size=256,
        compression_factor=8  # Compress to 1/8 resolution
    )

    aligned_more, _ = model_more_comp(lidar_features, image_features)
    print(f"Aligned features shape (more compression): {aligned_more.shape}")
    assert aligned_more.shape == lidar_features.shape, "Output shape must match input shape!"
    print("✓ Shape preservation verified with more compression")
