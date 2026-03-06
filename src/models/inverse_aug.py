"""
Inverse Augmentation module for DeepFusion.
FIXED VERSION — forward() sekarang menerima:
  - None                          → no-op
  - AugmentationParams            → single (lama, backward-compatible)
  - list[AugmentationParams|None] → per-sample dalam batch (BARU)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class AugmentationParams:
    """Container for augmentation parameters per sample."""

    def __init__(
        self,
        rotation_angle: float = 0.0,
        flip_x: bool = False,
        flip_y: bool = False,
        scale: float = 1.0,
        translate_x: float = 0.0,
        translate_y: float = 0.0
    ):
        self.rotation_angle = rotation_angle
        self.flip_x        = flip_x
        self.flip_y        = flip_y
        self.scale         = scale
        self.translate_x   = translate_x
        self.translate_y   = translate_y

    def is_identity(self) -> bool:
        """True jika params ini tidak melakukan transformasi apapun."""
        return (
            self.rotation_angle == 0.0
            and not self.flip_x
            and not self.flip_y
            and self.scale == 1.0
            and self.translate_x == 0.0
            and self.translate_y == 0.0
        )

    def to_dict(self) -> Dict:
        return {
            'rotation_angle': self.rotation_angle,
            'flip_x':        self.flip_x,
            'flip_y':        self.flip_y,
            'scale':         self.scale,
            'translate_x':   self.translate_x,
            'translate_y':   self.translate_y
        }

    @classmethod
    def from_dict(cls, params: Dict) -> 'AugmentationParams':
        return cls(**params)

    @classmethod
    def identity(cls) -> 'AugmentationParams':
        """Kembalikan params identity (no-op)."""
        return cls()


class InverseAugmentation(nn.Module):
    """
    Inverse Augmentation untuk DeepFusion.

    Mendukung aug_params berupa:
      - None                     → return as-is
      - AugmentationParams       → satu params untuk seluruh batch
      - list[AugmentationParams] → satu params per sample (diproses per-sample)
    """

    def __init__(self):
        super().__init__()

    # ── private helpers ───────────────────────────────────────────────────────

    def _inverse_rotation(self, feat: torch.Tensor, angle: float) -> torch.Tensor:
        if angle == 0.0:
            return feat
        cos_a = float(np.cos(-angle))
        sin_a = float(np.sin(-angle))
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0]],
            dtype=feat.dtype, device=feat.device
        ).unsqueeze(0)                          # (1, 2, 3)
        grid = torch.nn.functional.affine_grid(
            theta, feat.unsqueeze(0).size(), align_corners=False
        )
        return torch.nn.functional.grid_sample(
            feat.unsqueeze(0), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(0)

    def _inverse_flip(self, feat: torch.Tensor, flip_x: bool, flip_y: bool) -> torch.Tensor:
        if flip_x:
            feat = torch.flip(feat, dims=[2])   # W dimension
        if flip_y:
            feat = torch.flip(feat, dims=[1])   # H dimension
        return feat

    def _inverse_scale(self, feat: torch.Tensor, scale: float) -> torch.Tensor:
        if scale == 1.0:
            return feat
        C, H, W = feat.shape
        inv = 1.0 / scale
        scaled = torch.nn.functional.interpolate(
            feat.unsqueeze(0),
            size=(max(1, int(H * inv)), max(1, int(W * inv))),
            mode='bilinear', align_corners=False
        )
        # Resize kembali ke ukuran asli agar konsisten
        return torch.nn.functional.interpolate(
            scaled, size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0)

    def _apply_one(self, image_feat: torch.Tensor, p: AugmentationParams) -> torch.Tensor:
        """
        Terapkan inverse aug pada satu sampel image feature (C, H, W).
        Urutan inverse: scale → flip → rotation (kebalikan dari augmentasi).
        """
        if p is None or p.is_identity():
            return image_feat

        feat = image_feat
        if p.scale != 1.0:
            feat = self._inverse_scale(feat, p.scale)
        if p.flip_x or p.flip_y:
            feat = self._inverse_flip(feat, p.flip_x, p.flip_y)
        if p.rotation_angle != 0.0:
            feat = self._inverse_rotation(feat, p.rotation_angle)
        return feat

    # ── public forward ────────────────────────────────────────────────────────

    def forward(
        self,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor,
        aug_params: Optional[Union[
            'AugmentationParams',
            List[Optional['AugmentationParams']]
        ]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lidar_features: (B, C, H, W)
            image_features: (B, C, H, W)
            aug_params:
                None                          → no-op
                AugmentationParams            → sama untuk semua sample
                list[AugmentationParams|None] → per-sample (panjang = B)

        Returns:
            aligned_lidar : (B, C, H, W)  — tidak berubah
            aligned_image : (B, C, H, W)  — sudah di-inverse-aug
        """
        # ── kasus None atau semua identity → cepat ───────────────────────────
        if aug_params is None:
            return lidar_features, image_features

        # ── normalisasi ke list[B] ────────────────────────────────────────────
        B = lidar_features.shape[0]

        if isinstance(aug_params, AugmentationParams):
            # satu params → broadcast ke semua sample
            params_list: List[Optional[AugmentationParams]] = [aug_params] * B
        elif isinstance(aug_params, list):
            # sudah list — pastikan panjangnya B
            if len(aug_params) != B:
                # kalau tidak cocok (mis. collate edge case), pakai identity
                params_list = [None] * B
            else:
                params_list = aug_params
        else:
            # tipe tidak dikenal → no-op
            return lidar_features, image_features

        # Kalau semua None → shortcut
        if all(p is None for p in params_list):
            return lidar_features, image_features

        # ── proses per-sample ─────────────────────────────────────────────────
        aligned_images = []
        for i in range(B):
            p = params_list[i]
            aligned = self._apply_one(image_features[i], p)   # (C, H, W)
            aligned_images.append(aligned)

        aligned_image_batch = torch.stack(aligned_images, dim=0)   # (B, C, H, W)
        return lidar_features, aligned_image_batch

    # ── point cloud helper (tidak berubah) ───────────────────────────────────

    def inverse_augment_point_cloud(
        self,
        points: torch.Tensor,
        aug_params: 'AugmentationParams'
    ) -> torch.Tensor:
        if aug_params is None or aug_params.is_identity():
            return points

        pts = points.clone()
        x, y = pts[:, 0], pts[:, 1]

        if aug_params.scale != 1.0:
            x = x / aug_params.scale
            y = y / aug_params.scale

        if aug_params.flip_x:
            x = -x
        if aug_params.flip_y:
            y = -y

        if aug_params.rotation_angle != 0.0:
            cos_a = float(np.cos(-aug_params.rotation_angle))
            sin_a = float(np.sin(-aug_params.rotation_angle))
            x, y = x * cos_a - y * sin_a, x * sin_a + y * cos_a

        if aug_params.translate_x != 0.0 or aug_params.translate_y != 0.0:
            x = x - aug_params.translate_x
            y = y - aug_params.translate_y

        pts[:, 0], pts[:, 1] = x, y
        return pts


def get_inverse_augmentation_matrix(aug_params: AugmentationParams) -> np.ndarray:
    """Buat 3×3 inverse transformation matrix dari aug_params."""
    M = np.eye(3)
    M[0, 2] = -aug_params.translate_x
    M[1, 2] = -aug_params.translate_y

    if aug_params.scale != 1.0:
        s = np.diag([1.0 / aug_params.scale, 1.0 / aug_params.scale, 1.0])
        M = s @ M

    if aug_params.flip_x or aug_params.flip_y:
        F = np.eye(3)
        if aug_params.flip_x: F[0, 0] = -1
        if aug_params.flip_y: F[1, 1] = -1
        M = F @ M

    if aug_params.rotation_angle != 0.0:
        a = -aug_params.rotation_angle
        R = np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a),  np.cos(a), 0],
            [0,          0,         1]
        ])
        M = R @ M

    return M