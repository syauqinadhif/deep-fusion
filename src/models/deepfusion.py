"""
DeepFusion Model — Main integration.
FIXED VERSION:
  - forward() menerima aug_params sebagai list[AugmentationParams|None]
    (satu per sample dalam batch), bukan satu object tunggal
  - compute_loss() signature dibersihkan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from .pointpillars import PointPillarsBackbone
from .image_encoder import ImageEncoder
from .inverse_aug import InverseAugmentation, AugmentationParams
from .learnable_align import LearnableAlignment
from .detection_head import DetectionHead, ObjectDetectionLoss


class DeepFusion(nn.Module):
    """
    DeepFusion model untuk 3D object detection.

    forward() menerima aug_params berupa:
      - None
      - AugmentationParams          (backward-compat, broadcast ke semua sample)
      - list[AugmentationParams]    (per-sample, dari collate_fn)
    """

    def __init__(
        self,
        lidar_channels:       int  = 256,
        image_channels:       int  = 256,
        hidden_dim:           int  = 256,
        num_heads:            int  = 8,
        num_layers:           int  = 1,
        num_classes:          int  = 3,
        max_objects:          int  = 512,
        image_backbone:       str  = 'resnet34',
        pretrained_image:     bool = True,
        max_points_per_pillar: int = 100,
        max_pillars:          int  = 12000,
        voxel_size:           list = [0.16, 0.16, 4.0],
        point_range:          list = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        self.num_classes = num_classes
        self.max_objects = max_objects
        self.hidden_dim  = hidden_dim

        # ── sub-modules ───────────────────────────────────────────────────────
        self.lidar_backbone = PointPillarsBackbone(
            in_channels=4,
            out_channels=lidar_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
            voxel_size=voxel_size,
            point_range=point_range
        )

        self.image_encoder = ImageEncoder(
            backbone=image_backbone,
            pretrained=pretrained_image,
            out_features=image_channels
        )

        # InverseAugmentation sudah di-fix: menerima list[AugmentationParams]
        self.inverse_aug = InverseAugmentation()

        self.learnable_align = LearnableAlignment(
            lidar_channels=lidar_channels,
            image_channels=image_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.detection_head = DetectionHead(
            in_channels=hidden_dim,
            num_classes=num_classes,
            max_objects=max_objects
        )

        self.loss_fn = ObjectDetectionLoss(
            num_classes=num_classes,
            alpha=0.25,
            beta=2.0,
            gamma=2.0,
            offset_weight=1.0,
            size_weight=1.0,
            rotation_weight=1.0,
            z_weight=1.0
        )

    def forward(
        self,
        points:          torch.Tensor,
        images:          torch.Tensor,
        aug_params:      Optional[Union[
                             AugmentationParams,
                             List[Optional[AugmentationParams]]
                         ]] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points:      (B, N, 4)  LiDAR point cloud [x, y, z, intensity]
            images:      (B, 3, H, W) camera images
            aug_params:  None | AugmentationParams | list[AugmentationParams|None]
            return_attention: apakah kembalikan attention map

        Returns:
            dict dengan key: predictions, lidar_features, image_features,
                             fused_features, (attention_weights opsional)
        """
        # 1. LiDAR features
        lidar_features = self.lidar_backbone(points)          # (B, C, H_bev, W_bev)

        # 2. Image features
        image_features, _ = self.image_encoder(images)        # (B, C, H_img, W_img)

        # 3. Resize image features ke ukuran BEV
        if image_features.shape[2:] != lidar_features.shape[2:]:
            image_features = F.interpolate(
                image_features,
                size=lidar_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # 4. Inverse augmentation — aug_params bisa list[B] atau None
        #    InverseAugmentation.forward() sudah handle semua kasus
        aligned_lidar, aligned_image = self.inverse_aug(
            lidar_features,
            image_features,
            aug_params          # ← diteruskan apa adanya, tidak di-index
        )

        # 5. Learnable alignment (cross-attention fusion)
        fused_features, attention_weights = self.learnable_align(
            aligned_lidar,
            aligned_image,
            return_attention=return_attention
        )

        # 6. Detection head
        predictions = self.detection_head(fused_features)

        output = {
            'predictions':    predictions,
            'lidar_features': lidar_features,
            'image_features': image_features,
            'fused_features': fused_features
        }
        if return_attention:
            output['attention_weights'] = attention_weights

        return output

    def compute_loss(
        self,
        output:  Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            output:  dict hasil forward() — ambil key 'predictions'
            targets: dict ground-truth dari DataLoader
        """
        return self.loss_fn(output['predictions'], targets)

    def inference(
        self,
        points:         torch.Tensor,
        images:         torch.Tensor,
        conf_threshold: float = 0.3,
        nms_threshold:  float = 0.5
    ) -> list:
        self.eval()
        with torch.no_grad():
            output = self.forward(points, images)
            from .detection_head import DetectionDecoder
            decoder = DetectionDecoder(
                num_classes=self.num_classes,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                max_objects=self.max_objects
            )
            return decoder.decode(output['predictions'])

    def get_model_info(self) -> Dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters':     total,
            'trainable_parameters': trainable,
            'num_classes':          self.num_classes,
            'hidden_dim':           self.hidden_dim,
            'max_objects':          self.max_objects
        }


class DeepFusionLite(nn.Module):
    """Versi ringan DeepFusion untuk Jetson deployment."""

    def __init__(
        self,
        lidar_channels:       int  = 128,
        image_channels:       int  = 128,
        hidden_dim:           int  = 128,
        num_heads:            int  = 4,
        num_classes:          int  = 3,
        max_objects:          int  = 512,
        max_points_per_pillar: int = 100,
        max_pillars:          int  = 12000,
        voxel_size:           list = [0.16, 0.16, 4.0],
        point_range:          list = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.hidden_dim  = hidden_dim

        self.lidar_backbone = PointPillarsBackbone(
            in_channels=4, out_channels=lidar_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars, voxel_size=voxel_size, point_range=point_range
        )
        from .image_encoder import LightweightImageEncoder
        self.image_encoder  = LightweightImageEncoder(out_features=image_channels)
        self.inverse_aug    = InverseAugmentation()
        self.learnable_align = LearnableAlignment(
            lidar_channels=lidar_channels, image_channels=image_channels,
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=1
        )
        self.detection_head = DetectionHead(
            in_channels=hidden_dim, num_classes=num_classes, max_objects=max_objects
        )
        self.loss_fn = ObjectDetectionLoss(num_classes=num_classes)

    def forward(self, points, images, aug_params=None, return_attention=False):
        lidar_features = self.lidar_backbone(points)
        image_features = self.image_encoder(images)

        if image_features.shape[2:] != lidar_features.shape[2:]:
            image_features = F.interpolate(
                image_features, size=lidar_features.shape[2:],
                mode='bilinear', align_corners=False
            )

        aligned_lidar, aligned_image = self.inverse_aug(
            lidar_features, image_features, aug_params
        )
        fused_features, _ = self.learnable_align(aligned_lidar, aligned_image)
        predictions = self.detection_head(fused_features)

        return {
            'predictions':    predictions,
            'lidar_features': lidar_features,
            'image_features': image_features,
            'fused_features': fused_features
        }

    def compute_loss(self, output, targets):
        return self.loss_fn(output['predictions'], targets)


def build_deepfusion_model(config: dict, lite: bool = False) -> nn.Module:
    mc = config.get('model', {})
    if lite:
        return DeepFusionLite(
            lidar_channels=mc.get('image_features', 128),
            image_channels=mc.get('image_features', 128),
            hidden_dim=mc.get('hidden_dim', 128),
            num_heads=mc.get('n_heads', 4),
            num_classes=mc.get('num_classes', 3),
            max_objects=mc.get('max_objects_per_image', 512),
            max_points_per_pillar=mc.get('max_points_per_pillar', 100),
            max_pillars=mc.get('max_pillars', 12000),
            voxel_size=mc.get('voxel_size', [0.16, 0.16, 4.0]),
            point_range=mc.get('point_range', [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
        )
    return DeepFusion(
        lidar_channels=mc.get('image_features', 256),
        image_channels=mc.get('image_features', 256),
        hidden_dim=mc.get('hidden_dim', 256),
        num_heads=mc.get('n_heads', 8),
        num_layers=mc.get('n_layers', 1),
        num_classes=mc.get('num_classes', 3),
        max_objects=mc.get('max_objects_per_image', 512),
        image_backbone=mc.get('image_backbone', 'resnet34'),
        pretrained_image=mc.get('pretrained', True),
        max_points_per_pillar=mc.get('max_points_per_pillar', 100),
        max_pillars=mc.get('max_pillars', 12000),
        voxel_size=mc.get('voxel_size', [0.16, 0.16, 4.0]),
        point_range=mc.get('point_range', [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
    )