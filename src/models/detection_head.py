"""
Detection Head for DeepFusion 3D Object Detection.
FIXED VERSION — perbaikan IndexError pada loss functions:
  mask dari heatmap shape (B, num_classes, H, W) tidak bisa langsung
  dipakai untuk index tensor dengan channel berbeda (2, 3, 1).
  Fix: collapse mask ke spatial (B, H, W) dengan .any(dim=1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class DetectionHead(nn.Module):
    """
    Detection Head for 3D object detection (CenterPoint-style).

    Args:
        in_channels:      Number of input feature channels
        num_classes:      Number of object classes
        max_objects:      Maximum number of objects per image
        feature_channels: Number of intermediate feature channels
    """

    def __init__(
        self,
        in_channels:      int = 256,
        num_classes:      int = 3,
        max_objects:      int = 512,
        feature_channels: int = 256
    ):
        super().__init__()

        self.num_classes = num_classes
        self.max_objects = max_objects

        self.conv1 = nn.Conv2d(in_channels,      feature_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(feature_channels)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(feature_channels)
        self.relu  = nn.ReLU(inplace=True)

        # Heatmap: (B, num_classes, H, W)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        )

        # Offset: (B, 2, H, W)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 2, kernel_size=1)
        )

        # Size: (B, 3, H, W)  — w, l, h
        self.size_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 3, kernel_size=1)
        )

        # Rotation: (B, 1, H, W)
        self.rotation_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 1, kernel_size=1)
        )

        # Z-coordinate: (B, 1, H, W)
        self.z_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 1, kernel_size=1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(features)))
        x = self.relu(self.bn2(self.conv2(x)))

        return {
            'heatmap':  self.heatmap_conv(x),   # (B, num_classes, H, W)
            'offset':   self.offset_conv(x),    # (B, 2, H, W)
            'size':     self.size_conv(x),       # (B, 3, H, W)
            'rotation': self.rotation_conv(x),  # (B, 1, H, W)
            'z_center': self.z_conv(x)           # (B, 1, H, W)
        }


class ObjectDetectionLoss(nn.Module):
    """Loss function for CenterPoint-style 3D detection."""

    def __init__(
        self,
        num_classes:     int   = 3,
        alpha:           float = 0.25,
        beta:            float = 2.0,
        gamma:           float = 2.0,
        offset_weight:   float = 1.0,
        size_weight:     float = 1.0,
        rotation_weight: float = 1.0,
        z_weight:        float = 1.0
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.alpha           = alpha
        self.beta            = beta
        self.gamma           = gamma
        self.offset_weight   = offset_weight
        self.size_weight     = size_weight
        self.rotation_weight = rotation_weight
        self.z_weight        = z_weight

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _spatial_mask(heatmap_target: torch.Tensor) -> torch.Tensor:
        """
        Buat spatial mask (B, H, W) dari heatmap target (B, num_classes, H, W).

        True di posisi mana saja yang ada objek di setidaknya satu class.
        Ini digunakan untuk mask pada regression heads yang channel-nya ≠ num_classes.
        """
        return heatmap_target.gt(0).any(dim=1)   # (B, H, W)

    # ── loss components ───────────────────────────────────────────────────────

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Modified focal loss untuk heatmap."""
        pred_s = torch.sigmoid(torch.clamp(pred, -10, 10))

        # Posisi objek (target ≈ 1) → focal weight = (1 - p)^gamma
        # Posisi background (target < 1) → down-weighted oleh (1 - target)^beta
        pos_mask = target.eq(1).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = torch.log(pred_s + 1e-6) * (1 - pred_s).pow(self.gamma) * pos_mask
        neg_loss = (torch.log(1 - pred_s + 1e-6)
                    * pred_s.pow(self.gamma)
                    * (1 - target).pow(self.beta)
                    * neg_mask)

        num_pos = pos_mask.sum().clamp(min=1)
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss

    def _regression_loss(
        self,
        pred:       torch.Tensor,   # (B, C_pred, H, W)
        target:     torch.Tensor,   # (B, C_pred, H, W)
        mask:       torch.Tensor,   # (B, H, W)  ← spatial mask
        loss_type:  str = 'l1'
    ) -> torch.Tensor:
        """
        Generic regression loss dengan spatial mask.

        FIX: mask shape (B, H, W) di-expand ke (B, C_pred, H, W)
        agar kompatibel dengan tensor pred/target yang C-nya bervariasi.
        """
        num_pos = mask.sum()
        if num_pos == 0:
            return pred.sum() * 0.0   # gradient tetap ada, nilai 0

        # Expand mask: (B, H, W) → (B, 1, H, W) → (B, C_pred, H, W)
        C = pred.shape[1]
        mask_expanded = mask.unsqueeze(1).expand_as(pred)   # (B, C_pred, H, W)

        pred_masked   = pred[mask_expanded]     # (num_pos * C,)
        target_masked = target[mask_expanded]   # (num_pos * C,)

        if loss_type == 'l1':
            return F.l1_loss(pred_masked, target_masked)
        elif loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred_masked, target_masked, beta=0.1)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: output dari DetectionHead.forward()
            targets:     dict dari KITTIDataset._create_targets()
                         keys: heatmap, offset, size, rotation, z_center

        Returns:
            total_loss, loss_dict
        """
        # ── spatial mask dari heatmap target ────────────────────────────────
        # shape: (B, H, W) — True di mana ada objek (any class)
        spatial_mask = self._spatial_mask(targets['heatmap'])   # (B, H, W)

        # ── heatmap loss (bekerja pada (B, num_classes, H, W)) ───────────────
        heatmap_loss = self._focal_loss(
            predictions['heatmap'],
            targets['heatmap']
        )

        # ── regression losses — semua pakai spatial_mask (B, H, W) ──────────
        offset_loss = self._regression_loss(
            predictions['offset'],     # (B, 2, H, W)
            targets['offset'],         # (B, 2, H, W)
            spatial_mask,
            loss_type='l1'
        )

        size_loss = self._regression_loss(
            predictions['size'],       # (B, 3, H, W)
            targets['size'],           # (B, 3, H, W)
            spatial_mask,
            loss_type='l1'
        )

        rotation_loss = self._regression_loss(
            predictions['rotation'],   # (B, 1, H, W)
            targets['rotation'],       # (B, 1, H, W)
            spatial_mask,
            loss_type='smooth_l1'
        )

        z_loss = self._regression_loss(
            predictions['z_center'],   # (B, 1, H, W)
            targets['z_center'],       # (B, 1, H, W)
            spatial_mask,
            loss_type='smooth_l1'
        )

        # ── total ────────────────────────────────────────────────────────────
        total_loss = (
            heatmap_loss
            + self.offset_weight   * offset_loss
            + self.size_weight     * size_loss
            + self.rotation_weight * rotation_loss
            + self.z_weight        * z_loss
        )

        loss_dict = {
            'total':    total_loss,
            'heatmap':  heatmap_loss,
            'offset':   offset_loss,
            'size':     size_loss,
            'rotation': rotation_loss,
            'z':        z_loss
        }

        return total_loss, loss_dict


class DetectionDecoder:
    """Decode DetectionHead outputs ke 3D bounding boxes."""

    def __init__(
        self,
        num_classes:    int   = 3,
        conf_threshold: float = 0.3,
        nms_threshold:  float = 0.5,
        max_objects:    int   = 512,
        voxel_size:     List[float] = [0.16, 0.16, 4.0],
        point_range:    List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        self.num_classes    = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold
        self.max_objects    = max_objects
        self.x_min          = point_range[0]
        self.y_min          = point_range[1]
        self.voxel_x        = voxel_size[0]
        self.voxel_y        = voxel_size[1]

    def decode(
        self,
        predictions: Dict[str, torch.Tensor],
        return_scores: bool = True
    ) -> List[Dict]:
        B = predictions['heatmap'].shape[0]
        all_dets = []

        for b in range(B):
            hm  = torch.sigmoid(predictions['heatmap'][b])   # (C, H, W)
            off = predictions['offset'][b]
            sz  = predictions['size'][b]
            rot = predictions['rotation'][b]
            z   = predictions['z_center'][b]

            # Peak extraction via max-pool
            pad  = 1
            maxp = F.max_pool2d(hm, kernel_size=2*pad+1, stride=1, padding=pad)
            peaks = (hm == maxp) & (hm > self.conf_threshold)

            dets = self._extract(peaks, hm, off, sz, rot, z)
            boxes, labels, scores = self._nms(dets['boxes'], dets['labels'], dets['scores'])

            entry = {'boxes': boxes, 'labels': labels}
            if return_scores:
                entry['scores'] = scores
            all_dets.append(entry)

        return all_dets

    def _extract(self, peaks, hm, offset, size, rotation, z_center):
        locs = torch.nonzero(peaks)   # (N, 3): cls, y, x
        if len(locs) == 0:
            dev = hm.device
            return {
                'boxes':  torch.zeros((0, 7), device=dev),
                'labels': torch.zeros((0,), dtype=torch.long, device=dev),
                'scores': torch.zeros((0,), device=dev)
            }

        boxes, labels, scores = [], [], []
        for loc in locs:
            cls, y, x = loc
            score = hm[cls, y, x].item()
            xc = x.item() + offset[0, y, x].item()
            yc = y.item() + offset[1, y, x].item()
            wx = self.x_min + xc * self.voxel_x
            wy = self.y_min + yc * self.voxel_y
            wz = z_center[0, y, x].item()
            w, l, h = size[0, y, x].item(), size[1, y, x].item(), size[2, y, x].item()
            yaw = rotation[0, y, x].item()
            boxes.append(torch.tensor([wx, wy, wz, w, l, h, yaw]))
            labels.append(cls.item())
            scores.append(score)

        return {
            'boxes':  torch.stack(boxes),
            'labels': torch.tensor(labels, dtype=torch.long),
            'scores': torch.tensor(scores)
        }

    def _nms(self, boxes, labels, scores):
        if len(boxes) == 0:
            return boxes, labels, scores

        keep_b, keep_l, keep_s = [], [], []
        for cls in range(self.num_classes):
            m = labels == cls
            if m.sum() == 0:
                continue
            cb, cs = boxes[m], scores[m]
            idx = torch.argsort(cs, descending=True)
            cb, cs = cb[idx], cs[idx]
            keep = self._bev_nms(cb, cs, self.nms_threshold)
            keep_b.append(cb[keep])
            keep_l.append(torch.full((len(keep),), cls, dtype=torch.long))
            keep_s.append(cs[keep])

        if not keep_b:
            return boxes[:0], labels[:0], scores[:0]
        return torch.cat(keep_b), torch.cat(keep_l), torch.cat(keep_s)

    def _bev_nms(self, boxes, scores, thr):
        idx = torch.argsort(scores, descending=True)
        keep = []
        while len(idx) > 0:
            i = idx[0].item()
            keep.append(i)
            if len(idx) == 1:
                break
            ious = self._bev_iou(boxes[i].unsqueeze(0), boxes[idx[1:]])
            idx  = idx[1:][ious.squeeze(0) < thr]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _bev_iou(self, b1, b2):
        def corners(b):
            x, y, w, l = b[..., 0], b[..., 1], b[..., 3], b[..., 4]
            return x - w/2, x + w/2, y - l/2, y + l/2

        x1a, x1b, y1a, y1b = corners(b1.unsqueeze(1))
        x2a, x2b, y2a, y2b = corners(b2.unsqueeze(0))
        xi = torch.clamp(torch.min(x1b, x2b) - torch.max(x1a, x2a), min=0)
        yi = torch.clamp(torch.min(y1b, y2b) - torch.max(y1a, y2a), min=0)
        inter = xi * yi
        a1 = b1[..., 3] * b1[..., 4]
        a2 = b2[..., 3] * b2[..., 4]
        union = a1.unsqueeze(1) + a2.unsqueeze(0) - inter
        return inter / (union + 1e-6)


# ── sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    B, C, H, W = 2, 256, 252, 252
    feats = torch.randn(B, C, H, W)
    head  = DetectionHead(in_channels=C, num_classes=3)
    preds = head(feats)
    print("Predictions:")
    for k, v in preds.items():
        print(f"  {k}: {v.shape}")

    targets = {
        'heatmap':  torch.zeros(B, 3, H, W).random_(0, 2).float(),
        'offset':   torch.randn(B, 2, H, W),
        'size':     torch.randn(B, 3, H, W),
        'rotation': torch.randn(B, 1, H, W),
        'z_center': torch.randn(B, 1, H, W)
    }

    loss_fn             = ObjectDetectionLoss(num_classes=3)
    total_loss, loss_d  = loss_fn(preds, targets)
    print("\nLosses:")
    for k, v in loss_d.items():
        print(f"  {k}: {v.item():.4f}")
    print("✓ No shape errors")