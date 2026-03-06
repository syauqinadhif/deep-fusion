"""
Detection Head for DeepFusion 3D Object Detection.
Predicts 3D bounding boxes, class labels, and confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class DetectionHead(nn.Module):
    """
    Detection Head for 3D object detection.

    Uses a center-based detection approach with heatmaps for object localization
    and regression heads for bounding box parameters.

    Args:
        in_channels: Number of input feature channels
        num_classes: Number of object classes
        max_objects: Maximum number of objects per image
        feature_channels: Number of intermediate feature channels
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        max_objects: int = 512,
        feature_channels: int = 256
    ):
        super().__init__()

        self.num_classes = num_classes
        self.max_objects = max_objects

        # Feature pyramid for multi-scale detection
        self.conv1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channels)
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels)

        # Heatmap head (for object center detection)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        )

        # Offset head (for sub-pixel center refinement)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 2, kernel_size=1)
        )

        # Size head (for object dimensions: w, l, h)
        self.size_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 3, kernel_size=1)
        )

        # Rotation head (for yaw angle)
        self.rotation_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 1, kernel_size=1)  # Single angle value
        )

        # Z-coordinate head (for height)
        self.z_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, 1, kernel_size=1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of detection head.

        Args:
            features: (B, C, H, W) input features

        Returns:
            Dictionary containing:
                - heatmap: (B, num_classes, H, W) object center heatmaps
                - offset: (B, 2, H, W) sub-pixel offsets
                - size: (B, 3, H, W) object sizes (w, l, h)
                - rotation: (B, 1, H, W) rotation angles
                - z_center: (B, 1, H, W) z-coordinates
        """
        # Process features
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Predict all outputs
        heatmap = self.heatmap_conv(x)
        offset = self.offset_conv(x)
        size = self.size_conv(x)
        rotation = self.rotation_conv(x)
        z_center = self.z_conv(x)

        return {
            'heatmap': heatmap,
            'offset': offset,
            'size': size,
            'rotation': rotation,
            'z_center': z_center
        }


class DetectionDecoder:
    """
    Decode detection head outputs into 3D bounding boxes.

    Args:
        num_classes: Number of object classes
        conf_threshold: Confidence threshold for detections
        nms_threshold: IoU threshold for NMS
        max_objects: Maximum number of objects to return
    """

    def __init__(
        self,
        num_classes: int = 3,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_objects: int = 512,
        voxel_size: List[float] = [0.16, 0.16, 4.0],
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_objects = max_objects
        self.voxel_size = voxel_size
        self.point_range = point_range

        # Calculate BEV dimensions
        self.x_min, self.y_min = point_range[0], point_range[1]
        self.voxel_x, self.voxel_y = voxel_size[0], voxel_size[1]

    def decode(
        self,
        predictions: Dict[str, torch.Tensor],
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Decode predictions into 3D bounding boxes.

        Args:
            predictions: Dictionary with model outputs
            return_scores: Whether to return confidence scores

        Returns:
            List of detections per batch element, each containing:
                - boxes: (N, 7) array [x, y, z, w, l, h, yaw]
                - labels: (N,) class labels
                - scores: (N,) confidence scores (optional)
        """
        batch_size = predictions['heatmap'].shape[0]

        all_detections = []

        for b in range(batch_size):
            # Extract predictions for this batch
            heatmap = predictions['heatmap'][b]  # (num_classes, H, W)
            offset = predictions['offset'][b]     # (2, H, W)
            size = predictions['size'][b]         # (3, H, W)
            rotation = predictions['rotation'][b] # (1, H, W)
            z_center = predictions['z_center'][b] # (1, H, W)

            # Get heatmap peaks (local maxima)
            heatmap = torch.sigmoid(heatmap)

            # Find peaks using max pooling
            padding = 1
            max_pool = F.max_pool2d(
                heatmap,
                kernel_size=2 * padding + 1,
                stride=1,
                padding=padding
            )
            peaks = (heatmap == max_pool) & (heatmap > self.conf_threshold)

            detections = self._extract_detections(
                peaks,
                heatmap,
                offset,
                size,
                rotation,
                z_center
            )

            # Apply NMS per class
            final_boxes, final_labels, final_scores = self._nms_per_class(
                detections['boxes'],
                detections['labels'],
                detections['scores']
            )

            output_dict = {
                'boxes': final_boxes,
                'labels': final_labels
            }

            if return_scores:
                output_dict['scores'] = final_scores

            all_detections.append(output_dict)

        return all_detections

    def _extract_detections(
        self,
        peaks: torch.Tensor,
        heatmap: torch.Tensor,
        offset: torch.Tensor,
        size: torch.Tensor,
        rotation: torch.Tensor,
        z_center: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract detections from heatmap peaks."""
        num_classes, H, W = heatmap.shape

        # Find all peak locations
        peak_locations = torch.nonzero(peaks)  # (N, 3) -> (class, y, x)

        if len(peak_locations) == 0:
            return {
                'boxes': torch.zeros((0, 7), device=heatmap.device),
                'labels': torch.zeros((0,), dtype=torch.long, device=heatmap.device),
                'scores': torch.zeros((0,), device=heatmap.device)
            }

        boxes_list = []
        labels_list = []
        scores_list = []

        for loc in peak_locations:
            cls, y, x = loc

            # Get confidence score
            score = heatmap[cls, y, x].item()

            # Apply offset for sub-pixel accuracy
            x_offset = offset[0, y, x].item()
            y_offset = offset[1, y, x].item()
            x_center = x.item() + x_offset
            y_center = y.item() + y_offset

            # Convert to world coordinates
            world_x = self.x_min + x_center * self.voxel_x
            world_y = self.y_min + y_center * self.voxel_y
            world_z = z_center[0, y, x].item()

            # Get size
            w = size[0, y, x].item()
            l = size[1, y, x].item()
            h = size[2, y, x].item()

            # Get rotation
            yaw = rotation[0, y, x].item()

            # Create box
            box = torch.tensor([world_x, world_y, world_z, w, l, h, yaw])

            boxes_list.append(box)
            labels_list.append(cls.item())
            scores_list.append(score)

        return {
            'boxes': torch.stack(boxes_list),
            'labels': torch.tensor(labels_list, dtype=torch.long),
            'scores': torch.tensor(scores_list)
        }

    def _nms_per_class(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Non-Maximum Suppression per class."""
        if len(boxes) == 0:
            return boxes, labels, scores

        keep_boxes = []
        keep_labels = []
        keep_scores = []

        for cls in range(self.num_classes):
            cls_mask = labels == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            # Sort by score (descending)
            sorted_indices = torch.argsort(cls_scores, descending=True)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]

            # NMS using BEV IoU
            keep = self._bev_nms(cls_boxes, cls_scores, self.nms_threshold)

            keep_boxes.append(cls_boxes[keep])
            keep_labels.append(torch.full((len(keep),), cls, dtype=torch.long))
            keep_scores.append(cls_scores[keep])

        # Concatenate
        final_boxes = torch.cat(keep_boxes) if keep_boxes else boxes[:0]
        final_labels = torch.cat(keep_labels) if keep_labels else labels[:0]
        final_scores = torch.cat(keep_scores) if keep_scores else scores[:0]

        return final_boxes, final_labels, final_scores

    def _bev_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float
    ) -> torch.Tensor:
        """BEV NMS using 2D IoU."""
        if len(boxes) == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)

        keep = []
        while len(sorted_indices) > 0:
            # Keep the box with highest score
            current = sorted_indices[0].item()
            keep.append(current)

            if len(sorted_indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current].unsqueeze(0)
            remaining_boxes = boxes[sorted_indices[1:]]
            ious = self._compute_bev_iou(current_box, remaining_boxes)

            # Keep boxes with IoU below threshold
            mask = ious.squeeze(0) < iou_threshold
            sorted_indices = sorted_indices[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _compute_bev_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute BEV IoU between two sets of boxes."""
        # Extract 2D bounding boxes (x, y, w, l)
        boxes1_bev = boxes1[:, [0, 1, 3, 4]]  # (N, 4)
        boxes2_bev = boxes2[:, [0, 1, 3, 4]]  # (M, 4)

        # Compute areas
        area1 = boxes1_bev[:, 2] * boxes1_bev[:, 3]  # (N,)
        area2 = boxes2_bev[:, 2] * boxes2_bev[:, 3]  # (M,)

        # Expand for broadcasting
        boxes1_bev = boxes1_bev.unsqueeze(1)  # (N, 1, 4)
        boxes2_bev = boxes2_bev.unsqueeze(0)  # (1, M, 4)

        # Get coordinates
        x1 = boxes1_bev[..., 0]
        y1 = boxes1_bev[..., 1]
        w1 = boxes1_bev[..., 2]
        l1 = boxes1_bev[..., 3]

        x2 = boxes2_bev[..., 0]
        y2 = boxes2_bev[..., 1]
        w2 = boxes2_bev[..., 2]
        l2 = boxes2_bev[..., 3]

        # Compute intersection
        x_left = torch.max(x1 - w1/2, x2 - w2/2)
        x_right = torch.min(x1 + w1/2, x2 + w2/2)
        y_top = torch.max(y1 - l1/2, y2 - l2/2)
        y_bottom = torch.min(y1 + l1/2, y2 + l2/2)

        intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)

        # Compute union
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        return iou


class ObjectDetectionLoss(nn.Module):
    """
    Loss function for 3D object detection.

    Args:
        num_classes: Number of object classes
        alpha: Focal loss alpha parameter
        beta: Focal loss beta parameter
        gamma: Focal loss gamma parameter
    """

    def __init__(
        self,
        num_classes: int = 3,
        alpha: float = 0.25,
        beta: float = 2.0,
        gamma: float = 2.0,
        offset_weight: float = 1.0,
        size_weight: float = 1.0,
        rotation_weight: float = 1.0,
        z_weight: float = 1.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.offset_weight = offset_weight
        self.size_weight = size_weight
        self.rotation_weight = rotation_weight
        self.z_weight = z_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute detection loss.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth

        Returns:
            total_loss: Total loss
            loss_dict: Dictionary with individual losses
        """
        # Heatmap loss (focal loss)
        heatmap_loss = self._focal_loss(
            predictions['heatmap'],
            targets['heatmap']
        )

        # Offset loss (L1)
        offset_loss = self._offset_loss(
            predictions['offset'],
            targets['offset'],
            targets['heatmap'] > 0
        )

        # Size loss (L1)
        size_loss = self._size_loss(
            predictions['size'],
            targets['size'],
            targets['heatmap'] > 0
        )

        # Rotation loss (smooth L1)
        rotation_loss = self._rotation_loss(
            predictions['rotation'],
            targets['rotation'],
            targets['heatmap'] > 0
        )

        # Z-coordinate loss (smooth L1)
        z_loss = self._z_loss(
            predictions['z_center'],
            targets['z_center'],
            targets['heatmap'] > 0
        )

        # Total loss
        total_loss = (
            heatmap_loss +
            self.offset_weight * offset_loss +
            self.size_weight * size_loss +
            self.rotation_weight * rotation_loss +
            self.z_weight * z_loss
        )

        loss_dict = {
            'total': total_loss,
            'heatmap': heatmap_loss,
            'offset': offset_loss,
            'size': size_loss,
            'rotation': rotation_loss,
            'z': z_loss
        }

        return total_loss, loss_dict

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss for heatmap."""
        pred = torch.clamp(pred, min=-10, max=10)  # For numerical stability

        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        pred_sigmoid = torch.sigmoid(pred)

        # Focal loss
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        focal_weight = (1 - pt).pow(gamma)

        loss = focal_weight * (
            -target * torch.log(pred_sigmoid + 1e-6) * alpha -
            beta * (1 - target) * torch.log(1 - pred_sigmoid + 1e-6) * (1 - alpha)
        )

        return loss.mean()

    def _offset_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 loss for offset."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[mask]
        target = target[mask]

        return F.l1_loss(pred, target)

    def _size_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """L1 loss for size."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[mask]
        target = target[mask]

        return F.l1_loss(pred, target)

    def _rotation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Smooth L1 loss for rotation."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[mask]
        target = target[mask]

        return F.smooth_l1_loss(pred, target, beta=0.1)

    def _z_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Smooth L1 loss for z-coordinate."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred = pred[mask]
        target = target[mask]

        return F.smooth_l1_loss(pred, target, beta=0.1)


if __name__ == "__main__":
    # Test the detection head
    batch_size = 2
    in_channels = 256
    height = 128
    width = 128

    # Create dummy features
    features = torch.randn(batch_size, in_channels, height, width)

    # Create detection head
    det_head = DetectionHead(
        in_channels=in_channels,
        num_classes=3,
        max_objects=512
    )

    # Forward pass
    predictions = det_head(features)

    print("Detection Head Outputs:")
    for name, pred in predictions.items():
        print(f"  {name}: {pred.shape}")

    # Test loss
    targets = {
        'heatmap': torch.zeros(batch_size, 3, height, width).random_(0, 2),
        'offset': torch.randn(batch_size, 2, height, width),
        'size': torch.randn(batch_size, 3, height, width),
        'rotation': torch.randn(batch_size, 1, height, width),
        'z_center': torch.randn(batch_size, 1, height, width)
    }

    loss_fn = ObjectDetectionLoss(num_classes=3)
    total_loss, loss_dict = loss_fn(predictions, targets)

    print(f"\nLosses:")
    for name, loss in loss_dict.items():
        print(f"  {name}: {loss.item():.4f}")

    print(f"\nTotal parameters: {sum(p.numel() for p in det_head.parameters()):,}")
