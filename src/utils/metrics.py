"""
Evaluation metrics for 3D object detection.
Implements KITTI-style evaluation metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict


class DetectionMetrics:
    """
    Compute detection metrics (AP, Precision, Recall, F1).

    Implements KITTI-style evaluation with IoU thresholds.
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_names: List[str] = None,
        iou_thresholds: List[float] = None,
        difficulty_levels: List[str] = None
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.iou_thresholds = iou_thresholds or [0.5, 0.7]
        self.difficulty_levels = difficulty_levels or ['easy', 'moderate', 'hard']

        # Reset metrics
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)

    def update(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        difficulty: str = 'moderate'
    ):
        """
        Update metrics with new predictions and ground truths.

        Args:
            predictions: List of predictions per image
                Each prediction dict has:
                    - boxes: (N, 7) array [x, y, z, w, l, h, yaw]
                    - labels: (N,) class labels
                    - scores: (N,) confidence scores
            ground_truths: List of ground truths per image
                Each GT dict has:
                    - boxes: (M, 7) array
                    - labels: (M,) class labels
                    - difficulty: (M,) difficulty levels
        """
        for pred, gt in zip(predictions, ground_truths):
            for cls in range(self.num_classes):
                # Filter by class
                cls_mask = gt['labels'] == cls
                cls_gt_boxes = gt['boxes'][cls_mask]
                cls_gt_difficulty = gt['difficulty'][cls_mask] if 'difficulty' in gt else None

                # Filter by difficulty level
                if cls_gt_difficulty is not None:
                    diff_indices = {
                        'easy': 0,
                        'moderate': 1,
                        'hard': 2
                    }
                    diff_mask = cls_gt_difficulty <= diff_indices.get(difficulty, 1)
                    cls_gt_boxes = cls_gt_boxes[diff_mask]

                self.ground_truths[(cls, difficulty)].append(cls_gt_boxes)

                # Filter predictions by class
                cls_pred_mask = pred['labels'] == cls
                cls_pred_boxes = pred['boxes'][cls_pred_mask]
                cls_pred_scores = pred['scores'][cls_pred_mask]

                self.predictions[(cls, difficulty)].append({
                    'boxes': cls_pred_boxes,
                    'scores': cls_pred_scores
                })

    def compute(self) -> Dict[str, float]:
        """
        Compute average precision and other metrics.

        Returns:
            Dictionary with metrics per class and difficulty
        """
        results = {}

        for cls in range(self.num_classes):
            for difficulty in self.difficulty_levels:
                key = f'{self.class_names[cls]}_{difficulty}'

                # Get predictions and ground truths
                preds = self.predictions.get((cls, difficulty), [])
                gts = self.ground_truths.get((cls, difficulty), [])

                if len(gts) == 0:
                    results[key] = {
                        'ap': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0
                    }
                    continue

                # Compute AP for each IoU threshold
                aps = []
                for iou_thresh in self.iou_thresholds:
                    ap = self._compute_ap(preds, gts, iou_thresh)
                    aps.append(ap)

                # Use AP@0.7 for KITTI
                ap_70 = aps[1] if len(aps) > 1 else aps[0]

                # Compute precision/recall at optimal threshold
                precision, recall, f1 = self._compute_precision_recall(preds, gts, 0.7)

                results[key] = {
                    'ap': ap_70,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

        # Compute mean metrics
        mean_ap = np.mean([results[key]['ap'] for key in results.keys()])
        mean_precision = np.mean([results[key]['precision'] for key in results.keys()])
        mean_recall = np.mean([results[key]['recall'] for key in results.keys()])
        mean_f1 = np.mean([results[key]['f1'] for key in results.keys()])

        results['mean'] = {
            'ap': mean_ap,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1': mean_f1
        }

        return results

    def _compute_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[np.ndarray],
        iou_threshold: float
    ) -> float:
        """Compute Average Precision for a single IoU threshold."""
        # Collect all predictions and ground truths
        all_preds = []
        all_gts = []

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            for j, box in enumerate(pred['boxes']):
                all_preds.append({
                    'image_id': i,
                    'box': box,
                    'score': pred['scores'][j]
                })
            for j, box in enumerate(gt):
                all_gts.append({
                    'image_id': i,
                    'box': box,
                    'detected': False
                })

        if len(all_gts) == 0:
            return 0.0

        # Sort predictions by score (descending)
        all_preds.sort(key=lambda x: x['score'], reverse=True)

        # Compute true positives and false positives
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for i, pred in enumerate(all_preds):
            # Find ground truth boxes in same image
            gt_boxes = [gt for gt in all_gts if gt['image_id'] == pred['image_id']]

            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            # Compute IoU with all ground truths
            max_iou = 0
            max_gt_idx = -1

            for j, gt in enumerate(gt_boxes):
                iou = self._compute_3d_iou(pred['box'], gt['box'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j

            # Check if detection is correct
            if max_iou >= iou_threshold:
                if not gt_boxes[max_gt_idx]['detected']:
                    tp[i] = 1
                    gt_boxes[max_gt_idx]['detected'] = True
                else:
                    fp[i] = 1  # Duplicate detection
            else:
                fp[i] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(all_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        return ap

    def _compute_precision_recall(
        self,
        predictions: List[Dict],
        ground_truths: List[np.ndarray],
        iou_threshold: float
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 at optimal threshold."""
        results = []

        for iou_thresh in [iou_threshold]:
            for conf_thresh in [0.1, 0.3, 0.5, 0.7]:
                tp = 0
                fp = 0
                fn = 0

                for pred_dict, gt_boxes in zip(predictions, ground_truths):
                    # Filter by confidence
                    conf_mask = pred_dict['scores'] >= conf_thresh
                    pred_boxes = pred_dict['boxes'][conf_mask]

                    # Match predictions to ground truth
                    detected = [False] * len(gt_boxes)

                    for pred_box in pred_boxes:
                        max_iou = 0
                        max_gt_idx = -1

                        for gt_idx, gt_box in enumerate(gt_boxes):
                            iou = self._compute_3d_iou(pred_box, gt_box)
                            if iou > max_iou:
                                max_iou = iou
                                max_gt_idx = gt_idx

                        if max_iou >= iou_thresh:
                            if not detected[max_gt_idx]:
                                tp += 1
                                detected[max_gt_idx] = True
                            else:
                                fp += 1
                        else:
                            fp += 1

                    fn += sum(not d for d in detected)

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

                results.append((precision, recall, f1))

        # Return best F1 score
        if not results:
            return 0.0, 0.0, 0.0

        best = max(results, key=lambda x: x[2])
        return best

    def _compute_3d_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """
        Compute 3D IoU between two bounding boxes.

        Args:
            box1: (7,) array [x, y, z, w, l, h, yaw]
            box2: (7,) array [x, y, z, w, l, h, yaw]

        Returns:
            3D IoU value
        """
        # For simplicity, compute BEV IoU
        return self._compute_bev_iou(box1, box2)

    def _compute_bev_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """Compute Bird's Eye View IoU."""
        # Extract 2D box parameters
        x1, y1, w1, l1, yaw1 = box1[0], box1[1], box1[3], box1[4], box1[6]
        x2, y2, w2, l2, yaw2 = box2[0], box2[1], box2[3], box2[4], box2[6]

        # Get corners of both boxes
        corners1 = self._get_bev_corners(x1, y1, w1, l1, yaw1)
        corners2 = self._get_bev_corners(x2, y2, w2, l2, yaw2)

        # Compute intersection area
        intersection = self._polygon_intersection(corners1, corners2)

        # Compute areas
        area1 = w1 * l1
        area2 = w2 * l2

        # Compute union
        union = area1 + area2 - intersection

        # Compute IoU
        iou = intersection / (union + 1e-6)

        return iou

    def _get_bev_corners(
        self,
        x: float,
        y: float,
        w: float,
        l: float,
        yaw: float
    ) -> np.ndarray:
        """Get BEV corner points of a rotated box."""
        # Direction vector
        dx = w / 2
        dy = l / 2

        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Local corners
        local_corners = np.array([
            [-dx, -dy],
            [+dx, -dy],
            [+dx, +dy],
            [-dx, +dy]
        ])

        # Rotate and translate
        corners = np.zeros((4, 2))
        for i in range(4):
            corners[i, 0] = x + local_corners[i, 0] * cos_yaw - local_corners[i, 1] * sin_yaw
            corners[i, 1] = y + local_corners[i, 0] * sin_yaw + local_corners[i, 1] * cos_yaw

        return corners

    def _polygon_intersection(
        self,
        poly1: np.ndarray,
        poly2: np.ndarray
    ) -> float:
        """Compute intersection area of two polygons."""
        try:
            from shapely.geometry import Polygon
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            intersection = p1.intersection(p2).area
            return intersection
        except ImportError:
            # Fallback: approximate with axis-aligned boxes
            x_min = max(np.min(poly1[:, 0]), np.min(poly2[:, 0]))
            x_max = min(np.max(poly1[:, 0]), np.max(poly2[:, 0]))
            y_min = max(np.min(poly1[:, 1]), np.min(poly2[:, 1]))
            y_max = min(np.max(poly1[:, 1]), np.max(poly2[:, 1]))

            if x_max < x_min or y_max < y_min:
                return 0.0

            return (x_max - x_min) * (y_max - y_min)


class LossTracker:
    """
    Track and compute loss statistics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.total_loss = 0.0
        self.losses = defaultdict(float)
        self.count = 0

    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """Update with new loss values."""
        self.count += 1

        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.losses[key] += value.item()
            else:
                self.losses[key] += value

        if 'total' in loss_dict:
            if isinstance(loss_dict['total'], torch.Tensor):
                self.total_loss += loss_dict['total'].item()
            else:
                self.total_loss += loss_dict['total']

    def get_average(self) -> Dict[str, float]:
        """Get average losses."""
        result = {}
        for key, value in self.losses.items():
            result[key] = value / self.count if self.count > 0 else 0.0
        return result

    def get_total_average(self) -> float:
        """Get average total loss."""
        return self.total_loss / self.count if self.count > 0 else 0.0


if __name__ == "__main__":
    # Test detection metrics
    print("Testing DetectionMetrics...")

    metrics = DetectionMetrics(
        num_classes=3,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        iou_thresholds=[0.5, 0.7],
        difficulty_levels=['easy', 'moderate', 'hard']
    )

    # Create dummy predictions and ground truths
    for i in range(10):
        pred = {
            'boxes': np.random.rand(5, 7) * 10,
            'labels': np.random.randint(0, 3, 5),
            'scores': np.random.rand(5)
        }
        gt = {
            'boxes': np.random.rand(3, 7) * 10,
            'labels': np.random.randint(0, 3, 3),
            'difficulty': np.random.randint(0, 3, 3)
        }
        metrics.update([pred], [gt])

    results = metrics.compute()

    print("\nDetection Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for metric_name, metric_value in value.items():
                print(f"  {metric_name}: {metric_value:.4f}")
