"""
Visualization utilities for DeepFusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
from typing import Dict, List, Tuple, Optional
import torch


class Visualizer:
    """
    Visualize 3D detection results.
    """

    def __init__(
        self,
        class_names: List[str] = None,
        colors: List[Tuple[int, int, int]] = None
    ):
        self.class_names = class_names or ['Car', 'Pedestrian', 'Cyclist']

        # Default colors (BGR format for OpenCV)
        self.colors = colors or [
            (0, 255, 0),    # Green - Car
            (255, 0, 0),    # Blue - Pedestrian
            (0, 0, 255)     # Red - Cyclist
        ]

    def visualize_bev(
        self,
        points: np.ndarray,
        boxes: np.ndarray = None,
        labels: np.ndarray = None,
        scores: np.ndarray = None,
        point_range: List[float] = [-40.0, -40.0, 40.0, 40.0],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize Bird's Eye View (BEV) of point cloud and detections.

        Args:
            points: (N, 4) point cloud [x, y, z, intensity]
            boxes: (M, 7) detected boxes [x, y, z, w, l, h, yaw]
            labels: (M,) class labels
            scores: (M,) confidence scores
            point_range: [x_min, y_min, x_max, y_max]
            save_path: Path to save visualization (optional)

        Returns:
            BEV image (H, W, 3)
        """
        # Create image
        x_min, y_min, x_max, y_max = point_range
        img_size = 800
        scale = img_size / (x_max - x_min)

        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Transform points to image coordinates
        x = points[:, 0]
        y = points[:, 1]
        intensity = points[:, 3]

        # Map to image coordinates
        x_img = ((x - x_min) * scale).astype(int)
        y_img = ((y - y_min) * scale).astype(int)

        # Clip to image bounds
        valid = (x_img >= 0) & (x_img < img_size) & (y_img >= 0) & (y_img < img_size)
        x_img = x_img[valid]
        y_img = y_img[valid]
        intensity = intensity[valid]

        # Draw points (colored by intensity)
        for i in range(len(x_img)):
            color_val = int(intensity[i] * 255)
            color = (color_val, color_val, color_val)  # Grayscale
            cv2.circle(img, (x_img[i], y_img[i]), 1, color, -1)

        # Draw bounding boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                self._draw_bev_box(img, box, point_range, scale,
                                 labels[i] if labels is not None else 0,
                                 scores[i] if scores is not None else None)

        # Add legend
        self._add_legend(img)

        # Save if requested
        if save_path is not None:
            cv2.imwrite(save_path, img)

        return img

    def _draw_bev_box(
        self,
        img: np.ndarray,
        box: np.ndarray,
        point_range: List[float],
        scale: float,
        label: int,
        score: Optional[float] = None
    ):
        """Draw a single bounding box in BEV."""
        x, y, z, w, l, h, yaw = box
        x_min, y_min, x_max, y_max = point_range

        # Get box corners
        corners = self._get_bev_corners(x, y, w, l, yaw)

        # Transform to image coordinates
        corners_img = []
        for corner in corners:
            cx = int((corner[0] - x_min) * scale)
            cy = int((corner[1] - y_min) * scale)
            corners_img.append((cx, cy))

        # Draw box
        color = self.colors[label % len(self.colors)]
        for i in range(4):
            pt1 = corners_img[i]
            pt2 = corners_img[(i + 1) % 4]
            cv2.line(img, pt1, pt2, color, 2)

        # Draw center and label
        cx = int((x - x_min) * scale)
        cy = int((y - y_min) * scale)

        label_text = self.class_names[label % len(self.class_names)]
        if score is not None:
            label_text += f' {score:.2f}'

        cv2.putText(img, label_text, (cx - 20, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _get_bev_corners(
        self,
        x: float,
        y: float,
        w: float,
        l: float,
        yaw: float
    ) -> np.ndarray:
        """Get BEV corner points of a rotated box."""
        dx = w / 2
        dy = l / 2

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        local_corners = np.array([
            [-dx, -dy],
            [+dx, -dy],
            [+dx, +dy],
            [-dx, +dy]
        ])

        corners = np.zeros((4, 2))
        for i in range(4):
            corners[i, 0] = x + local_corners[i, 0] * cos_yaw - local_corners[i, 1] * sin_yaw
            corners[i, 1] = y + local_corners[i, 0] * sin_yaw + local_corners[i, 1] * cos_yaw

        return corners

    def _add_legend(self, img: np.ndarray):
        """Add class legend to image."""
        y_offset = 20
        for i, class_name in enumerate(self.class_names):
            color = self.colors[i % len(self.colors)]
            # Draw color box
            cv2.rectangle(img, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            # Draw text
            cv2.putText(img, class_name, (35, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 25

    def visualize_3d(
        self,
        points: np.ndarray,
        boxes: np.ndarray = None,
        labels: np.ndarray = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize 3D point cloud and detections using matplotlib.

        Args:
            points: (N, 4) point cloud
            boxes: (M, 7) detected boxes
            labels: (M,) class labels
            save_path: Path to save visualization
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=points[:, 3], cmap='viridis', s=0.1, alpha=0.5)

        # Plot boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                self._draw_3d_box(ax, box,
                                 labels[i] if labels is not None else 0)

        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([-40, 40])
        ax.set_ylim([-40, 40])
        ax.set_zlim([-3, 3])

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=100)
            plt.close()
        else:
            plt.show()

    def _draw_3d_box(self, ax, box: np.ndarray, label: int):
        """Draw a 3D bounding box."""
        x, y, z, w, l, h, yaw = box

        # Create 3D box corners
        dx = w / 2
        dy = l / 2
        dz = h / 2

        # Local corners
        local_corners = np.array([
            [-dx, -dy, -dz], [+dx, -dy, -dz], [+dx, +dy, -dz], [-dx, +dy, -dz],
            [-dx, -dy, +dz], [+dx, -dy, +dz], [+dx, +dy, +dz], [-dx, +dy, +dz]
        ])

        # Rotate
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        corners = np.zeros((8, 3))
        for i in range(8):
            corners[i, 0] = x + local_corners[i, 0] * cos_yaw - local_corners[i, 1] * sin_yaw
            corners[i, 1] = y + local_corners[i, 0] * sin_yaw + local_corners[i, 1] * cos_yaw
            corners[i, 2] = z + local_corners[i, 2]

        # Draw edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]

        color = plt.cm.tab10(label)

        for edge in edges:
            points = corners[list(edge)]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1)

    def visualize_image_with_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray = None,
        labels: np.ndarray = None,
        scores: np.ndarray = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize 2D image with projected 3D detections.

        Args:
            image: (H, W, 3) RGB image
            boxes: (M, 7) 3D boxes
            labels: (M,) class labels
            scores: (M,) confidence scores
            save_path: Path to save visualization

        Returns:
            Image with drawn detections
        """
        img = image.copy()

        if boxes is not None:
            for i, box in enumerate(boxes):
                # For simplicity, just draw label and score
                # In practice, you'd project the 3D box to 2D image plane
                label = labels[i] if labels is not None else 0
                score = scores[i] if scores is not None else 0.0

                class_name = self.class_names[label % len(self.class_names)]
                text = f'{class_name}: {score:.2f}'

                # Draw text at top-left
                cv2.putText(img, text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if save_path is not None:
            cv2.imwrite(save_path, img)

        return img

    def visualize_attention(
        self,
        attention_map: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention map from learnable alignment.

        Args:
            attention_map: (H, W) or (num_heads, H, W) attention weights
            save_path: Path to save visualization
        """
        if attention_map.ndim == 3:
            # Average over heads
            attention_map = attention_map.mean(axis=0)

        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Attention Weight')
        plt.title('Cross-Attention Map')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=100)
            plt.close()
        else:
            plt.show()


def create_detection_video(
    images: List[np.ndarray],
    predictions: List[Dict],
    output_path: str,
    visualizer: Visualizer = None,
    fps: int = 10
):
    """
    Create a video from detection results.

    Args:
        images: List of RGB images
        predictions: List of prediction dictionaries
        output_path: Output video path
        visualizer: Visualizer instance
        fps: Frames per second
    """
    if visualizer is None:
        visualizer = Visualizer()

    # Get image size
    H, W = images[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for img, pred in zip(images, predictions):
        # Draw detections
        vis_img = visualizer.visualize_image_with_detections(
            img,
            pred.get('boxes'),
            pred.get('labels'),
            pred.get('scores')
        )

        # Convert to BGR for OpenCV
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # Write frame
        video_writer.write(vis_img)

    video_writer.release()


if __name__ == "__main__":
    # Test visualizer
    print("Testing Visualizer...")

    # Create dummy data
    num_points = 10000
    points = np.random.randn(num_points, 4)
    points[:, :3] *= 10
    points[:, 3] = np.abs(points[:, 3])

    boxes = np.array([
        [0.0, 0.0, -1.5, 1.6, 3.9, 1.5, 0.0],
        [5.0, 3.0, -1.5, 1.6, 3.9, 1.5, 0.5]
    ])
    labels = np.array([0, 1])
    scores = np.array([0.9, 0.8])

    # Create visualizer
    viz = Visualizer()

    # Visualize BEV
    bev_img = viz.visualize_bev(points, boxes, labels, scores)
    print(f"BEV image shape: {bev_img.shape}")

    # Save BEV visualization
    output_path = '/tmp/bev_test.jpg'
    viz.visualize_bev(points, boxes, labels, scores, save_path=output_path)
    print(f"Saved BEV visualization to {output_path}")
