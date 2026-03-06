"""
Data augmentation transforms for DeepFusion training.

Implements multi-modal augmentation that maintains consistency
between LiDAR point clouds and camera images.
"""

import random
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import cv2

# Absolute import that works when running from scripts/
try:
    from models.inverse_aug import AugmentationParams
except ImportError:
    # Fallback for relative import
    from ..models.inverse_aug import AugmentationParams


class DataAugmentation:
    """
    Combined LiDAR and camera data augmentation.

    Ensures geometric consistency between modalities.
    """

    def __init__(
        self,
        train_mode: bool = True,
        # Point cloud augmentation
        point_aug_range: float = 0.2,
        point_noise_std: float = 0.01,
        # Image augmentation
        img_flip: float = 0.5,
        img_rotate: float = 0.5,
        img_brightness: float = 0.3,
        img_contrast: float = 0.2,
        # Shared geometric augmentation
        rotation_range: float = np.pi / 6,  # +/- 30 degrees
        flip_x_prob: float = 0.5,
        flip_y_prob: float = 0.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_range: float = 0.2,  # +/- 0.2m
        # Point cloud range for cropping
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        self.train_mode = train_mode

        # Point cloud augmentation
        self.point_aug_range = point_aug_range
        self.point_noise_std = point_noise_std

        # Image augmentation
        self.img_flip = img_flip
        self.img_rotate = img_rotate
        self.img_brightness = img_brightness
        self.img_contrast = img_contrast

        # Shared geometric augmentation
        self.rotation_range = rotation_range
        self.flip_x_prob = flip_x_prob
        self.flip_y_prob = flip_y_prob
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.point_range = point_range

    def __call__(
        self,
        points: np.ndarray,
        image: np.ndarray,
        labels: np.ndarray,
        calib: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AugmentationParams]:
        """
        Apply augmentation to both LiDAR and camera data.

        Args:
            points: (N, 4) point cloud [x, y, z, intensity]
            image: (3, H, W) image (normalized 0-1)
            labels: (M, 8) labels [x, y, z, w, l, h, yaw, class]
            calib: Calibration dictionary

        Returns:
            Augmented points, image, labels, and augmentation parameters
        """
        if not self.train_mode:
            return points, image, labels, None

        aug_params_list = []

        # 1. Shared geometric augmentation (applied to both modalities)
        points, image, labels, geom_params = self._apply_geometric_augmentation(
            points, image, labels, calib
        )
        aug_params_list.append(geom_params)

        # 2. Point cloud specific augmentation
        points = self._augment_point_cloud(points)

        # 3. Image specific augmentation
        image = self._augment_image(image)

        # Combine augmentation parameters
        aug_params = aug_params_list[0] if aug_params_list else None

        return points, image, labels, aug_params

    def _apply_geometric_augmentation(
        self,
        points: np.ndarray,
        image: np.ndarray,
        labels: np.ndarray,
        calib: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AugmentationParams]:
        """Apply geometric transformations to both modalities."""
        # Determine transformations
        rotation_angle = random.uniform(-self.rotation_range, self.rotation_range)
        flip_x = random.random() < self.flip_x_prob
        flip_y = random.random() < self.flip_y_prob
        scale = random.uniform(*self.scale_range)
        translate_x = random.uniform(-self.translate_range, self.translate_range)
        translate_y = random.uniform(-self.translate_range, self.translate_range)

        # Create augmentation parameters
        aug_params = AugmentationParams(
            rotation_angle=rotation_angle,
            flip_x=flip_x,
            flip_y=flip_y,
            scale=scale,
            translate_x=translate_x,
            translate_y=translate_y
        )

        # Apply to point cloud
        points = self._transform_points(points, rotation_angle, flip_x, flip_y, scale, translate_x, translate_y)

        # Apply to labels
        labels = self._transform_labels(labels, rotation_angle, flip_x, flip_y, scale, translate_x, translate_y)

        # Apply to image
        image = self._transform_image(image, rotation_angle, flip_x, flip_y, scale)

        return points, image, labels, aug_params

    def _transform_points(
        self,
        points: np.ndarray,
        rotation_angle: float,
        flip_x: bool,
        flip_y: bool,
        scale: float,
        translate_x: float,
        translate_y: float
    ) -> np.ndarray:
        """Transform point cloud."""
        transformed = points.copy()

        # Extract coordinates
        x = transformed[:, 0]
        y = transformed[:, 1]
        z = transformed[:, 2]

        # Translation
        x = x + translate_x
        y = y + translate_y

        # Scaling
        x = x * scale
        y = y * scale
        z = z * scale

        # Flipping
        if flip_x:
            x = -x
        if flip_y:
            y = -y

        # Rotation
        if rotation_angle != 0:
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a
            x = x_new
            y = y_new

        transformed[:, 0] = x
        transformed[:, 1] = y
        transformed[:, 2] = z

        return transformed

    def _transform_labels(
        self,
        labels: np.ndarray,
        rotation_angle: float,
        flip_x: bool,
        flip_y: bool,
        scale: float,
        translate_x: float,
        translate_y: float
    ) -> np.ndarray:
        """Transform 3D bounding box labels."""
        if len(labels) == 0:
            return labels

        transformed = labels.copy()

        # Transform center
        x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]
        w, l, h = transformed[:, 3], transformed[:, 4], transformed[:, 5]
        yaw = transformed[:, 6]

        # Translation
        x = x + translate_x
        y = y + translate_y

        # Scaling
        x = x * scale
        y = y * scale
        z = z * scale
        w = w * scale
        l = l * scale
        h = h * scale

        # Flipping
        if flip_x:
            x = -x
            yaw = -yaw
        if flip_y:
            y = -y
            yaw = np.pi - yaw

        # Rotation
        if rotation_angle != 0:
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a
            x = x_new
            y = y_new
            yaw = yaw + rotation_angle

        # Normalize yaw to [-pi, pi]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        transformed[:, 0] = x
        transformed[:, 1] = y
        transformed[:, 2] = z
        transformed[:, 3] = w
        transformed[:, 4] = l
        transformed[:, 5] = h
        transformed[:, 6] = yaw

        return transformed

    def _transform_image(
        self,
        image: np.ndarray,
        rotation_angle: float,
        flip_x: bool,
        flip_y: bool,
        scale: float
    ) -> np.ndarray:
        """Transform image (H, W, C for opencv)."""
        # Convert to HWC for opencv
        image_hwc = np.transpose(image, (1, 2, 0))

        # Get image dimensions
        H, W = image_hwc.shape[:2]

        # Create transformation matrix
        # Center of rotation
        cx, cy = W / 2, H / 2

        # Build transformation matrix
        M = np.eye(3)

        # Scale
        if scale != 1.0:
            scale_matrix = np.array([
                [scale, 0, (1 - scale) * cx],
                [0, scale, (1 - scale) * cy],
                [0, 0, 1]
            ])
            M = scale_matrix @ M

        # Rotation
        if rotation_angle != 0:
            angle_deg = np.degrees(rotation_angle)
            rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
            M = rotation_matrix @ M

        # Flip
        flip_code = None
        if flip_x and flip_y:
            flip_code = -1
        elif flip_x:
            flip_code = 1
        elif flip_y:
            flip_code = 0

        if flip_code is not None:
            image_hwc = cv2.flip(image_hwc, flip_code)

        # Apply affine transformation
        if rotation_angle != 0 or scale != 1.0:
            image_hwc = cv2.warpAffine(
                image_hwc,
                M[:2, :],
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

        # Convert back to CHW
        image = np.transpose(image_hwc, (2, 0, 1))

        return image

    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Apply point cloud specific augmentation."""
        # Add random noise
        if self.point_noise_std > 0:
            noise = np.random.normal(0, self.point_noise_std, points[:, :3].shape)
            points[:, :3] += noise

        # Random jitter
        if self.point_aug_range > 0:
            jitter = np.random.uniform(
                -self.point_aug_range,
                self.point_aug_range,
                points[:, :3].shape
            )
            points[:, :3] += jitter

        # Clip intensity
        points[:, 3] = np.clip(points[:, 3], 0, 1)

        return points

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image specific augmentation."""
        # Convert to HWC
        image_hwc = np.transpose(image, (1, 2, 0))

        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(1 - self.img_brightness, 1 + self.img_brightness)
            image_hwc = np.clip(image_hwc * brightness_factor, 0, 1)

        # Contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(1 - self.img_contrast, 1 + self.img_contrast)
            mean = image_hwc.mean()
            image_hwc = np.clip((image_hwc - mean) * contrast_factor + mean, 0, 1)

        # Saturation adjustment
        if random.random() < 0.3:
            # Convert to HSV
            hsv = cv2.cvtColor((image_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)

            # Adjust saturation
            sat_factor = random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

            # Convert back
            hsv = hsv.astype(np.uint8)
            image_hwc = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0

        # Convert back to CHW
        image = np.transpose(image_hwc, (2, 0, 1))

        return image


class PointCloudTransform:
    """
    Point cloud specific transforms (for preprocessing).
    """

    def __init__(
        self,
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0],
        max_points: int = 50000
    ):
        self.point_range = point_range
        self.max_points = max_points

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply point cloud preprocessing."""
        # Filter by range
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_range

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        )

        points = points[mask]

        # Shuffle and sample
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        return points


class ImageTransform:
    """
    Image specific transforms (for preprocessing).
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (384, 1280),
        normalize: bool = True
    ):
        self.output_size = output_size
        self.normalize = normalize

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing."""
        # Image is expected to be (3, H, W) and normalized 0-1

        # Resize
        if self.output_size is not None:
            image_hwc = np.transpose(image, (1, 2, 0))
            image_hwc = cv2.resize(
                image_hwc,
                (self.output_size[1], self.output_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            image = np.transpose(image_hwc, (2, 0, 1))

        # Normalize (ImageNet stats)
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image = (image - mean) / std

        return image


class ComposeTransforms:
    """
    Compose multiple transforms together.
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(
        self,
        points: np.ndarray,
        image: np.ndarray,
        labels: np.ndarray,
        calib: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[AugmentationParams]]:
        """Apply all transforms sequentially."""
        aug_params = None

        for transform in self.transforms:
            if isinstance(transform, DataAugmentation):
                points, image, labels, aug_params = transform(points, image, labels, calib)
            elif isinstance(transform, PointCloudTransform):
                points = transform(points)
            elif isinstance(transform, ImageTransform):
                image = transform(image)

        return points, image, labels, aug_params


def get_training_transforms(config: dict) -> ComposeTransforms:
    """Create training transforms from config."""
    data_config = config.get('data', {})

    transforms = [
        DataAugmentation(
            train_mode=True,
            point_aug_range=data_config.get('point_aug_range', 0.2),
            point_noise_std=data_config.get('point_noise_std', 0.01),
            img_flip=data_config.get('img_flip', 0.5),
            img_rotate=data_config.get('img_rotate', 0.5),
            img_brightness=data_config.get('img_brightness', 0.3),
            img_contrast=data_config.get('img_contrast', 0.2)
        ),
        PointCloudTransform(
            point_range=config.get('model', {}).get('point_range', [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
        ),
        ImageTransform(output_size=(384, 1280))
    ]

    return ComposeTransforms(transforms)


def get_val_transforms(config: dict) -> ComposeTransforms:
    """Create validation transforms from config."""
    transforms = [
        DataAugmentation(train_mode=False),
        PointCloudTransform(
            point_range=config.get('model', {}).get('point_range', [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0])
        ),
        ImageTransform(output_size=(384, 1280))
    ]

    return ComposeTransforms(transforms)


if __name__ == "__main__":
    # Test data augmentation
    print("Testing data augmentation...")

    # Create dummy data
    num_points = 10000
    points = np.random.randn(num_points, 4)
    points[:, :3] *= 10
    points[:, 3] = np.abs(points[:, 3])

    image = np.random.rand(3, 384, 1280).astype(np.float32)

    labels = np.array([
        [5.0, 0.0, -1.5, 1.6, 3.9, 1.5, 0.0, 0]  # Car at origin
    ])

    calib = {}

    # Create augmentation
    aug = DataAugmentation(train_mode=True)

    # Apply augmentation
    aug_points, aug_image, aug_labels, aug_params = aug(points, image, labels, calib)

    print(f"Original points shape: {points.shape}")
    print(f"Augmented points shape: {aug_points.shape}")
    print(f"Original image shape: {image.shape}")
    print(f"Augmented image shape: {aug_image.shape}")
    print(f"Original labels: {labels}")
    print(f"Augmented labels: {aug_labels}")

    if aug_params is not None:
        print(f"\nAugmentation parameters:")
        print(f"  Rotation: {aug_params.rotation_angle:.4f} rad")
        print(f"  Flip X: {aug_params.flip_x}")
        print(f"  Flip Y: {aug_params.flip_y}")
        print(f"  Scale: {aug_params.scale:.4f}")
        print(f"  Translate X: {aug_params.translate_x:.4f}")
        print(f"  Translate Y: {aug_params.translate_y:.4f}")
