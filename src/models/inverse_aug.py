"""
Inverse Augmentation module for DeepFusion.
Handles geometric alignment between LiDAR and camera features after data augmentation.

Key concept: When data is augmented (rotated, flipped, scaled), the LiDAR and camera
coordinate systems become misaligned. This module reverses the augmentation transforms
to bring features back to the original coordinate system for proper fusion.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class AugmentationParams:
    """
    Container for augmentation parameters.

    Attributes:
        rotation_angle: Rotation angle in radians
        flip_x: Whether to flip along x axis
        flip_y: Whether to flip along y axis
        scale: Scaling factor
        translate_x: Translation in x direction (meters)
        translate_y: Translation in y direction (meters)
    """

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
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.scale = scale
        self.translate_x = translate_x
        self.translate_y = translate_y

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'rotation_angle': self.rotation_angle,
            'flip_x': self.flip_x,
            'flip_y': self.flip_y,
            'scale': self.scale,
            'translate_x': self.translate_x,
            'translate_y': self.translate_y
        }

    @classmethod
    def from_dict(cls, params: Dict) -> 'AugmentationParams':
        """Create from dictionary."""
        return cls(**params)


class InverseAugmentation(nn.Module):
    """
    Inverse Augmentation module for DeepFusion.

    This module reverses data augmentation transforms to maintain geometric
    consistency between LiDAR and camera features during fusion.

    The key insight: After augmentation, LiDAR points and image pixels are in
    different coordinate systems. We need to reverse the augmentation on one
    modality so both are aligned again.
    """

    def __init__(self):
        super().__init__()

    def inverse_rotation(
        self,
        features: torch.Tensor,
        angle: float,
        center_x: float,
        center_y: float
    ) -> torch.Tensor:
        """
        Apply inverse rotation to feature map.

        Args:
            features: (B, C, H, W) feature map
            angle: Rotation angle in radians
            center_x: Center x coordinate
            center_y: Center y coordinate

        Returns:
            Rotated feature map
        """
        if angle == 0.0:
            return features

        # Create inverse rotation transformation
        cos_a = np.cos(-angle)  # Inverse rotation
        sin_a = np.sin(-angle)

        # Use grid sampling for rotation
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=features.dtype, device=features.device)

        theta = theta.unsqueeze(0).repeat(features.shape[0], 1, 1)

        # Create grid
        grid = torch.nn.functional.affine_grid(
            theta[:, :2, :],
            features.size(),
            align_corners=False
        )

        # Sample
        rotated = torch.nn.functional.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='zeros',  # MPS doesn't support 'border'
            align_corners=False
        )

        return rotated

    def inverse_flip(self, features: torch.Tensor, flip_x: bool, flip_y: bool) -> torch.Tensor:
        """
        Apply inverse flip to feature map.

        Args:
            features: (B, C, H, W) feature map
            flip_x: Whether to flip along width dimension
            flip_y: Whether to flip along height dimension

        Returns:
            Flipped feature map
        """
        if not flip_x and not flip_y:
            return features

        # Flip along width dimension (W)
        if flip_x:
            features = torch.flip(features, dims=[3])

        # Flip along height dimension (H)
        if flip_y:
            features = torch.flip(features, dims=[2])

        return features

    def inverse_scale(
        self,
        features: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        Apply inverse scaling to feature map.

        Args:
            features: (B, C, H, W) feature map
            scale: Scaling factor (value < 1 means zoom out, > 1 means zoom in)

        Returns:
            Scaled feature map
        """
        if scale == 1.0:
            return features

        # Inverse scale
        inv_scale = 1.0 / scale

        # Calculate new size
        _, _, H, W = features.shape
        new_H = int(H * inv_scale)
        new_W = int(W * inv_scale)

        # Resize
        scaled = torch.nn.functional.interpolate(
            features,
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )

        # Resize back to original size if needed
        if new_H != H or new_W != W:
            scaled = torch.nn.functional.interpolate(
                scaled,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return scaled

    def forward(
        self,
        lidar_features: torch.Tensor,
        image_features: torch.Tensor,
        aug_params: Optional[AugmentationParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply inverse augmentation to align features.

        Args:
            lidar_features: (B, C_lidar, H, W) LiDAR BEV features
            image_features: (B, C_image, H, W) image features (projected to BEV)
            aug_params: Augmentation parameters

        Returns:
            aligned_lidar: Aligned LiDAR features
            aligned_image: Aligned image features (inverse augmented)
        """
        if aug_params is None:
            # No augmentation, return as is
            return lidar_features, image_features

        # We inverse augment the IMAGE features to match LiDAR's original coordinates
        # (LiDAR features are already in the augmented coordinate system from the backbone)
        aligned_image = image_features.clone()

        # Apply inverse transforms in reverse order
        # Order: scale -> flip -> rotation (reverse of augmentation order)

        # 1. Inverse scale
        if aug_params.scale != 1.0:
            aligned_image = self.inverse_scale(aligned_image, aug_params.scale)

        # 2. Inverse flip
        if aug_params.flip_x or aug_params.flip_y:
            aligned_image = self.inverse_flip(aligned_image, aug_params.flip_x, aug_params.flip_y)

        # 3. Inverse rotation
        if aug_params.rotation_angle != 0.0:
            # Get center of feature map
            _, _, H, W = aligned_image.shape
            center_x = W / 2.0
            center_y = H / 2.0
            aligned_image = self.inverse_rotation(aligned_image, aug_params.rotation_angle, center_x, center_y)

        # LiDAR features remain as-is (already augmented)
        aligned_lidar = lidar_features

        return aligned_lidar, aligned_image

    def inverse_augment_point_cloud(
        self,
        points: torch.Tensor,
        aug_params: AugmentationParams
    ) -> torch.Tensor:
        """
        Apply inverse augmentation directly to point cloud.

        This is useful for coordinate transformations.

        Args:
            points: (N, 3+) point cloud [x, y, z, ...]
            aug_params: Augmentation parameters

        Returns:
            Inverse augmented points
        """
        if aug_params is None:
            return points

        points_aug = points.clone()

        # Extract x, y coordinates
        x = points_aug[:, 0]
        y = points_aug[:, 1]

        # Inverse scale
        if aug_params.scale != 1.0:
            x = x / aug_params.scale
            y = y / aug_params.scale

        # Inverse flip
        if aug_params.flip_x:
            x = -x
        if aug_params.flip_y:
            y = -y

        # Inverse rotation
        if aug_params.rotation_angle != 0.0:
            cos_a = np.cos(-aug_params.rotation_angle)
            sin_a = np.sin(-aug_params.rotation_angle)

            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a

            x = x_new
            y = y_new

        # Inverse translation
        if aug_params.translate_x != 0.0 or aug_params.translate_y != 0.0:
            x = x - aug_params.translate_x
            y = y - aug_params.translate_y

        points_aug[:, 0] = x
        points_aug[:, 1] = y

        return points_aug


def get_inverse_augmentation_matrix(aug_params: AugmentationParams) -> np.ndarray:
    """
    Get the 3x3 inverse augmentation transformation matrix.

    Args:
        aug_params: Augmentation parameters

    Returns:
        3x3 transformation matrix
    """
    # Start with identity
    matrix = np.eye(3)

    # Apply inverse translation
    matrix[0, 2] = -aug_params.translate_x
    matrix[1, 2] = -aug_params.translate_y

    # Apply inverse scale
    if aug_params.scale != 1.0:
        scale_matrix = np.eye(3)
        inv_scale = 1.0 / aug_params.scale
        scale_matrix[0, 0] = inv_scale
        scale_matrix[1, 1] = inv_scale
        matrix = scale_matrix @ matrix

    # Apply inverse flip
    if aug_params.flip_x or aug_params.flip_y:
        flip_matrix = np.eye(3)
        if aug_params.flip_x:
            flip_matrix[0, 0] = -1
        if aug_params.flip_y:
            flip_matrix[1, 1] = -1
        matrix = flip_matrix @ matrix

    # Apply inverse rotation
    if aug_params.rotation_angle != 0.0:
        cos_a = np.cos(-aug_params.rotation_angle)
        sin_a = np.sin(-aug_params.rotation_angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        matrix = rotation_matrix @ matrix

    return matrix


if __name__ == "__main__":
    # Test the inverse augmentation module
    batch_size = 2
    channels = 64
    height = 128
    width = 128

    # Create dummy features
    lidar_features = torch.randn(batch_size, channels, height, width)
    image_features = torch.randn(batch_size, channels, height, width)

    # Create augmentation parameters
    aug_params = AugmentationParams(
        rotation_angle=np.pi / 6,  # 30 degrees
        flip_x=True,
        flip_y=False,
        scale=0.95,
        translate_x=0.1,
        translate_y=0.1
    )

    # Create inverse augmentation module
    inverse_aug = InverseAugmentation()

    # Apply inverse augmentation
    aligned_lidar, aligned_image = inverse_aug(lidar_features, image_features, aug_params)

    print(f"LiDAR features shape: {lidar_features.shape}")
    print(f"Aligned LiDAR features shape: {aligned_lidar.shape}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Aligned image features shape: {aligned_image.shape}")

    # Test point cloud inverse augmentation
    num_points = 1000
    points = torch.randn(num_points, 4)  # x, y, z, intensity

    inverse_augmented_points = inverse_aug.inverse_augment_point_cloud(points, aug_params)

    print(f"\nOriginal points shape: {points.shape}")
    print(f"Inverse augmented points shape: {inverse_augmented_points.shape}")
    print(f"Sample point difference: {(points[:5] - inverse_augmented_points[:5]).abs().sum()}")
