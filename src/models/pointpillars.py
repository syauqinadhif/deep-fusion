"""
PointPillars backbone for LiDAR feature extraction.
Based on "PointPillars: Fast Encoders for Object Detection from Point Clouds"
Lang et al., CVPR 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network - converts point cloud pillars to feature pillars.

    Args:
        in_channels: Input channels (x, y, z, intensity = 4)
        out_channels: Output feature channels
        max_points_per_pillar: Maximum number of points per pillar
        max_pillars: Maximum number of pillars
        pillar_x_size: Voxel size in x direction
        pillar_y_size: Voxel size in y direction
        pillar_z_size: Voxel size in z direction
        point_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        max_points_per_pillar: int = 100,
        max_pillars: int = 12000,
        pillar_x_size: float = 0.16,
        pillar_y_size: float = 0.16,
        pillar_z_size: float = 4.0,
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        self.pillar_x_size = pillar_x_size
        self.pillar_y_size = pillar_y_size
        self.pillar_z_size = pillar_z_size

        self.point_range = point_range
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = point_range

        # Calculate grid size
        self.x_w = self.x_max - self.x_min
        self.y_w = self.y_max - self.y_min
        self.z_w = self.z_max - self.z_min

        self.nx = int(np.round(self.x_w / self.pillar_x_size))
        self.ny = int(np.round(self.y_w / self.pillar_y_size))
        self.nz = int(np.round(self.z_w / self.pillar_z_size))

        # Linear layer for enhancement (x_c, y_c, x_offset, y_offset) + features
        # We have 4 channels: x_c, y_c (center of pillar), x_offset, y_offset (distance to center)
        # + original features (x, y, z, intensity)
        self.total_in_channels = in_channels + 4  # 4 for enhanced features

        # Conv layers for pillar feature extraction
        self.conv1 = nn.Conv2d(
            self.total_in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of PillarFeatureNet.

        Args:
            points: (N, 4) tensor where N is total number of points
                    Format: [x, y, z, intensity]

        Returns:
            pillar_features: (max_pillars, out_channels) tensor
            coords: (max_pillars, 3) tensor with pillar coordinates
        """
        batch_size = points.shape[0]

        # Process each batch element
        all_pillar_features = []
        all_coords = []

        for i in range(batch_size):
            batch_points = points[i]  # (N, 4)

            # Filter points outside range
            mask = (
                (batch_points[:, 0] >= self.x_min) &
                (batch_points[:, 0] < self.x_max) &
                (batch_points[:, 1] >= self.y_min) &
                (batch_points[:, 1] < self.y_max) &
                (batch_points[:, 2] >= self.z_min) &
                (batch_points[:, 2] < self.z_max)
            )
            batch_points = batch_points[mask]

            if batch_points.shape[0] == 0:
                # No valid points, return empty pillars
                pillar_feats = torch.zeros(
                    self.max_pillars, self.out_channels,
                    device=points.device
                )
                coords = torch.zeros(
                    self.max_pillars, 3,
                    device=points.device, dtype=torch.long
                )
                all_pillar_features.append(pillar_feats)
                all_coords.append(coords)
                continue

            # Calculate pillar indices
            x_indices = torch.floor(
                (batch_points[:, 0] - self.x_min) / self.pillar_x_size
            ).long()
            y_indices = torch.floor(
                (batch_points[:, 1] - self.y_min) / self.pillar_y_size
            ).long()

            # Get unique pillars
            pillar_indices = torch.stack([x_indices, y_indices], dim=1)
            unique_pillars, inverse_indices = torch.unique(
                pillar_indices, dim=0, return_inverse=True
            )

            # Limit to max_pillars
            if len(unique_pillars) > self.max_pillars:
                unique_pillars = unique_pillars[:self.max_pillars]
                mask = inverse_indices < self.max_pillars
                inverse_indices = inverse_indices[mask]
                batch_points = batch_points[mask]

            # Create pillar features
            num_pillars = len(unique_pillars)
            pillar_feats_list = []

            for pillar_idx in range(num_pillars):
                # Get points in this pillar
                pillar_mask = inverse_indices == pillar_idx
                pillar_points = batch_points[pillar_mask]

                # Sort by z (descending) and pad/truncate
                pillar_points = pillar_points[torch.argsort(pillar_points[:, 2], descending=True)]

                if pillar_points.shape[0] > self.max_points_per_pillar:
                    pillar_points = pillar_points[:self.max_points_per_pillar]
                else:
                    padding = torch.zeros(
                        self.max_points_per_pillar - pillar_points.shape[0],
                        4, device=pillar_points.device
                    )
                    pillar_points = torch.cat([pillar_points, padding], dim=0)

                # Calculate enhanced features
                x_c = unique_pillars[pillar_idx, 0] * self.pillar_x_size + self.x_min + self.pillar_x_size / 2
                y_c = unique_pillars[pillar_idx, 1] * self.pillar_y_size + self.y_min + self.pillar_y_size / 2

                x_offset = pillar_points[:, 0] - x_c
                y_offset = pillar_points[:, 1] - y_c

                # Concatenate enhanced features
                enhanced = torch.stack([
                    torch.full((self.max_points_per_pillar,), x_c, device=pillar_points.device),
                    torch.full((self.max_points_per_pillar,), y_c, device=pillar_points.device),
                    x_offset,
                    y_offset
                ], dim=1)

                pillar_feat = torch.cat([pillar_points, enhanced], dim=1)  # (max_points, 8)

                pillar_feats_list.append(pillar_feat)

            # Stack and process through conv layers
            if len(pillar_feats_list) > 0:
                pillar_feats = torch.stack(pillar_feats_list, dim=0)  # (num_pillars, max_points, 8)
                pillar_feats = pillar_feats.permute(0, 2, 1).unsqueeze(2)  # (num_pillars, 8, 1, max_points)
                pillar_feats = pillar_feats[:, :, :, :self.max_points_per_pillar]

                # Apply conv layers
                pillar_feats = self.conv1(pillar_feats)
                pillar_feats = self.bn1(pillar_feats)
                pillar_feats = self.relu(pillar_feats)

                pillar_feats = self.conv2(pillar_feats)
                pillar_feats = self.bn2(pillar_feats)
                pillar_feats = self.relu(pillar_feats)

                # Max pool over points dimension
                pillar_feats = F.max_pool2d(pillar_feats, kernel_size=(1, self.max_points_per_pillar))
                pillar_feats = pillar_feats.squeeze(-1).squeeze(-1)  # (num_pillars, out_channels)

                # Pad to max_pillars
                if pillar_feats.shape[0] < self.max_pillars:
                    padding = torch.zeros(
                        self.max_pillars - pillar_feats.shape[0],
                        self.out_channels,
                        device=pillar_feats.device
                    )
                    pillar_feats = torch.cat([pillar_feats, padding], dim=0)

                # Create coordinates
                coords = unique_pillars
                if coords.shape[0] < self.max_pillars:
                    padding = torch.zeros(
                        self.max_pillars - coords.shape[0],
                        2, device=coords.device, dtype=torch.long
                    )
                    coords = torch.cat([coords, padding], dim=0)

                # Add batch dimension
                batch_coords = torch.zeros(coords.shape[0], 3, device=coords.device, dtype=torch.long)
                batch_coords[:, 1:] = coords
                batch_coords[:, 0] = 0  # batch index

            else:
                pillar_feats = torch.zeros(
                    self.max_pillars, self.out_channels, device=points.device
                )
                batch_coords = torch.zeros(self.max_pillars, 3, device=points.device, dtype=torch.long)

            all_pillar_features.append(pillar_feats)
            all_coords.append(batch_coords)

        # Stack batch
        all_pillar_features = torch.stack(all_pillar_features, dim=0)
        all_coords = torch.stack(all_coords, dim=0)

        return all_pillar_features, all_coords


class PointPillarsScatter(nn.Module):
    """
    Scatters pillar features back to a 2D BEV map.

    Args:
        nx: Number of pillars in x direction
        ny: Number of pillars in y direction
        out_channels: Number of feature channels
    """

    def __init__(
        self,
        nx: int = 496,  # 80m / 0.16m
        ny: int = 496,  # 80m / 0.16m
        out_channels: int = 64
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.out_channels = out_channels

    def forward(
        self,
        pillar_features: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Scatter pillar features to 2D BEV map.

        Args:
            pillar_features: (batch_size, max_pillars, out_channels)
            coords: (batch_size, max_pillars, 3) [batch_idx, x_idx, y_idx]

        Returns:
            bev_map: (batch_size, out_channels, nx, ny)
        """
        batch_size = pillar_features.shape[0]
        bev_map = torch.zeros(
            batch_size,
            self.out_channels,
            self.nx,
            self.ny,
            device=pillar_features.device
        )

        for i in range(batch_size):
            # Get valid pillars (where coords != 0)
            mask = coords[i, :, 1:].sum(dim=1) > 0
            valid_coords = coords[i][mask]
            valid_features = pillar_features[i][mask]

            # Scatter to BEV map
            for j in range(valid_coords.shape[0]):
                batch_idx, x_idx, y_idx = valid_coords[j]
                if 0 <= x_idx < self.nx and 0 <= y_idx < self.ny:
                    bev_map[i, :, x_idx, y_idx] = valid_features[j]

        return bev_map


class PointPillarsBackbone(nn.Module):
    """
    Complete PointPillars backbone for LiDAR feature extraction.

    Args:
        in_channels: Input channels (default: 4 for x,y,z,intensity)
        out_channels: Output feature channels (default: 64)
        max_points_per_pillar: Maximum points per pillar (default: 100)
        max_pillars: Maximum number of pillars (default: 12000)
        voxel_size: Voxel size [x, y, z] (default: [0.16, 0.16, 4.0])
        point_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 64,
        max_points_per_pillar: int = 100,
        max_pillars: int = 12000,
        voxel_size: List[float] = [0.16, 0.16, 4.0],
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        x_w = point_range[3] - point_range[0]
        y_w = point_range[4] - point_range[1]

        nx = int(np.round(x_w / voxel_size[0]))
        ny = int(np.round(y_w / voxel_size[1]))

        self.pillar_net = PillarFeatureNet(
            in_channels=in_channels,
            out_channels=out_channels,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
            pillar_x_size=voxel_size[0],
            pillar_y_size=voxel_size[1],
            pillar_z_size=voxel_size[2],
            point_range=point_range
        )

        self.scatter = PointPillarsScatter(
            nx=nx,
            ny=ny,
            out_channels=out_channels
        )

        # 2D CNN backbone (similar to SECOND)
        self.conv1 = nn.Conv2d(out_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Upsampling to maintain resolution
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.relu = nn.ReLU()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PointPillars backbone.

        Args:
            points: (batch_size, N, 4) point cloud tensor

        Returns:
            features: (batch_size, 256, H, W) BEV feature map
        """
        # Extract pillar features
        pillar_features, coords = self.pillar_net(points)

        # Scatter to BEV map
        bev_map = self.scatter(pillar_features, coords)

        # 2D CNN processing
        x = self.conv1(bev_map)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Upsample
        x = self.up1(x)
        x = self.relu(x)
        x = self.up2(x)
        x = self.relu(x)

        return x


if __name__ == "__main__":
    # Test the PointPillars backbone
    batch_size = 2
    num_points = 1000

    # Create dummy point cloud
    points = torch.randn(batch_size, num_points, 4)
    points[:, :, :3] *= 10  # Scale coordinates
    points[:, :, 3] = torch.abs(points[:, :, 3])  # Positive intensity

    # Create model
    model = PointPillarsBackbone(
        in_channels=4,
        out_channels=64,
        max_points_per_pillar=100,
        max_pillars=12000,
        voxel_size=[0.16, 0.16, 4.0],
        point_range=[-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    )

    # Forward pass
    features = model(points)

    print(f"Input shape: {points.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
