"""
KITTI Dataset loader for DeepFusion 3D Object Detection.

This module handles loading and preprocessing KITTI dataset for 3D object detection.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path


class KITTIDataset(Dataset):
    """
    KITTI Dataset for 3D object detection.

    Args:
        root_path: Root directory of KITTI dataset
        split: Data split ('train', 'val', 'test')
        split_file: Path to split file (optional)
        transform: Data augmentation transforms (optional)
        class_names: List of class names to detect
        max_objects: Maximum number of objects per sample
    """

    # Class names in KITTI
    CLASS_NAMES = [
        'Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck',
        'Person_sitting', 'Tram', 'Misc', 'DontCare'
    ]

    # Default classes to use
    DEFAULT_CLASSES = ['Car', 'Pedestrian', 'Cyclist']

    def __init__(
        self,
        root_path: str,
        split: str = 'train',
        split_file: Optional[str] = None,
        transform=None,
        class_names: List[str] = None,
        max_objects: int = 512,
        voxel_size: List[float] = [0.16, 0.16, 4.0],
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.transform = transform
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.max_objects = max_objects

        # BEV configuration
        self.voxel_size = voxel_size
        self.point_range = point_range

        # Calculate BEV dimensions based on PointPillars output
        # PointPillars: conv layers with stride 2, then upsample with stride 2
        # Input BEV: 500x500 (80m / 0.16m)
        # After 3 stride-2 convs: 500 / 8 = 62.5 → ~62
        # After 2 stride-2 transposed convs: 62 * 4 = ~248
        # Due to padding, the actual output is 252x252
        self.bev_h = 252  # Expected output height from PointPillars
        self.bev_w = 252  # Expected output width from PointPillars

        # Directories
        self.velodyne_dir = self.root_path / 'velodyne'
        self.image_dir = self.root_path / 'image_2'  # Left color camera
        self.calib_dir = self.root_path / 'calib'
        self.label_dir = self.root_path / 'label_2'

        # Load split file
        self.indices = self._load_split(split_file)

        print(f"Loaded {len(self.indices)} samples for {split} split")

    def _load_split(self, split_file: Optional[str]) -> List[str]:
        """Load data split from file or create default split."""
        if split_file is not None and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                indices = [line.strip() for line in f.readlines()]
            return indices

        # Default: use all available files
        velodyne_files = sorted(self.velodyne_dir.glob('*.bin'))
        indices = [f.stem for f in velodyne_files]

        # Simple train/val split (80/20)
        if self.split == 'train':
            indices = indices[:int(len(indices) * 0.8)]
        elif self.split == 'val':
            indices = indices[int(len(indices) * 0.8):]

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - points: (N, 4) point cloud
                - image: (3, H, W) image
                - calib: Calibration info
                - labels: (M, 7) ground truth labels [x, y, z, w, l, h, yaw, class]
                - aug_params: Augmentation parameters (if transform applied)
        """
        index = self.indices[idx]

        # Load point cloud
        points = self._load_velodyne(index)

        # Load image
        image = self._load_image(index)

        # Load calibration
        calib = self._load_calib(index)

        # Load labels
        labels = self._load_labels(index)

        # Filter labels by class
        labels = self._filter_labels(labels)

        # Transform point cloud to camera coordinates if needed
        # (KITTI velodyne is already in camera frame)

        # Apply transforms if available
        aug_params = None
        if self.transform is not None:
            points, image, labels, aug_params = self.transform(
                points, image, labels, calib
            )

        # Create targets for detection head
        targets = self._create_targets(labels, calib, image.shape[:2])

        return {
            'points': torch.from_numpy(points).float(),
            'image': torch.from_numpy(image).float(),
            'targets': targets,
            'calib': calib,
            'aug_params': aug_params,
            'index': index
        }

    def _load_velodyne(self, index: str) -> np.ndarray:
        """Load point cloud from velodyne.bin file."""
        file_path = self.velodyne_dir / f'{index}.bin'
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points

    def _load_image(self, index: str) -> np.ndarray:
        """Load image from PNG file."""
        file_path = self.image_dir / f'{index}.png'
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return image

    def _load_calib(self, index: str) -> Dict:
        """Load calibration data."""
        file_path = self.calib_dir / f'{index}.txt'

        calib = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])

        # Reshape matrices
        calib['P2'] = calib['P2'].reshape(3, 4)  # Left camera projection
        calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)  # Rectification matrix
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)  # Velodyne to camera transform

        return calib

    def _load_labels(self, index: str) -> np.ndarray:
        """Load 3D object labels."""
        file_path = self.label_dir / f'{index}.txt'

        labels = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                obj_type = parts[0]

                # KITTI format: type, truncated, occluded, alpha, bbox, dims, location, rotation_y
                # We only need: location (x,y,z), dims (h,w,l), rotation_y, type
                location = np.array([float(parts[11]), float(parts[12]), float(parts[13])])  # camera coords
                dims = np.array([float(parts[8]), float(parts[9]), float(parts[10])])  # h, w, l
                rotation_y = float(parts[14])

                # Store in LiDAR coordinates (z forward, x right, y down)
                # For now, keep in camera coordinates
                label = np.array([
                    location[0],  # x
                    location[1],  # y
                    location[2],  # z
                    dims[1],      # width (in KITTI: w)
                    dims[2],      # length (in KITTI: l)
                    dims[0],      # height (in KITTI: h)
                    rotation_y
                ])

                # Add class index
                try:
                    class_idx = self.CLASS_NAMES.index(obj_type)
                except ValueError:
                    class_idx = -1  # Unknown class

                labels.append(np.append(label, class_idx))

        return np.array(labels) if labels else np.empty((0, 8))

    def _filter_labels(self, labels: np.ndarray) -> np.ndarray:
        """Filter labels by class name."""
        if len(labels) == 0:
            return labels

        class_indices = [self.CLASS_NAMES.index(name) for name in self.class_names]
        mask = np.isin(labels[:, 7], class_indices)

        # Remap class indices to 0..N-1
        filtered = labels[mask]
        for i, class_idx in enumerate(class_indices):
            filtered[filtered[:, 7] == class_idx, 7] = i

        return filtered

    def _create_targets(
        self,
        labels: np.ndarray,
        calib: Dict,
        image_shape: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Create training targets for detection head.

        This creates:
        - Heatmap for object centers
        - Offset for sub-pixel refinement
        - Size for object dimensions
        - Rotation for yaw angles
        - Z-coordinate for height

        IMPORTANT: Targets are created with BEV dimensions (not image dimensions)
        to match the PointPillars backbone output.
        """
        H, W = self.bev_h, self.bev_w  # Use BEV dimensions, not image dimensions
        num_classes = len(self.class_names)

        # Initialize targets with BEV dimensions
        heatmap = np.zeros((num_classes, H, W), dtype=np.float32)
        offset = np.zeros((2, H, W), dtype=np.float32)
        size = np.zeros((3, H, W), dtype=np.float32)
        rotation = np.zeros((1, H, W), dtype=np.float32)
        z_center = np.zeros((1, H, W), dtype=np.float32)

        # TODO: Implement proper target generation
        # This requires projecting 3D boxes to BEV and creating heatmaps
        # For now, return empty targets with correct dimensions

        return {
            'heatmap': torch.from_numpy(heatmap),
            'offset': torch.from_numpy(offset),
            'size': torch.from_numpy(size),
            'rotation': torch.from_numpy(rotation),
            'z_center': torch.from_numpy(z_center)
        }


class KITTIBEVOnly(Dataset):
    """
    KITTI dataset with BEV-only processing (simplified version).

    This version focuses on BEV representation for faster training.
    """

    def __init__(
        self,
        root_path: str,
        split: str = 'train',
        transform=None,
        voxel_size: List[float] = [0.16, 0.16, 4.0],
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.transform = transform
        self.voxel_size = voxel_size
        self.point_range = point_range

        # Create KITTI dataset
        self.kitti = KITTIDataset(root_path, split)

    def __len__(self) -> int:
        return len(self.kitti)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get BEV-only sample."""
        sample = self.kitti[idx]

        # Optionally process point cloud to BEV
        # This can be done in the model

        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Handles variable-sized point clouds and different numbers of objects.
    """
    batch_size = len(batch)

    # Stack points (may need padding for different sizes)
    max_points = max([s['points'].shape[0] for s in batch])
    points = torch.zeros(batch_size, max_points, 4)
    for i, sample in enumerate(batch):
        n = sample['points'].shape[0]
        points[i, :n] = sample['points']

    # Stack images (already same size)
    images = torch.stack([s['image'] for s in batch])

    # Stack targets
    targets = {
        'heatmap': torch.stack([s['targets']['heatmap'] for s in batch]),
        'offset': torch.stack([s['targets']['offset'] for s in batch]),
        'size': torch.stack([s['targets']['size'] for s in batch]),
        'rotation': torch.stack([s['targets']['rotation'] for s in batch]),
        'z_center': torch.stack([s['targets']['z_center'] for s in batch])
    }

    return {
        'points': points,
        'images': images,
        'targets': targets,
        'aug_params': [s['aug_params'] for s in batch],
        'indices': [s['index'] for s in batch]
    }


if __name__ == "__main__":
    # Test the KITTI dataset loader
    import sys

    # You can specify the KITTI root path as argument
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        # Default path (adjust as needed)
        root_path = '/path/to/kitti'

    print(f"Loading KITTI dataset from: {root_path}")

    try:
        dataset = KITTIDataset(
            root_path=root_path,
            split='train'
        )

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]

            print(f"\nSample keys: {sample.keys()}")
            print(f"Points shape: {sample['points'].shape}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"Targets:")
            for name, target in sample['targets'].items():
                print(f"  {name}: {target.shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please specify the correct KITTI dataset path:")
        print(f"  python {sys.argv[0]} /path/to/kitti")
