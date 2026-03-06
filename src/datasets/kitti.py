"""
KITTI Dataset loader for DeepFusion 3D Object Detection.
FIXED VERSION - Perbaikan bug kritis:
  1. collate_fn: padding point cloud lebih efisien + pre-truncate
  2. _load_image: tambah resize agar shape konsisten
  3. _create_targets: implementasi heatmap Gaussian yang benar
  4. aug_params: dikembalikan sebagai list per-sample (tidak di-index [0])
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path

# ── konstanta resize image ──────────────────────────────────────────────────
TARGET_IMG_H = 376   # KITTI native height (sudah pas, tidak perlu crop)
TARGET_IMG_W = 1248  # Dibuat kelipatan 32 agar kompatibel dengan ResNet stride

# ── ukuran BEV output PointPillars ─────────────────────────────────────────
# range x: [-40, 40] → 80m / 0.16 = 500 pillar
# setelah backbone (3× stride-2 conv, 2× stride-2 deconv) → 252
BEV_H = 252
BEV_W = 252

# ── Gaussian radius helper ──────────────────────────────────────────────────
def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> int:
    """Compute Gaussian radius for heatmap target (CornerNet formula)."""
    h, w = det_size
    a1 = 1
    b1 = (h + w)
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return max(0, int(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int):
    """Draw Gaussian blob on heatmap at given center."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x = np.arange(0, diameter, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = radius
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    cx, cy = int(center[0]), int(center[1])
    H, W = heatmap.shape

    left   = min(cx, radius)
    right  = min(W - cx, radius + 1)
    top    = min(cy, radius)
    bottom = min(H - cy, radius + 1)

    masked_hm  = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_g   = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_hm, masked_g, out=masked_hm)


class KITTIDataset(Dataset):
    CLASS_NAMES = [
        'Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck',
        'Person_sitting', 'Tram', 'Misc', 'DontCare'
    ]
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
        point_range: List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0],
        # ── FIX: batasi jumlah point agar collate cepat ──
        max_num_points: int = 80_000,
    ):
        self.root_path      = Path(root_path)
        self.split          = split
        self.transform      = transform
        self.class_names    = class_names or self.DEFAULT_CLASSES
        self.max_objects    = max_objects
        self.voxel_size     = voxel_size
        self.point_range    = point_range
        self.max_num_points = max_num_points   # ← BARU

        self.bev_h = BEV_H
        self.bev_w = BEV_W

        self.velodyne_dir = self.root_path / 'velodyne'
        self.image_dir    = self.root_path / 'image_2'
        self.calib_dir    = self.root_path / 'calib'
        self.label_dir    = self.root_path / 'label_2'

        self.indices = self._load_split(split_file)
        print(f"Loaded {len(self.indices)} samples for {split} split")

    # ── split loading (tidak berubah) ───────────────────────────────────────
    def _load_split(self, split_file: Optional[str]) -> List[str]:
        if split_file is not None and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f]

        velodyne_files = sorted(self.velodyne_dir.glob('*.bin'))
        indices = [f.stem for f in velodyne_files]
        if self.split == 'train':
            return indices[:int(len(indices) * 0.8)]
        elif self.split == 'val':
            return indices[int(len(indices) * 0.8):]
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        index  = self.indices[idx]
        points = self._load_velodyne(index)   # (N, 4)
        image  = self._load_image(index)      # (3, H, W)  — sudah resize
        calib  = self._load_calib(index)
        labels = self._load_labels(index)
        labels = self._filter_labels(labels)

        aug_params = None
        if self.transform is not None:
            points, image, labels, aug_params = self.transform(
                points, image, labels, calib
            )

        targets = self._create_targets(labels, calib)

        return {
            'points':     torch.from_numpy(points).float(),
            'image':      torch.from_numpy(image).float(),
            'targets':    targets,
            'calib':      calib,
            'aug_params': aug_params,   # bisa None kalau transform=None
            'index':      index
        }

    # ── loaders ─────────────────────────────────────────────────────────────

    def _load_velodyne(self, index: str) -> np.ndarray:
        """Load & truncate point cloud agar ukuran terbatas."""
        pts = np.fromfile(
            self.velodyne_dir / f'{index}.bin', dtype=np.float32
        ).reshape(-1, 4)

        # ── FIX #1: truncate point cloud di sini, bukan di collate ──────────
        # Range filter dulu (buang point di luar BEV range)
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_range
        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )
        pts = pts[mask]

        # Kalau masih terlalu banyak, random sample
        if len(pts) > self.max_num_points:
            idx = np.random.choice(len(pts), self.max_num_points, replace=False)
            pts = pts[idx]

        return pts  # (≤ max_num_points, 4)

    def _load_image(self, index: str) -> np.ndarray:
        """Load image dan resize ke ukuran fixed agar bisa di-stack."""
        img = cv2.imread(str(self.image_dir / f'{index}.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── FIX #2: resize agar semua image sama shape ───────────────────────
        img = cv2.resize(img, (TARGET_IMG_W, TARGET_IMG_H),
                         interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))   # CHW

    def _load_calib(self, index: str) -> Dict:
        calib = {}
        with open(self.calib_dir / f'{index}.txt', 'r') as f:
            for line in f:
                if line == '\n':
                    continue
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])

        calib['P2']           = calib['P2'].reshape(3, 4)
        calib['R0_rect']      = calib['R0_rect'].reshape(3, 3)
        calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
        return calib

    def _load_labels(self, index: str) -> np.ndarray:
        labels = []
        with open(self.label_dir / f'{index}.txt', 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                obj_type = parts[0]
                if obj_type == 'DontCare':
                    continue

                loc  = np.array([float(parts[11]), float(parts[12]), float(parts[13])])
                dims = np.array([float(parts[8]),  float(parts[9]),  float(parts[10])])
                ry   = float(parts[14])

                label = np.array([
                    loc[0], loc[1], loc[2],   # x, y, z  (camera coords)
                    dims[1], dims[2], dims[0], # w, l, h
                    ry
                ])
                try:
                    cls_idx = self.CLASS_NAMES.index(obj_type)
                except ValueError:
                    cls_idx = -1

                labels.append(np.append(label, cls_idx))

        return np.array(labels) if labels else np.empty((0, 8))

    def _filter_labels(self, labels: np.ndarray) -> np.ndarray:
        if len(labels) == 0:
            return labels

        class_indices = [self.CLASS_NAMES.index(n) for n in self.class_names]
        mask     = np.isin(labels[:, 7], class_indices)
        filtered = labels[mask].copy()
        for i, ci in enumerate(class_indices):
            filtered[filtered[:, 7] == ci, 7] = i
        return filtered

    # ── FIX #3: target generation yang benar ────────────────────────────────
    def _create_targets(
        self,
        labels: np.ndarray,
        calib: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Buat heatmap CenterPoint-style di BEV space.
        Koordinat objek diproyeksikan dari camera coords ke BEV grid.
        """
        H, W       = self.bev_h, self.bev_w
        num_cls    = len(self.class_names)
        xmin, ymin, _, xmax, ymax, _ = self.point_range

        heatmap  = np.zeros((num_cls, H, W), dtype=np.float32)
        offset   = np.zeros((2,       H, W), dtype=np.float32)
        size_map = np.zeros((3,       H, W), dtype=np.float32)
        rot_map  = np.zeros((1,       H, W), dtype=np.float32)
        z_map    = np.zeros((1,       H, W), dtype=np.float32)

        if len(labels) == 0:
            return self._pack_targets(heatmap, offset, size_map, rot_map, z_map)

        # Transform: camera → velodyne → BEV pixel
        R0    = calib['R0_rect']              # (3,3)
        Tr    = calib['Tr_velo_to_cam']       # (3,4)

        # Inverse transform: cam → velo
        R0_inv = np.linalg.inv(R0)
        R      = Tr[:, :3]
        t      = Tr[:, 3]
        R_inv  = R.T
        t_inv  = -R.T @ t

        for label in labels:
            x_cam, y_cam, z_cam = label[0], label[1], label[2]
            w, l, h             = label[3], label[4], label[5]
            ry                  = label[6]
            cls                 = int(label[7])

            if cls < 0 or cls >= num_cls:
                continue

            # Camera → rectified camera → velodyne
            pt_cam  = np.array([x_cam, y_cam, z_cam])
            pt_rect = R0_inv @ pt_cam
            pt_velo = R_inv @ pt_rect + t_inv   # (3,) — velodyne coords

            xv, yv, zv = pt_velo[0], pt_velo[1], pt_velo[2]

            # BEV pixel index (x forward, y left in velodyne)
            bev_x = (xv - xmin) / (xmax - xmin) * W
            bev_y = (yv - ymin) / (ymax - ymin) * H

            bev_xi = int(bev_x)
            bev_yi = int(bev_y)

            if not (0 <= bev_xi < W and 0 <= bev_yi < H):
                continue

            # Gaussian radius berdasarkan ukuran objek di BEV
            bev_l = l / (xmax - xmin) * W
            bev_w = w / (ymax - ymin) * H
            radius = gaussian_radius((bev_l, bev_w))
            radius = max(1, radius)

            draw_gaussian(heatmap[cls], (bev_xi, bev_yi), radius)

            # Sub-pixel offset
            offset[0, bev_yi, bev_xi] = bev_x - bev_xi
            offset[1, bev_yi, bev_xi] = bev_y - bev_yi

            # Size, rotation, z
            size_map[0, bev_yi, bev_xi] = w
            size_map[1, bev_yi, bev_xi] = l
            size_map[2, bev_yi, bev_xi] = h
            rot_map[0,  bev_yi, bev_xi] = ry
            z_map[0,    bev_yi, bev_xi] = zv

        return self._pack_targets(heatmap, offset, size_map, rot_map, z_map)

    @staticmethod
    def _pack_targets(heatmap, offset, size_map, rot_map, z_map) -> Dict[str, torch.Tensor]:
        return {
            'heatmap':  torch.from_numpy(heatmap),
            'offset':   torch.from_numpy(offset),
            'size':     torch.from_numpy(size_map),
            'rotation': torch.from_numpy(rot_map),
            'z_center': torch.from_numpy(z_map)
        }


# ── FIX #4: collate_fn yang efisien ─────────────────────────────────────────
def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function yang efisien.
    - Point cloud di-pad ke max di batch (bukan 120k), karena sudah di-truncate
      di _load_velodyne.
    - Image sudah sama shape → langsung torch.stack.
    - aug_params dikembalikan sebagai list (tidak di-index [0]).
    """
    B = len(batch)

    # ── points: pad ke max dalam batch ini ──────────────────────────────────
    max_pts = max(s['points'].shape[0] for s in batch)
    points  = torch.zeros(B, max_pts, 4)
    for i, s in enumerate(batch):
        n = s['points'].shape[0]
        points[i, :n] = s['points']

    # ── images: sudah sama size, langsung stack ──────────────────────────────
    images = torch.stack([s['image'] for s in batch])   # (B, 3, H, W)

    # ── targets: stack per key ───────────────────────────────────────────────
    targets = {
        k: torch.stack([s['targets'][k] for s in batch])
        for k in batch[0]['targets']
    }

    # ── aug_params: list of dicts/None ──────────────────────────────────────
    # FIX #5: jangan ambil [0] saja — kembalikan semua
    aug_params = [s['aug_params'] for s in batch]

    return {
        'points':     points,
        'images':     images,
        'targets':    targets,
        'aug_params': aug_params,   # list[B] ← gunakan ini di train loop
        'indices':    [s['index'] for s in batch]
    }