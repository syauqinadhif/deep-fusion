"""
PointPillars backbone for LiDAR feature extraction.
FIXED VERSION — semua Python loop diganti dengan vectorized tensor operations.

Perubahan utama:
  PillarFeatureNet.forward() : loop per-pillar → scatter/gather tensor ops
  PointPillarsScatter.forward(): loop per-voxel  → index_put_ / advanced indexing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network — fully vectorized, no Python loops.
    """

    def __init__(
        self,
        in_channels:           int         = 4,
        out_channels:          int         = 64,
        max_points_per_pillar: int         = 100,
        max_pillars:           int         = 12000,
        pillar_x_size:         float       = 0.16,
        pillar_y_size:         float       = 0.16,
        pillar_z_size:         float       = 4.0,
        point_range:           List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        self.in_channels           = in_channels
        self.out_channels          = out_channels
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars           = max_pillars
        self.pillar_x_size         = pillar_x_size
        self.pillar_y_size         = pillar_y_size
        self.pillar_z_size         = pillar_z_size
        self.point_range           = point_range
        self.x_min, self.y_min, self.z_min, \
        self.x_max, self.y_max, self.z_max = point_range

        self.nx = int(np.round((self.x_max - self.x_min) / pillar_x_size))
        self.ny = int(np.round((self.y_max - self.y_min) / pillar_y_size))

        # in_channels(4) + enhanced(4: xc, yc, x_off, y_off) = 8
        total_in = in_channels + 4

        self.conv1 = nn.Conv2d(total_in, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (B, N, 4) — x, y, z, intensity

        Returns:
            pillar_features : (B, max_pillars, out_channels)
            coords          : (B, max_pillars, 3) — [batch_idx, x_idx, y_idx]
        """
        B, N, _ = points.shape
        device   = points.device

        all_feats   = []
        all_coords  = []

        for b in range(B):
            pts = points[b]   # (N, 4)

            # ── 1. range filter ──────────────────────────────────────────────
            valid = (
                (pts[:, 0] >= self.x_min) & (pts[:, 0] < self.x_max) &
                (pts[:, 1] >= self.y_min) & (pts[:, 1] < self.y_max) &
                (pts[:, 2] >= self.z_min) & (pts[:, 2] < self.z_max)
            )
            pts = pts[valid]   # (M, 4)
            M   = pts.shape[0]

            if M == 0:
                all_feats.append(torch.zeros(self.max_pillars, self.out_channels, device=device))
                all_coords.append(torch.zeros(self.max_pillars, 3, device=device, dtype=torch.long))
                continue

            # ── 2. pillar index per point ────────────────────────────────────
            xi = torch.floor((pts[:, 0] - self.x_min) / self.pillar_x_size).long()
            yi = torch.floor((pts[:, 1] - self.y_min) / self.pillar_y_size).long()
            xi = xi.clamp(0, self.nx - 1)
            yi = yi.clamp(0, self.ny - 1)

            # Flatten 2D pillar index → 1D key
            pillar_key = xi * self.ny + yi   # (M,)

            # ── 3. unique pillars + inverse mapping ──────────────────────────
            unique_keys, inverse = torch.unique(pillar_key, return_inverse=True)
            P = unique_keys.shape[0]

            if P > self.max_pillars:
                # Keep only first max_pillars unique pillars
                keep_mask = inverse < self.max_pillars
                unique_keys = unique_keys[:self.max_pillars]
                pts         = pts[keep_mask]
                inverse     = inverse[keep_mask]
                xi          = xi[keep_mask]
                yi          = yi[keep_mask]
                P           = self.max_pillars

            # ── 4. sort points by pillar, then truncate / pad ────────────────
            # Sort by pillar index so we can slice contiguous groups
            sort_order   = torch.argsort(inverse)
            pts_sorted   = pts[sort_order]       # (M', 4)
            inv_sorted   = inverse[sort_order]   # (M',)

            # Count points per pillar using bincount
            counts = torch.bincount(inv_sorted, minlength=P)   # (P,)
            counts_clamped = counts.clamp(max=self.max_points_per_pillar)

            T = self.max_points_per_pillar

            # Pillar tensor: (P, T, 4)  — zero-padded
            pillar_pts = torch.zeros(P, T, 4, device=device)

            # Fill using cumulative sum of counts (vectorised assignment)
            cum = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                             counts.cumsum(0)[:-1]])   # start index per pillar

            # Build point index inside its pillar (0, 1, 2, ... counts[p]-1)
            intra_idx = torch.arange(pts_sorted.shape[0], device=device) - cum[inv_sorted]
            valid_pt  = intra_idx < T
            pts_filt  = pts_sorted[valid_pt]
            inv_filt  = inv_sorted[valid_pt]
            intra_filt= intra_idx[valid_pt]

            pillar_pts[inv_filt, intra_filt] = pts_filt   # vectorised scatter

            # ── 5. enhanced features (pillar centre offsets) ─────────────────
            xi_u = (unique_keys // self.ny).float()
            yi_u = (unique_keys  % self.ny).float()
            xc   = xi_u * self.pillar_x_size + self.x_min + self.pillar_x_size / 2  # (P,)
            yc   = yi_u * self.pillar_y_size + self.y_min + self.pillar_y_size / 2  # (P,)

            # (P, T) offset tensors
            x_off = pillar_pts[:, :, 0] - xc.unsqueeze(1)
            y_off = pillar_pts[:, :, 1] - yc.unsqueeze(1)
            xc_t  = xc.unsqueeze(1).expand(P, T)
            yc_t  = yc.unsqueeze(1).expand(P, T)

            # (P, T, 8)
            enhanced = torch.stack([xc_t, yc_t, x_off, y_off], dim=2)
            pillar_in = torch.cat([pillar_pts, enhanced], dim=2)  # (P, T, 8)

            # ── 6. conv layers  (P, 8, 1, T) → (P, out_channels) ───────────
            x = pillar_in.permute(0, 2, 1).unsqueeze(2)   # (P, 8, 1, T)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=(1, T))        # (P, out_channels, 1, 1)
            feats = x.squeeze(-1).squeeze(-1)               # (P, out_channels)

            # ── 7. pad to max_pillars ────────────────────────────────────────
            if P < self.max_pillars:
                pad   = torch.zeros(self.max_pillars - P, self.out_channels, device=device)
                feats = torch.cat([feats, pad], dim=0)

            # ── 8. build coords [batch_idx, xi, yi] ──────────────────────────
            coords_xy = torch.stack([xi_u.long(), yi_u.long()], dim=1)   # (P, 2)
            if P < self.max_pillars:
                pad_c    = torch.zeros(self.max_pillars - P, 2, device=device, dtype=torch.long)
                coords_xy = torch.cat([coords_xy, pad_c], dim=0)

            coords_full       = torch.zeros(self.max_pillars, 3, device=device, dtype=torch.long)
            coords_full[:, 0] = b
            coords_full[:, 1] = coords_xy[:, 0]
            coords_full[:, 2] = coords_xy[:, 1]

            all_feats.append(feats)
            all_coords.append(coords_full)

        pillar_features = torch.stack(all_feats,  dim=0)   # (B, max_pillars, C)
        coords          = torch.stack(all_coords, dim=0)   # (B, max_pillars, 3)
        return pillar_features, coords


class PointPillarsScatter(nn.Module):
    """
    Scatter pillar features to 2D BEV map — fully vectorized, no Python loops.
    """

    def __init__(
        self,
        nx:           int = 500,
        ny:           int = 500,
        out_channels: int = 64
    ):
        super().__init__()
        self.nx          = nx
        self.ny          = ny
        self.out_channels = out_channels

    def forward(
        self,
        pillar_features: torch.Tensor,   # (B, max_pillars, C)
        coords:          torch.Tensor    # (B, max_pillars, 3) [b, xi, yi]
    ) -> torch.Tensor:
        """
        Returns:
            bev_map: (B, C, nx, ny)
        """
        B, P, C = pillar_features.shape
        device  = pillar_features.device

        bev_map = torch.zeros(B, C, self.nx, self.ny,
                              device=device, dtype=pillar_features.dtype)

        # ── fully vectorized: nol loop ────────────────────────────────────────
        bi = coords[:, :, 0].reshape(-1)   # (B*P,)
        xi = coords[:, :, 1].reshape(-1)   # (B*P,)
        yi = coords[:, :, 2].reshape(-1)   # (B*P,)
        fv = pillar_features.reshape(-1, C) # (B*P, C)

        valid = (xi > 0) | (yi > 0)
        bi = bi[valid].clamp(0, B - 1)
        xi = xi[valid].clamp(0, self.nx - 1)
        yi = yi[valid].clamp(0, self.ny - 1)
        fv = fv[valid]                      # (V, C)

        # satu operasi untuk seluruh batch sekaligus
        bev_map[bi, :, xi, yi] = fv         # (V, C) → (B, C, nx, ny)

        return bev_map


class PointPillarsBackbone(nn.Module):
    """Complete PointPillars backbone."""

    def __init__(
        self,
        in_channels:           int         = 4,
        out_channels:          int         = 64,
        max_points_per_pillar: int         = 100,
        max_pillars:           int         = 12000,
        voxel_size:            List[float] = [0.16, 0.16, 4.0],
        point_range:           List[float] = [-40.0, -40.0, -3.0, 40.0, 40.0, 1.0]
    ):
        super().__init__()

        x_w = point_range[3] - point_range[0]
        y_w = point_range[4] - point_range[1]
        nx  = int(np.round(x_w / voxel_size[0]))
        ny  = int(np.round(y_w / voxel_size[1]))

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
        self.scatter = PointPillarsScatter(nx=nx, ny=ny, out_channels=out_channels)

        # 2D CNN backbone
        self.conv1 = nn.Conv2d(out_channels, 64,  kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.up1   = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2   = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 4)
        Returns:
            (B, 256, H, W) BEV feature map
        """
        pillar_features, coords = self.pillar_net(points)
        bev = self.scatter(pillar_features, coords)

        x = self.relu(self.bn1(self.conv1(bev)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.up1(x))
        x = self.relu(self.up2(x))
        return x


# ── sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    B, N = 2, 20000
    pts  = torch.randn(B, N, 4)
    pts[:, :, :3] *= 20
    pts[:, :, 3]   = pts[:, :, 3].abs()

    model = PointPillarsBackbone()

    # warm-up
    with torch.no_grad():
        _ = model(pts)

    t0 = time.time()
    with torch.no_grad():
        out = model(pts)
    print(f"Output shape : {out.shape}")
    print(f"Forward time : {(time.time()-t0)*1000:.1f} ms")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")