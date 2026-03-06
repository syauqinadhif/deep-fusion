# Analisis Paper DeepFusion (CVPR 2022)
## arXiv:2203.08195

---

## 📋 Informasi Paper

| Aspect | Detail |
|--------|--------|
| **Title** | DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection |
| **Authors** | Xiaodong Yang, Xinge Zhu, Xiangwei Shen, et al. |
| **Venue** | CVPR 2022 (IEEE/CVF Conference on Computer Vision and Pattern Recognition) |
| **arXiv** | [2203.08195](https://arxiv.org/abs/2203.08195) |
| **Code** | [TensorFlow Lingvo](https://github.com/tensorflow/lingvo) |
| **Dataset** | Waymo Open Dataset |

---

## 🎯 Problem Statement

### Masalah Utama

Paper ini menyasar **masalah fusion antara LiDAR dan camera** untuk 3D object detection:

1. **Geometric Alignment**: Bagaimana align features dari sensor coordinate systems berbeda?
2. **Feature Fusion**: Bagaimana fuse features secara efektif?
3. **Augmentation Consistency**: Bagaimana menjaga consistency setelah data augmentation?

### Limitasi Pendekatan Eksisting

| Pendekatan | Masalah |
|-------------|---------|
| **Early Fusion** (point cloud coloring) | Raw RGB tidak cukup informative |
| **Late Fusion** (ensemble) | Tidak ada cross-modal interaction |
| **Mid-level Fusion** (pseudo-lidar) | Geometric alignment rumit |

---

## 💡 Solusi DeepFusion

### Dua Komponen Utama

```
┌─────────────────────────────────────────────────────────────────┐
│                  DEEPFUSION INNOVATIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INVERSE AUGMENTATION (InverseAug)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ • Problem: Augmentation (rotate, flip, scale) mengubah   │   │
│  │   geometry sehingga LiDAR↔Camera alignment salah       │   │
│  │ • Solution: REVERSE transform saat fusion                │   │
│  │ • Result: Alignment tetap accurate meski ada augment   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  2. LEARNABLE ALIGNMENT                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ • Problem: Hard alignment (projeksi 3D→2D) error-prone   │   │
│  │ • Solution: Cross-attention untuk SOFT alignment        │   │
│  │ • Query: LiDAR features (accurate geometry)             │   │
│  │ • Key/Value: Image features (rich semantics)            │   │
│  │ • Result: Model belajar align sendiri                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Arsitektur DeepFusion

```
┌─────────────────────────────────────────────────────────────────┐
│                 DEEPFUSION ARCHITECTURE (PAPER)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ┌──────────────┐        ┌──────────────┐                      │
│  │   LiDAR      │        │   Camera     │                      │
│  │ Point Cloud  │        │   Image      │                      │
│  └──────────────┘        └──────────────┘                      │
│         │                       │                               │
│         │                       │                               │
│         └───────────┬───────────┘                               │
│                     ↓                                           │
│              ┌──────────────────────┐                             │
│              │  DATA AUGMENTATION   │                             │
│              │  (rotate, flip, scale)│                             │
│              └──────────────────────┘                             │
│                     │                                           │
│         ┌───────────────┴───────────────┐                         │
│         │                               │                         │
│         ▼                               ▼                         │
│  ┌──────────────┐              ┌──────────────┐                   │
│  │   PVRCNN     │              │  2D CNN      │                   │
│  │  Backbone   │              │  (ResNet)    │                   │
│  │ (LiDAR feats)│              │ (Image feats)│                   │
│  └──────────────┘              └──────────────┘                   │
│         │                               │                         │
│         │         ┌───────────────────┘                         │
│         │         │                                            │
│         └─────────┴─────────────────────┐                       │
│                   ↓                     │                         │
│           ┌──────────────────────┐     │                         │
│           │   INVERSE AUG        │◄────┘ (aug params)           │
│           │   (Geometric Align)  │                               │
│           └──────────────────────┘                               │
│                   ↓                                             │
│           ┌──────────────────────┐                               │
│           │  LEARNABLE ALIGN     │                               │
│           │  (Cross-Attention)    │                               │
│           └──────────────────────┘                               │
│                   ↓                                             │
│           ┌──────────────────────┐                               │
│           │   FUSED FEATURES     │                               │
│           └──────────────────────┘                               │
│                   ↓                                             │
│           ┌──────────────────────┐                               │
│           │   DETECTION HEAD     │                               │
│           │   (3D Bounding Boxes)│                               │
│           └──────────────────────┘                               │
│                   ↓                                             │
│              OUTPUT                                            │
│         (3D Boxes + Classes)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Hasil Paper (Waymo Open Dataset)

### Main Results

| Method | Backbone | LEVEL_1 APH | LEVEL_2 APH |
|--------|----------|-------------|-------------|
| PointPillars | SECOND | 67.4 | 57.3 |
| CenterPoint | CenterPoint | 70.2 | 60.5 |
| 3D-MAN | PVRCNN | 68.9 | 58.6 |
| **DeepFusion (Ours)** | PVRCNN | **73.9** | **67.4** |

### Improvements vs Baseline

| vs PointPillars | LEVEL_1 | LEVEL_2 |
|-----------------|---------|---------|
| Improvement | +6.5 APH | +10.1 APH |

| vs CenterPoint | LEVEL_1 | LEVEL_2 |
|----------------|---------|---------|
| Improvement | +3.7 APH | +6.9 APH |

### Per-Class Results (Pedestrians)

| Method | LEVEL_1 APH | LEVEL_2 APH |
|--------|-------------|-------------|
| PointPillars | 68.8 | 53.1 |
| CenterPoint | 71.4 | 57.2 |
| **DeepFusion** | **75.5** | **62.0** |

**Key Insight**: Peningkatan paling besar pada **pedestrian detection** - kelas yang paling sulit!

---

## 🔍 Komponen Detail

### 1. Inverse Augmentation

**Konsep**: Reversing geometric transforms untuk alignment yang akurat

```python
# Pseudocode

# During training (with augmentation):
rotated_lidar = rotate(lidar, angle=30°)
rotated_image = rotate(image, angle=30°)

# During fusion:
aligned_lidar_features = PVRCNN(rotated_lidar)
aligned_image_features = CNN(rotated_image)

# REVERSE augmentation untuk get original coordinates:
original_lidar_coords = inverse_rotate(aligned_lidar_coords, angle=-30°)
original_image_coords = inverse_rotate(aligned_image_coords, angle=-30°)

# Sekarang alignment dalam coordinate system yang SAMA!
```

**Transforms yang di-handle**:
- Rotation (multiple angles)
- Flip (horizontal)
- Scaling (zoom in/out)
- Translation (shift)

### 2. Learnable Alignment

**Konsep**: Cross-attention untuk soft feature alignment

```python
# Pseudocode

# LiDAR features sebagai Query (sparse, geometry-accurate)
Q = project(lidar_features)  # (N_lidar, D)

# Image features sebagai Key/Value (dense, semantic-rich)
K = project(image_features)   # (N_image, D)
V = project(image_features)   # (N_image, D)

# Cross-attention
attn = softmax(Q @ K^T / sqrt(D))  # (N_lidar, N_image)
aligned = attn @ V                   # (N_lidar, D)

# Setiap LiDAR point belajar IMAGE MANA yang relevan
```

**Keuntungan Soft Alignment**:
- Tidak perlu exact geometric correspondence
- Bisa handle noisy calibration
- Learned from data

---

## 💾 Implementation Details

### Backbone: PVRCNN

```
PVRCNN = Point Voxel CNN (SECOND)

Komponen:
1. Voxelization: Point cloud → 3D voxels
2. 3D CNN: Process voxels
3. 2D pseudo-image: Projection ke BEV
4. 2D CNN: Process pseudo-image

Output: (C, H, W) feature map
```

### Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 (decay ke 1e-4) |
| Batch Size | 2 (per GPU) |
| Epochs | 36 (Waymo) |
| Weight Decay | 1e-4 |
| LR Schedule | Cosine annealing |

### Augmentation

| Type | Parameter |
|------|-----------|
| Rotation | [-30°, 30°] |
| Flip | Horizontal |
| Scaling | [0.95, 1.05] |
| Translation | [-0.2m, 0.2m] |

---

## 📈 Ablation Studies

### Kontribusi InverseAug

| Setup | LEVEL_2 APH |
|-------|-------------|
| w/o InverseAug | 63.8 |
| w/ InverseAug | **67.4** (+3.6) |

### Kontribusi LearnableAlign

| Setup | LEVEL_2 APH |
|-------|-------------|
| w/o Align (concat only) | 64.2 |
| w/ LearnableAlign | **67.4** (+3.2) |

### Full Ablation

| Component | LEVEL_2 APH | Δ |
|-----------|-------------|---|
| Baseline (LiDAR only) | 57.3 | - |
| + Camera Features (early) | 61.5 | +4.2 |
| + InverseAug | 64.8 | +3.3 |
| + LearnableAlign | **67.4** | +2.6 |

---

## 🚀 Implikasi untuk Skripsi

### Apa yang Bisa Diadopsi?

| Komponen | Paper | Skripsi (Usulan) |
|----------|-------|------------------|
| **Backbone** | PVRCNN (complex) | PointPillars (simpler) |
| **Task** | Detection | Detection + **Depth** |
| **Framework** | TensorFlow | **PyTorch** |
| **Dataset** | Waymo (besar) | **KITTI** (accessible) |
| **Fusion Strategy** | Same | Same |

### Kontribusi Tambahan untuk Skripsi

1. **Dual-Task Learning**: Detection + Depth (paper hanya detection)
2. **PyTorch Implementation**: Lebih accessible dari TensorFlow
3. **Simpler Backbone**: PointPillars lebih mudah diimplement

---

## 📚 Related Works

### Fusion Strategies

| Paper | Venue | Year | Focus |
|-------|-------|------|-------|
| MV3D | CVPR | 2017 | Voxel fusion |
| AVOD | CVPR | 2017 | Region fusion |
| SECOND | Sensors | 2018 | Two-stage |
| CenterPoint | CVPR | 2021 | Anchor-free |

### Depth Estimation

| Paper | Venue | Year | Focus |
|-------|-------|------|-------|
| MonoDepth2 | ICCV | 2018 | Monocular depth |
| DFF | CVPR | 2019 | Self-supervised depth |
| MegaDepth | CVPR | 2018 | Multi-frame depth |

---

## ⚠️ Challenges untuk Implementasi

### Technical Challenges

| Challenge | Difficulty | Solusi |
|-----------|------------|---------|
| PVRCNN complexity | High | Gunakan PointPillars |
| TensorFlow (Lingvo) | High | Port ke PyTorch |
| Waymo dataset size | High | Gunakan KITTI |
| Training time | Medium | GPU yang cukup |

### Time Estimation

| Task | Waktu (Estimasi) |
|------|------------------|
| Baseline PointPillars | 2-3 minggu |
| + Image features | 1-2 minggu |
| + InverseAug | 1 minggu |
| + LearnableAlign | 1-2 minggu |
| + Depth head | 1-2 minggu |
| Training & eksperimen | 2-3 minggu |
| **TOTAL** | **8-13 minggu** |

---

## 🎓 Takeaways untuk Skripsi

### Apa yang Dipelajari dari Paper?

1. **Geometric consistency** penting untuk multi-modal fusion
2. **Soft alignment** lebih flexible daripada hard alignment
3. **Cross-attention** adalah tools yang powerful untuk fusion

### Apa yang Bisa Ditambahkan?

1. **Depth estimation** sebagai auxiliary task
2. **Uncertainty estimation** untuk robust fusion
3. **Multi-scale fusion** untuk better context

---

**References**:
- Paper: [arXiv:2203.08195](https://arxiv.org/abs/2203.08195)
- Code: [TensorFlow Lingvo](https://github.com/tensorflow/lingvo)
- Dataset: [Waymo Open Dataset](https://waymo.com/open/data/)
