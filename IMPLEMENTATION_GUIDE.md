# Panduan Implementasi DeepFusion untuk Skripsi S1
## Object Detection Only - Jetson AGX Orin Deployment

---

## 📋 Daftar Isi

1. [Overview Arsitektur](#overview-arsitektur)
2. [Struktur Project & Source Code](#struktur-project--source-code)
3. [Quick Start](#quick-start)
4. [Detail Komponen](#detail-komponen)
5. [Training Strategy](#training-strategy)
6. [Evaluasi](#evaluasi)
7. [Jetson Deployment](#jetson-deployment)

---

## 🏗️ Overview Arsitektur

### Komponen Utama DeepFusion (Detection Only)

```
┌─────────────────────────────────────────────────────────────────┐
│                  DEEPFUSION COMPONENTS (DETECTION ONLY)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ┌──────────────┐        ┌──────────────┐                      │
│  │   LiDAR      │        │   Camera     │                      │
│  │ Point Cloud  │        │   Image      │                      │
│  └──────────────┘        └──────────────┘                      │
│         │                       │                               │
│         └───────────┬───────────┘                               │
│                     ▼                                           │
│  1. POINTPILLARS BACKBONE (LiDAR Feature Extraction)           │
│     • Voxelization: Point cloud → pillars                       │
│     • PillarFeatureNet: Learn pillar features                   │
│     • Scatter: Pillars → BEV feature map                        │
│     • 2D CNN: Process BEV features                             │
│     • Output: (B, 256, H, W) feature map                       │
│                                                                 │
│  2. IMAGE ENCODER (Camera Feature Extraction)                 │
│     • ResNet-34 pretrained backbone                            │
│     • Multi-scale feature extraction                           │
│     • Feature aggregation                                       │
│     • Output: (B, 256, H', W') feature map                     │
│                                                                 │
│  3. INVERSE AUGMENTATION (Geometric Alignment)                 │
│     • Reverse rotation/flip/scale transforms                    │
│     • Align LiDAR ↔ Camera features to same coordinate system  │
│     • Handles data augmentation consistency                     │
│                                                                 │
│  4. LEARNABLE ALIGNMENT (Feature Fusion)                       │
│     • Cross-attention: LiDAR queries, Image keys/values        │
│     • Soft alignment in feature space                          │
│     • Single multi-head attention layer                        │
│     • Output: (B, 256, H, W) fused features                    │
│                                                                 │
│  5. DETECTION HEAD (3D Bounding Box Prediction)                │
│     • Heatmap: Object center detection                         │
│     • Offset: Sub-pixel refinement                            │
│     • Size: Box dimensions (w, l, h)                          │
│     • Rotation: Yaw angle                                      │
│     • Z-center: Height coordinate                              │
│     • Output: (M, 7) = (x,y,z,w,l,h,yaw, class)               │
│                                                                 │
│  OUTPUT                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  3D Bounding Boxes + Class Labels + Confidence Scores    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Project & Source Code

### Directory Structure

```
deepfusion_project/
├── README.md                          ← Project overview
├── config.yaml                        ← Training configuration
├── requirements.txt                   ← Dependencies
├── IMPLEMENTATION_GUIDE.md            ← File ini
│
├── code/                              ← Source code
│   ├── __init__.py
│   │
│   ├── models/                        ← Model architectures
│   │   ├── __init__.py
│   │   ├── pointpillars.py           ← PointPillars backbone
│   │   ├── image_encoder.py          ← ResNet image encoder
│   │   ├── inverse_aug.py            ← Inverse augmentation
│   │   ├── learnable_align.py        ← Cross-attention fusion
│   │   ├── detection_head.py         ← 3D detection head
│   │   └── deepfusion.py             ← Main DeepFusion model
│   │
│   ├── datasets/                      ← Data loading
│   │   ├── __init__.py
│   │   ├── kitti.py                  ← KITTI dataset loader
│   │   └── transforms.py             ← Data augmentation
│   │
│   ├── utils/                         ← Utilities
│   │   ├── __init__.py
│   │   ├── common.py                 ← Common functions
│   │   ├── metrics.py                ← Evaluation metrics
│   │   └── visualization.py          ← Visualization tools
│   │
│   └── scripts/                       ← Training/eval scripts
│       ├── train.py                  ← Training script
│       ├── evaluate.py               ← Evaluation script
│       ├── export.py                 ← Model export
│       └── jetson_deploy.py          ← Jetson deployment
│
└── docs/                              ← Documentation
    ├── PAPER_ANALYSIS.md             ← Paper analysis
    └── JETSON_DEPLOYMENT.md          ← Deployment guide
```

### Source Code Files

#### 1. Models (`code/models/`)

| File | Description | Key Classes/Functions |
|------|-------------|---------------------|
| `pointpillars.py` | PointPillars backbone for LiDAR | `PointPillarsBackbone`, `PillarFeatureNet` |
| `image_encoder.py` | Camera feature extraction | `ImageEncoder`, `LightweightImageEncoder` |
| `inverse_aug.py` | Geometric alignment | `InverseAugmentation`, `AugmentationParams` |
| `learnable_align.py` | Feature fusion | `LearnableAlignment`, `MultiHeadAttention` |
| `detection_head.py` | 3D detection | `DetectionHead`, `ObjectDetectionLoss`, `DetectionDecoder` |
| `deepfusion.py` | Main model | `DeepFusion`, `DeepFusionLite` |

#### 2. Datasets (`code/datasets/`)

| File | Description | Key Classes/Functions |
|------|-------------|---------------------|
| `kitti.py` | KITTI dataset loader | `KITTIDataset`, `collate_fn` |
| `transforms.py` | Data augmentation | `DataAugmentation`, `ComposeTransforms` |

#### 3. Utils (`code/utils/`)

| File | Description | Key Classes/Functions |
|------|-------------|---------------------|
| `common.py` | Common utilities | `load_config`, `save_checkpoint`, `EarlyStopping` |
| `metrics.py` | Evaluation metrics | `DetectionMetrics`, `LossTracker` |
| `visualization.py` | Visualization | `Visualizer`, `create_detection_video` |

#### 4. Scripts (`code/scripts/`)

| File | Description | Usage |
|------|-------------|-------|
| `train.py` | Training script | `python train.py --config ../config.yaml --data_path /path/to/kitti` |
| `evaluate.py` | Evaluation script | `python evaluate.py --checkpoint best.pth.tar --data_path /path/to/kitti` |
| `export.py` | Model export | `python export.py --checkpoint best.pth.tar --format all` |
| `jetson_deploy.py` | Jetson inference | `python jetson_deploy.py --model_path model.pth --mode benchmark` |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone project (optional, already created)
cd deepfusion_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download KITTI dataset (3D Object Detection)
# http://www.cvlibs.net/datasets/kitti/

# Organize as:
# /path/to/kitti/
# ├── velodyne/     ← Point clouds (.bin)
# ├── image_2/      ← Images (.png)
# ├── calib/        ← Calibration (.txt)
# └── label_2/      ← Labels (.txt)
```

### 3. Training

```bash
cd code/scripts

# Basic training
python train.py \
    --config ../../config.yaml \
    --data_path /path/to/kitti \
    --output_dir ../../results

# Resume from checkpoint
python train.py \
    --config ../../config.yaml \
    --data_path /path/to/kitti \
    --resume ../../results/checkpoints/best.pth.tar

# Train lite model (for faster iteration)
python train.py \
    --config ../../config.yaml \
    --data_path /path/to/kitti \
    --model_type lite
```

### 4. Evaluation

```bash
# Evaluate on validation set
python evaluate.py \
    --config ../../config.yaml \
    --checkpoint ../../results/checkpoints/best.pth.tar \
    --data_path /path/to/kitti \
    --save_predictions
```

### 5. Export for Jetson

```bash
# Export all formats (TorchScript, FP16, ONNX for TensorRT)
python export.py \
    --config ../../config.yaml \
    --checkpoint ../../results/checkpoints/best.pth.tar \
    --output_dir ../../exported_models \
    --format all
```

---

## 📝 Detail Komponen

### PointPillars Backbone

**File**: `code/models/pointpillars.py`

Key features:
- Voxelization: Point cloud → uniform pillars
- PillarFeatureNet: Linear layers + BatchNorm
- 2D CNN backbone: Simplified ResNet-style blocks
- Output: 256-channel BEV feature map

```python
# Usage example
from models import PointPillarsBackbone

model = PointPillarsBackbone(
    in_channels=4,          # x, y, z, intensity
    out_channels=256,       # Output features
    max_points_per_pillar=100,
    max_pillars=12000,
    voxel_size=[0.16, 0.16, 4.0],
    point_range=[-40, -40, -3, 40, 40, 1]
)

# Forward pass
points = torch.randn(1, 10000, 4)  # (B, N, 4)
features = model(points)  # (B, 256, H, W)
```

### Image Encoder

**File**: `code/models/image_encoder.py`

Key features:
- ResNet-34 pretrained backbone
- Multi-scale feature extraction
- Feature aggregation with upsampling
- Lightweight version for Jetson

```python
# Usage example
from models import ImageEncoder

model = ImageEncoder(
    backbone='resnet34',
    pretrained=True,
    out_features=256
)

# Forward pass
images = torch.randn(1, 3, 384, 1280)  # (B, 3, H, W)
features, feat_dict = model(images)  # (B, 256, H', W')
```

### Inverse Augmentation

**File**: `code/models/inverse_aug.py`

Key features:
- Reverse rotation, flip, scale transforms
- Grid sampling for rotation
- Applied to image features to match LiDAR coordinates

```python
# Usage example
from models import InverseAugmentation, AugmentationParams

inverse_aug = InverseAugmentation()

# Create augmentation parameters
aug_params = AugmentationParams(
    rotation_angle=np.pi/6,  # 30 degrees
    flip_x=True,
    scale=0.95
)

# Apply inverse augmentation
aligned_lidar, aligned_image = inverse_aug(
    lidar_features,
    image_features,
    aug_params
)
```

### Learnable Alignment

**File**: `code/models/learnable_align.py`

Key features:
- Multi-head cross-attention
- LiDAR features as queries (geometry-accurate)
- Image features as keys/values (semantic-rich)
- Positional encoding variant available

```python
# Usage example
from models import LearnableAlignment

align = LearnableAlignment(
    lidar_channels=256,
    image_channels=256,
    hidden_dim=256,
    num_heads=8,
    num_layers=1
)

# Forward pass
fused_features, attn_weights = align(
    lidar_features,     # (B, 256, H, W)
    image_features,     # (B, 256, H, W)
    return_attention=True
)
```

### Detection Head

**File**: `code/models/detection_head.py`

Key features:
- Heatmap for object centers (focal loss)
- Offset for sub-pixel refinement (L1 loss)
- Size for dimensions (L1 loss)
- Rotation for yaw (smooth L1 loss)
- Z-center for height (smooth L1 loss)

```python
# Usage example
from models import DetectionHead, ObjectDetectionLoss

det_head = DetectionHead(
    in_channels=256,
    num_classes=3,        # Car, Pedestrian, Cyclist
    max_objects=512
)

# Forward pass
predictions = det_head(features)

# Compute loss
loss_fn = ObjectDetectionLoss(num_classes=3)
total_loss, loss_dict = loss_fn(predictions, targets)
```

### Main DeepFusion Model

**File**: `code/models/deepfusion.py`

Two variants:
1. **DeepFusion**: Full model for maximum accuracy
2. **DeepFusionLite**: Optimized for Jetson deployment

```python
# Usage example
from models import DeepFusion, DeepFusionLite

# Standard model
model = DeepFusion(
    lidar_channels=256,
    image_channels=256,
    hidden_dim=256,
    num_heads=8,
    num_classes=3
)

# Lite model (for Jetson)
lite_model = DeepFusionLite(
    lidar_channels=128,
    image_channels=128,
    hidden_dim=128,
    num_heads=4,
    num_classes=3
)

# Forward pass
output = model(points, images, aug_params=None)
predictions = output['predictions']

# Inference
detections = model.inference(points, images, conf_threshold=0.3)
```

---

## 🎯 Training Strategy

### Configuration

Edit `config.yaml`:

```yaml
training:
  batch_size: 4
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine
  warmup_epochs: 5
  min_lr: 0.00001

  # Detection only
  lambda_detection: 1.0
```

### Training Phases

```
PHASE 1: Baseline (Epoch 1-20)
├── Focus: Train detection head
├── Expected: mAP > 60%
└── Checkpoint: epoch_20.pth.tar

PHASE 2: Image Features (Epoch 21-40)
├── Focus: Fine-tune with camera
├── Expected: mAP > 65%
└── Checkpoint: epoch_40.pth.tar

PHASE 3: Full Fusion (Epoch 41-60)
├── Focus: Learnable alignment
├── Expected: mAP > 70%
└── Checkpoint: best.pth.tar

PHASE 4: Optimization (Epoch 61-100)
├── Focus: Fine-tune all components
├── Expected: mAP > 72%
└── Final: best.pth.tar
```

### Loss Components

```
Total Loss = Heatmap Loss + Offset Loss + Size Loss + Rotation Loss + Z Loss

Where:
- Heatmap Loss: Focal loss (α=0.25, γ=2.0)
- Offset Loss: L1 loss (only at object centers)
- Size Loss: L1 loss (only at object centers)
- Rotation Loss: Smooth L1 loss (β=0.1)
- Z Loss: Smooth L1 loss (β=0.1)
```

---

## 📊 Evaluasi

### Metrics

**File**: `code/utils/metrics.py`

Implements KITTI-style evaluation:
- Average Precision (AP) at IoU 0.5 and 0.7
- Precision, Recall, F1 score
- Per-class metrics
- Per-difficulty metrics (easy, moderate, hard)

### Running Evaluation

```bash
# Evaluate on validation set
python evaluate.py \
    --config ../../config.yaml \
    --checkpoint ../../results/checkpoints/best.pth.tar \
    --data_path /path/to/kitti \
    --save_predictions \
    --output_dir ../../results/eval

# Output:
# - results/eval/results.json
# - results/eval/bev_*.jpg (visualizations)
```

### Expected Results

| Metric | Easy | Moderate | Hard |
|--------|------|----------|------|
| AP (IoU=0.7) | > 75% | > 70% | > 65% |
| Precision | > 80% | > 75% | > 70% |
| Recall | > 75% | > 70% | > 65% |

---

## 🔧 Jetson Deployment

### Export Model

```bash
# Export for Jetson
python export.py \
    --config ../../config.yaml \
    --checkpoint ../../results/checkpoints/best.pth.tar \
    --model_type lite \
    --format all

# Outputs:
# - deepfusion_scripted.pt (TorchScript)
# - deepfusion_fp16.pth (FP16 checkpoint)
# - deepfusion.onnx (for TensorRT)
# - convert_to_tensorrt.sh (conversion script)
```

### Transfer to Jetson

```bash
# Copy exported models to Jetson
scp exported_models/* jetson@<ip>:~/deepfusion/

# Or convert to TensorRT on Jetson
ssh jetson@<ip>
cd ~/deepfusion
bash convert_to_tensorrt.sh
```

### Run Inference on Jetson

```bash
# Benchmark
python jetson_deploy.py \
    --model_path deepfusion_fp16.pth \
    --mode benchmark \
    --fp16 \
    --iterations 100

# Expected:
# - Latency: < 33ms
# - FPS: > 30
# - Power: < 25W
```

### Optimization Tips

1. **Use FP16**: 2x faster with minimal accuracy loss
2. **TensorRT**: Best performance, requires conversion
3. **Lite Model**: Smaller, faster for real-time
4. **Batch Size**: Use 1 for lowest latency

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size in config.yaml
training:
  batch_size: 2  # or 1
```

**Issue**: Slow training
```bash
# Solution: Use fewer workers or disable cudnn benchmark
system:
  num_workers: 2
  cudnn_benchmark: false
```

**Issue**: Poor detection accuracy
```bash
# Solution: Check data augmentation and learning rate
# 1. Verify dataset is loaded correctly
# 2. Reduce learning rate: 0.0001
# 3. Increase warmup epochs: 10
```

---

## 📚 Referensi

- [DeepFusion Paper](https://arxiv.org/abs/2203.08195)
- [PointPillars Paper](https://arxiv.org/abs/1812.05784)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [TensorRT](https://developer.nvidia.com/tensorrt/)
- [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/jetson-orin/)

---

**Last Updated**: 2025-01-XX
**Version**: 1.0.0 (Detection Only)
