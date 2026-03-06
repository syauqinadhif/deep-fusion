# DeepFusion: Lidar-Camera Fusion untuk 3D Object Detection
## Skripsi S1 - Autonomous Driving (Jetson AGX Orin Deployment)

---

## 📋 Ringkasan Proyek

**Paper**: [DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection](https://arxiv.org/abs/2203.08195) (CVPR 2022)

**Fokus Skripsi**: Implementasi DeepFusion untuk **3D Object Detection** dengan deploy ke Jetson AGX Orin

**Perubahan dari Paper Asli**:
- ❌ TANPA Depth Estimation (fokus detection saja)
- ✅ Simplified backbone (PointPillars, bukan PVRCNN)
- ✅ PyTorch implementation (bukan TensorFlow)
- ✅ Optimized untuk Jetson Orin deployment

---

## 🎯 Target Skripsi

### Output

```
INPUTS
├── LiDAR Point Cloud → (N, 4) = [x, y, z, intensity]
└── Camera Image → (3, H, W) RGB

OUTPUT
└── 3D Bounding Boxes → (M, 7) = [x, y, z, w, l, h, yaw, class]
```

### Target Performance di Jetson Orin

| Metric | Target |
|--------|--------|
| **FPS** | ≥ 30 (real-time) |
| **Latency** | ≤ 33ms |
| **mAP (KITTI)** | > 70% (moderate) |
| **Power** | < 25W |

---

## 🏗️ Arsitektur (Object Detection Only)

```
┌─────────────────────────────────────────────────────────────────┐
│              DEEPFUSION ARCHITECTURE (DETECTION ONLY)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUTS                                                         │
│  ┌──────────────┐        ┌──────────────┐                      │
│  │   LiDAR      │        │   Camera     │                      │
│  │ Point Cloud  │        │   Image      │                      │
│  └──────────────┘        └──────────────┘                      │
│         │                       │                               │
│         └───────────┬───────────┘                               │
│                     ▼                                           │
│              ┌──────────────────────┐                             │
│              │   INVERSE AUG        │ ← Geometric Alignment    │
│              │   (Align)             │                             │
│              └──────────────────────┘                             │
│                     ▼                                           │
│              ┌──────────────────────┐                             │
│              │  POINTPILLARS         │ ← Voxel Encoder          │
│              │  BACKBONE            │                             │
│              └──────────────────────┘                             │
│                     ▼                                           │
│              ┌──────────────────────┐                             │
│              │  IMAGE ENCODER        │ ← ResNet Feature Extractor│
│              │  (ResNet34)           │                             │
│              └──────────────────────┘                             │
│                     │                   │                            │
│         LiDAR Features         Image Features                   │
│                     │                   │                            │
│                     └─────────┬─────────┘                            │
│                               ▼                                 │
│              ┌──────────────────────┐                             │
│              │   LEARNABLE ALIGN     │ ← Cross-Attention Fusion  │
│              │   (Soft Fusion)        │                             │
│              └──────────────────────┘                             │
│                               ▼                                 │
│              ┌──────────────────────┐                             │
│              │    FUSED FEATURES     │                             │
│              └──────────────────────┘                             │
│                               ▼                                 │
│              ┌──────────────────────┐                             │
│              │  DETECTION HEAD       │ ← 3D Bounding Box Predictor│
│              │  (3D BBox Predictor)  │                             │
│              └──────────────────────┘                             │
│                               ▼                                 │
│              ┌──────────────────────┐                             │
│              │   OUTPUT             │                             │
│              │   • Bounding Boxes    │                             │
│              │   • Class Labels      │                             │
│              │   • Confidence Scores  │                             │
│              └──────────────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Struktur Project (Detection Only)

```
deepfusion_project/
├── README.md                          ← File ini (overview)
│
├── docs/
│   ├── PAPER_ANALYSIS.md              ← Analisis paper DeepFusion
│   ├── ARCHITECTURE.md               ← Detail arsitektur
│   └── JETSON_DEPLOYMENT.md           ← Panduan deploy Jetson
│
├── code/
│   ├── models/
│   │   ├── pointpillars.py            ← PointPillars backbone
│   │   ├── image_encoder.py           ← ResNet image encoder
│   │   ├── inverse_aug.py             ← Inverse augmentation
│   │   ├── learnable_align.py         ← Cross-attention alignment
│   │   ├── detection_head.py          ← 3D bounding box head
│   │   └── deepfusion.py              ← Main DeepFusion model
│   │
│   ├── datasets/
│   │   ├── kitti.py                   ← KITTI dataset loader
│   │   ├── carla.py                   ← CARLA dataset loader
│   │   └── transforms.py              ← Data augmentation
│   │
│   ├── utils/
│   │   ├── metrics.py                 ← Evaluation metrics (mAP)
│   │   ├── visualization.py           ← Visualisasi hasil
│   │   └── common.py                  ← Utility functions
│   │
│   ├── train.py                       ← Training script
│   ├── evaluate.py                    ← Evaluation script
│   ├── export.py                      ← Export ke TorchScript/TensorRT
│   └── jetson_deploy.py              ← Deployment script untuk Jetson
│
├── config.yaml                        ← Konfigurasi training
│
└── requirements.txt                   ← Dependencies
```

---

## 🎯 Rumusan Masalah Skripsi

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROBLEM STATEMENT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MASALAH UTAMA:                                                 │
│  "Bagaimana fuse LiDAR dan camera untuk 3D object detection       │
│   yang akurat DAN efisien untuk real-time deployment          │
│   di edge device (Jetson AGX Orin)?"                           │
│                                                                 │
│  SUB-MASALAH:                                                   │
│  1. Fusion Strategy: Bagaimana fuse multi-modal features?        │
│  2. Alignment Challenge: Bagaimana align LiDAR 3D ↔ Camera 2D?   │
│  3. Real-Time Target: Bagaimana capai ≥30 FPS di Jetson Orin?    │
│  4. Robustness: Bagaimana maintain akurasi di kondisi buruk?      │
│                                                                 │
│  SOLUSI:                                                       │
│  1. Mid-Level Fusion: Fuse di feature level (bukan input/output) │
│  2. Inverse Augmentation: Geometric alignment untuk coord       │
│  3. Learnable Alignment: Cross-attention untuk soft fusion        │
│  4. Model Optimizations: FP16, TensorRT, INT8 untuk Jetson     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Expected Hasil untuk Skripsi

### Table 1: Performance Comparison (KITTI Val)

| Method | Backbone | mAP (Easy) | mAP (Mod) | mAP (Hard) |
|--------|----------|------------|-----------|------------|
| PointPillars (baseline) | - | 68.2 | 62.1 |
| PointPillars + Camera | - | 70.5 | 64.3 |
| **DeepFusion (Ours)** | PointPillars | **72.8** | **66.7** |

### Table 2: Ablation Study

| Component | mAP (Mod) | Δ mAP | Catatan |
|-----------|-----------|-------|---------|
| Baseline (LiDAR only) | 68.2 | - | PointPillars |
| + Camera Features (early) | 70.5 | +2.3 | Simple concat |
| + InverseAug | 71.8 | +1.3 | Geometric align |
| + LearnableAlign | **72.8** | +1.0 | Soft fusion |

### Table 3: Jetson Orin Performance

| Config | Precision | FPS | Latency | Power | mAP |
|--------|-----------|-----|---------|-------|-----|
| FP32 (baseline) | FP32 | 12-15 | 66-83ms | 30W | 72.8% |
| FP16 | FP16 | 20-25 | 40-50ms | 20W | 72.6% |
| FP16 + TensorRT | FP16 | **30-35** | **28-33ms** | 18W | 72.5% |
| INT8 + TensorRT | INT8 | **35-40** | **25-28ms** | 12W | 71.8% |

**TARGET: ✓ 30 FPS achievable dengan FP16 + TensorRT!**

---

## 📝 Struktur Skripsi

### Bab 1: Pendahuluan
- 1.1 Latar Belakang: Autonomous driving & object detection
- 1.2 Rumusan Masalah: Multi-modal fusion & edge deployment
- 1.3 Tujuan: Implementasi DeepFusion + deploy ke Jetson Orin
- 1.4 Kontribusi:
  - PyTorch implementation
  - Simplified backbone
  - Jetson deployment optimization
- 1.5 Batasan: PointPillars backbone, KITTI dataset

### Bab 2: Tinjauan Pustaka
- 2.1 3D Object Detection (PointPillars, SECOND, CenterPoint)
- 2.2 Multi-Modal Fusion (early, mid, late)
- 2.3 DeepFusion Paper (CVPR 2022)
- 2.4 Edge Deployment (TensorRT, quantization)

### Bab 3: Metodologi
- 3.1 PointPillars Backbone
- 3.2 Image Encoder (ResNet)
- 3.3 Inverse Augmentation
- 3.4 Learnable Alignment (Cross-Attention)
- 3.5 Detection Head
- 3.6 Loss Function
- 3.7 Training Strategy

### Bab 4: Hasil dan Pembahasan
- 4.1 Experimental Setup
- 4.2 Baseline Comparison
- 4.3 Ablation Studies
- 4.4 KITTI Results
- 4.5 Jetson Deployment Results
- 4.6 Performance Analysis

### Bab 5: Kesimpulan
- 5.1 Kesimpulan
- 5.2 Limitations
- 5.3 Future Work

---

## 🎓 Timeline Skripsi (4 Bulan - Disarankan)

```
BULAN 1: FUNDASI (Minggu 1-4)
├── Minggu 1-2: Study paper, setup environment
├── Minggu 3-4: Implement PointPillars baseline
└── Output: Baseline detection model

BULAN 2: FUSION (Minggu 5-8)
├── Minggu 5-6: Implement image encoder + early fusion
├── Minggu 7-8: Implement InverseAug + LearnableAlign
└── Output: Complete DeepFusion model

BULAN 3: OPTIMASI & DEPLOY (Minggu 9-12)
├── Minggu 9: Training lengkap
├── Minggu 10: Export ke TensorRT
├── Minggu 11: Deploy ke Jetson Orin
└── Output: Real-time inference on Jetson

BULAN 4: PENULISAN (Minggu 13-16)
├── Minggu 13-14: Ablation studies
├── Minggu 15-16: Tulis skripsi
└── Output: Siap sidang
```

---

## 🚀 Keunggulan Pendekatan Ini

### 1. Lebih Fokus (Single Task)

```
BEFORE (Dual Task):
- Object Detection
- Depth Estimation
→ Complex, lama, sulit real-time

AFTER (Detection Only):
- Object Detection
→ Lebih simple, lebih cepat, lebih feasible
```

### 2. Lebih Mudah Deploy ke Jetson

```
COMPLEXITY REDUCTION:
┌─────────────────────────────────────────────────────────────┐
│  WITH DEPTH          │  WITHOUT DEPTH                        │
│  ┌────────────────┐  │  ┌────────────────┐                    │
│  │ Detection Head │  │  │ Detection Head │                    │
│  │ Depth Head    │  │  └────────────────┘                    │
│  └────────────────┘  │                                       │
│  ┌────────────────┐  │                                       │
│  │ Loss: Det+Depth│  │  │ Loss: Detection Only               │
│  └────────────────┘  │  └─────────────────────────────────────┘
│  ↓                 │  ↓                                      │
│  SLOW             │  FASTER                                 │
│  15-25 FPS        │  30-40 FPS ✓                           │
└─────────────────────────────────────────────────────────────┘
```

### 3. Lebih Jelas Fokus untuk Skripsi

```
KONTRIBUSI JELAS:
1. PyTorch implementation DeepFusion (paper: TensorFlow)
2. SOTA fusion strategy untuk 3D detection
3. Real-time deployment di Jetson Orin
4. Comprehensive optimasi study

TUJUAN FOKUS:
• Bagaimana align LiDAR ↔ Camera?
• Bagaimana fuse secara efektif?
• Bagaimana optimasi untuk edge device?
```

---

## 📊 Comparison: DeepFusion vs Adaptive Gating (Final)

Dengan modifikasi ini (detection only):

| Aspect | DeepFusion (Detection) | Adaptive Gating |
|--------|------------------------|-----------------|
| **Task** | Object Detection | End-to-End Driving |
| **Output** | Bounding Boxes | Waypoints |
| **Fusion** | Mid-level (Cross-Attention) | Multi-scale (Transformer) |
| **Complexity** | Medium | Medium |
| **Waktu Implementasi** | 4 bulan | 3 bulan |
| **Jetson FPS** | 30-35 (dengan TensorRT) | 30-40 (native) |
| **Deploy Ready** | ✓ (butuh optimasi) | ✓ (native) |

---

## ✅ Keputusan

**REKOMENDASI: DeepFusion (Detection Only) untuk Skripsi**

Kenapa?
1. ✅ Object Detection adalah topik penting dan relevan
2. ✅ Paper DeepFusion SOTA (CVPR 2022)
3. ✅ Bisa deploy ke Jetson dengan 30+ FPS (real-time)
4. ✅ Masih feasible untuk timeline 4 bulan
5. ✅ Kontribusi jelas: PyTorch implementation + Jetson deployment

---

## 📚 Referensi Utama

- [DeepFusion Paper](https://arxiv.org/abs/2203.08195)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [PointPillars Paper](https://arxiv.org/abs/1812.05784)
- [TensorRT](https://developer.nvidia.com/tensorrt/)
- [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/jetson-orin/)

---

**Siap untuk mulai implementasi DeepFusion (Detection Only)! 🚗💨**

Untuk detail implementasi, lihat file lain di `docs/` dan `IMPLEMENTATION_GUIDE.md` (akan diupdate).
