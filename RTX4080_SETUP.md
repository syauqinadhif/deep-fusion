# Quick Start Guide - DeepFusion Training on RTX 4080

## Your PC Specs (Monster! 🔥)
- GPU: RTX 4080 16GB VRAM
- CPU: AMD Ryzen 7 7800X3D (16-core)
- RAM: 32GB
- CUDA: 13.0

## Expected Performance on RTX 4080

| Metric | M4 Pro | RTX 4080 | Speedup |
|--------|---------|----------|---------|
| Per epoch | ~10-15 min | ~1-2 min | **6-8x faster** |
| 80 epochs | ~13-20 hours | **~2-4 hours** | **6-8x faster** |
| Batch size | 2 | 8-16 | - |
| FP16 | Limited | Full support | - |

## Step 1: Copy Project to PC

```bash
# On your Mac, create tarball
cd ~/Documents/Skripsi
tar -czf deepfusion_project.tar.gz deepfusion_project/

# Transfer to PC (choose one):
# 1. Via network share/SMB
# 2. Via Google Drive/Dropbox
# 3. Via external SSD
# 4. Via scp (if you have SSH access)
```

## Step 2: Setup Environment on PC

```bash
# Navigate to project directory
cd /media/mf/MAIN/deepfusion_project

# Create conda environment (recommended)
conda create -n deepfusion python=3.10
conda activate deepfusion

# Install PyTorch with CUDA 13.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install PyYAML tqdm opencv-python matplotlib scipy

# Install additional packages
pip install scikit-learn tensorboard
```

## Step 3: Prepare Dataset

```bash
# Update dataset path in config
nano config_rtx4080.yaml

# Change this line:
root_path: /media/mf/MAIN/KITTI

# Save and exit (Ctrl+X, Y, Enter)
```

## Step 4: Start Training!

```bash
cd src/scripts

# Start training
python train_rtx4080.py \
    --config ../config_rtx4080.yaml \
    --data_path /media/mf/MAIN/KITTI \
    --output_dir ../../results
```

## Monitor Training

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f results/logs/latest.log
```

## Expected Output

```
✓ Using CUDA (NVIDIA GPU)
  GPU: NVIDIA GeForce RTX 4080
  VRAM: 16.0 GB
  CUDA Version: 13.0

============================================================
Building Model
============================================================
Total parameters: 23,456,789

============================================================
Loading Datasets
============================================================
Training samples: 5984
Validation samples: 1497

============================================================
Starting Training on RTX 4080
============================================================

Epoch 1/80: 100%|████| 748/748 [01:23<00:00,  8.92it/s]
  Train Loss: 2.3456
  Val Loss:   2.1234
  Time:       1.2 min/epoch
  ETA:        1.6 hours
```

## Resume If Interrupted

```bash
python train_rtx4080.py \
    --config ../config_rtx4080.yaml \
    --data_path /media/mf/MAIN/KITTI \
    --resume ../../results/checkpoints/latest.pth.tar
```

## Tips for Maximum Performance

1. **Close unnecessary apps** - Free up RAM
2. **Use performance mode** in NVIDIA Control Panel
3. **Cooling** - Ensure good airflow
4. **Disable Windows key logging** (if gaming PC)
5. **Use XDG for desktop** instead of Wayland (Linux)

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size in config_rtx4080.yaml
```yaml
training:
  batch_size: 4  # or even 2
```

### Issue: Slow data loading
**Solution**: Reduce num_workers
```yaml
system:
  num_workers: 8  # or 4
```

### Issue: CUDA error
**Solution**: Update NVIDIA drivers
```bash
# Check current driver
nvidia-smi

# Update if needed
sudo apt update && sudo apt install nvidia-driver-580
```

---

**Training di RTX 4080 akan jauh lebih cepat! 🚀**

Estimasi waktu: 2-4 jam untuk 80 epochs (vs 13-20 jam di M4 Pro)
