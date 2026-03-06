# Training DeepFusion on MacBook Pro M4 Pro

## Storage Requirements ✅

You have **250 GB free** - this is enough!

```
Required Storage:
├── KITTI Dataset:        ~25 GB
├── Virtual Environment:  ~5 GB
├── Checkpoints:          ~10 GB
├── Logs:                 ~3 GB
└── Buffer:               ~10 GB
─────────────────────────────
Total:                   ~53 GB  ← You have 250 GB! ✅
```

## Step 1: Download KITTI Dataset

```bash
# Create dataset directory
mkdir -p ~/Datasets/KITTI

# Download KITTI 3D Object Detection
cd ~/Datasets/KITTI

# Download using wget or curl
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# Unzip
unzip data_object_velodyne.zip -d .
unzip data_object_image_2.zip -d .
unzip data_object_calib.zip -d .
unzip data_object_label_2.zip -d .

# Clean up zip files
rm *.zip

# Organize structure
# Should have:
# ~/Datasets/KITTI/
# ├── training/    (or velodyne/, image_2/, etc.)
# └── testing/
```

**Alternative**: Use torrent download or download individual subsets.

## Step 2: Setup Environment

```bash
cd ~/Documents/Skripsi/deepfusion_project

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

## Step 3: Verify MPS Support

```bash
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output: `MPS available: True`

## Step 4: Start Training

```bash
cd code/scripts

# Start training
python train_m4.py \
    --config ../config_m4_pro.yaml \
    --data_path ~/Datasets/KITTI \
    --output_dir ../../results
```

## Training Expectations

### Time Estimates (M4 Pro)
```
Lite Model (recommended):
├── Per epoch: ~10-15 minutes
├── 80 epochs: ~13-20 hours
└── Storage for checkpoints: ~5 GB

Standard Model:
├── Per epoch: ~20-30 minutes
├── 80 epochs: ~26-40 hours
└── Storage for checkpoints: ~10 GB
```

### Memory Usage
```
Expected MPS Memory:
├── Model: ~1-2 GB
├── Batch (size=2): ~2-4 GB
├── Optimizer states: ~1-2 GB
└── Peak: ~6-8 GB (well within M4 Pro limits)
```

## Tips for Training on M4 Pro

### 1. Monitor Resources
```bash
# In another terminal, monitor memory
sudo powermetrics --samplers gpu_power -i 1000
```

### 2. Prevent Thermal Throttling
- Keep MacBook on a hard surface
- Elevate for better airflow
- Avoid intensive background tasks

### 3. Training While Sleeping
```bash
# Prevent sleep during training
caffeinate -d &
python train_m4.py ...
```

### 4. Resume from Checkpoint
```bash
# If training is interrupted
python train_m4.py \
    --config ../config_m4_pro.yaml \
    --data_path ~/Datasets/KITTI \
    --resume ../../results/checkpoints/latest.pth.tar
```

## Common Issues

### Issue: Out of Memory
**Solution**: Reduce batch size in config
```yaml
training:
  batch_size: 1  # or even 1
```

### Issue: Slow Training
**Solution**:
- Use `DeepFusionLite` instead of `DeepFusion`
- Reduce dataset size for testing
- Close other applications

### Issue: Storage Space
**Solution**:
- Delete old checkpoints
- Reduce `save_freq` in config
- Move dataset to external drive

## Expected Results

After 80 epochs, you should achieve:
```
KITTI Val Set:
├── Car (Moderate):     mAP > 65%
├── Pedestrian (Mod):   mAP > 55%
└── Cyclist (Mod):      mAP > 70%
```

## Next Steps After Training

1. **Evaluate**:
   ```bash
   python evaluate.py --checkpoint ../../results/checkpoints/best.pth.tar
   ```

2. **Export for Jetson**:
   ```bash
   python export.py --checkpoint ../../results/checkpoints/best.pth.tar --format all
   ```

3. **Transfer to Jetson** for deployment!

---

**Good luck with your training! 🚀**
