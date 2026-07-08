# PointPillars Run Guide

This repo supports two modes:

- **LiDAR-only PointPillars**: the original model.
- **Multimodal PointPillars**: LiDAR + KITTI `image_2` early fusion using `--multimodal`.

Use the already installed conda environment:

```bash
/data/home/zahra/miniconda3/envs/3detection/bin/python
```

Do not install packages unless you intentionally want to change the environment.

## 1. Dataset Layout

Prepare KITTI object detection data like this:

```text
kitti/
  training/
    calib/
    image_2/
    label_2/
    velodyne/
  testing/
    calib/
    image_2/
    velodyne/
```

Images are required for preprocessing and for multimodal training/evaluation. The preprocessing script does not copy images; it reads them from `image_2` and records their metadata.

## 2. Preprocess KITTI

From the repo root:

```bash
cd /data/home/zahra/PointPillars
/data/home/zahra/miniconda3/envs/3detection/bin/python pre_process_kitti.py \
  --data_root /path/to/kitti
```

After preprocessing, the KITTI folder should contain:

```text
kitti/
  training/
    velodyne_reduced/
  testing/
    velodyne_reduced/
  kitti_infos_train.pkl
  kitti_infos_val.pkl
  kitti_infos_trainval.pkl
  kitti_infos_test.pkl
  kitti_gt_database/
  kitti_dbinfos_train.pkl
```

## 3. Train LiDAR-Only PointPillars

```bash
cd /data/home/zahra/PointPillars
/data/home/zahra/miniconda3/envs/3detection/bin/python train.py \
  --data_root /path/to/kitti \
  --saved_path pillar_logs_lidar
```

Checkpoints are written to:

```text
pillar_logs_lidar/checkpoints/
```

## 4. Train Multimodal PointPillars

```bash
cd /data/home/zahra/PointPillars
/data/home/zahra/miniconda3/envs/3detection/bin/python train.py \
  --data_root /path/to/kitti \
  --saved_path pillar_logs_multimodal \
  --multimodal
```

Notes:

- Multimodal mode loads `image_2` images, resizes them to `384x1280`, and fuses ResNet/FPN image features with pillar point features.
- Geometry-changing LiDAR augmentations are enabled in multimodal mode. The dataset carries each point's original LiDAR coordinates for camera-feature sampling; database-sampled points receive zero image features.
- Batch size may need to be smaller than LiDAR-only mode because the image branch uses extra GPU memory.

Example with smaller batch size:

```bash
/data/home/zahra/miniconda3/envs/3detection/bin/python train.py \
  --data_root /path/to/kitti \
  --saved_path pillar_logs_multimodal \
  --batch_size 2 \
  --multimodal
```

## 5. Evaluate LiDAR-Only Checkpoint

```bash
cd /data/home/zahra/PointPillars
/data/home/zahra/miniconda3/envs/3detection/bin/python evaluate.py \
  --data_root /path/to/kitti \
  --ckpt pretrained/epoch_160.pth \
  --saved_path results_lidar
```

Results are written to:

```text
results_lidar/
  results.pkl
  submit/
```

## 6. Evaluate Multimodal Checkpoint

Use a checkpoint trained with `--multimodal`:

```bash
cd /data/home/zahra/PointPillars
/data/home/zahra/miniconda3/envs/3detection/bin/python evaluate.py \
  --data_root /path/to/kitti \
  --ckpt pillar_logs_multimodal/checkpoints/epoch_160.pth \
  --saved_path results_multimodal \
  --multimodal
```

## 7. Quick Smoke Checks

Verify imports:

```bash
/data/home/zahra/miniconda3/envs/3detection/bin/python -c \
  "from pointpillars.model import PointPillars, MultimodalPointPillars; print('ok')"
```

Verify script options:

```bash
/data/home/zahra/miniconda3/envs/3detection/bin/python train.py --help
/data/home/zahra/miniconda3/envs/3detection/bin/python evaluate.py --help
```

## 8. Common Issues

`ModuleNotFoundError: No module named 'torchvision'`

Use the `3detection` env. The `pointpillars` env has PyTorch but does not have `torchvision`.

```bash
/data/home/zahra/miniconda3/envs/3detection/bin/python train.py --help
```

`Could not read KITTI image`

Check that `image_2` exists under both `training/` and `testing/`, and that the `--data_root` path points to the KITTI root.

CUDA out of memory in multimodal mode

Reduce `--batch_size`, for example:

```bash
--batch_size 1
```
