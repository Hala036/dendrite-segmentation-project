#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
# /Users/halaabutair/Desktop/Year 4/Image Processing/dendrite-segmentation

# 0) (Optional) install deps if needed
python3 -m pip install -U ultralytics opencv-python pyyaml

# 1) Build tiled dataset
python3 pipeline_A_yolo/tile_dataset.py \
  --input data/annotated \
  --output data/tiled \
  --tile-size 640 \
  --overlap 0.2

# 2) Train model (YOLO26)
RUN_NAME="dendrite_$(date +%Y%m%d_%H%M)"
python3 pipeline_A_yolo/train.py \
  --data data/tiled/data.yaml \
  --model yolo26n-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch -1 \
  --output outputs/weights \
  --run-name "${RUN_NAME}"

# 3) Predict on full test images
python3 pipeline_A_yolo/predict.py \
  --weights outputs/weights/best.pt \
  --folder data/annotated/test/images \
  --output outputs/masks \
  --conf 0.25 \
  --iou 0.45 \
  --visualize
