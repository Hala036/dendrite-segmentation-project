# Lithium Dendrite Segmentation in SEM Images

Final project for Image Processing at Azrieli College of Engineering.

This project compares two methods for segmenting lithium dendrites from SEM images:

1. A classical computer vision pipeline
2. A tiled YOLO segmentation pipeline

The goal is to compare both approaches visually and quantitatively and show their main strengths and weaknesses on the same dataset.

## What This Repository Includes

- preprocessing of grayscale SEM images
- classical segmentation, mask cleanup, and skeletonization
- YOLO dataset tiling, training, and tiled inference
- metric-based comparison using IoU, Dice, Precision, and Recall
- visual comparison artifacts and a short demo notebook

This is a project repository, not a production package.

## Main Pipelines

### Pipeline B: Classical Computer Vision

Implemented in:

- [preprocessing.py](pipeline_B_classic/preprocessing.py)
- [segmentation.py](pipeline_B_classic/segmentation.py)
- [postprocessing.py](pipeline_B_classic/postprocessing.py)
- [skeletonization.py](pipeline_B_classic/skeletonization.py)
- [run_pipeline_b.py](run_pipeline_b.py)

Typical flow:

1. Load grayscale SEM image
2. Normalize intensities
3. Apply CLAHE and bilateral filtering
4. Segment with thresholding
5. Clean the mask with morphology
6. Remove unreasonable large compact regions with simple heuristics
7. Skeletonize the final mask

### Pipeline A: YOLO Tiled Segmentation

Implemented in:

- [tile_dataset.py](pipeline_A_yolo/tile_dataset.py)
- [train.py](pipeline_A_yolo/train.py)
- [predict_tiled.py](pipeline_A_yolo/predict_tiled.py)
- [run_A.bash](run_A.bash)

Typical flow:

1. Tile the training split
2. Train a YOLO segmentation model
3. Run tiled inference on full-size test images
4. Stitch tile predictions back into one full-image mask

## Repository Layout

```text
dendrite-segmentation/
├── comparison_results/
│   ├── artifacts.py
│   └── artifacts_combined/
├── data/
│   ├── annotated/
│   ├── raw/
│   └── tiled/
├── notebooks/
│   ├── demo.ipynb
│   └── tune_pipeline_b_new.ipynb
├── outputs/
│   ├── masks/
│   ├── masks_tiled/
│   ├── metrics_results.csv
│   ├── skeletons/
│   ├── visuals/
│   └── weights/
├── pipeline_A_yolo/
├── pipeline_B_classic/
├── metrics.py
├── run_A.bash
└── run_pipeline_b.py
```

## Data Notes

The labeled dataset used for comparison is under `data/annotated/`.

If dataset geometry changes, both pipelines should be regenerated before comparing metrics again.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install ultralytics opencv-python numpy matplotlib scikit-image scipy pyyaml nbformat
```

## How To Run

### 1. Run the classical pipeline on the annotated test set

```bash
.venv/bin/python run_pipeline_b.py
```

This generates the classical masks, skeletons, and comparison visuals for the annotated test set.

### 2. Run the YOLO pipeline

```bash
bash run_A.bash
```

This script:

1. builds the tiled dataset
2. trains YOLO
3. runs tiled inference on the test set

### 3. Compute metrics

```bash
.venv/bin/python metrics.py
```

This writes [metrics_results.csv](outputs/metrics_results.csv).

### 4. Create side-by-side comparison artifacts

```bash
.venv/bin/python comparison_results/artifacts.py
```

This combines the original image, classical mask, YOLO mask, and skeleton into one figure per test image.

## Demo Notebook

For a short live recording, use:

- [demo.ipynb](notebooks/demo.ipynb)

It shows:

1. raw image and histogram
2. classical preprocessing
3. classical segmentation
4. classical postprocessing
5. skeletonization
6. YOLO tiled inference
7. final side-by-side comparison

Before running it, set:

- `IMAGE_PATH`
- `WEIGHTS_PATH`

in the first code cell.

## Published Outputs

This public version keeps only selected final outputs:

- `outputs/masks` for the final classical evaluation masks
- `outputs/masks_tiled` for the final YOLO tiled evaluation masks
- `outputs/skeletons` for the final classical skeletons
- `outputs/metrics_results.csv` for the comparison table
- `outputs/visuals/demo_comparison.png` for the demo figure
- `outputs/weights/best.pt` for the trained YOLO weights

## Evaluation

Metrics are computed on the annotated test set using:

- IoU
- Dice
- Precision
- Recall

The classical pipeline is easier to explain step by step, while the YOLO pipeline is usually stronger when raw segmentation quality is the main goal.

## Suggested Entry Points

If you only want the most useful files:

- [run_pipeline_b.py](run_pipeline_b.py)
- [run_A.bash](run_A.bash)
- [metrics.py](metrics.py)
- [comparison_results/artifacts.py](comparison_results/artifacts.py)
- [notebooks/demo.ipynb](notebooks/demo.ipynb)
