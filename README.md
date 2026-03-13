# Lithium Dendrite Segmentation in SEM Images

Final project for Image Processing at Azrieli College of Engineering.

This repository compares two ways of segmenting lithium dendrites from SEM images:

1. A classical computer vision pipeline based on preprocessing, thresholding, morphology, and skeletonization.
2. A deep learning pipeline based on tiled YOLO instance segmentation.

The project scope is not just “make a mask”. It is to compare both approaches visually and quantitatively, inspect their failure cases, and produce clear artifacts for a report or short live demo.

## Project Scope

The repository covers:

- SEM image preprocessing for dendrite visibility improvement
- Classical segmentation and postprocessing
- Skeleton extraction for branch visualization
- YOLO dataset preparation, training, and tiled inference
- Quantitative comparison with IoU, Dice, Precision, and Recall
- Visual comparison artifacts and a short demo notebook

The repository does not try to be a production system. It is a project comparison setup for a small labeled dataset.

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

The labeled evaluation images are under `data/annotated/`.

Important detail:

- The classical pipeline can optionally crop the image bottom to remove an SEM metadata bar.
- For the current annotated evaluation dataset, extra cropping is disabled in the evaluation path so masks stay aligned with the labeled images.

If you change dataset geometry, retrain or regenerate predictions before comparing metrics.

## Environment Setup

Create a virtual environment and install the needed packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install ultralytics opencv-python numpy matplotlib scikit-image scipy pyyaml nbformat
```

If you already have a working `.venv`, use that instead.

## How To Run

### 1. Run the classical pipeline on the annotated test set

```bash
.venv/bin/python run_pipeline_b.py
```

This generates:

- original images for comparison
- classical mask overlays
- skeleton overlays

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

This writes:

- [metrics_results.csv](outputs/metrics_results.csv)

### 4. Create side-by-side comparison artifacts

```bash
.venv/bin/python comparison_results/artifacts.py
```

This combines:

- original image
- classical mask
- YOLO mask
- skeleton

into a single figure per test image.

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

## Current Outputs

Main output folders:

- `outputs/masks` for classical masks
- `outputs/masks_tiled` for tiled YOLO masks
- `outputs/skeletons` for classical skeletons
- `outputs/visuals` for saved figures
- `outputs/weights` for YOLO weights

Combined visual artifacts are written to:

- `comparison_results/artifacts_combined`

## Notes on Evaluation

Metrics are computed against the annotated test set using:

- IoU
- Dice
- Precision
- Recall

The project focuses on comparing behavior, not claiming that one model is universally best. The classical pipeline is easier to explain step by step, while the YOLO pipeline is usually stronger when segmentation quality is the main goal.

## Suggested Entry Points

If you only want the most useful files:

- [run_pipeline_b.py](run_pipeline_b.py)
- [run_A.bash](run_A.bash)
- [metrics.py](metrics.py)
- [comparison_results/artifacts.py](comparison_results/artifacts.py)
- [notebooks/demo.ipynb](notebooks/demo.ipynb)
