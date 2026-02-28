# Morphological Segmentation of Lithium Dendrites in SEM Images

Final Project – Image Processing  
Azrieli College of Engineering  

---

## Overview

Lithium dendrites are microscopic metallic structures that grow on battery anodes during charging cycles. Their fractal geometry may lead to separator puncture and internal short circuits (thermal runaway).

This project implements and compares two segmentation approaches for dendrite extraction from SEM (Scanning Electron Microscope) images:

1. **Classical Computer Vision Pipeline**
2. **Deep Learning (YOLOv8-Seg)**

The goal is to evaluate accuracy, robustness, and failure modes of each method.

---

## Project Structure

```text
dendrite-segmentation/
├── data/
│   ├── raw/          # raw SEM images
│   ├── processed/    # intermediate/processed images
│   └── annotated/    # labels/annotations
├── pipeline_a_classic/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── postprocessing.py
│   └── run_pipeline_a.py
├── pipeline_b_yolo/
├── outputs/
│   ├── masks/
│   ├── skeletons/
│   ├── visuals/
│   └── weights/
├── evaluation/
├── notebooks/
├── utils/
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run (Implemented Step)

Run preprocessing on one image:

```bash
python3 pipeline_a_classic/preprocessing.py data/raw/2e-9_100s_002.tif
```

Expected output figure:

- `outputs/visuals/preprocessing_check.png`
