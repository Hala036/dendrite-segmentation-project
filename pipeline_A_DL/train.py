"""
dl_pipeline/train.py

Train a YOLOv8 segmentation model on the converted dendrite dataset.

FULL WORKFLOW:
    Step 1 — Convert Roboflow COCO export:
        python dl_pipeline/coco_to_yolo.py \\
            --coco-dir path/to/roboflow_download/ \\
            --output-dir data/annotated/

    Step 2 — Train:
        python dl_pipeline/train.py \\
            --data dl_pipeline/dataset.yaml \\
            --epochs 100

    Step 3 — Evaluate:
        python evaluate.py \\
            --test-images data/annotated/images/test/ \\
            --test-masks  data/annotated/masks/test/ \\
            --yolo-weights runs/train/dendrite_seg/weights/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── SEM-specific augmentation settings ───────────────────────────────────────
SEM_AUGMENTATION = dict(
    hsv_h=0.0,        # no hue shift (grayscale images)
    hsv_s=0.0,        # no saturation shift (grayscale images)
    hsv_v=0.2,        # brightness ±20% — simulates SEM beam current variation
    flipud=0.0,       # NO vertical flip — dendrites always grow upward from electrode
    fliplr=0.5,       # horizontal flip valid — no left/right electrode preference
    degrees=15,       # small rotation only — keep electrode baseline near bottom
    translate=0.1,    # slight translation
    scale=0.3,        # zoom — simulates different magnification levels
    shear=2.0,        # minimal shear
    perspective=0.0003,
    mosaic=1.0,       # 4-image mosaic — helps sparse dendrite distribution
    mixup=0.0,        # off — not useful for segmentation
    copy_paste=0.2,   # copy-paste — increases branch diversity
    erasing=0.0,      # off — YOLOv8 applies cutout internally already
    close_mosaic=10,  # disable mosaic in final 10 epochs for stable convergence
)


def verify_dataset(data_yaml: str) -> bool:
    """Verify dataset structure before training."""
    import yaml

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        print(f"[Error] dataset.yaml not found: {data_yaml}")
        print("  Run first: python dl_pipeline/coco_to_yolo.py --coco-dir <download_path>")
        return False

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config.get('path', '.')).resolve()
    ok = True

    for split in ['train', 'val', 'test']:
        rel_path = config.get(split, f'images/{split}')
        img_dir = dataset_path / rel_path
        lbl_dir = dataset_path / rel_path.replace('images', 'labels')

        if not img_dir.exists():
            print(f"  [Warning] {split} images not found: {img_dir}")
            if split == 'train':
                ok = False
            continue

        n_imgs = len(list(img_dir.glob('*.*')))
        n_lbls = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
        status = "✓" if n_lbls > 0 else "⚠ no labels"
        print(f"  [{split}] {n_imgs} images | {n_lbls} labels {status}")

        if split == 'train' and n_lbls == 0:
            ok = False

    return ok


def get_model_for_dataset_size(data_yaml: str) -> str:
    """Auto-select smallest appropriate YOLO model based on training set size."""
    try:
        import yaml
        with open(data_yaml) as f:
            config = yaml.safe_load(f)
        dataset_path = Path(config.get('path', '.')).resolve()
        train_dir = dataset_path / config.get('train', 'images/train')
        n = len(list(train_dir.glob('*.*')))
    except Exception:
        n = 0

    if n < 50:
        model = 'yolov8n-seg.pt'
        reason = f"nano (dataset={n} images, minimizes overfitting)"
    elif n < 200:
        model = 'yolov8s-seg.pt'
        reason = f"small (dataset={n} images)"
    else:
        model = 'yolov8m-seg.pt'
        reason = f"medium (dataset={n} images)"

    print(f"[Auto] Model selected: {model} — {reason}")
    return model


def train(args):
    """Train YOLOv8 segmentation with SEM-optimized configuration."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] Run: pip install ultralytics")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  YOLOv8 Dendrite Segmentation — Training")
    print(f"{'='*60}\n")

    # Verify dataset
    print("[Step 1] Verifying dataset structure...")
    if not verify_dataset(args.data):
        sys.exit(1)
    print()

    # Auto-select model if not specified
    if args.model == 'auto':
        args.model = get_model_for_dataset_size(args.data)

    # Load model
    print(f"[Step 2] Loading model: {args.model}")
    model = YOLO(args.resume if args.resume else args.model)

    print(f"[Step 3] Starting training...")
    print(f"  Data:    {args.data}")
    print(f"  Epochs:  {args.epochs} (patience={args.patience})")
    print(f"  Batch:   {args.batch} | ImgSz: {args.imgsz}")
    print(f"  Output:  {args.project}/{args.name}")
    print(f"  Device:  {args.device or 'auto'}")
    print(f"  Freeze:  first {args.freeze} backbone layers (transfer learning)")
    print()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        resume=bool(args.resume),

        # SEM augmentation
        **SEM_AUGMENTATION,

        # Optimizer
        optimizer='AdamW',
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # Output
        save=True,
        save_period=-1,
        val=True,
        plots=True,
        verbose=True,

        # Segmentation-specific
        overlap_mask=True,
        mask_ratio=4,
        retina_masks=False,

        # Transfer learning — freeze backbone layers
        # With only ~33 training images, the COCO-pretrained backbone
        # already understands edges, textures and shapes. Freezing the
        # first 10 layers preserves these general features and forces
        # training to focus on the segmentation head only.
        # Result: less overfitting, faster convergence, better IoU on small datasets.
        freeze=args.freeze,
    )

    best = Path(args.project) / args.name / 'weights' / 'best.pt'
    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"  Best weights: {best}")

    try:
        d = results.results_dict
        print(f"  Mask mAP50:    {d.get('metrics/mAP50(M)', 'N/A'):.4f}")
        print(f"  Mask mAP50-95: {d.get('metrics/mAP50-95(M)', 'N/A'):.4f}")
    except Exception:
        pass

    print(f"\nNext:")
    print(f"  python dl_pipeline/inference.py --weights {best} --source data/raw/")
    print(f"  python evaluate.py --yolo-weights {best}")
    print(f"{'='*60}\n")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 for SEM Dendrite Segmentation")
    p.add_argument('--data', default='dl_pipeline/dataset.yaml')
    p.add_argument('--model', default='auto',
                   help="yolov8n/s/m/l/x-seg.pt or 'auto'")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr0', type=float, default=1e-3)
    p.add_argument('--lrf', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--device', default='')
    p.add_argument('--project', default='runs/train')
    p.add_argument('--name', default='dendrite_seg')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', default='')
    p.add_argument('--freeze', type=int, default=10,
                   help='Number of backbone layers to freeze for transfer learning. '
                        '10 = freeze backbone, train head only (recommended for <50 images). '
                        '0 = train all layers (fine-tune everything).')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)