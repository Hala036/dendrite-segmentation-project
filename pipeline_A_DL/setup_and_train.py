"""
setup_and_train.py

One-script runner for the complete DL pipeline:
  1. Convert Roboflow COCO download → YOLO format
  2. Verify dataset structure
  3. Train YOLOv8
  4. Run inference on test set
  5. Print summary metrics

Usage:
    python setup_and_train.py --coco-dir path/to/roboflow_download/

Example with all options:
    python setup_and_train.py \\
        --coco-dir ~/Downloads/dendrite-dataset-coco/ \\
        --output-dir data/annotated/ \\
        --epochs 100 \\
        --model yolov8n-seg.pt \\
        --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Full DL Pipeline Setup + Training")
    p.add_argument('--coco-dir', required=True,
                   help="Path to extracted Roboflow COCO zip folder")
    p.add_argument('--output-dir', default='data/annotated/',
                   help="Where to store converted YOLO dataset")
    p.add_argument('--model', default='auto',
                   help="YOLO model size or 'auto'")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='',
                   help="'' = auto, 'cpu', '0' = first GPU")
    p.add_argument('--skip-convert', action='store_true',
                   help="Skip conversion if already done")
    p.add_argument('--skip-train', action='store_true',
                   help="Skip training (convert only)")
    return p.parse_args()


def main():
    args = parse_args()

    yaml_path = str(Path(args.output_dir).resolve() / 'dataset.yaml')

    # ── Step 1: Convert COCO → YOLO ──────────────────────────────────────────
    if not args.skip_convert:
        print("\n" + "="*60)
        print("  STEP 1: Converting Roboflow COCO -> YOLO format")
        print("="*60)
        from coco_to_yolo import convert_roboflow_coco
        convert_roboflow_coco(
            coco_dir=args.coco_dir,
            output_dir=args.output_dir,
            yaml_path=yaml_path
        )
    else:
        print("\n[Skip] COCO conversion (--skip-convert)")

    if args.skip_train:
        print("\n[Skip] Training (--skip-train). Dataset ready at:", args.output_dir)
        return

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2: Training YOLOv8 Segmentation")
    print("="*60)

    # Build args namespace for train()
    import types
    train_args = types.SimpleNamespace(
        data=yaml_path,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=1e-3,
        lrf=0.01,
        patience=30,
        device=args.device,
        project='runs/train',
        name='dendrite_seg',
        workers=2,
        seed=42,
        resume='',
        freeze=10   # freeze backbone for transfer learning
    )

    from train import train
    results = train(train_args)

    best_weights = Path('runs/train/dendrite_seg/weights/best.pt')

    # ── Step 3: Inference on test set ─────────────────────────────────────────
    if best_weights.exists():
        try:
            print("\n" + "="*60)
            print("  STEP 3: Running inference on test set")
            print("="*60)

            import types
            inf_args = types.SimpleNamespace(
                weights=str(best_weights),
                source=str(Path(args.output_dir) / 'images/test'),
                output='results/yolo_test/',
                conf=0.25,
                iou=0.45,
                imgsz=args.imgsz,
                device=args.device,
                save_overlay=True,
                save_skeleton=True
            )

            from inference import run_inference
            run_inference(inf_args)
        except ImportError:
            print("\n[Skip] STEP 3: inference.py not found. Run manually:")
            print(f"  python pipeline_A_DL/inference.py --weights {best_weights} --source data/raw/")

    print("\n" + "="*60)
    print("  ALL DONE!")
    print("="*60)
    print(f"  Weights:    runs/train/dendrite_seg/weights/best.pt")
    print(f"  Inference:  results/yolo_test/")
    print(f"\n  To evaluate against ground truth:")
    print(f"  python evaluate.py \\")
    print(f"      --test-images {args.output_dir}/images/test/ \\")
    print(f"      --test-masks  {args.output_dir}/masks/test/ \\")
    print(f"      --yolo-weights runs/train/dendrite_seg/weights/best.pt")


if __name__ == '__main__':
    main()