"""
Train the YOLO segmentation model on the tiled dendrite dataset.

This script loads a pretrained YOLO model, trains it on the prepared dataset,
and copies the best weights to `outputs/weights/best.pt` for later prediction.
"""

import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime

# Check ultralytics is installed
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not installed. Run: pip install ultralytics"
    )


# ==============================================================================
# DEVICE DETECTION
# ==============================================================================

def get_device() -> str:
    """
    Pick the best available device for training.

    Returns:
        Device string used by Ultralytics.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = '0'
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Device: CUDA GPU — {gpu_name} ({vram:.1f} GB VRAM)")
            return device
        elif torch.backends.mps.is_available():
            print(f"  Device: Apple MPS (Metal)")
            return 'mps'
        else:
            print(f"  Device: CPU (no GPU found — training will be slow)")
            return 'cpu'
    except ImportError:
        print("  Device: CPU (torch not found)")
        return 'cpu'


# ==============================================================================
# DATASET VALIDATION
# ==============================================================================

def validate_dataset(data_yaml_path: str) -> dict:
    """
    Check that the dataset structure looks correct before training.

    Args:
        data_yaml_path: Path to `data.yaml`.

    Returns:
        Parsed yaml data.

    Raises:
        AssertionError: If the dataset is missing something important.
    """
    yaml_path = Path(data_yaml_path)
    assert yaml_path.exists(), \
        f"data.yaml not found at {data_yaml_path}\n" \
        f"Did you run tile_dataset.py first?"

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Resolve paths relative to yaml location
    root = Path(data.get('path', yaml_path.parent))

    print("\nDataset validation:")

    for split in ['train', 'val', 'test']:
        if split not in data:
            if split == 'test':
                print(f"  {split:6s}: not defined (optional, skipping)")
                continue
            else:
                raise AssertionError(f"'{split}' key missing from data.yaml")

        img_dir = root / data[split]
        lbl_dir = img_dir.parent.parent / 'labels' / img_dir.name \
                  if 'images' in str(img_dir) \
                  else img_dir.parent / 'labels'

        # Handle Roboflow structure: images/ and labels/ are siblings
        lbl_dir = Path(str(img_dir).replace('images', 'labels'))

        assert img_dir.exists(), \
            f"  {split} images dir not found: {img_dir}"

        images = list(img_dir.glob('*.png')) + \
                 list(img_dir.glob('*.jpg')) + \
                 list(img_dir.glob('*.jpeg'))

        labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []

        # Count non-empty label files (empty = tile with no dendrites, OK for train)
        non_empty_labels = [l for l in labels if l.stat().st_size > 0]

        print(f"  {split:6s}: {len(images):4d} images | "
              f"{len(labels):4d} label files | "
              f"{len(non_empty_labels):4d} with annotations")

        if split in ['train', 'val']:
            assert len(images) > 0, \
                f"No images found in {split} split at {img_dir}"
            assert len(non_empty_labels) > 0, \
                f"No annotated labels found in {split} split — " \
                f"did tile_dataset.py run correctly?"

    print(f"  Classes: {data.get('names', 'not specified')}")
    print(f"  Dataset root: {root}")
    print("  ✓ Dataset looks valid\n")

    return data


# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================

def build_train_config(args) -> dict:
    """
    Build the config dictionary passed to `model.train()`.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Training config dictionary.
    """
    return {
        # ── Data ────────────────────────────────────────────────────────────
        'data': args.data,
            # Path to data.yaml — tells YOLO where train/val/test images are

        # ── Training duration ────────────────────────────────────────────────
        'epochs': args.epochs,
            # Number of full passes through the training data.
            # With 70-80 tiles: 100 epochs is a good starting point.
            # More epochs → better fit, but risk of overfitting on small datasets.
            # Watch val/loss in the output — if it stops decreasing, you can stop.

        'patience': args.patience,
            # Early stopping: stop training if val loss doesn't improve
            # for this many consecutive epochs. Prevents overfitting.
            # 20 means "give up if no improvement for 20 epochs."

        # ── Image size ───────────────────────────────────────────────────────
        'imgsz': args.imgsz,
            # Input image size. Must match your tile size (640).
            # YOLO internally resizes all images to this before processing.

        # ── Batch size ───────────────────────────────────────────────────────
        'batch': args.batch,
            # Number of tiles processed per gradient update step.
            # Larger batch = more stable gradients, needs more VRAM.
            # -1 = auto-detect based on available VRAM (recommended).
            # If you get CUDA out-of-memory errors, reduce to 8 or 4.

        # ── Output ───────────────────────────────────────────────────────────
        'project': str(Path(args.output).resolve()),
            # Parent directory for all training outputs

        'name': args.run_name,
            # Subdirectory name for this specific training run.
            # Outputs saved to: args.output / args.run_name /

        'exist_ok': False,
            # If True, overwrites existing run directory.
            # False = creates a new numbered directory (run1, run2, etc.)

        # ── Transfer learning ────────────────────────────────────────────────
        'freeze': args.freeze,
            # Number of backbone layers to freeze (not update during training).
            # None = fine-tune the entire model (best for small datasets).
            # 10 = freeze first 10 layers, only train the detection head.
            # For 20 images, freezing nothing gives the model more flexibility.

        # ── Optimizer ────────────────────────────────────────────────────────
        'optimizer': 'AdamW',
            # AdamW is more stable than SGD for fine-tuning on small datasets.
            # It handles sparse gradients better — important when most tiles
            # have few dendrite pixels relative to background.

        'lr0': args.lr,
            # Initial learning rate. How big each weight update step is.
            # Too high → training diverges (loss goes up instead of down).
            # Too low → training is slow, may get stuck.
            # 0.001 is conservative and safe for transfer learning.

        'lrf': 0.01,
            # Final learning rate as a fraction of lr0.
            # Learning rate decays from lr0 → lr0 * lrf during training.
            # This "cooling down" helps the model fine-tune at the end.

        'weight_decay': 0.0005,
            # Regularization: penalizes large weights to prevent overfitting.
            # Small value is standard for fine-tuning.

        # ── Augmentation ─────────────────────────────────────────────────────
        'hsv_h': 0.0,
            # Hue augmentation. Set to 0 for grayscale SEM images —
            # there is no color information, so hue shifts are meaningless.

        'hsv_s': 0.0,
            # Saturation augmentation. Same reason — SEM images have no color.

        'hsv_v': 0.3,
            # Brightness augmentation. Randomly varies image brightness ±30%.
            # This is useful because SEM images vary in exposure.

        'fliplr': 0.5,
            # Horizontal flip probability. Dendrites have no preferred direction,
            # so flipping is a valid augmentation that doubles effective dataset size.

        'flipud': 0.3,
            # Vertical flip probability. Less common but valid for top-down (Type 2)
            # images where there's no physical up/down constraint.

        'degrees': 15.0,
            # Random rotation up to ±15 degrees. Dendrites grow in various
            # directions so rotation augmentation is physically valid.

        'scale': 0.3,
            # Random scale variation ±30%. Helps model handle dendrites
            # at slightly different zoom levels.

        'mosaic': 0.0,
            # Mosaic augmentation (combines 4 tiles into one). Disabled here
            # because our tiles are already small and densely annotated —
            # mosaic would make the effective image too crowded.

        # ── Hardware ─────────────────────────────────────────────────────────
        'device': get_device(),
            # Auto-detected above: 'cuda:0', 'mps', or 'cpu'

        'workers': 4,
            # Number of CPU threads for loading training data in parallel.
            # 4 is safe. Reduce to 0 if you get multiprocessing errors on Windows.

        # ── Logging & saving ─────────────────────────────────────────────────
        'save': True,
            # Save model checkpoints during training.

        'save_period': 10,
            # Save a checkpoint every N epochs (in addition to best.pt).
            # Useful for resuming if training crashes.

        'val': True,
            # Run validation after each epoch and report metrics.
            # This is how YOLO decides which checkpoint is "best."

        'plots': True,
            # Save training plots: loss curves, PR curves, confusion matrix.
            # Very useful for your report's results section.

        'verbose': True,
            # Print per-epoch metrics to console.
    }


# ==============================================================================
# POST-TRAINING: COPY BEST WEIGHTS
# ==============================================================================

def copy_best_weights(run_dir: Path, output_weights_dir: str) -> Path:
    """
    Copy the best saved weights to a fixed output path.

    Args:
        run_dir: Training run folder.
        output_weights_dir: Folder where `best.pt` will be copied.

    Returns:
        Path to the copied weights file.
    """
    src = run_dir / 'weights' / 'best.pt'
    dst_dir = Path(output_weights_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / 'best.pt'

    if src.exists():
        shutil.copy2(str(src), str(dst))
        print(f"\n  Best weights copied to: {dst}")
    else:
        print(f"\n  WARNING: best.pt not found at {src}")
        print(f"  Check {run_dir / 'weights'} for available checkpoints")

    return dst


def resolve_run_dir(results, args) -> Path:
    """
    Find the run directory that Ultralytics created.
    """
    save_dir = getattr(results, 'save_dir', None)
    if save_dir:
        return Path(save_dir)

    # Fallback: search expected output location
    output_path = Path(args.output)
    run_dirs = sorted(output_path.glob(f"{args.run_name}*"),
                      key=lambda p: p.stat().st_mtime)
    if run_dirs:
        return run_dirs[-1]

    # Last resort fallback for Ultralytics default nesting behavior
    return Path('runs') / 'segment' / output_path / args.run_name


# ==============================================================================
# TRAINING SUMMARY
# ==============================================================================

def print_training_summary(results, run_dir: Path, best_weights_path: Path) -> None:
    """
    Print a short summary after training finishes.

    Args:
        results: YOLO training results object.
        run_dir: Training run directory.
        best_weights_path: Copied best weights path.
    """
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)

    # Extract final metrics if available
    try:
        metrics = results.results_dict
        box_map  = metrics.get('metrics/mAP50(B)',   'N/A')
        seg_map  = metrics.get('metrics/mAP50(M)',   'N/A')
        seg_map95 = metrics.get('metrics/mAP50-95(M)', 'N/A')

        print(f"\n  Segmentation mAP@50:       "
              f"{seg_map:.3f}" if isinstance(seg_map, float) else f"  {seg_map}")
        print(f"  Segmentation mAP@50-95:    "
              f"{seg_map95:.3f}" if isinstance(seg_map95, float) else f"  {seg_map95}")
        print(f"  Box mAP@50:                "
              f"{box_map:.3f}" if isinstance(box_map, float) else f"  {box_map}")
    except Exception:
        print("  (metrics not available in results object)")

    print(f"\n  Run directory:   {run_dir}")
    print(f"  Best weights:    {best_weights_path}")
    print(f"  Training plots:  {run_dir}")
    print(f"\nNext step: python pipeline_A_yolo/predict.py "
          f"--weights {best_weights_path} --folder data/annotated/test/images")
    print("=" * 55)


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train(args) -> None:
    """
    Run the full training pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    print("=" * 55)
    print("PIPELINE A — YOLO26 TRAINING")
    print("=" * 55)
    print(f"  Model:    {args.model}")
    print(f"  Data:     {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Run name: {args.run_name}")

    # Step 1: Validate dataset structure
    validate_dataset(args.data)

    # Step 2: Load pretrained model
    # YOLO auto-downloads the weights on first run if not found locally
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"  ✓ Model loaded — {sum(p.numel() for p in model.model.parameters()):,} parameters")

    # Step 3: Build config and train
    config = build_train_config(args)

    print(f"\nStarting training...")
    print(f"  Output: {args.output}/{args.run_name}/")
    print(f"  Early stopping patience: {args.patience} epochs\n")

    results = model.train(**config)

    # Step 4: Resolve actual run directory YOLO created
    run_dir = resolve_run_dir(results, args)

    # Step 5: Copy best weights
    best_weights = copy_best_weights(run_dir, 'outputs/weights')

    # Step 6: Summary
    print_training_summary(results, run_dir, best_weights)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train YOLO26-seg on tiled dendrite dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument(
        '--data', type=str,
        default='data/tiled/data.yaml',
        help='Path to data.yaml from tile_dataset.py'
    )

    # Model
    parser.add_argument(
        '--model', type=str,
        default='yolo26n-seg.pt',
        help='YOLO model to use. Options: yolo26n-seg.pt (fast), '
             'yolo26s-seg.pt (balanced), yolo26m-seg.pt (accurate).'
    )

    # Training duration
    parser.add_argument(
        '--epochs', type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--patience', type=int,
        default=20,
        help='Early stopping patience (epochs without improvement)'
    )

    # Image size — must match tile size from tile_dataset.py
    parser.add_argument(
        '--imgsz', type=int,
        default=640,
        help='Input image size (must match tile size used in tile_dataset.py)'
    )

    # Batch
    parser.add_argument(
        '--batch', type=int,
        default=-1,
        help='Batch size (-1 = auto based on VRAM)'
    )

    # Optimizer
    parser.add_argument(
        '--lr', type=float,
        default=0.001,
        help='Initial learning rate'
    )

    # Transfer learning
    parser.add_argument(
        '--freeze', type=int,
        default=None,
        help='Number of backbone layers to freeze (None = fine-tune all)'
    )

    # Output
    parser.add_argument(
        '--output', type=str,
        default='outputs/weights',
        help='Directory to save training runs'
    )
    parser.add_argument(
        '--run-name', type=str,
        default=f'dendrite_{datetime.now().strftime("%Y%m%d_%H%M")}',
        help='Name for this training run (used as output subdirectory name)'
    )

    # Resume
    parser.add_argument(
        '--resume', type=str,
        default=None,
        help='Path to last.pt to resume interrupted training'
    )

    args = parser.parse_args()

    # Handle resume separately — YOLO's resume doesn't use train config
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        train(args)
