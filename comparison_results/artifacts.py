from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

ORIG_DIR = ROOT / 'classic' / 'original'
CLASSIC_MASK_DIR = ROOT / 'classic' / 'mask_overlay'
YOLO_OVERLAY_DIR = ROOT / 'yolo_tiled' / 'overlays'
SKELETON_DIR = ROOT / 'classic' / 'skeleton_overlay'

OUT_DIR = ROOT / 'artifacts_combined'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_base_name(original_name: str) -> str:
    return original_name.replace('_original.png', '')


def find_expected_files(base: str) -> dict:
    return {
        'original': ORIG_DIR / f'{base}_original.png',
        'classic_mask': CLASSIC_MASK_DIR / f'{base}_mask_overlay.png',
        'yolo': YOLO_OVERLAY_DIR / f'{base}_tiled_overlay.png',
        'skeleton': SKELETON_DIR / f'{base}_skeleton_overlay.png',
    }


def combine_one(base: str) -> Path:
    files = find_expected_files(base)

    for label, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} for '{base}': {p}")

    original = read_rgb(files['original'])
    h, w = original.shape[:2]

    classic_mask = cv2.resize(read_rgb(files['classic_mask']), (w, h), interpolation=cv2.INTER_LINEAR)
    yolo = cv2.resize(read_rgb(files['yolo']), (w, h), interpolation=cv2.INTER_LINEAR)
    skeleton = cv2.resize(read_rgb(files['skeleton']), (w, h), interpolation=cv2.INTER_LINEAR)

    out_path = OUT_DIR / f'{base}_artifact.png'

    panels = [
        ('Original', original),
        ('Classic Mask Overlay', classic_mask),
        ('YOLO Overlay', yolo),
        ('Skeleton Overlay', skeleton),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(base, fontsize=12)

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Add visible spacing between panels and leave room for title
    plt.subplots_adjust(wspace=0.06, top=0.88)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    return out_path


def main() -> None:
    originals = sorted(ORIG_DIR.glob('*_original.png'))
    if not originals:
        raise FileNotFoundError(f'No originals found in {ORIG_DIR}')

    print(f'Found {len(originals)} originals')

    written = []
    for p in originals:
        base = get_base_name(p.name)
        out = combine_one(base)
        written.append(out)
        print(f'  Saved: {out.name}')

    print(f'\nDone. Wrote {len(written)} combined artifacts to: {OUT_DIR}')


if __name__ == '__main__':
    main()
