from pathlib import Path
import re
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

ORIG_DIR = ROOT / 'classic' / 'original'
CLASSIC_MASK_DIR = REPO_ROOT / 'outputs' / 'masks'
YOLO_MASK_DIR = REPO_ROOT / 'outputs' / 'masks_tiled'
SKELETON_DIR = REPO_ROOT / 'outputs' / 'skeletons'

OUT_DIR = ROOT / 'artifacts_combined'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_base_name(original_name: str) -> str:
    return original_name.replace('_original.png', '')


def resolve_classic_stem(base: str) -> str:
    exact = CLASSIC_MASK_DIR / f'{base}_mask.png'
    if exact.exists():
        return f'{base}_mask'

    match_2e9 = re.match(r'^2e-9_100s_(\d+)_', base)
    if match_2e9:
        return match_2e9.group(1) + '_mask'

    match_70nm = re.match(r'^70nm_diameter_100nm_pitch_(\d+)_', base)
    if match_70nm:
        return f'70nm_diameter_{match_70nm.group(1)}_mask'

    match_ag = re.match(r'^Ag_2e?-?9?_?(\d+[a-z]?)_', base, re.IGNORECASE)
    if match_ag:
        return f'Ag_{match_ag.group(1)}_mask'

    return f'{base}_mask'


def find_expected_files(base: str) -> dict:
    classic_stem = resolve_classic_stem(base)
    skeleton_stem = classic_stem.replace('_mask', '_skeleton')

    return {
        'original': ORIG_DIR / f'{base}_original.png',
        'classic_mask': CLASSIC_MASK_DIR / f'{classic_stem}.png',
        'yolo': YOLO_MASK_DIR / f'{base}_yolo_tiled_mask.png',
        'skeleton': SKELETON_DIR / f'{skeleton_stem}.png',
    }


def combine_one(base: str) -> Path:
    files = find_expected_files(base)

    for label, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} for '{base}': {p}")

    original = read_rgb(files['original'])
    h, w = original.shape[:2]

    classic_mask = cv2.resize(read_rgb(files['classic_mask']), (w, h), interpolation=cv2.INTER_NEAREST)
    yolo = cv2.resize(read_rgb(files['yolo']), (w, h), interpolation=cv2.INTER_NEAREST)
    skeleton = cv2.resize(read_rgb(files['skeleton']), (w, h), interpolation=cv2.INTER_NEAREST)

    out_path = OUT_DIR / f'{base}_artifact.png'

    panels = [
        ('Original', original),
        ('Classic Mask', classic_mask),
        ('YOLO Mask', yolo),
        ('Skeleton', skeleton),
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
