"""
02_tissue_mask.py — Generate a binary tissue mask from a whole-slide image
using Otsu thresholding plus connected-component filtering to remove ink
markers, dust specks, and other non-tissue artifacts.

Usage:
    python scripts/02_tissue_mask.py data/tumor/tumor_001.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import (
    closing, opening, disk, remove_small_objects
)


def generate_tissue_mask(slide_path: Path, output_dir: Path,
                         mask_level: int | None = None,
                         min_object_size: int = 15000) -> None:
    slide = openslide.OpenSlide(str(slide_path))

    if mask_level is None:
        mask_level = slide.level_count - 4
    mask_level = max(0, min(mask_level, slide.level_count - 1))

    width, height = slide.level_dimensions[mask_level]
    print(f"Reading level {mask_level}: {width} x {height} pixels")

    region = slide.read_region((0, 0), mask_level, (width, height))
    rgb = np.array(region.convert("RGB"))

    # Otsu on luminance: tissue (and ink) is darker than background
    gray = rgb2gray(rgb)
    threshold = threshold_otsu(gray)
    tissue = gray < threshold
    print(f"Otsu threshold: {threshold:.3f}")
    print(f"After Otsu:                 {tissue.mean():.1%} flagged")

    # Morphological cleanup: close small holes, smooth edges
    tissue = closing(tissue, disk(4))
    tissue = opening(tissue, disk(2))
    print(f"After morphology:           {tissue.mean():.1%} flagged")

    # Connected-component size filter: remove anything too small to be tissue.
    # This eliminates ink markers, dust specks, and isolated artifacts.
    tissue = remove_small_objects(tissue, min_size=min_object_size)
    print(f"After size filter (>={min_object_size}px): "
          f"{tissue.mean():.1%} flagged")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = slide_path.stem

    mask_png = output_dir / f"{stem}_mask.png"
    Image.fromarray((tissue * 255).astype(np.uint8)).save(mask_png)
    print(f"Mask PNG saved:    {mask_png}")

    mask_npy = output_dir / f"{stem}_mask.npy"
    np.save(mask_npy, tissue)
    print(f"Mask NPY saved:    {mask_npy}")

    overlay = rgb.copy()
    overlay[~tissue] = (overlay[~tissue] * 0.3).astype(np.uint8)
    overlay_png = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_png)
    print(f"Overlay PNG saved: {overlay_png}")

    slide.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--mask-level", type=int, default=None)
    parser.add_argument("--min-object-size", type=int, default=15000,
                        help="Minimum connected-component size in pixels "
                             "(default 5000; raise for tighter filtering)")
    args = parser.parse_args()

    if not args.slide.exists():
        sys.exit(f"Error: {args.slide} does not exist")

    generate_tissue_mask(args.slide, args.output_dir,
                         args.mask_level, args.min_object_size)


if __name__ == "__main__":
    main()