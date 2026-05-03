"""
02_tissue_mask.py — Generate a binary tissue mask from a whole-slide image
using Otsu thresholding on a low-resolution level, with cleanup steps to
exclude ink markers and edge artifacts.

Usage:
    python scripts/02_tissue_mask.py data/tumor/tumor_001.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk


def generate_tissue_mask(slide_path: Path, output_dir: Path,
                         mask_level: int | None = None,
                         min_saturation: float = 0.05) -> None:
    slide = openslide.OpenSlide(str(slide_path))

    if mask_level is None:
        mask_level = slide.level_count - 4
    mask_level = max(0, min(mask_level, slide.level_count - 1))

    width, height = slide.level_dimensions[mask_level]
    print(f"Reading level {mask_level}: {width} x {height} pixels")

    region = slide.read_region((0, 0), mask_level, (width, height))
    rgb = np.array(region.convert("RGB"))

    # Otsu on luminance: tissue is darker than the white-ish background
    gray = rgb2gray(rgb)
    threshold = threshold_otsu(gray)
    tissue = gray < threshold
    print(f"Otsu threshold: {threshold:.3f}")
    print(f"After Otsu:                  {tissue.mean():.1%} flagged")

    # Reject low-saturation pixels (ink markers, dark scanner artifacts).
    # H&E stain has high saturation; ink and shadows are near-grey.
    hsv = rgb2hsv(rgb)
    saturation = hsv[..., 1]
    is_stained = saturation > min_saturation
    tissue = tissue & is_stained
    print(f"After saturation filter:     {tissue.mean():.1%} flagged")

    # Morphological cleanup
    tissue = binary_closing(tissue, disk(4))   # fill small holes inside tissue
    tissue = binary_opening(tissue, disk(5))   # remove specks and ink remnants
    print(f"After morphological cleanup: {tissue.mean():.1%} flagged")

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
    parser.add_argument("slide", type=Path, help="Path to a .tif WSI")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--mask-level", type=int, default=None)
    parser.add_argument("--min-saturation", type=float, default=0.05,
                        help="Pixels below this HSV saturation are excluded "
                             "(removes ink markers; default 0.05)")
    args = parser.parse_args()

    if not args.slide.exists():
        sys.exit(f"Error: {args.slide} does not exist")

    generate_tissue_mask(args.slide, args.output_dir,
                         args.mask_level, args.min_saturation)


if __name__ == "__main__":
    main()