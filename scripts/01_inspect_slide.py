"""
01_inspect_slide.py — Open a whole-slide image, print its metadata,
and save a low-resolution thumbnail.

Usage:
    python scripts/01_inspect_slide.py data/tumor/tumor_001.tif
"""

import argparse
import sys
from pathlib import Path

import openslide


def inspect_slide(slide_path: Path, output_dir: Path) -> None:
    slide = openslide.OpenSlide(str(slide_path))

    print(f"File:              {slide_path.name}")
    print(f"Dimensions (L0):   {slide.dimensions}")
    print(f"Level count:       {slide.level_count}")
    print(f"Level dimensions:")
    for i, dims in enumerate(slide.level_dimensions):
        downsample = slide.level_downsamples[i]
        print(f"  L{i}: {dims}  (downsample {downsample:.2f}x)")

    # Selected properties — the full dict has dozens of vendor-specific keys
    interesting_keys = [
        "openslide.vendor",
        "openslide.objective-power",
        "openslide.mpp-x",
        "openslide.mpp-y",
        "tiff.ImageDescription",
    ]
    print("\nSelected properties:")
    for key in interesting_keys:
        value = slide.properties.get(key, "<not set>")
        print(f"  {key}: {value}")

    # Thumbnail — 1024 px on the long side
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail = slide.get_thumbnail((1024, 1024))
    thumbnail_path = output_dir / f"{slide_path.stem}_thumbnail.png"
    thumbnail.save(thumbnail_path)
    print(f"\nThumbnail saved:   {thumbnail_path}")

    slide.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide", type=Path, help="Path to a .tif WSI")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Where to save the thumbnail (default: output/)",
    )
    args = parser.parse_args()

    if not args.slide.exists():
        print(f"Error: {args.slide} does not exist", file=sys.stderr)
        sys.exit(1)

    inspect_slide(args.slide, args.output_dir)


if __name__ == "__main__":
    main()
