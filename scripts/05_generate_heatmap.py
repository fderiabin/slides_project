"""
05_generate_heatmap.py — Overlay anomaly scores on the slide thumbnail
to produce a heatmap visualisation.

Usage:
    python scripts/05_generate_heatmap.py data/tumor/tumor_001.tif
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide

def generate_heatmap(slide_path: Path, output_dir: Path,
                     tile_level: int = 1, tile_size: int = 256,
                     thumbnail_level: int | None = None) -> None:

    stem = slide_path.stem
    scores_path = output_dir / f"{stem}_scores.csv"
    if not scores_path.exists():
        sys.exit(f"Error: {scores_path} not found. Run 04b first.")

    slide = openslide.OpenSlide(str(slide_path))

    # The thumbnail level we'll paint on. Default: the same one as the mask.
    if thumbnail_level is None:
        thumbnail_level = slide.level_count - 4
    thumbnail_level = max(0, min(thumbnail_level, slide.level_count - 1))

    thumb_w, thumb_h = slide.level_dimensions[thumbnail_level]
    thumb_downsample = slide.level_downsamples[thumbnail_level]
    tile_downsample = slide.level_downsamples[tile_level]
    print(f"Thumbnail at level {thumbnail_level}: {thumb_w}x{thumb_h}")

    # Build a heatmap array shaped like the thumbnail. Each tile paints
    # a rectangle of size (tile_size * tile_downsample / thumb_downsample).
    heatmap = np.full((thumb_h, thumb_w), np.nan, dtype=np.float32)
    paint_size = max(1, int(round(
        tile_size * tile_downsample / thumb_downsample)))
    print(f"Each tile paints a {paint_size}x{paint_size} block on heatmap")

    with open(scores_path) as f:
        rows = list(csv.DictReader(f))
    print(f"Painting {len(rows)} tiles onto heatmap...")

    for row in rows:
        x0_l0 = int(row["x0_l0"])
        y0_l0 = int(row["y0_l0"])
        score = float(row["anomaly_score"])
        # Convert L0 coords to thumbnail coords
        x0 = int(x0_l0 / thumb_downsample)
        y0 = int(y0_l0 / thumb_downsample)
        x1 = min(x0 + paint_size, thumb_w)
        y1 = min(y0 + paint_size, thumb_h)
        heatmap[y0:y1, x0:x1] = score

    # Render: thumbnail with the heatmap blended in.
    thumbnail = np.array(slide.get_thumbnail(
        (thumb_w, thumb_h)).convert("RGB"))

    finite = heatmap[np.isfinite(heatmap)]
    vmin, vmax = np.percentile(finite, [5, 99])
    print(f"Heatmap percentile range: [{vmin:.4f}, {vmax:.4f}]")

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    axes[0].imshow(thumbnail)
    axes[0].set_title("Slide thumbnail")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title("Anomaly score heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.04)

    axes[2].imshow(thumbnail)
    overlay = axes[2].imshow(heatmap, cmap="jet", alpha=0.5,
                             vmin=vmin, vmax=vmax)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.colorbar(overlay, ax=axes[2], fraction=0.04)

    out_path = output_dir / f"{stem}_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {out_path}")

    slide.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tile-level", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--thumbnail-level", type=int, default=None)
    args = parser.parse_args()

    if not args.slide.exists():
        sys.exit(f"Error: {args.slide} does not exist")

    generate_heatmap(args.slide, args.output_dir,
                     args.tile_level, args.tile_size, args.thumbwnail_level)


if __name__ == "__main__":
    main()