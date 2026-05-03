"""
03_tile_slide.py — Extract non-empty tiles from a whole-slide image,
using a precomputed tissue mask to skip background regions.

Usage:
    python scripts/03_tile_slide.py data/tumor/tumor_001.tif

The tile level defaults to 1 (effective 20x for a 40x-scanned slide).
Tile size defaults to 256x256 pixels.

Outputs:
    output/<stem>_tiles/         Directory of PNG tiles
    output/<stem>_tile_index.csv Tile coordinates and mask coverage
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import openslide
from PIL import Image


def tile_slide(slide_path: Path,
               output_dir: Path,
               tile_level: int = 1,
               tile_size: int = 256,
               tissue_threshold: float = 0.5) -> None:

    slide = openslide.OpenSlide(str(slide_path))
    stem = slide_path.stem

    # Load the precomputed tissue mask
    mask_npy = output_dir / f"{stem}_mask.npy"
    if not mask_npy.exists():
        sys.exit(f"Error: {mask_npy} not found. Run 02_tissue_mask.py first.")
    tissue_mask = np.load(mask_npy)
    mask_height, mask_width = tissue_mask.shape

    # Figure out which mask level was used: match dimensions
    mask_level = None
    for i, (w, h) in enumerate(slide.level_dimensions):
        if (w, h) == (mask_width, mask_height):
            mask_level = i
            break
    if mask_level is None:
        sys.exit(f"Error: mask dimensions {mask_width}x{mask_height} "
                 f"don't match any pyramid level.")

    print(f"Tiling at level {tile_level}, "
          f"deciding from mask at level {mask_level}")

    # Downsample factors relative to level 0
    tile_downsample = slide.level_downsamples[tile_level]
    mask_downsample = slide.level_downsamples[mask_level]

    # The slide grid at the tile level
    tile_level_width, tile_level_height = slide.level_dimensions[tile_level]
    n_cols = tile_level_width // tile_size
    n_rows = tile_level_height // tile_size
    print(f"Grid at L{tile_level}: {n_cols} cols x {n_rows} rows "
          f"= {n_cols * n_rows} candidate tiles")

    # How much of the mask does each tile cover?
    # tile_size pixels at tile_level == tile_size * tile_downsample at L0
    # == tile_size * tile_downsample / mask_downsample at mask level
    mask_step = int(round(tile_size * tile_downsample / mask_downsample))
    if mask_step < 1:
        mask_step = 1

    # Output directory for tiles
    tiles_dir = output_dir / f"{stem}_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / f"{stem}_tile_index.csv"
    kept = 0
    skipped = 0

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_id", "row", "col", "x0_l0", "y0_l0",
                         "tissue_fraction", "filename"])

        for row in range(n_rows):
            for col in range(n_cols):
                # Mask region for this tile
                m_y0 = row * mask_step
                m_x0 = col * mask_step
                m_y1 = min(m_y0 + mask_step, mask_height)
                m_x1 = min(m_x0 + mask_step, mask_width)
                if m_y1 <= m_y0 or m_x1 <= m_x0:
                    skipped += 1
                    continue

                tissue_fraction = tissue_mask[m_y0:m_y1, m_x0:m_x1].mean()
                if tissue_fraction < tissue_threshold:
                    skipped += 1
                    continue

                # Level-0 coordinates for read_region (it always takes L0 coords)
                x0_l0 = int(col * tile_size * tile_downsample)
                y0_l0 = int(row * tile_size * tile_downsample)

                tile = slide.read_region(
                    (x0_l0, y0_l0), tile_level, (tile_size, tile_size)
                ).convert("RGB")

                tile_filename = f"tile_r{row:04d}_c{col:04d}.png"
                tile.save(tiles_dir / tile_filename)

                writer.writerow([kept, row, col, x0_l0, y0_l0,
                                 f"{tissue_fraction:.3f}", tile_filename])
                kept += 1

                if kept % 200 == 0:
                    print(f"  {kept} tiles kept...")

    total = kept + skipped
    print(f"\nDone: kept {kept}, skipped {skipped} ({kept/total:.1%} retained)")
    print(f"Tiles written to: {tiles_dir}")
    print(f"Tile index:       {index_path}")

    slide.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tile-level", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tissue-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.slide.exists():
        sys.exit(f"Error: {args.slide} does not exist")

    tile_slide(args.slide, args.output_dir,
               args.tile_level, args.tile_size, args.tissue_threshold)


if __name__ == "__main__":
    main()