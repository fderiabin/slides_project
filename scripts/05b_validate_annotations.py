"""
05b_validate_annotations.py — Overlay CAMELYON16 ground-truth tumor
polygons on the anomaly heatmap to visually validate whether the
unsupervised score localises with annotated tumour regions.

Usage:
    python scripts/05b_validate_annotations.py data/tumor/tumor_001.tif \\
        --annotation data/annotations/tumor_001.xml

The XML is the ASAP format: a tree of <Annotation> elements, each with
<Coordinate X="..." Y="..."> children in level-0 pixel space.
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath


def parse_annotation_polygons(xml_path: Path) -> list[dict]:
    """Return one dict per polygon: name, group, vertices (Nx2 L0 pixels)."""
    root = ET.parse(xml_path).getroot()
    polygons = []
    for ann in root.iter("Annotation"):
        verts = [
            (float(c.get("X")), float(c.get("Y")))
            for c in ann.findall("./Coordinates/Coordinate")
        ]
        polygons.append({
            "name": ann.get("Name", "?"),
            "group": ann.get("PartOfGroup", "?"),
            "vertices": np.array(verts, dtype=np.float32),
        })
    return polygons


def quantify_overlap(rows: list[dict], polygons: list[dict],
                     tile_l0_size: float) -> None:
    """Print recall/precision/lift of the heatmap against tumour polygons.

    Tiles are considered "inside" if their centre (level-0 coords) falls
    in any polygon. "Hot" is the top 5% of anomaly scores across the slide.
    """
    cxs = np.array([float(r["x0_l0"]) + tile_l0_size / 2 for r in rows])
    cys = np.array([float(r["y0_l0"]) + tile_l0_size / 2 for r in rows])
    scores = np.array([float(r["anomaly_score"]) for r in rows])
    centres = np.column_stack([cxs, cys])

    inside = np.zeros(len(rows), dtype=bool)
    for p in polygons:
        inside |= MplPath(p["vertices"]).contains_points(centres)

    n = len(rows)
    n_in = int(inside.sum())
    print()
    print("Quantitative overlap (tile centres vs tumour polygons):")
    print(f"  Total tiles                : {n}")
    print(f"  Inside tumour polygon      : {n_in}  ({n_in/n:.1%})")
    if n_in == 0:
        print("  No tiles fall inside any polygon — skipping stats.")
        return
    print(f"  Mean score inside polygon  : {scores[inside].mean():.4f}")
    print(f"  Mean score outside polygon : {scores[~inside].mean():.4f}")

    thr = float(np.percentile(scores, 95))
    hot = scores >= thr
    tp = int((hot & inside).sum())
    recall = tp / n_in
    precision = tp / max(int(hot.sum()), 1)
    lift = precision / (n_in / n)
    print(f"  Hottest 5% (score >= {thr:.4f}): {int(hot.sum())} tiles")
    print(f"    Recall    (tumour tiles in hot): "
          f"{tp}/{n_in} ({recall:.1%})")
    print(f"    Precision (hot tiles in tumour): "
          f"{tp}/{int(hot.sum())} ({precision:.1%})")
    print(f"    Lift over baseline             : {lift:.1f}x")


def validate_annotations(slide_path: Path, annotation_path: Path,
                         output_dir: Path,
                         tile_level: int = 1, tile_size: int = 256,
                         thumbnail_level: int | None = None,
                         quantify: bool = True) -> None:

    stem = slide_path.stem
    scores_path = output_dir / f"{stem}_scores.csv"
    if not scores_path.exists():
        sys.exit(f"Error: {scores_path} not found. Run 04b first.")
    if not annotation_path.exists():
        sys.exit(f"Error: {annotation_path} not found.")

    with openslide.OpenSlide(str(slide_path)) as slide:
        if thumbnail_level is None:
            thumbnail_level = slide.level_count - 4
        thumbnail_level = max(0, min(thumbnail_level, slide.level_count - 1))

        thumb_w, thumb_h = slide.level_dimensions[thumbnail_level]
        thumb_downsample = slide.level_downsamples[thumbnail_level]
        tile_downsample = slide.level_downsamples[tile_level]
        print(f"Thumbnail at level {thumbnail_level}: {thumb_w}x{thumb_h}")

        # Heatmap construction — same logic as script 05.
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
            x0 = int(x0_l0 / thumb_downsample)
            y0 = int(y0_l0 / thumb_downsample)
            x1 = min(x0 + paint_size, thumb_w)
            y1 = min(y0 + paint_size, thumb_h)
            heatmap[y0:y1, x0:x1] = score

        thumbnail = np.array(slide.get_thumbnail(
            (thumb_w, thumb_h)).convert("RGB"))

    polygons = parse_annotation_polygons(annotation_path)
    print(f"Loaded {len(polygons)} polygon(s) from {annotation_path.name}:")
    for p in polygons:
        print(f"  {p['name']:>14s} | {p['group']:>10s} | "
              f"{len(p['vertices'])} vertices")

    finite = heatmap[np.isfinite(heatmap)]
    vmin, vmax = np.percentile(finite, [5, 99])
    print(f"Heatmap percentile range: [{vmin:.4f}, {vmax:.4f}]")

    fig, axes = plt.subplots(1, 4, figsize=(24, 8))

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
    axes[2].set_title("Heatmap overlay")
    axes[2].axis("off")
    plt.colorbar(overlay, ax=axes[2], fraction=0.04)

    axes[3].imshow(thumbnail)
    axes[3].imshow(heatmap, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    for p in polygons:
        verts_thumb = p["vertices"] / thumb_downsample
        patch = Polygon(verts_thumb, closed=True, fill=False,
                        edgecolor="magenta", linewidth=2.5)
        axes[3].add_patch(patch)
    axes[3].set_title("Overlay + ground-truth tumour polygons")
    axes[3].axis("off")

    out_path = output_dir / f"{stem}_validated.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Validation figure saved: {out_path}")

    if quantify:
        tile_l0_size = tile_size * tile_downsample
        quantify_overlap(rows, polygons, tile_l0_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide", type=Path)
    parser.add_argument("--annotation", type=Path, default=None,
                        help="ASAP XML file. Default: "
                             "~/projects/pathology/shared/slides/<stem>.xml")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tile-level", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--thumbnail-level", type=int, default=None)
    parser.add_argument("--quantify", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Print recall/precision/lift of the heatmap "
                             "against the polygons (default: on)")
    args = parser.parse_args()

    if not args.slide.exists():
        sys.exit(f"Error: {args.slide} does not exist")

    annotation = args.annotation
    if annotation is None:
        annotation = (Path.home() / "projects/pathology/shared/slides"
                      / f"{args.slide.stem}.xml")

    validate_annotations(args.slide, annotation, args.output_dir,
                         args.tile_level, args.tile_size,
                         args.thumbnail_level, args.quantify)


if __name__ == "__main__":
    main()
