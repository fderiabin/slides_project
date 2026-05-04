"""
04b_score_anomalies.py — Score every tile in the target slide by its
distance from normal-tissue features.

For each tile in <target> we compute the average cosine distance to its
k nearest neighbours among <reference> tiles. High score == unusual.

Usage:
    python scripts/04b_score_anomalies.py \\
        --reference data/normal/normal_001.tif \\
        --target    data/tumor/tumor_001.tif

Outputs:
    output/<target_stem>_scores.csv  tile_id, row, col, score
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Cosine similarity then == dot product."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def score_anomalies(reference_slide: Path, target_slide: Path,
                    output_dir: Path, k: int = 5) -> None:

    ref_stem = reference_slide.stem
    tgt_stem = target_slide.stem

    ref_feats_path = output_dir / f"{ref_stem}_features.npy"
    tgt_feats_path = output_dir / f"{tgt_stem}_features.npy"
    tgt_index_path = output_dir / f"{tgt_stem}_tile_index.csv"

    for p in (ref_feats_path, tgt_feats_path, tgt_index_path):
        if not p.exists():
            sys.exit(f"Error: {p} not found.")

    ref_feats = l2_normalise(np.load(ref_feats_path))
    tgt_feats = l2_normalise(np.load(tgt_feats_path))
    print(f"Reference features: {ref_feats.shape}")
    print(f"Target features:    {tgt_feats.shape}")

    # Cosine distance: 1 - cosine similarity. Higher score = more dissimilar.
    print(f"Computing {k}-nearest-neighbour distances...")
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(ref_feats)
    distances, _ = nn.kneighbors(tgt_feats)   # (N_target, k)
    scores = distances.mean(axis=1)           # (N_target,)

    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Score mean:  {scores.mean():.4f} +/- {scores.std():.4f}")

    # Join with the tile index to keep row/col coordinates
    with open(tgt_index_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(scores):
        sys.exit(f"Error: tile index has {len(rows)} rows but "
                 f"features have {len(scores)}.")

    scores_path = output_dir / f"{tgt_stem}_scores.csv"
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile_id", "row", "col", "x0_l0", "y0_l0",
                         "tissue_fraction", "filename", "anomaly_score"])
        for row, score in zip(rows, scores):
            writer.writerow([
                row["tile_id"], row["row"], row["col"],
                row["x0_l0"], row["y0_l0"],
                row["tissue_fraction"], row["filename"],
                f"{score:.6f}",
            ])
    print(f"Scores saved: {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Slide whose features define 'normal'")
    parser.add_argument("--target", type=Path, required=True,
                        help="Slide to score")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("-k", "--k-neighbors", type=int, default=5)
    args = parser.parse_args()

    score_anomalies(args.reference, args.target, args.output_dir,
                    args.k_neighbors)


if __name__ == "__main__":
    main()