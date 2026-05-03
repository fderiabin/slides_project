"""
04a_extract_features.py — Run Phikon (Owkin) on every tile of a slide
and save the resulting feature vectors as a numpy array.

Usage:
    python scripts/04a_extract_features.py data/tumor/tumor_001.tif

Outputs:
    output/<stem>_features.npy   shape (N, 768), float32
    output/<stem>_features.csv   tile_id -> filename mapping
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "owkin/phikon"   # 768-dim features, ViT-Base
BATCH_SIZE = 64               # reduce to 16 or 8 if you run out of RAM
NUM_WORKERS = 2               # data loader threads


class TileDataset(Dataset):
    def __init__(self, tile_dir: Path, filenames: list[str], processor):
        self.tile_dir = tile_dir
        self.filenames = filenames
        self.processor = processor

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img = Image.open(self.tile_dir / self.filenames[idx]).convert("RGB")
        # The processor handles resize + normalisation. It returns a dict
        # of tensors; we want pixel_values, shape (1, 3, 224, 224).
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"]
        return pixel_values.squeeze(0)


def extract_features(slide_path: Path, output_dir: Path) -> None:
    stem = slide_path.stem

    # Locate the tile index from script 03
    index_path = output_dir / f"{stem}_tile_index.csv"
    tiles_dir = output_dir / f"{stem}_tiles"
    if not index_path.exists():
        sys.exit(f"Error: {index_path} not found. Run 03_tile_slide.py first.")

    # Read the tile index in order so features and CSV row indices match
    with open(index_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    filenames = [row["filename"] for row in rows]
    print(f"Found {len(filenames)} tiles to process")

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Phikon
    print(f"Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    # Build the dataloader
    dataset = TileDataset(tiles_dir, filenames, processor)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=False, pin_memory=(device.type == "cuda"),
    )

    # Run inference
    all_features = []
    start = time.time()
    with torch.inference_mode():
        for batch_idx, pixel_values in enumerate(loader):
            pixel_values = pixel_values.to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values)
            # CLS token of the last hidden state — Phikon's recommended feature
            feats = outputs.last_hidden_state[:, 0, :]   # (B, 768)
            all_features.append(feats.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                done = (batch_idx + 1) * BATCH_SIZE
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(filenames) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(filenames)} tiles "
                      f"({rate:.1f}/s, ETA {eta/60:.1f} min)")

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Total time: {(time.time() - start)/60:.1f} min")

    # Save
    features_path = output_dir / f"{stem}_features.npy"
    np.save(features_path, features)
    print(f"Features saved: {features_path}")

    # Mapping CSV: just the filename order, so script 04b can join with index
    mapping_path = output_dir / f"{stem}_features_index.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_row", "filename"])
        for i, fn in enumerate(filenames):
            writer.writerow([i, fn])
    print(f"Mapping saved:  {mapping_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    if not args.slide.exists():
        sys.exit(f"Error: {args.slide} does not exist")

    extract_features(args.slide, args.output_dir)


if __name__ == "__main__":
    main()