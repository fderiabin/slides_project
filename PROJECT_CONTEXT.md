# Digital Pathology Pipeline — Project Context

This document captures the state of the project, the design decisions made, and the open questions, so that work can resume in a fresh Claude session (or by a collaborator) without losing context. It is meant to be uploaded alongside relevant scripts at the start of any future conversation.

## Author and motivation

Built by Filipp Deriabin (B.Sc. Physics, ETH Zürich; B.Sc. International Management, Bocconi) as a portfolio project to demonstrate readiness for a Senior Field Application Engineer role in digital pathology AI. The project was triggered by a recruiter screening call (Will Booth, digital pathology headhunter) where the candidate could not confidently answer questions about hardware architecture and Linux for imaging. The goal is to produce a GitHub repository that demonstrates end-to-end whole-slide image processing on Linux.

## What this pipeline does

Takes a CAMELYON16 whole-slide image (`.tif`, ~1–2 GB, scanned at 40× magnification on a Philips IntelliSite scanner) and produces a heatmap visualisation of unusual tissue regions. The intent of the heatmap is to flag candidate metastatic regions on a sentinel lymph node section.

The pipeline is six scripts long, each runnable independently from the command line.

## Pipeline architecture

The full chain, with one example invocation per script:

```
01_inspect_slide.py    Print slide metadata, save thumbnail
02_tissue_mask.py      Otsu thresholding + connected-component filter
03_tile_slide.py       Extract 256×256 tiles at level 1 (effective 20×)
04a_extract_features   Run Phikon (Owkin) ViT-Base on each tile → 768-dim features
04b_score_anomalies    k-NN cosine distance from normal-tissue features
05_generate_heatmap    Paint scores onto thumbnail, save PNG
06_report.py           NOT YET WRITTEN — HTML/PDF summary
```

Each step writes to `output/` and reads from previous steps' outputs. The contract between steps is a small set of conventions:

- The slide stem (e.g. `tumor_001`) is the primary identifier; output filenames are derived from it.
- Tile coordinates are stored in level-0 pixels in `*_tile_index.csv`, because OpenSlide's `read_region` always uses level-0 coordinates regardless of which level you're reading from.
- Feature vectors are stored as `(N, 768)` float32 NumPy arrays; the row order matches the tile index.
- The mask is stored both as PNG (for inspection) and as `.npy` (for use in script 03).

## Hardware / environment

- Windows 10 host, WSL2 Ubuntu, NVIDIA RTX 2070 (8 GB VRAM)
- Python venv at `~/projects/slides_project/.venv`
- PyTorch 2.11+cu130, CUDA visible from inside WSL via host driver 595
- Phikon feature extraction runs at ~95 tiles/second on the 2070; full 7,615-tile slide processed in 1.3 minutes

## Dataset

- Two slides downloaded from the AWS Open Data registry: `s3://camelyon-dataset/CAMELYON16/`
  - `images/normal_001.tif` — reference (1981 tiles after filtering)
  - `images/tumor_001.tif` — target with metastases (7615 tiles after filtering)
- Plus `annotations/tumor_001.xml` for ground-truth tumor polygons (not yet used; reserved for v2 supervised classifier)

## Key design decisions

### Working at level 1, not level 0
The slide is at 40× (mpp ≈ 0.243). Most pathology models — including Phikon — are trained at 20× equivalent. Tiling at level 1 gives the model the field of view it expects, while quartering the tile count vs level 0.

### Phikon (Owkin) as the feature extractor
Chosen over ImageNet ResNet-18 (no pathology signal) and over CAMELYON16-specific GitHub repos (dependency hell). Phikon is a ViT-Base self-supervised on 40M H&E patches; produces 768-dim features per tile from the CLS token. Currently using `owkin/phikon` v1; v2 (1024-dim, ViT-Large) is a possible upgrade.

### Unsupervised anomaly scoring (k-NN cosine distance)
Reference slide features define "normal." For each target tile, we compute the average cosine distance to its 5 nearest reference neighbours; high distance = unusual.

This was a deliberate choice over training a supervised classifier on the XML annotations. Rationale: keeps the v1 pipeline simpler and demonstrates the foundation-model paradigm honestly. The v1 result confirmed that unsupervised distance is **not sufficient** for clinical-grade metastasis detection on heterogeneous lymph node tissue (see "Open issues" below) — which is itself an honest finding to report.

### Tissue mask: Otsu + morphology + size filter
Three-stage approach. Otsu thresholds on luminance (tissue is darker than glass). Morphological closing/opening cleans small holes and specks. Connected-component size filter (min 15,000 pixels) removes ink markers and dust artifacts.

The connected-component approach was chosen after a saturation-thresholding alternative was tried and failed (see "Debugging anecdotes" — useful for the README).

## Current state per script

| Script | Status | Notes |
|---|---|---|
| 01_inspect_slide.py | ✅ Done, reviewed line-by-line in chat | |
| 02_tissue_mask.py | ✅ Done, partially reviewed (chunks 1-7 of 8) | Defaults updated post-debug: `min_object_size=15000`. Deprecation warnings fixed (closing/opening). |
| 03_tile_slide.py | ✅ Done, not yet reviewed | Default `tissue_threshold=0.85` post-debug. |
| 04a_extract_features.py | ✅ Done, not yet reviewed | Phikon weights cached in `~/.cache/huggingface/`. BATCH_SIZE=64, NUM_WORKERS=2 tuned for RTX 2070. |
| 04b_score_anomalies.py | ✅ Done, not yet reviewed | Dead-code stub deleted. |
| 05_generate_heatmap.py | ✅ Done, not yet reviewed | Unused imports cleaned up. |
| 06_report.py | ❌ Not written | HTML or PDF summary; see project doc step 6. |
| README.md | ❌ Stub only | Major deliverable; needs full write-up. |

## Debugging anecdotes (gold for the README)

These are real things that happened, documented because they showcase methodical debugging on imaging data — exactly the FAE day-job.

### The saturation thresholding dud
First attempt to remove ink markers (the two crosses at the top of `tumor_001`) used HSV saturation: "ink should be near-grey, low saturation." First run with `min_saturation=0.05` partially worked (removed cross interiors, left edges). Raising to 0.15 didn't help. Diagnostic measurement on actual ink-cross pixels showed median saturation 0.27 — squarely in the H&E tissue range. Conclusion: Philips printer ink is **not** grey, it's bluish-black with non-trivial chromaticity. Saturation thresholding cannot discriminate ink from tissue on this slide.

### The connected-component fix
Second attempt: ink markers and tissue have very different sizes. Tissue blobs are 21k+ pixels; ink crosses are 9-12k pixels. A `remove_small_objects(min_size=15000)` call removes the markers without touching real tissue. This is a more general approach because it doesn't assume anything about ink colour or marker placement, only that artifacts are smaller than tissue.

### The orientation confusion
Initial diagnostic of "are the crosses still in the mask?" used `mask[:int(h*0.1)]` to check the top 10% of rows. Got an unexpected reading (8% coverage, identical to total mask coverage) which was misinterpreted as "the entire mask is in the top 10%." Real explanation: the slide is tall and portrait-oriented, the tissue happens to occupy the middle of the slide, and the slicing was correct — the high coverage in the top band was specifically the ink markers, while the tissue lived in the middle bands. A row-by-row coverage histogram (10 bands) made the layout obvious.

### The default-values audit
After the connected-component fix, `min_object_size` was set to 15000 via the `--min-object-size` flag, but the script's argparse default was still 5000. Same issue with `tissue_threshold` in script 03 (used 0.85 via flag, default still 0.5). These were caught in a later code-review pass and fixed. Lesson: when tuning a parameter, update the default *and* commit, don't just remember to pass the flag next time.

## Open issues / future work

### V1 heatmap quality
After fixing the artifact pipeline, the cleaned-up heatmap still does not show clear localised tumor regions. Most tissue reads as cool blue (scores 0.13–0.20) with marginal hot spots. This is consistent with the literature: unsupervised feature distance is insufficient for sentinel lymph node metastasis detection because lymphoid tissue is genuinely heterogeneous and metastases occupy a region of feature space that overlaps with normal variation.

### V2 supervised classifier (recommended next step)
The `tumor_001.xml` annotation file contains polygon vertices marking ground-truth tumor regions. Labelling tiles by polygon membership and fitting a small classifier (logistic regression) on top of Phikon features would produce a meaningful, accurate heatmap. Estimated effort: 1-2 hours.

### Stain normalisation
Not implemented. Production digital pathology pipelines use Macenko, Reinhard, or Vahadane stain normalisation to handle inter-slide colour variation. The current pipeline assumes the reference and target slides have similar colour distributions — fine for two slides from the same dataset, fragile across hospitals/scanners.

### Multi-reference normality
Currently a single normal slide defines "normal." A larger reference set (3-5 normal slides) would partially mitigate batch effects but doesn't solve the fundamental problem.

### Storage management
Tiles are saved as individual PNG files. ~1-4 GB per slide. For larger experiments, switch to HDF5 or LMDB.

### Context manager for OpenSlide
Currently uses manual `slide.close()`. Recent OpenSlide versions support `with openslide.OpenSlide(...) as slide:` which would be more robust against exceptions.

## Conventions worth preserving

- Each script is invokable as `python scripts/NN_name.py <slide_path>` with sensible defaults
- All outputs go to `output/`, named with the slide stem
- Tile coordinates are always level-0 pixels in CSVs
- Features are L2-normalised before cosine distance
- Heatmaps use percentile-based colour scaling (5th–99th) to be robust against outliers
