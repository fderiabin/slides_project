# What I Learned — Digital Pathology Pipeline Project

Consolidated reference of every technical concept covered during the two-week build. Organised by topic, not chronology. Cross-references the scripts in parentheses where relevant.

---

## Part 1 — Python and tooling

### Paths (`pathlib`)

`Path` is the modern way to handle filesystem paths. Use it instead of string concatenation, which breaks across operating systems (Windows backslashes vs Unix slashes).

The slash operator is overloaded for path joining:

```python
output_dir / f"{stem}_thumbnail.png"   # works on Linux, macOS, Windows
```

Five attributes worth memorising:

```python
p = Path("data/tumor/tumor_001.tif")
p.name      # "tumor_001.tif"      filename + extension
p.stem      # "tumor_001"          filename without extension
p.parent    # Path("data/tumor")   directory containing the file
p.suffix    # ".tif"               the extension
p.exists()  # True/False           does this path exist on disk?
```

Standard methods: `p.is_file()`, `p.is_dir()`, `p.mkdir(parents=True, exist_ok=True)`.

The `mkdir` flags: `parents=True` creates intermediate directories if missing; `exist_ok=True` doesn't error if the directory already exists. Together they make the call idempotent — safe to run multiple times.

### Parsing and argparse

**Parsing** = taking raw text input and breaking it into structured pieces a program can use.

Examples: Python parses `3 + 4 * 2` into an arithmetic tree; a browser parses HTML into a DOM tree; `csv.DictReader` parses comma-separated text into dictionaries.

**Argparse** parses command-line arguments. The OS hands the script a flat list of strings; argparse turns those into typed Python variables:

```python
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("slide", type=Path)                                 # positional
parser.add_argument("--output-dir", type=Path, default=Path("output"))  # optional
args = parser.parse_args()
```

Key conventions:
- Positional args (no `--`) are required. Position in the command matters.
- Optional args (`--`-prefixed) have defaults. Order doesn't matter.
- Dash-to-underscore conversion is automatic: `--min-object-size` becomes `args.min_object_size` because Python identifiers can't contain hyphens.
- `description=__doc__` reuses the module docstring as `--help` text. Write the docstring once, get help output for free.

### Type hints

```python
def f(x: int, y: int | None = None) -> None:
```

The `int | None` syntax (Python 3.10+) means "either int or None." Type hints don't enforce anything at runtime — they document intent and let editors catch mistakes.

### Context managers (the `with` pattern)

```python
with openslide.OpenSlide(str(slide_path)) as slide:
    # use slide here
# slide.close() is called automatically, even if an exception was raised
```

Better than manual `slide.close()` because cleanup happens deterministically. Same pattern as `with open(...) as f:` for files.

### Defensive value clamping

```python
mask_level = max(0, min(mask_level, slide.level_count - 1))
```

Reads inside-out: first cap the upper bound (`min(mask_level, max_valid)`), then lift the lower bound (`max(0, ...)`). Standard idiom for forcing a value into a valid range.

### List comprehensions

```python
filenames = [row["filename"] for row in rows]
```

Compact equivalent of:

```python
filenames = []
for row in rows:
    filenames.append(row["filename"])
```

Reads as "filenames is the row-filename for each row in rows."

### The `if __name__ == "__main__":` guard

```python
if __name__ == "__main__":
    main()
```

Means: "only run `main()` if this file is being executed as a script." If the file is imported as a module, `main()` doesn't run. Lets the same file serve both as a runnable script and as an importable library.

---

## Part 2 — NumPy

### Lists vs arrays

Python lists are flexible (any types, any structure) but slow because they're sequences of Python objects. NumPy arrays are rigid (one dtype, contiguous memory) but fast because operations dispatch to compiled C code.

For image data — millions of pixels, three numbers each — the difference is roughly 100×.

### Shape conventions

For an image, the shape convention is `(height, width, channels)` — height first, even though we describe images verbally as "1528 by 3456" (width × height). NumPy follows matrix conventions where rows (vertical position) come before columns; OpenSlide follows graphics conventions where x (horizontal) comes before y. Both are internally consistent; they just disagree.

```python
rgb.shape   # (3456, 1528, 3)  for tumor slide at level 6
```

### Indexing and slicing

The first index selects along axis 0 (row), the second along axis 1 (column). A colon `:` means "all values along this axis."

```python
a[0, :]   # row 0, all columns         → 1D array
a[:, 1]   # all rows, column 1         → 1D array
a[0, 1]   # row 0, column 1            → scalar
```

The `...` ellipsis means "as many `:` as needed":

```python
hsv[..., 1]   # for shape (H, W, 3): selects channel 1 at every pixel
              # equivalent to hsv[:, :, 1]
```

### Broadcasting and elementwise comparisons

```python
tissue = gray < threshold
```

NumPy compares every element of `gray` against the scalar `threshold` in one operation, producing a boolean array of the same shape. No Python loop. This is broadcasting.

### Boolean indexing — the extract-modify-assign pattern

```python
overlay[~tissue] = (overlay[~tissue] * 0.3).astype(np.uint8)
```

Three operations: `~tissue` produces a boolean mask of background pixels; `overlay[~tissue]` extracts those pixels into a flat 1D array; the multiplication produces dimmed values; the assignment writes them back to the same positions. Tissue pixels untouched.

When you boolean-index a multi-channel image of shape `(H, W, 3)` with a 2D mask of shape `(H, W)`, the result is shape `(N, 3)` where N is the number of `True` values. Spatial layout is lost (flattened to a list), but the channel dimension is preserved.

### Boolean array tricks

`True` is treated as 1 and `False` as 0:

```python
tissue.mean()    # fraction of True pixels
tissue.sum()     # count of True pixels
```

### Pointer semantics

Every variable in Python is a reference (pointer) to an object. For mutable objects (lists, arrays), aliasing is observable:

```python
a = [1, 2, 3]
b = a
b.append(4)
print(a)   # [1, 2, 3, 4]   — same list, two names
```

To copy: `b = a.copy()` for lists, `b = a.copy()` or `b = np.array(a)` for arrays. Without this, mutations propagate silently.

### `keepdims=True` and broadcasting

```python
norms = np.linalg.norm(x, axis=1, keepdims=True)   # shape (N, 1)
return x / norms                                   # broadcasts to (N, 768)
```

Without `keepdims=True`, `norms` would be shape `(N,)` and the division would fail because NumPy aligns shapes from the right and `(N, 768) / (N,)` doesn't match. `keepdims=True` preserves the singleton axis for broadcasting to work.

### NaN as a sentinel

```python
heatmap = np.full((H, W), np.nan, dtype=np.float32)
```

NaN ("not a number") is a special float value meaning "missing data." Used in script 05 to distinguish "no score available" from "score is zero." NaN propagates through most operations and is rendered as transparent by matplotlib.

To extract finite values for statistics:

```python
finite = heatmap[np.isfinite(heatmap)]
vmin, vmax = np.percentile(finite, [5, 99])
```

### F-string formatting

```python
f"{score:.6f}"        # 0.143251       (6 decimal places)
f"{rate:.1f}/s"       # 95.3/s         (1 decimal place)
f"{coverage:.1%}"     # 9.2%           (1 decimal, percentage)
```

The percent sign is part of the format spec and multiplies the value by 100 automatically.

---

## Part 3 — OpenSlide and WSI fundamentals

### What it is

A C library (with Python bindings) for reading multi-resolution whole-slide images. Handles all major vendor formats: Aperio (.svs), Hamamatsu (.ndpi), Philips (.tif), MIRAX (.mrxs), and others.

### The pyramid

Whole-slide images are stored as multi-resolution pyramids. Each level halves the linear dimensions of the previous, so quarters the pixel count. Level 0 is the original full-resolution scan; higher level numbers are progressively smaller, lower-detail versions.

For a 10-level slide at 40× scan:

```
Level 0:   1×    (original, biggest, most detail, ~40×)
Level 1:   2×    (effective 20×)
Level 2:   4×    (effective 10×)
...
Level 9: 512×    (smallest, lowest detail, basically a thumbnail)
```

**Higher level number = lower resolution = lower magnification.** This is the convention that catches everyone.

Total storage of all pyramid levels combined is only ~33% more than level 0 alone (geometric series convergence) — that's why pyramidal TIFF works as a format.

### Reading regions

```python
slide.read_region((x, y), level, (width, height))
```

Three things to know:

1. `(x, y)` is **always in level-0 coordinates**, regardless of which level you're reading. The OpenSlide gotcha that catches everyone first time.
2. `(width, height)` is in pixels at the chosen level.
3. Returns a PIL Image in RGBA mode. Convert with `.convert("RGB")` to drop alpha.

### Microns per pixel and magnification

`slide.properties["openslide.mpp-x"]` gives the physical resolution in micrometres per pixel. The conversion to magnification:

```
objective_power ≈ 10 / mpp_x
```

Rough calibration: 0.5 µm/px ≈ 20×, 0.25 µm/px ≈ 40×, 1.0 µm/px ≈ 10×.

Magnification is **linear**, not areal. Don't multiply mpp_x by mpp_y to get a "total magnification" — that gives you area per pixel, a different quantity.

### Downsampling vs magnification — opposites

| Higher pyramid level | Higher downsample factor | LOWER magnification | LESS detail |
| Lower pyramid level | Lower downsample factor | HIGHER magnification | MORE detail |

Going "up" the pyramid throws away detail. Going "down" gains it. Each step is a factor of 2 in linear dimensions, factor of 4 in pixel count.

### Why two levels for our pipeline

Tissue masking at level 6 (low resolution): fast, cheap, coarse — good enough for "where is tissue?" Tile inference at level 1 (high resolution): detailed enough for Phikon to recognise cellular structure.

The pipeline juggles three coordinate systems simultaneously: tile-level (level 1), mask-level (level 6), and level 0 for `read_region` calls.

---

## Part 4 — Image processing fundamentals

### RGB → greyscale

`skimage.color.rgb2gray` collapses 3 channels using a perceptually-weighted sum:

```
gray ≈ 0.299·R + 0.587·G + 0.114·B
```

Green is weighted highest because human eyes are most sensitive to green. Output is in **floats 0–1** (not integers 0–255), because that's the natural domain for image-processing math.

### Colour spaces and HSV

RGB is one way to describe a colour with three numbers. HSV is another, often more useful for processing:

- **Hue** — which colour (angle on a colour wheel, 0–360°). Red 0°, green 120°, blue 240°.
- **Saturation** — how vivid (0 = grey, 1 = pure primary).
- **Value** — how bright (0 = black, 1 = full intensity).

Saturation formula:

```
S = (max(R, G, B) − min(R, G, B)) / max(R, G, B)
```

Intuitively: how much do the three channels disagree? Equal channels = grey = saturation 0. One dominant channel = vivid = saturation high.

### Otsu thresholding

Automatically picks the threshold that best separates a bimodal intensity histogram (like H&E slides: dark tissue + bright glass).

For each candidate T, split pixels into two classes (≤ T and > T). Otsu minimises **within-class variance**:

```
σ²_within(T) = ω₀ · σ²₀ + ω₁ · σ²₁
```

where ω is class weight (fraction of pixels in that class), σ² is variance within that class.

The variance decomposition identity:

```
σ²_total = σ²_within(T) + σ²_between(T)
```

Since σ²_total is fixed (independent of T), minimising σ²_within is mathematically equivalent to maximising σ²_between:

```
σ²_between(T) = ω₀ · ω₁ · (μ₀ − μ₁)²
```

This formula needs only weights and means — faster to compute. Implementations maximise σ²_between for speed.

**Where Otsu fails:** assumes a bimodal histogram. Degrades on low-contrast images (modes blur), multi-class distributions (one threshold can't separate three groups), or heavily imbalanced classes (minor mode gets drowned out).

### Morphological operations

Operations on binary images using a "structuring element" — a small shape that slides over the image.

The two atomic operations:

- **Erosion** — a pixel stays True only if the structuring element fits entirely inside the True region centred on that pixel. Shrinks True regions.
- **Dilation** — a pixel becomes True if the structuring element overlaps any True region. Grows True regions.

The two compounds:

- **Closing** = dilate then erode. Net effect: fills small holes, otherwise preserves shape.
- **Opening** = erode then dilate. Net effect: removes small specks, otherwise preserves shape.

Order matters in our pipeline: close first to make tissue solid, then open to remove dust specks. Reversing the order would lose small tissue features.

### Connected-component analysis

A graph algorithm, not an image-processing algorithm. The mask is treated as a graph: True pixels are nodes, adjacent True pixels are edges. Components are identified using flood-fill or union-find.

The two-pass labelling algorithm: walk through pixels assigning tentative labels based on already-visited neighbours, remembering equivalences when labels collide; second pass relabels using the canonical representative of each equivalence class.

`remove_small_objects(mask, min_size=N)` removes any connected component with fewer than N pixels.

**Why connected-component beats colour for artifact removal:** It doesn't care what colour the artifact is or where it sits. It exploits the more general property that tissue blobs are large (>20,000 pixels in our case) while artifacts (ink markers, dust) are small (<15,000 pixels). More robust to slide variation.

Default is 4-connectivity (pixels share an edge, not just a corner).

---

## Part 5 — PyTorch and deep learning

### Tensors and devices

PyTorch tensors are like NumPy arrays but can live on GPU and support automatic differentiation.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The `device` is an abstraction for "where data and computations live." CPU tensors are in regular RAM; CUDA tensors are in GPU VRAM. They can't directly interact — moves are explicit via `.to(device)` and `.cpu()`.

### Memory hygiene

GPU VRAM is scarce (8 GB on RTX 2070); CPU RAM is plentiful (typically 16+ GB). Best practice: keep only the actively-computing batch on GPU; immediately move results to CPU.

```python
feats = outputs.last_hidden_state[:, 0, :]   # on GPU
all_features.append(feats.cpu().numpy())     # move to CPU, convert to NumPy
```

The `.cpu()` triggers a GPU→CPU copy because NumPy can't operate on CUDA tensors. The original GPU tensor gets garbage-collected after no more references exist.

### Lists hold pointers

A Python list is always a CPU object, but its elements can live anywhere. When you append a GPU tensor to a list, the list stores a *reference* (pointer) to GPU memory, not a copy. The list itself is small metadata; the underlying data is wherever it was created.

This is why we explicitly `.cpu()` before appending — to make sure list contents are accessible to CPU-only operations like `np.concatenate`.

### Inference mode

```python
with torch.inference_mode():
    outputs = model(pixel_values=pixel_values)
```

Disables gradient tracking and autograd graph construction for the entire block. Significantly reduces memory and slightly speeds things up. Forgetting it doesn't break correctness, just wastes resources.

The companion call `.eval()` puts dropout and batch normalisation in inference mode (some layers behave differently during training).

### Dataset/DataLoader pattern

A standard PyTorch idiom for lazy, batched, parallel data feeding.

```python
class TileDataset(Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int): ...

loader = DataLoader(dataset, batch_size=64, num_workers=2,
                    shuffle=False, pin_memory=True)
```

The `Dataset` interface just requires `__len__` and `__getitem__`. `DataLoader` wraps it and handles batching, parallelism, and memory pinning.

Solves three problems at once:

1. **Memory** — only a few batches in memory at any time, not the whole dataset
2. **Speed** — workers load batch N+1 from disk while the GPU processes batch N (loading and compute overlap)
3. **Decoupling** — the model code doesn't care whether tiles come from PNGs, S3, HDF5, or a database

### Vision Transformer mental model

A ViT preprocesses an image into a sequence of patches (16×16 patches for a 224×224 input = 196 patches), each treated as a "token" in an NLP-style sequence. Plus one extra **CLS token** — a learnable summary slot prepended at position 0 — that doesn't correspond to any image patch.

The transformer's attention layers let every patch token interact with every other token including the CLS. The model is trained so the CLS token ends up being a holistic summary of the whole image.

So `outputs.last_hidden_state` has shape `(B, 197, 768)` for a ViT-Base:
- B = batch size
- 197 = 1 CLS + 196 patches
- 768 = hidden dimension

`outputs.last_hidden_state[:, 0, :]` extracts the CLS token for each batch item. Shape `(B, 768)`. One feature vector per image.

### Foundation models as feature extractors

The dominant paradigm in modern computational ML: use a big pretrained model to compute features, then do something simpler on top.

Phikon was pretrained by Owkin on 40M H&E patches with self-supervised learning. It produces 768-dim features per tile. We don't fine-tune it — we just use it to extract features that downstream code consumes.

The feature extraction is the expensive, generalisable part. The downstream task (k-NN scoring, in our case; classification in other pipelines) is cheap and task-specific.

### Hugging Face `transformers`

The de facto standard library for using pretrained models. The `Auto*` classes are clever wrappers that automatically pick the right architecture based on a model name string:

```python
processor = AutoImageProcessor.from_pretrained("owkin/phikon")
model = AutoModel.from_pretrained("owkin/phikon").to(device).eval()
```

The model is downloaded (if not cached) from Hugging Face into `~/.cache/huggingface/`. Subsequent runs find it cached and load from disk in seconds.

### `pin_memory` and `non_blocking`

```python
loader = DataLoader(..., pin_memory=True)         # in loader construction
pixel_values.to(device, non_blocking=True)        # in the loop
```

`pin_memory` allocates CPU buffers in a special way that supports direct DMA transfer to GPU. `non_blocking=True` lets the transfer happen asynchronously with other CPU work, hiding latency. Together they enable CPU loading and GPU compute to overlap, giving you ~95 tiles/sec instead of ~40.

---

## Part 6 — Distance, similarity, and k-NN

### Cosine vs Euclidean distance

For high-dimensional feature vectors from neural networks, the **direction** of the vector usually carries more meaning than its **magnitude**. Two tiles whose vectors point the same way are similar in content regardless of how strongly the model activated for them.

Cosine similarity:

```
cos_sim(a, b) = (a · b) / (|a| × |b|)
```

If both vectors are L2-normalised (unit length), the denominator is 1, and cosine similarity reduces to the dot product. Cosine distance = 1 − cosine similarity. Higher distance = more dissimilar.

### L2 normalisation

```python
def l2_normalise(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
```

Divides each row by its Euclidean length, producing unit-length rows. The `norms[norms == 0] = 1.0` is defensive — prevents division by zero if any vector is all zeros.

### k-Nearest Neighbours for anomaly detection

```python
nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
nn.fit(reference_features)
distances, _ = nn.kneighbors(target_features)    # shape (N_target, k)
scores = distances.mean(axis=1)                  # shape (N_target,)
```

For each target tile, find its k closest reference tiles and average those k distances. Higher distance = the target tile is far from any reference tile in feature space = anomalous.

**Why k=5 and not k=1?** Averaging over multiple neighbours smooths out noise. A target tile might happen to land near one weird outlier reference tile; averaging over 5 prevents that from dominating.

`algorithm="brute"` computes all pairwise distances exhaustively. Fast enough for our scale (~1981 × 7615 = 15 million comparisons in under a second). For larger reference sets, tree-based or approximate methods (KD-tree, Annoy, HNSW) exist.

---

## Part 7 — Validation and metrics

### Recall, precision, lift

For the heatmap evaluation against ground-truth tumor polygons:

- **Recall** = (tumor tiles in hot region) / (total tumor tiles). "What fraction of true tumor did we catch?"
- **Precision** = (tumor tiles in hot region) / (total hot tiles). "What fraction of our flagged tiles is actually tumor?"
- **Lift** = (precision) / (random baseline probability). "How many times better than chance is our targeting?"

Our pipeline: 83% recall, 13% precision, 16.6× lift. Sensitive but unspecific.

### Hot region

Defined as tiles in the top 5% of anomaly scores. The threshold value is the 95th percentile of the score distribution.

### Percentile-based colour scaling

Instead of `vmin = scores.min(), vmax = scores.max()`, use:

```python
vmin, vmax = np.percentile(finite, [5, 99])
```

This dedicates almost the entire colour scale to the bulk of the distribution. Outliers get clipped to the extremes but don't dominate. Critical for readable heatmaps when score distributions have tails.

---

## Part 8 — Matplotlib

### Figure construction

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
```

Returns a `(fig, axes)` tuple. `axes` is an array of subplot regions — 1D for `(1, N)` or `(N, 1)`, 2D for grids. `figsize` is in inches at the default 72 dpi.

### `imshow`

Dual-mode function:

- **RGB(A) input** (3D array, last axis 3 or 4): rendered directly as colour image, no colormap applied.
- **Scalar input** (2D array): a colormap is applied to translate numbers to colours. Use `cmap`, `vmin`, `vmax`.

```python
axes[0].imshow(thumbnail)                                      # RGB mode
axes[1].imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)      # colour-mapped
```

Stacking two `imshow` calls on the same Axes layers them. The second draws on top:

```python
axes[2].imshow(thumbnail)                                      # background
axes[2].imshow(heatmap, cmap="jet", alpha=0.5, ...)            # overlay
```

`alpha` controls transparency. NaN cells in scalar arrays are always transparent regardless of alpha.

### Colormaps

`cmap="jet"` is the conventional "blue → red" gradient. Other options:
- `"viridis"` — perceptually uniform (scientifically preferred)
- `"hot"` — black → red → yellow → white
- `"coolwarm"` — blue → white → red (for signed data)

### Colorbars

```python
im = axes[1].imshow(...)
plt.colorbar(im, ax=axes[1], fraction=0.04)
```

The first argument is the `AxesImage` returned by `imshow`. `ax=axes[1]` specifies which subplot to attach to. `fraction=0.04` controls colorbar width (4% of panel width).

### Saving

```python
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
```

`tight_layout` adjusts spacing so titles and colorbars don't overlap. `dpi=150` sets rendering resolution. `bbox_inches="tight"` crops to exclude empty margins. `plt.close()` releases figure memory.

---

## Part 9 — File formats

### `.npy` vs `.csv`

- `.npy` — NumPy's native binary serialisation. Preserves shape and dtype. Fast to read/write. Good for numerical arrays. Use `np.save(path, array)` / `np.load(path)`.
- `.csv` — human-readable, opens in any spreadsheet. Slow to parse, but everything readable. Good for metadata. Use `csv.writer` / `csv.DictReader`.

Our pipeline saves the mask twice: PNG for human inspection, NPY for fast loading by the next script. This dual-output pattern is worth keeping in mind.

### PIL `Image.fromarray`

```python
Image.fromarray((tissue * 255).astype(np.uint8)).save(mask_png)
```

Converts a NumPy array to a PIL Image. PIL inspects shape and dtype to figure out the colour mode:
- 2D `uint8` → greyscale ("L" mode)
- 3D `uint8` with last dim 3 → RGB
- 3D `uint8` with last dim 4 → RGBA

The `.astype(np.uint8)` is critical — PIL needs 8-bit unsigned integers; floats or higher-bit integers don't work for standard PNGs.

### CSV writing

```python
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([header_columns])
    for row in data:
        writer.writerow([values])
```

The `newline=""` argument is a Windows quirk fix: it disables the file object's automatic newline translation so `csv.writer` can manage line endings itself without double-translation.

### XML parsing

```python
import xml.etree.ElementTree as ET

root = ET.parse(xml_path).getroot()
for ann in root.iter("Annotation"):
    name = ann.get("Name")
    verts = [(float(c.get("X")), float(c.get("Y")))
             for c in ann.findall("./Coordinates/Coordinate")]
```

Standard library. No external dependencies. `root.iter("tag")` walks all descendants with that tag; `element.get("attr")` reads attributes; `.findall("./path")` queries with XPath-like syntax.

---

## Part 10 — Conceptual frameworks worth keeping

### Separation of concerns in pipelines

Each script does one thing. The pipeline:

```
01 inspect → 02 mask → 03 tile → 04a features → 04b score → 05 heatmap
```

Each script reads outputs from earlier scripts and writes its own. No script calls another directly — communication happens through files. This means you can re-run any step independently and swap implementations without touching the rest of the pipeline.

The cost is some redundancy (the tile-index CSV gets passed through three scripts). The benefit is modularity.

### Conventions as contracts

The whole pipeline rests on conventions: tiles are saved in CSV order; feature row `i` corresponds to tile `i`; output filenames are derived from the slide stem. None of these are enforced by the type system or asserted at every step — they're just respected by every script. Trust between modules.

The places where we *do* enforce contracts (the `len(rows) != len(scores)` check in 04b) are exactly the places where silent failure would be catastrophic. Cheap defensive code there pays back enormously.

### Pyramidal data structures

The general principle: when you have data at one scale but need answers at another, store multiple resolutions. WSI pyramidal TIFF is one instance; mipmaps in computer graphics are another; database B-tree indices are a third.

The cost is ~33% extra storage. The benefit is multi-scale random access without re-reading the full-resolution version.

### Foundation models and feature extraction

Build expensive, generalisable representations once. Reuse them for many cheap, task-specific downstream models.

This is the modern ML paradigm. BERT, GPT, CLIP, SAM, Phikon — all foundation models. Most production ML systems today have a "frozen foundation model + small trained head" structure rather than training the whole stack from scratch.

### Geometric vs biological signal

A theme throughout: distinguishing pipeline behaviours that reflect physical/computational artifacts (tissue boundary tiles scoring high because they include glass pixels) from those that reflect what the user actually cares about (tumor cells scoring high because they look unusual under H&E).

The validation script's job was exactly to disentangle these. The 83% recall at 13% precision result tells us the pipeline has real biological signal *and* significant geometric noise — both real, both worth naming.

### Honest evaluation

The 13% precision is not a failure to be hidden — it's a real measurement to be reported. Documenting limitations openly distinguishes serious technical work from marketing. A paper or portfolio that says "we got 90% accuracy" without defining what that means is suspect; one that says "83% recall, 13% precision, 16.6× lift, with the following confounds..." is the work of someone you'd want to hire.

---

## Bonus — English idioms picked up along the way

Small style notes from the conversation:

- "Here's the heatmap" reads more natural than "Here is the heatmap" in chat. Same with "Here's some context."
- "Onto work" or "Let's get to work" beats "On to work!" — the verb form sounds more native.
- "Paste" not "copy" when asking someone to put text back in the chat: "Could you paste the exercise again?"
- "Everything gucci" is fine slang but very informal; in a German workplace, "all good" or "all set" is the equivalent register.
- "The size of the images will be immense" is grammatical but melodramatic for engineering; "the storage requirements get huge" reads cleaner.
- Capitalise sentence beginnings even in lists ("First: …" not "first: …").
- "Run" something (a model, a script, an algorithm) — don't say "do" the model.
- "I'm not sure" sounds more native than "I am not quite sure"; contractions matter for register.
- "Sharp observation" works as a compliment without sounding stiff.
