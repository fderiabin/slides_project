# Learning Notes — Digital Pathology Pipeline

A consolidated reference of the conceptual material we covered while building the pipeline. Organised by topic, not by chronology. Cross-references to the scripts are in parentheses.

## Python tooling

### Paths (`pathlib`)

`Path` is the modern way to handle filesystem paths. Instead of string concatenation (which breaks across operating systems), you create `Path` objects that know how to do path operations correctly. The slash operator is overloaded:

```python
output_dir / f"{stem}_thumbnail.png"  # works on Linux, macOS, Windows
```

Useful attributes:

```python
p = Path("data/tumor/tumor_001.tif")
p.name      # "tumor_001.tif"      filename + extension
p.stem      # "tumor_001"          filename without extension
p.parent    # Path("data/tumor")   directory containing the file
p.suffix    # ".tif"               the extension
```

Methods worth remembering: `p.exists()`, `p.is_file()`, `p.is_dir()`, `p.mkdir(parents=True, exist_ok=True)` (the flags mean "create intermediate dirs if missing" and "don't error if it already exists").

### Argparse and parsing

**Parsing** = taking raw text input and breaking it into structured pieces a program can use. Examples: Python parses `3 + 4 * 2` into an arithmetic tree; a browser parses HTML into a DOM tree; CSV readers parse comma-separated text into row dictionaries.

**Argparse** parses the command-line arguments your OS hands the script as a flat list of strings. It turns this:

```bash
python scripts/01_inspect_slide.py data/tumor/tumor_001.tif --output-dir output
```

into structured attributes you can read in code:

```python
args.slide       # Path("data/tumor/tumor_001.tif")
args.output_dir  # Path("output")
```

You teach argparse the rules with `add_argument` calls — one per parameter — and it gives you for free: type conversion, defaults, `--help` text auto-generated from the docstring (via `description=__doc__`), and standard error messages.

The convention is positional args have no `--` prefix and are required; optional args have `--` prefix and (usually) defaults.

### Type hints

```python
def f(x: int, y: int | None = None) -> None:
```

The `int | None` (Python 3.10+) syntax means "either int or None." Type hints don't enforce anything at runtime — they document intent and let editors catch mistakes.

### Context managers (the `with` pattern)

```python
with openslide.OpenSlide(str(slide_path)) as slide:
    # use slide here
# .close() is called automatically, even if an exception was raised
```

Better than manual `slide.close()` because cleanup happens deterministically. Same pattern as `with open(...) as f:` for files.

### Defensive value clamping

```python
mask_level = max(0, min(mask_level, slide.level_count - 1))
```

Reads inside-out: first cap the upper bound (`min(mask_level, max_valid)`), then lift the lower bound (`max(0, ...)`). Standard idiom for forcing a value into a valid range without a four-line if/elif block.

## NumPy

### Lists vs arrays

Python lists are flexible (any types, any structure) but slow because they're sequences of Python objects. NumPy arrays are rigid (one dtype, contiguous memory) but fast because operations dispatch to compiled C code.

For image data — millions of pixels, three numbers each — the difference is roughly 100×.

### Shape

For an image, the shape convention is `(height, width, channels)` — height first, even though we describe images verbally as "1528 by 3456" (width × height). NumPy follows mathematical matrix conventions where rows come before columns; OpenSlide follows graphics conventions where x comes before y. Both are internally consistent, they just disagree.

```python
rgb = np.array(region.convert("RGB"))
rgb.shape   # (3456, 1528, 3)  for our slide at level 6
```

### Indexing and slicing

The first index selects along axis 0 (row), the second along axis 1 (column), etc. A colon `:` means "all values along this axis."

```python
a[0, :]   # row 0, all columns         → 1D array
a[:, 1]   # all rows, column 1         → 1D array
a[0, 1]   # row 0, column 1            → scalar
```

The `...` ellipsis means "as many `:` as needed":

```python
hsv[..., 1]   # for shape (H, W, 3): selects channel 1 (saturation) at every pixel
              # equivalent to hsv[:, :, 1]
```

### Broadcasting and elementwise comparisons

```python
tissue = gray < threshold
```

NumPy compares every element of `gray` against the scalar `threshold` in one operation, producing a boolean array of the same shape. No Python loop. This is broadcasting.

### Boolean array tricks

`True` is treated as 1 and `False` as 0:

```python
tissue.mean()    # fraction of True pixels
tissue.sum()     # count of True pixels
```

## OpenSlide

### What it is

A C library (with Python bindings) for reading multi-resolution whole-slide images. Handles all major vendor formats transparently: Aperio (.svs), Hamamatsu (.ndpi), Philips (.tif), MIRAX (.mrxs), and others.

### The pyramid

Whole-slide images are stored as multi-resolution pyramids. Each level halves the linear dimensions of the previous (so quarters the pixel count). Level 0 is the original full-resolution scan; higher level numbers are progressively smaller, lower-detail versions.

For a 10-level slide:

```
Level 0:   1×    (original, biggest, most detail)
Level 1:   2×    (downsample factor; effective magnification halves)
Level 2:   4×
...
Level 9: 512×    (smallest, lowest detail, basically a thumbnail)
```

**Higher level number = lower resolution = lower magnification.** This catches everyone: in microscope conversation, "high magnification" means "more zoom"; in OpenSlide indexing, "high level" means "less zoom." They're inverted.

Total storage of all pyramid levels combined is only ~33% more than level 0 alone — geometric series convergence — which is why pyramid storage is essentially free.

### Reading regions

```python
slide.read_region((x, y), level, (width, height))
```

Three things to know:

1. `(x, y)` is **always in level-0 coordinates**, regardless of which level you're reading. This is the OpenSlide gotcha that catches everyone first time.
2. `(width, height)` is in pixels at the chosen level.
3. Returns a PIL Image in RGBA mode. Convert with `.convert("RGB")` to drop alpha.

### Microns per pixel and magnification

`slide.properties["openslide.mpp-x"]` gives the physical resolution in micrometres per pixel. The conversion to magnification:

```
objective_power ≈ 10 / mpp_x
```

Calibration of the formula: 0.5 µm/px ≈ 20×, 0.25 µm/px ≈ 40×, 1.0 µm/px ≈ 10×.

Magnification is **linear**, not areal. Don't multiply mpp_x and mpp_y to get a "total magnification" — that gives you area per pixel, a different quantity.

To view a 40×-scanned slide as if at 20×, read from level 1. As if at 10×, read from level 2. Each pyramid step halves effective magnification.

## Image processing fundamentals

### RGB → greyscale

`skimage.color.rgb2gray` collapses the 3-channel array to a single channel using a perceptually-weighted sum:

```
gray ≈ 0.299·R + 0.587·G + 0.114·B
```

Green is weighted highest because human eyes are most sensitive to green. The output is in **floats 0–1** (not integers 0–255), because that's the natural domain for image-processing math.

### Colour spaces and HSV

RGB is one way to describe a colour with three numbers. HSV is another, often more useful for processing:

- **Hue** — which colour (angle on a colour wheel, 0–360°). Red 0°, green 120°, blue 240°.
- **Saturation** — how vivid (0 = grey, 1 = pure primary).
- **Value** — how bright (0 = black, 1 = full intensity).

Saturation is computed as:

```
S = (max(R, G, B) − min(R, G, B)) / max(R, G, B)
```

Intuitively: how much do the three channels disagree with each other? Equal channels = grey = saturation 0. One dominant channel = vivid = saturation high.

This is what we used (and abandoned) when trying to discriminate ink markers from tissue.

### Otsu thresholding

Automatically picks the threshold that best separates a bimodal intensity histogram (like H&E slides: dark tissue + bright glass).

**The setup.** You want to split pixels into two classes: those with intensity ≤ T (foreground) and those with intensity > T (background). Otsu's question: which T?

**The algorithm.** For each candidate T, compute:

```
σ²_within(T) = ω₀ · σ²₀ + ω₁ · σ²₁
```

where ω is class weight (fraction of pixels in that class), σ² is variance within that class. The best T minimises this — the one that produces the two most internally consistent groups.

**The equivalence.** A statistical identity says:

```
σ²_total = σ²_within(T) + σ²_between(T)
```

where σ²_total is the overall variance (fixed, independent of T). Since the two right-hand terms must always sum to a constant, minimising one is equivalent to maximising the other. So Otsu equivalently maximises:

```
σ²_between(T) = ω₀ · ω₁ · (μ₀ − μ₁)²
```

This formula needs only weights and means (no subgroup variances), so it's faster to compute. Implementations maximise σ²_between for speed.

**Where Otsu fails.** The method assumes a bimodal histogram. Fails on: low-contrast images (the modes blur into one), images with three or more classes (a single threshold can't separate them), images where one class hugely dominates (the minor mode gets drowned out).

### Morphological operations

Operations on binary images using a "structuring element" — a small shape (we used `disk(4)` and `disk(2)`) that slides over the image.

The two atomic operations:

- **Erosion** — a pixel stays True only if the structuring element fits entirely inside the True region centred on that pixel. Shrinks True regions.
- **Dilation** — a pixel becomes True if the structuring element overlaps any True region. Grows True regions.

The two compounds we use:

- **Closing** = dilate then erode. Net effect: fills small holes, otherwise preserves shape.
- **Opening** = erode then dilate. Net effect: removes small specks, otherwise preserves shape.

Order matters in our pipeline: close first to make tissue solid (fill internal sinuses, vessel lumens), then open to remove dust specks. Reversing the order would lose small tissue features.

### Connected-component analysis

A graph algorithm, not an image-processing algorithm. The mask is treated as a graph: True pixels are nodes, adjacent True pixels are edges. Components are identified using flood-fill or union-find. "Adjacent" usually means 4-connected (sharing an edge), occasionally 8-connected (also sharing a corner).

`skimage.morphology.remove_small_objects(mask, min_size=N)` finds every connected component, measures its pixel count, and removes any with fewer than N pixels. This is what we use to remove ink markers — exploiting the fact that artifacts are small and isolated, while tissue forms large blobs.

The size threshold is dataset-specific. Lymph nodes have 200,000+ pixel blobs, so 15,000 is a conservative threshold that kills artifacts. Needle biopsies (small tissue cores) would need a lower threshold; whole organ slides could go higher.

## Pipeline-specific concepts

### Tile coordinates across pyramid levels

Script 03 has a tricky bit: it tiles at level 1 (effective 20×) but consults a mask at level 6 (the mask is generated by script 02 at low resolution). For each tile location, we need to:

1. Decide which mask region this tile maps to. Tile at level 1 is 256×256 pixels; that's 256 × 2 = 512 pixels at level 0; that's 512 ÷ 64 = 8 pixels at level 6. So each tile corresponds to an 8×8 region of the mask.

2. Translate the tile's level-1 grid position into level-0 coordinates for `read_region`. A tile at row `r`, col `c` at level 1 starts at `(c × 256 × 2, r × 256 × 2)` in level-0 pixels (because OpenSlide's `read_region` always wants level-0 coordinates).

The arithmetic looks scary but it's just unit conversion: tile-grid units → level-1 pixels → level-0 pixels → mask-level pixels.

### Phikon and feature vectors

Phikon is a Vision Transformer (ViT-Base) trained by Owkin on 40M H&E patches with self-supervised learning (iBOT). It's a **foundation model**: it doesn't classify, it produces features (768-dim vectors per tile from the CLS token). Downstream tasks use those features for classification, retrieval, segmentation, etc.

The pretraining data and self-supervised objective give Phikon useful inductive biases for histopathology — it has learned, without labels, what cellular structures, glands, stromal patterns, and stain variations look like.

### Cosine distance and L2 normalisation

For high-dimensional feature vectors from neural networks, cosine similarity (the angle between vectors, ignoring magnitude) is usually more meaningful than Euclidean distance.

To compute it, L2-normalise both vectors first (so each has unit length). Then their dot product equals the cosine of the angle between them, and `1 − dot_product` is the cosine distance.

For our anomaly score, we average cosine distances to k=5 nearest neighbours. Higher distance = more unusual.

## English idioms picked up along the way

A handful of small style notes from the conversation, since improving English was a stated goal:

- "Here's the heatmap" reads more natural than "Here is the heatmap" in chat. Same with "Here's some context."
- "Onto work" or "Let's get to work" beats "On to work!" — the verb form sounds more native.
- "Paste" not "copy" when asking someone to put text back in the chat: "Could you paste the exercise again?"
- "Everything gucci" is fine slang but very informal; in a German workplace, "all good" or "all set" is the equivalent register.
- "The size of the images will be immense" is grammatical but melodramatic for engineering; "the storage requirements get huge" or "the data gets prohibitively large" reads cleaner.
- Capitalise sentence beginnings even in lists ("First: …" not "first: …").
