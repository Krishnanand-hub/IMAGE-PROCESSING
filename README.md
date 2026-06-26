# Nuclei Segmentation in Fluorescence Microscopy — Classical Image-Processing Pipeline

A staged, **non-deep-learning** pipeline that segments cell nuclei in 16-bit fluorescence microscopy images and benchmarks each stage against expert ground-truth masks using IoU, Dice, and object-level F1.

Built on the public **BBBC039v1** benchmark (Broad Bioimage Benchmark Collection). The point of the project is to show that carefully engineered classical methods (Otsu, morphology, marker-controlled watershed, adaptive thresholding) get you most of the way on this task — and to measure exactly how far.

---

## Table of Contents

- [Aim & Abstract](#aim--abstract)
- [What This Project Does](#what-this-project-does)
- [Outcomes](#outcomes)
- [Dataset](#dataset)
- [Repository / Image Structure](#repository--image-structure)
- [How to Run It](#how-to-run-it)
- [Crucial Things to Keep in Mind](#crucial-things-to-keep-in-mind)
- [Common Errors & Fixes](#common-errors--fixes)
- [Tech Stack](#tech-stack)
- [License & Attribution](#license--attribution)

---

## Aim & Abstract

**Aim:** Segment individual cell nuclei from fluorescence microscopy images and quantify how much each processing stage improves segmentation quality against expert annotations.

**Abstract:** Identifying nuclei is usually the first step in any quantitative microscopy workflow — counting cells, measuring density, profiling morphology. This project implements a four-stage classical pipeline and evaluates it on Hoechst-stained U2OS nuclei. Starting from a global Otsu threshold baseline, it layers in morphological cleanup, marker-controlled watershed (to split touching nuclei), and adaptive local thresholding. Each stage is scored with pixel-level metrics (IoU, Dice, sensitivity, specificity) and an object-level F1 at IoU ≥ 0.5. The result is a reproducible benchmark showing that a tuned classical pipeline reaches a foreground IoU of ~0.91 and Dice of ~0.95 — without any trained model.

---

## What This Project Does

The pipeline runs in four cumulative stages, each building on the previous one:

| Stage | Method | What it adds |
|-------|--------|--------------|
| **S1 — Baseline** | Percentile contrast stretch (1–99.5%) → Gaussian smoothing (σ=1.5) → global **Otsu threshold** | A first binary segmentation |
| **S2 — Cleanup** | + binary opening (disk r=2), hole filling, removal of objects < 30 px | Removes speckle noise and fills nuclei |
| **S3 — Watershed** | + Euclidean distance transform → `peak_local_max` seeds (min_distance=7) → **marker-controlled watershed** | Splits touching/fused nuclei into instances |
| **S4 — Adaptive** | + **local adaptive threshold** (block size 51) gated by a 0.6× Otsu floor, then morphology + watershed | Recovers nuclei missed by a single global threshold |

Evaluation:
- **Pixel-level:** foreground IoU, mean IoU, Dice, sensitivity, specificity — per image and averaged.
- **Object-level:** precision / recall / F1 by matching predicted vs. ground-truth instances at IoU ≥ 0.5.
- **Artifacts produced:** `metrics_all.csv`, `metrics_summary.csv`, and `performance_by_stage.png`.

---

## Outcomes

Measured on the 5-image subset in this repository (mean across images):

| Stage | Foreground IoU | mIoU | Dice | Sensitivity | Specificity |
|-------|:---:|:---:|:---:|:---:|:---:|
| S1: Otsu | 0.887 | 0.930 | 0.940 | 0.907 | 0.996 |
| S2: + cleanup | 0.885 | 0.929 | 0.939 | 0.904 | 0.996 |
| S3: + watershed | 0.884 | 0.928 | 0.938 | 0.904 | 0.996 |
| **S4: + adaptive** | **0.909** | **0.944** | **0.952** | **0.950** | 0.991 |

**Object-level (example image):** precision 0.74, recall 0.87, **F1 0.80** — 111 nuclei predicted vs. 94 in ground truth.

**Honest read of the results:**
- The adaptive stage (S4) is the only one that meaningfully moves the needle — it lifts foreground IoU by ~2 points and sensitivity from 0.91 to 0.95 by recovering nuclei a single global threshold misses.
- S2 and S3 leave pixel metrics essentially flat. Watershed's value is **instance separation** (splitting touching nuclei), which pixel-level IoU does not reward — its benefit shows up in the object-level F1, not the IoU table.
- These numbers are on **5 images**. They are a demonstration, not a population estimate. The full BBBC039 set has 200 images and ~23,000 annotated nuclei; results on the full set will differ.

---

## Dataset

This is the **BBBC039v1** dataset — *"Nuclei of U2OS cells in a chemical screen"* — from the Broad Bioimage Benchmark Collection.

- **Content:** Fluorescence microscopy fields of Hoechst-stained U2OS cell nuclei (DNA channel).
- **Format:** 16-bit TIFF images at 520 × 696 px; ground-truth as PNG instance masks (touching nuclei encoded with distinct labels).
- **Full size:** 200 images (~23,000 annotated nuclei). **This repo includes a 5-image subset** distributed as coursework material by Newcastle University.
- **License:** Creative Commons Zero (CC0) — public domain.
- **Source:** https://bbbc.broadinstitute.org/BBBC039
- **GitHub mirrors:** `DeepTrackAI/cell_counting_dataset`, `Nicholas-Schaub/BBBC039`

**Citation:**
> Caicedo, J.C. et al. *Evaluation of Deep Learning Strategies for Nucleus Segmentation in Fluorescence Images.* Cytometry Part A, 95(9): 952–965 (2019). DOI: 10.1002/cyto.a.23863
> Ljosa, V. et al. *Annotated high-throughput microscopy image sets for validation.* Nature Methods, 9: 637 (2012). DOI: 10.1038/nmeth.2083

To run on the **full 200-image set**, download it from the BBBC039 page above and point the data root at the unpacked folders.

---

## Repository / Image Structure

The pipeline expects two sibling folders — images and masks — with **matching filenames** (same stem, different extension):

```
Microscopy_Data/
├── im/                                   # input images: 16-bit TIFF
│   ├── IXMtest_A02_s1_w1051DAA7C-....tif
│   ├── IXMtest_A06_s6_w1B9577918-....tif
│   └── ...
└── gt/                                   # ground-truth masks: PNG (instance-labelled)
    ├── IXMtest_A02_s1_w1051DAA7C-....png
    ├── IXMtest_A06_s6_w1B9577918-....png
    └── ...
```

**Pairing rule:** an image and its mask are matched by their **filename stem** (everything before the extension). `im/FOO.tif` pairs with `gt/FOO.png`. Files without a partner in the other folder are silently ignored. The code asserts at least one matched pair exists before proceeding.

Outputs (`metrics_all.csv`, `metrics_summary.csv`, `performance_by_stage.png`) are written to an `outputs/` folder created next to `im/` and `gt/`.

---

## How to Run It

The notebook as committed targets **Google Colab**. Two paths:

### Option A — Google Colab (as written, least friction)
1. Upload `Microscopy_Data/` to your Google Drive at `MyDrive/Microscopy_Data`.
2. Open `IP.ipynb` in Colab.
3. Run all cells. The `drive.mount(...)` cell will prompt for Drive authorisation.

### Option B — Run locally (recommended for a portfolio reviewer)
The notebook will **not** run locally as-is because of the Colab Drive mount. Two edits fix it:

1. Clone the repo:
   ```bash
   git clone https://github.com/Krishnanand-hub/IMAGE-PROCESSING.git
   cd IMAGE-PROCESSING
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy scikit-image tifffile pillow jupyter
   ```
3. In the notebook, **delete the Colab mount cell**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   and change the data root to the repo-local path:
   ```python
   ROOT = "Microscopy_Data"          # was "/content/drive/MyDrive/Microscopy_Data"
   ```
4. Run the notebook:
   ```bash
   jupyter notebook IP.ipynb
   ```

---

## Crucial Things to Keep in Mind

- **Filename stems must match** across `im/` and `gt/`. A mismatched extension or a renamed file means that pair is dropped without warning. If "Matched pairs: 0" appears, this is why.
- **Images are 16-bit.** Do not load them with a plain 8-bit reader — intensity range here is roughly `[120, 4095]`. The pipeline reads them via `tifffile`.
- **Masks are instance-labelled, not binary.** They are read as RGBA PNGs and collapsed to a 2-D label map; `> 0` gives the binary foreground. Touching nuclei carry different labels — that distinction is what the watershed stage and the object-level F1 depend on.
- **Tunable parameters** live in the stage functions: `sigma` (smoothing), `min_size` (small-object removal), `min_distance` (watershed seed spacing), `block_size` and `floor_frac` (adaptive threshold). These were tuned for this dataset; expect to retune for images at a different magnification or SNR.
- **Metrics scale with image count.** Five images is a demo. Don't quote these numbers as the method's performance on BBBC039 as a whole.

---

## Common Errors & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'google.colab'` | Running the Colab mount cell outside Colab | Delete the `drive.mount` cell; set `ROOT` to a local path (Option B). |
| `AssertionError: No matched pairs...` | `im/` and `gt/` filename stems don't match, or `ROOT` is wrong | Check `ROOT` points at the folder containing `im/` and `gt/`; confirm stems align. |
| `FileNotFoundError` on the data root | Drive not mounted, or data not uploaded to the expected path | Mount Drive (Colab) or correct the local `ROOT`. |
| `tifffile` import fails | Package missing | `pip install tifffile` (the notebook also auto-installs it at runtime). |
| Mask looks empty / all-background | A genuinely binary or single-channel PNG read with the wrong assumption | The loader handles grayscale, RGB, and RGBA; verify the mask actually encodes foreground as non-zero. |
| Memory spike on the full 200-image set | All images held in memory at once | Process in batches, or load images lazily inside the per-image loop. |

---

## Tech Stack

Python · NumPy · pandas · SciPy (ndimage) · scikit-image (filters, morphology, segmentation, measure) · tifffile · Pillow · matplotlib

---

## License & Attribution

- **Data:** BBBC039v1, Broad Bioimage Benchmark Collection — Creative Commons Zero (CC0). Cite Caicedo et al. (2019) and Ljosa et al. (2012) as above.
