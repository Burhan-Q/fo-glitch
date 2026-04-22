# fo-glitch

A [FiftyOne](https://docs.voxel51.com) plugin that generates realistic camera-corruption artifacts for training-data augmentation. Apply glitch effects — pixel sorting, macro-block corruption, channel shifting, and more — to dataset samples directly from the FiftyOne App.

## Why

Real-world cameras (CCTV, industrial inspection, drones) produce transient image corruption during data transmission. Models trained only on clean images produce false positives when encountering these artifacts in production. No mainstream augmentation library covers transmission-level corruption — this plugin fills that gap.

## Corruption Modes

| Mode | Effect | Real-world analogue |
|------|--------|---------------------|
| **Pixel Sort** | Sort contiguous pixel runs within each row by brightness | Artistic baseline / glitch-art hook |
| **Row Displacement** | Shift rows horizontally by random offsets | Scan-line displacement, signal timing errors |
| **Block Corruption** | Scramble / freeze / zero blocks at one or more sizes | H.264 / H.265 macro-block corruption |
| **Channel Shift** | Offset R/G/B planes by different amounts | Chroma sub-sampling errors |
| **Frame Tear** | Split the image at random y-coordinates and offset the halves | Torn frames from missing vsync |
| **Scan Line Noise** | Periodic horizontal intensity bands | EMI interference |
| **Interlacing** | Darken alternating rows | Field-based capture artifacts |

Each mode has an independent enable checkbox and a 0–100 % intensity number field. Block Corruption exposes three additional tuning fields when enabled: block size (% of the image's shorter edge), pattern (`uniform`, `localized`, `streak`), and layers (1–4 multi-pass count).

## Architecture

Single operator, no panel. The entire workflow — configure, preview, target, execute — lives in one dynamic operator form. A separate panel was prototyped but deliberately removed to avoid having two interfaces for the same functionality.

```
fiftyone.yml           plugin manifest
__init__.py            registers the ApplyGlitch operator
config.py              dataclasses, constants, env-var resolution, suffix helpers
glitch.py              numpy-only corruption engine (7 mode fns + apply_profile)
operators.py           ApplyGlitch (resolve_input split into per-section _render_* helpers)
```

## Installation

The plugin lives inside your FiftyOne plugins directory. Copy or symlink it there if it isn't already:

```bash
# Find your plugins directory
python -c "import fiftyone as fo; print(fo.config.plugins_dir)"

# Symlink (adjust paths as needed)
ln -s /path/to/fo-glitch "$PLUGINS_DIR/@Burhan-Q/fo-glitch"
```

No additional Python dependencies beyond FiftyOne itself — the engine uses only NumPy and Pillow, both of which ship with FiftyOne.

## Quick start

### 1. Open the operator

Launch the FiftyOne App with any image dataset loaded. Open the operator browser (press `` ` `` or click the search icon) and run **Apply Glitch Augmentations**.

The form is flat (not tabbed). Sections, top to bottom:

| Section | What it controls |
|---------|------------------|
| Corruption Modes | The 7 per-mode toggles + intensities (and Block Corruption sub-fields) |
| Noise Injection | Per-sample intensity jitter toggle + scale % |
| Preview | Inline preview of the configured effect on the first selected / in-view sample |
| Settings | Seed, filename-suffix template |
| Augment Samples | Target-selection dropdown + target-specific sub-fields (fraction, saved view, tags) |

### 2. Enable modes and set intensities

Toggle on one or more modes. Each mode's intensity field becomes editable; a value of `0` or an empty field causes that mode to be skipped at execute time (same effect as leaving it disabled). Example:

- **Block Corruption (Macro-block)** at 60 %, with size 2 % and 3 layers
- **Channel Shift (RGB Offset)** at 40 %
- **Scan Line Noise** at 30 %

### 3. (Optional) enable noise injection

Flip **Enable noise injection** to jitter each mode's intensity independently per sample. The scale is a percentage of each intensity — a scale of 15 % means each sample's effective intensity is perturbed by up to ±15 % of its configured value. This produces visibly distinct glitches across a batch.

### 4. (Optional) preview

Flip **Show preview** to render a glitched version of the first selected sample (or first sample in view if nothing is selected). The preview is written to a hidden file next to the source sample and served through FiftyOne's `/media` endpoint. The file is removed at the end of execute; dismissing the dialog without running may leave at most one orphan file per source directory, which is overwritten on the next preview.

### 5. Settings

- **Seed** — integer for reproducible output, blank for non-deterministic. With a seed set, sample *i* of the batch receives a deterministic sub-seed `base_seed + i`, so results are reproducible even when noise injection is enabled.
- **Filename suffix** — template appended to source filenames. Default: `_glitch_{TIMESTAMP}`. Supported placeholders:

  | Placeholder | Expands to | Example |
  |-------------|-----------|---------|
  | `{TIMESTAMP}` | Unix epoch (int) | `1776891021` |
  | `{DATETIME}` | ISO datetime (file-safe, UTC) | `2026-04-22T18-00-00` |
  | `{DATE}` | ISO date (UTC) | `2026-04-22` |
  | `{INDEX}` | Zero-padded batch index | `007` |
  | `{MODE}` | Comma-joined enabled mode names | `pixel_sort,block_corruption` |

  Any other `{...}` token (`{FOO}`, `{1234}`, `{AB !@}`, etc.) is flagged with an inline warning and **stripped** from the output filename at execute time. Output files are saved **alongside the source image**, using the pattern `{original_stem}{expanded_suffix}{ext}`.

### 6. Pick a target and execute

Choose **Augment Samples** from:

| Choice | What it means |
|--------|---------------|
| Selected sample(s) | Samples highlighted in the grid, or the current modal sample if none are highlighted |
| Samples with tag(s) | Multi-select a set of sample tags; matches `match_tags` |
| Current view | The active view / filter (default) |
| Saved view | A dataset-saved view selected by name |
| Random fraction of dataset | Random subset sized by the fraction slider (0.01–1.00) |
| Entire dataset | All samples |

When the resolved target exceeds the delegation threshold (default 100 samples; see below), a notice appears recommending **Schedule** instead of **Execute**. Use the button's ▾ menu to choose.

On execute, new samples are added to the dataset with:

- A `glitched` tag (plus all tags inherited from the source)
- `glitch_source_id` — the source sample's ID
- `glitch_profile` — a dict of the applied configuration
- `glitch_applied_params` — the actual intensities used after any noise jitter
- `glitch_seed` — the per-sample sub-seed, when a base seed was set

### 7. Filter and review

```python
view = dataset.match_tags("glitched")
session.view = view
```

## Configuration

| Environment variable | Default | Effect |
|----------------------|---------|--------|
| `FO_GLITCH_DELEGATE_THRESHOLD` | `100` | Sample-count threshold above which the form recommends Schedule rather than Execute. Resolved once at plugin-module load; set before launching FiftyOne to change it. |

Declared in `fiftyone.yml` under `secrets:`.

## Reproducibility

Given:
- identical inputs (same images),
- identical profile (same modes, intensities, noise settings),
- identical seed,

the plugin produces byte-identical output. Each sample's per-sample RNG is `numpy.random.default_rng(seed + index)` where `index` is its position in the target view iteration.

## Operators

| Operator | Description | Listed |
|----------|-------------|--------|
| `apply_glitch` | Configure, preview, and apply glitch augmentation | Yes |

## Roadmap

- **Save / export profiles** — persist named glitch configurations globally and share them across datasets or with teammates
- **Real-time canvas preview** — hybrid JS panel with live pixel manipulation instead of the current operator-form preview
- **Video support** — apply frame-level corruption patterns to video datasets
- **Cursor-aware suffix insertion** — small buttons next to the suffix field that insert `{TIMESTAMP}` etc. at the cursor position rather than the end

## License

Apache 2.0
