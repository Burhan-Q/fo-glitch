# FiftyOne Plugin Glitch Augmentation

<p align="center">
    <img
        width="600" 
        height="450" 
        alt="FiftyOne Glitch Augmentation plugin demo preview" 
        src="https://github.com/user-attachments/assets/940bd974-c714-4710-8847-59eddef68d99" 
    />
</p>

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

### CLI

```bash
fiftyone plugins download https://github.com/Burhan-Q/fo-glitch
```

### Python

```python
import fiftyone.plugins as fop

fop.download("https://github.com/Burhan-Q/fo-glitch")
```

### Requirements

- Python >= 3.11
- [FiftyOne](https://docs.voxel51.com) >= 1.14.1
- [NumPy](https://numpy.org/) >= 2.4.4
- [Pillow](https://pillow.readthedocs.io/) >= 12.2.0

NumPy and Pillow ship with FiftyOne; no extra install step is needed unless you want to pin newer versions yourself.

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

  An additional warning appears when the suffix contains `{MODE}` and three or more corruption modes are enabled at once: the joined mode list can push the final filename close to the 255-byte filesystem name limit. Either drop `{MODE}` from the suffix or enable fewer modes per run.

  **Character set / OS assumptions**: filenames are assumed to be standard ASCII, and source directories are assumed to be writable. If write fails for a given sample (permissions, disk full, path-length overflow) the error is logged to the FiftyOne server log and the batch continues; skipped samples are reported in the operator's completion notice and output summary.

### 6. Pick a target and execute

Choose **Augment Samples** from:

| Choice | What it means |
|--------|---------------|
| Selected sample(s) | Samples highlighted in the grid, or the current modal sample if none are highlighted |
| Samples with tag(s) | Multi-select a set of sample tags; matches `match_tags` |
| Current view | The active view / filter (default) |
| Saved view | A dataset-saved view selected by name |
| Random fraction of dataset | Random subset sized by the fraction field (0.01–1.00) |
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

## History

Version numbers track `pyproject.toml`.  Dates are the day the change landed on `main`.

### 0.1.0 (2026-04-22) — current

- **Plugin surface consolidated.** The standalone *Glitch Configurator* panel was removed so the plugin has a single UI entry point: the `Apply Glitch Augmentations` operator form. A panel had been prototyped for per-dataset profile persistence; maintaining two parallel interfaces with the same fields created drift risk and was dropped in favor of the operator form alone.
- **Filename-suffix input validation.** The suffix field now accepts arbitrary `{...}` tokens and flags any unrecognized ones inline; unknown tokens are stripped at execute time. A second inline warning fires when `{MODE}` is combined with three or more enabled modes to flag filename-length risk.
- **Preview inlined in the operator form.** A `Show preview` switch renders a glitched sample directly inside the form using FiftyOne's `/media` endpoint. The hidden preview file is cleaned up at the end of execute (success or empty target); dismissing the dialog may leave at most one orphan per source directory, overwritten on the next preview.
- **Target selection expanded.** Added *Selected sample(s)* and *Samples with tag(s)* to the existing view-based target choices.
- **Delegation recommendation** for large targets driven by `FO_GLITCH_DELEGATE_THRESHOLD` (default 100) — the form shows a notice recommending `Schedule` instead of `Execute` rather than auto-delegating.
- **Engine correctness fixes.** `0`/`None` intensity skips the mode entirely; row-displacement intensity scaling rebuilt as a probability mask to eliminate step discontinuity; interlacing reworked to darken alternating rows (previous implementation was a no-op on static images); block corruption gained size (% of image), pattern, and multi-pass layer tuning.
- **Removed dead code.** The `CleanPreviews` utility and the `{PROFILE}` filename placeholder were both removed — neither had a reachable write path in the current code.
- **Internal refactor.** `resolve_input` split into per-section `_render_*` helpers; `_safe_float`/`_safe_int` collapsed onto a shared `_coerce_number` helper; `GlitchProfile.name` field dropped alongside `{PROFILE}`.
- **Dependencies declared explicitly.** `numpy>=2.4.4` and `Pillow>=12.2.0` are now listed in `pyproject.toml` rather than relied on transitively via `fiftyone`.

## Roadmap

- **Save / export profiles** — persist named glitch configurations globally and share them across datasets or with teammates
- **Real-time canvas preview** — hybrid JS panel with live pixel manipulation instead of the current operator-form preview
- **Video support** — apply frame-level corruption patterns to video datasets
- **Cursor-aware suffix insertion** — small buttons next to the suffix field that insert `{TIMESTAMP}` etc. at the cursor position rather than the end

## Development

Clone the repository and symlink into your FiftyOne plugins directory:

```bash
git clone https://github.com/Burhan-Q/fo-glitch.git
ln -s "$(pwd)/fo-glitch" "$(fiftyone config plugins_dir)/@Burhan-Q/fo-glitch"
```

## License

Apache 2.0
