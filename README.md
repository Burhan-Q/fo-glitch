# fo-glitch

A [FiftyOne](https://docs.voxel51.com) plugin that generates realistic camera corruption artifacts for training data augmentation. Apply glitch effects — pixel sorting, macro-block corruption, channel shifting, and more — to dataset samples directly from the FiftyOne App.

## Why

Real-world cameras (CCTV, industrial inspection, drones) produce transient image corruption during data transmission. Models trained only on clean images produce false positives when encountering these artifacts in production. No mainstream augmentation library covers transmission-level corruption — this plugin fills that gap.

## Corruption Modes

| Mode | Effect | Real-World Artifact |
|------|--------|---------------------|
| **Pixel Sort** | Sort pixels within rows by luminance | Artistic baseline / glitch-art hook |
| **Row Displacement** | Shift rows horizontally by random offsets | Scan-line displacement, signal timing errors |
| **Block Corruption** | Scramble/freeze/zero 16x16 pixel blocks | H.264/H.265 macro-block corruption |
| **Channel Shift** | Offset R/G/B planes by different amounts | Chroma sub-sampling errors |
| **Frame Tear** | Composite halves at different horizontal offsets | Torn frames from missing vsync |
| **Scan Line Noise** | Periodic horizontal intensity bands | EMI interference patterns |
| **Interlacing** | Drop alternating rows and interpolate | Field-based capture artifacts |

Each mode has an independent **enable toggle** and **intensity slider** (0–100%).

## Installation

The plugin lives inside your FiftyOne plugins directory. If it isn't there already, copy or symlink it:

```bash
# Find your plugins directory
python -c "import fiftyone as fo; print(fo.config.plugins_dir)"

# Symlink (adjust paths as needed)
ln -s /path/to/fo-glitch "$PLUGINS_DIR/@Burhan-Q/fo-glitch"
```

No additional Python dependencies beyond FiftyOne itself are required — the corruption engine uses only NumPy and Pillow, both of which ship with FiftyOne.

## Quick Start

### 1. Open the operator

Launch the FiftyOne App with any image dataset loaded. Open the **operator browser** (press `` ` `` or click the search icon) and search for **Apply Glitch**.

The operator uses three tabs: **Corruption Modes**, **Settings**, and **Target**.

### 2. Enable corruption modes (Corruption Modes tab)

Toggle on one or more corruption modes. An intensity slider (0–100%) appears for each enabled mode. For example:

- Enable **Block Corruption (Macro-block)** at 60%
- Enable **Channel Shift (RGB Offset)** at 40%
- Enable **Scan Line Noise** at 30%

### 3. Configure settings (Settings tab)

- **Enable noise injection** — jitters each mode's intensity per sample so artifacts vary across the batch. A scale of 15% means each intensity varies by up to &plusmn;15% of its configured value.
- **Seed** — set an integer for reproducible results, or leave blank for random.
- **Filename suffix** — template appended to source filenames. Default: `_glitch_{TIMESTAMP}`.

Available placeholders:

| Placeholder | Expands To | Example |
|-------------|-----------|---------|
| `{TIMESTAMP}` | Unix epoch | `1713045600` |
| `{DATETIME}` | ISO datetime (file-safe) | `2026-04-13T18-00-00` |
| `{DATE}` | ISO date | `2026-04-13` |
| `{INDEX}` | Zero-padded batch index | `007` |
| `{PROFILE}` | Profile name (sanitized) | `cctv_corruption` |
| `{MODE}` | Enabled mode names | `pixel_sort,block_corruption` |

Output files are saved **alongside the source image** using the pattern `{original_stem}{suffix}{ext}`.

### 4. Preview before applying (Target tab)

Select **Preview selected sample** as the target and click **Execute**. This generates a temporary glitched copy of the currently selected sample so you can inspect the effect. Previous previews are cleaned up automatically.

Adjust modes and preview again until the result looks right.

### 5. Apply to a batch (Target tab)

Once satisfied with the preview, change the target to one of:

- **Current sample** — the single selected sample
- **Current view** — all samples in the active view/filter
- **Random fraction** — a random subset of the full dataset (a fraction slider appears)
- **Saved view** — a previously saved view by name
- **Entire dataset** — every sample

Click **Execute**. New samples are added to the dataset with:

- A `glitched` tag for easy filtering
- `glitch_source_id` linking back to the original sample
- `glitch_applied_params` recording the exact intensities used (after noise)
- `glitch_seed` for reproducibility (when a seed is set)

### 6. Filter and review

After applying, filter the dataset by the `glitched` tag to review all generated samples:

```python
view = dataset.match_tags("glitched")
session.view = view
```

## Reproducibility

Set the **Seed** field to an integer value. With a seed set, the same profile applied to the same images will produce byte-identical output. Each sample in a batch receives a deterministic sub-seed (`base_seed + sample_index`), so results are reproducible even when noise injection is enabled.

## Operators

| Operator | Description | Listed |
|----------|-------------|--------|
| `apply_glitch` | Configure, preview, and apply glitch augmentation | Yes |
| `clean_previews` | Remove temporary preview samples and files | No |

## Configuration Persistence

The **Glitch Configurator** panel (available from the `+` panel browser) saves settings per-dataset using FiftyOne's ExecutionStore. Close the panel and reopen it later — mode toggles, intensity values, noise settings, and the suffix template are all restored. The panel can also trigger the Apply Glitch operator with its current settings pre-filled.

## Roadmap

- **Save / export profiles** — persist named glitch configurations globally and share them across datasets or with teammates
- **Real-time canvas preview** — hybrid JS panel with live pixel manipulation instead of the current operator-based preview
- **Per-mode advanced parameters** — expose block size, sort direction, frequency, and other mode-specific knobs
- **Video support** — apply frame-level corruption patterns to video datasets

## License

Apache 2.0
