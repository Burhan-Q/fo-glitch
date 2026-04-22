"""Core corruption engine — pure numpy image manipulation functions.

Every corruption function has the signature::

    fn(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray

where *img* is (H, W, 3) uint8 RGB, *intensity* is 0–100, and *rng* is a
per-sample random generator for deterministic reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import GLITCH_MODES, GlitchProfile

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_image(filepath: str | Path) -> np.ndarray:
    """Load an image from disk as a numpy array (H, W, 3) uint8 RGB."""
    with Image.open(filepath) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_image(img: np.ndarray, filepath: str | Path) -> None:
    """Save a numpy array (H, W, 3) uint8 RGB to disk via PIL.

    Parent directories are created if they do not exist.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------


def pixel_sort(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Sort pixels within rows by luminance between brightness thresholds.

    For each row the function identifies contiguous spans of pixels whose
    brightness falls within a threshold window, then sorts those spans by
    luminance.  Higher *intensity* widens the window so more pixels are
    eligible for sorting.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls the width of the brightness threshold
            window.  At 0 almost no pixels qualify; at 100 the full
            brightness range is eligible.
        rng: Random generator for per-row threshold jitter.

    Returns:
        New image array with sorted pixel spans.
    """
    out = img.copy()
    h, w, _ = img.shape

    # Convert to grayscale for brightness
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    # Map intensity to threshold window
    window = intensity / 100.0 * 200  # max window = 200 brightness levels
    base_low = max(0, 128 - window / 2)
    base_high = min(255, 128 + window / 2)

    for y in range(h):
        # Small per-row jitter for natural look
        jitter = rng.uniform(-10, 10)
        lo = max(0, base_low + jitter)
        hi = min(255, base_high + jitter)

        row_gray = gray[y]
        mask = (row_gray >= lo) & (row_gray <= hi)

        # Find contiguous True spans
        spans = _contiguous_spans(mask)
        for start, end in spans:
            if end - start < 2:
                continue
            segment = out[y, start:end]
            lum = row_gray[start:end]
            order = np.argsort(lum)
            out[y, start:end] = segment[order]

    return out


def _contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True spans in a 1-D boolean array.

    Returns a list of ``(start, end)`` index pairs (end is exclusive).
    """
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def row_displacement(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Shift a subset of rows horizontally by random offsets.

    Simulates scan-line displacement / signal timing errors.  Both the
    *fraction* of rows that get shifted and the *magnitude* of each
    shift scale linearly with intensity, so the effect grows smoothly
    from subtle (a few rows shift by 1 px) to extreme (every row
    shifts by up to ~10 % of image width).

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  At 0 no rows move; at 100 every row is
            shifted by up to ~10 % of image width.
        rng: Random generator for row selection and offsets.

    Returns:
        New image with shifted rows (edges filled with black).
    """
    out = img.copy()
    h, w, _ = img.shape

    t = intensity / 100.0
    # At low intensity, shift only a few rows by 1 px (subtle).  As
    # intensity rises, both the affected fraction and the max offset
    # grow linearly, avoiding the threshold-pop behaviour of a purely
    # magnitude-based scaling.
    max_offset = max(1, int(round(w * t * 0.1)))
    selected = rng.random(h) < t

    offsets = rng.integers(-max_offset, max_offset + 1, size=h)
    for y in range(h):
        off = int(offsets[y])
        if not selected[y] or off == 0:
            continue
        out[y] = np.roll(img[y], off, axis=0)
        # Fill wrapped pixels with black
        if off > 0:
            out[y, :off] = 0
        else:
            out[y, off:] = 0

    return out


def _block_probs(
    rows: int,
    cols: int,
    base_prob: float,
    pattern: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a (rows, cols) array of per-block corruption probabilities.

    Args:
        rows: Number of block rows.
        cols: Number of block columns.
        base_prob: Average probability for a block to be corrupted (0–1).
        pattern: ``"uniform"``, ``"localized"``, or ``"streak"``.
        rng: Random generator for pattern randomness.
    """
    if pattern == "localized":
        cy = int(rng.integers(0, rows))
        cx = int(rng.integers(0, cols))
        yy, xx = np.meshgrid(
            np.arange(rows), np.arange(cols), indexing="ij"
        )
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = np.sqrt(rows**2 + cols**2) * 0.5
        falloff = np.clip(1.0 - dist / max_dist, 0.0, 1.0)
        # 2x boost so average corruption count stays near base_prob
        return base_prob * falloff * 2.0
    if pattern == "streak":
        num = int(rng.integers(1, 4))
        streak_rows = rng.integers(0, rows, size=num)
        probs = np.full((rows, cols), base_prob * 0.1)
        probs[streak_rows, :] = base_prob * 3.0
        return probs
    # uniform (default)
    return np.full((rows, cols), base_prob)


def _block_corruption_pass(
    out: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
    block: int,
    pattern: str,
) -> np.ndarray:
    """Apply a single pass of block corruption with a fixed block size.

    Mutates and returns *out* in place.  Used by :func:`block_corruption`
    for each layer in a multi-pass run.
    """
    h, w, _ = out.shape
    rows = h // block
    cols = w // block
    if rows == 0 or cols == 0:
        return out

    base_prob = (intensity / 100.0) * 0.5
    probs = _block_probs(rows, cols, base_prob, pattern, rng)
    src = out.copy()  # source for copy-action reads current state

    for br in range(rows):
        for bc in range(cols):
            if rng.random() > probs[br, bc]:
                continue
            y0, y1 = br * block, (br + 1) * block
            x0, x1 = bc * block, (bc + 1) * block

            action = rng.integers(0, 3)
            if action == 0:
                # Copy from a random other block position
                sy0 = int(rng.integers(0, rows)) * block
                sx0 = int(rng.integers(0, cols)) * block
                out[y0:y1, x0:x1] = src[sy0 : sy0 + block, sx0 : sx0 + block]
            elif action == 1:
                # Zero the block (black)
                out[y0:y1, x0:x1] = 0
            else:
                # Shift the block horizontally within its row
                shift = int(rng.integers(-block, block + 1))
                src_x0 = max(0, x0 + shift)
                src_x1 = min(w, x1 + shift)
                dst_w = src_x1 - src_x0
                if dst_w > 0:
                    out[y0:y1, x0 : x0 + dst_w] = src[y0:y1, src_x0:src_x1]

    return out


def block_corruption(
    img: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
    size_pct: float = 2.0,
    pattern: str = "uniform",
    layers: int = 1,
) -> np.ndarray:
    """Scramble, freeze, or zero pixel blocks at random positions.

    Simulates H.264/H.265 macro-block corruption from lost I-frames.
    Block size is computed as a percentage of ``min(H, W)`` so the
    effect scales correctly across image resolutions.  When *layers*
    is greater than 1 the function runs multiple passes at varied
    sizes (halving and doubling around the configured size), each at
    reduced intensity, producing overlapping corruption that mimics
    real codec behaviour with hierarchical block sizes.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Average probability that each block is
            corrupted (at 100, uniform pattern hits ~50 % of blocks).
            In multi-pass mode the intensity is divided across layers.
        rng: Random generator for block selection and corruption type.
        size_pct: Block size as a percentage of ``min(H, W)``.  A value
            of 2.0 means blocks are ~2 % of the image's shorter side.
        pattern: Spatial distribution of corruption — ``"uniform"``
            (even), ``"localized"`` (clustered around a random point),
            or ``"streak"`` (concentrated on random horizontal bands).
        layers: Number of corruption passes (1–4).  Each layer uses a
            different block size spread around *size_pct* and receives
            ``intensity / layers`` so cumulative density stays sane.

    Returns:
        New image with corrupted blocks.
    """
    out = img.copy()
    h, w, _ = out.shape
    base_px = max(2, int(min(h, w) * size_pct / 100.0))

    layers = max(1, min(int(layers), 4))
    per_pass_intensity = intensity / layers

    # Layer size factors spread around 1x: for layers=1 just [1x]; for
    # layers=3 [0.5x, 1x, 2x]; for layers=4 [0.5x, ~0.8x, ~1.3x, 2x].
    for i in range(layers):
        factor = 2.0 ** (i - (layers - 1) / 2.0)
        block_px = max(2, int(base_px * factor))
        _block_corruption_pass(out, per_pass_intensity, rng, block_px, pattern)

    return out


def channel_shift(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Offset R, G, B planes by different random pixel amounts.

    Simulates chroma sub-sampling errors during signal transmission.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls the maximum offset in pixels for each
            channel.  At 100 the max offset is ~3 % of image width.
        rng: Random generator for per-channel offsets.

    Returns:
        New image with shifted colour channels.
    """
    h, w, _ = img.shape
    max_off = max(1, int(w * (intensity / 100.0) * 0.03))
    out = np.zeros_like(img)

    for ch in range(3):
        dx = int(rng.integers(-max_off, max_off + 1))
        dy = int(rng.integers(-max_off, max_off + 1))
        out[:, :, ch] = np.roll(img[:, :, ch], (dy, dx), axis=(0, 1))

    return out


def frame_tear(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Split image at random y-coordinates and offset halves horizontally.

    Simulates torn frames from missing vsync in real-time capture.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls the number of tear lines (1–5) and
            the maximum horizontal offset magnitude.
        rng: Random generator for tear positions and offsets.

    Returns:
        New image with horizontal tear lines.
    """
    out = img.copy()
    h, w, _ = img.shape
    num_tears = max(1, int((intensity / 100.0) * 5))
    max_offset = max(1, int(w * (intensity / 100.0) * 0.15))

    tear_ys = sorted(rng.integers(1, h - 1, size=num_tears))
    prev_y = 0

    for tear_y in tear_ys:
        offset = int(rng.integers(-max_offset, max_offset + 1))
        if offset == 0:
            continue
        out[prev_y:tear_y] = np.roll(img[prev_y:tear_y], offset, axis=1)
        prev_y = tear_y

    return out


def scanline_noise(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Add periodic horizontal intensity bands simulating EMI interference.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls the amplitude of the noise bands.
            Higher values produce more visible banding.
        rng: Random generator for phase and frequency jitter.

    Returns:
        New image with horizontal scan-line noise overlaid.
    """
    h, w, _ = img.shape
    amplitude = (intensity / 100.0) * 80  # max ±80 brightness levels

    # Generate a sinusoidal pattern with slight randomness
    freq = rng.uniform(0.02, 0.08)  # cycles per pixel row
    phase = rng.uniform(0, 2 * np.pi)
    y_coords = np.arange(h, dtype=np.float32)
    wave = np.sin(2 * np.pi * freq * y_coords + phase) * amplitude

    # Add per-row noise for realism
    row_noise = rng.normal(0, amplitude * 0.2, size=h)
    wave += row_noise

    # Broadcast to (H, W, 3) and add
    noise_pattern = wave[:, np.newaxis, np.newaxis].astype(np.float32)
    result = img.astype(np.float32) + noise_pattern
    return np.clip(result, 0, 255).astype(np.uint8)


def interlace(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Darken alternating rows to produce visible scan-line artefacts.

    Simulates interlaced / field-based capture artefacts.  At low
    intensity the effect is subtle; at high intensity alternate rows
    are strongly darkened, creating the classic comb / scan-line look.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls how much alternate rows are
            darkened.  At 0 no change; at 100 dropped rows are reduced
            to ~15 % of original brightness.
        rng: Random generator for field selection (odd / even).

    Returns:
        New image with interlacing artefacts.
    """
    out = img.copy()
    drop_odd = bool(rng.integers(0, 2))
    start = 1 if drop_odd else 0

    # At intensity=100, alternate rows retain 15% brightness (~85% darken)
    factor = 1.0 - (intensity / 100.0) * 0.85
    darkened = (out[start::2].astype(np.float32) * factor).clip(0, 255)
    out[start::2] = darkened.astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

GLITCH_FUNCTIONS = {
    "pixel_sort": pixel_sort,
    "row_displacement": row_displacement,
    "block_corruption": block_corruption,
    "channel_shift": channel_shift,
    "frame_tear": frame_tear,
    "scanline_noise": scanline_noise,
    "interlace": interlace,
}
"""Maps mode names to their corruption functions."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def apply_profile(
    img: np.ndarray,
    profile: GlitchProfile,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply all enabled modes from a profile to an image.

    Applies noise jitter to each mode's intensity when
    ``profile.noise.enabled`` is ``True``.  Returns both the corrupted
    image and a dict of the *actual* intensities used (after jitter) so
    callers can record exactly what was applied.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        profile: Complete glitch configuration.
        rng: Random generator (should be unique per sample for
            reproducibility).

    Returns:
        ``(corrupted_image, applied_params)`` where *applied_params* is
        ``{mode_name: actual_intensity}`` for every enabled mode.
    """
    applied_params: dict[str, float] = {}

    for mode_name in GLITCH_MODES:
        mode_cfg = profile.mode_configs[mode_name]
        if not mode_cfg.enabled:
            continue

        # Skip modes with zero intensity (cleared field or explicit 0).
        # A checkbox toggled on but with no intensity value means the
        # user wanted this mode disabled — don't silently substitute
        # a default.
        if mode_cfg.intensity <= 0:
            continue

        intensity = mode_cfg.intensity

        # Noise injection — jitter the intensity per sample
        if profile.noise.enabled:
            scale = profile.noise.scale / 100.0
            jitter = rng.uniform(-scale, scale) * intensity
            intensity = float(np.clip(intensity + jitter, 0.0, 100.0))

        applied_params[mode_name] = round(intensity, 4)
        if mode_name == "block_corruption":
            img = block_corruption(
                img,
                intensity,
                rng,
                size_pct=profile.block_size_pct,
                pattern=profile.block_pattern,
                layers=profile.block_layers,
            )
        else:
            img = GLITCH_FUNCTIONS[mode_name](img, intensity, rng)

    return img, applied_params


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------


def generate_preview_image(
    filepath: str | Path,
    profile: GlitchProfile,
    seed: int | None,
    max_dim: int = 512,
) -> np.ndarray:
    """Load, downscale, and corrupt an image for preview.

    Args:
        filepath: Path to the source image on disk.
        profile: The glitch profile to apply.
        seed: Base seed for the RNG (``None`` for non-deterministic).
        max_dim: Maximum pixel dimension (width or height) of the result.

    Returns:
        Corrupted image array, shape (H, W, 3), dtype uint8.
    """
    img = load_image(filepath)
    h, w = img.shape[:2]
    ratio = min(max_dim / max(h, w), 1.0)
    if ratio < 1.0:
        new_h, new_w = int(h * ratio), int(w * ratio)
        pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
        img = np.asarray(pil, dtype=np.uint8)

    rng = np.random.default_rng(seed)
    corrupted, _ = apply_profile(img, profile, rng)
    return corrupted
