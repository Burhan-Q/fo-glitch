"""Core corruption engine — pure numpy image manipulation functions.

Every corruption function has the signature::

    fn(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray

where *img* is (H, W, 3) uint8 RGB, *intensity* is 0–100, and *rng* is a
per-sample random generator for deterministic reproducibility.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .config import GLITCH_MODES, GlitchProfile, ModeConfig

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
    spans: list[tuple[int, int]] = []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        spans.append((int(s), int(e)))
    return spans


def row_displacement(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Shift each row horizontally by a random offset.

    Simulates scan-line displacement / signal timing errors.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Maps to the maximum horizontal pixel offset.
            At 100 the max offset is ~10 % of image width.
        rng: Random generator for per-row offsets.

    Returns:
        New image with shifted rows (edges filled with black).
    """
    out = img.copy()
    h, w, _ = img.shape
    max_offset = int(w * (intensity / 100.0) * 0.1)
    if max_offset < 1:
        return out

    offsets = rng.integers(-max_offset, max_offset + 1, size=h)
    for y in range(h):
        off = int(offsets[y])
        if off == 0:
            continue
        out[y] = np.roll(img[y], off, axis=0)
        # Fill wrapped pixels with black
        if off > 0:
            out[y, :off] = 0
        else:
            out[y, off:] = 0

    return out


def block_corruption(
    img: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Scramble, freeze, or zero 16x16 pixel blocks at random positions.

    Simulates H.264/H.265 macro-block corruption from lost I-frames.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls the probability that each block is
            corrupted.  At 100 roughly half of all blocks are affected.
        rng: Random generator for block selection and corruption type.

    Returns:
        New image with corrupted blocks.
    """
    out = img.copy()
    h, w, _ = img.shape
    block = 16
    prob = (intensity / 100.0) * 0.5  # max 50 % of blocks

    rows = h // block
    cols = w // block

    for br in range(rows):
        for bc in range(cols):
            if rng.random() > prob:
                continue
            y0, y1 = br * block, (br + 1) * block
            x0, x1 = bc * block, (bc + 1) * block

            action = rng.integers(0, 3)
            if action == 0:
                # Copy from a random other block position
                src_r = rng.integers(0, rows)
                src_c = rng.integers(0, cols)
                sy0, sy1 = src_r * block, (src_r + 1) * block
                sx0, sx1 = src_c * block, (src_c + 1) * block
                out[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
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
                    out[y0:y1, x0 : x0 + dst_w] = img[y0:y1, src_x0:src_x1]

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
        plane = img[:, :, ch]
        shifted = np.roll(np.roll(plane, dx, axis=1), dy, axis=0)
        out[:, :, ch] = shifted

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
    """Drop alternating rows and interpolate from neighbours.

    Simulates interlaced / field-based capture artefacts.

    Args:
        img: Input image array, shape (H, W, 3), dtype uint8.
        intensity: 0–100.  Controls the blend factor between original and
            interpolated rows.  At 0 no change; at 100 dropped rows are
            fully replaced by neighbour averages.
        rng: Random generator for field selection (odd / even).

    Returns:
        New image with interlacing artefacts.
    """
    out = img.copy().astype(np.float32)
    h, w, _ = img.shape
    blend = intensity / 100.0

    # Choose which field to drop (odd or even rows)
    drop_odd = bool(rng.integers(0, 2))
    start = 1 if drop_odd else 0

    for y in range(start, h, 2):
        # Average of the row above and below
        above = max(0, y - 1)
        below = min(h - 1, y + 1)
        interpolated = (
            img[above].astype(np.float32) + img[below].astype(np.float32)
        ) / 2.0
        out[y] = img[y].astype(np.float32) * (1 - blend) + interpolated * blend

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

GlitchFunction = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]

GLITCH_FUNCTIONS: dict[str, GlitchFunction] = {
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
        mode_cfg: ModeConfig = profile.mode_configs[mode_name]
        if not mode_cfg.enabled:
            continue

        intensity = mode_cfg.intensity

        # Noise injection — jitter the intensity per sample
        if profile.noise.enabled:
            scale = profile.noise.scale / 100.0
            jitter = rng.uniform(-scale, scale) * intensity
            intensity = float(np.clip(intensity + jitter, 0.0, 100.0))

        applied_params[mode_name] = round(intensity, 4)
        img = GLITCH_FUNCTIONS[mode_name](img, intensity, rng)

    return img, applied_params


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------


def generate_preview_base64(
    filepath: str | Path,
    profile: GlitchProfile,
    seed: int | None,
    max_dim: int = 512,
    quality: int = 70,
) -> str:
    """Generate a base64-encoded JPEG preview of a glitched image.

    The source image is downscaled to *max_dim* pixels on its longest
    side before corruption is applied, keeping preview generation fast
    and the resulting base64 payload small (~30–60 KB).

    Args:
        filepath: Path to the source image on disk.
        profile: The glitch profile to apply.
        seed: Base seed for the RNG (``None`` for non-deterministic).
        max_dim: Maximum pixel dimension (width or height) for the
            preview thumbnail.
        quality: JPEG quality for the encoded output (1–95).

    Returns:
        A complete ``data:image/jpeg;base64,...`` URI string suitable
        for embedding in markdown or HTML ``<img>`` tags.
    """
    img = load_image(filepath)

    # Downscale for speed and payload size
    h, w = img.shape[:2]
    ratio = min(max_dim / max(h, w), 1.0)
    if ratio < 1.0:
        new_h, new_w = int(h * ratio), int(w * ratio)
        pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
        img = np.asarray(pil, dtype=np.uint8)

    rng = np.random.default_rng(seed if seed is not None else None)
    corrupted, _ = apply_profile(img, profile, rng)

    buf = io.BytesIO()
    Image.fromarray(corrupted).save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
