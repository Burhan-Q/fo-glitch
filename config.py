"""Glitch profile dataclasses, constants, and storage helpers."""

from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModeConfig:
    """Configuration for a single glitch corruption mode.

    Attributes:
        enabled: Whether this mode is active.
        intensity: Strength of the effect, 0.0 (off) to 100.0 (maximum).
    """

    enabled: bool = False
    intensity: float = 50.0

    @classmethod
    def from_dict(cls, data: dict[str, bool | float]) -> ModeConfig:
        """Create a ModeConfig from a plain dict, using defaults for missing keys."""
        return cls(
            enabled=bool(data.get("enabled", False)),
            intensity=float(data.get("intensity", 50.0)),
        )


@dataclass
class NoiseConfig:
    """Controls per-sample parameter jitter so glitches are not uniform.

    Attributes:
        enabled: Whether noise injection is active.
        scale: Jitter magnitude as a percentage of each mode's intensity
            (0.0 – 50.0).  A scale of 10 means each mode's intensity is
            randomly varied by up to +/-10 % of its configured value.
    """

    enabled: bool = False
    scale: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, bool | float]) -> NoiseConfig:
        """Create a NoiseConfig from a plain dict, using defaults for missing keys."""
        return cls(
            enabled=bool(data.get("enabled", False)),
            scale=float(data.get("scale", 10.0)),
        )


@dataclass
class GlitchProfile:
    """Complete glitch configuration — all modes, noise, seed, and naming.

    Attributes:
        name: Profile name (for future save/export).
        pixel_sort: Brightness-based row pixel sorting.
        row_displacement: Horizontal row offset (scan-line displacement).
        block_corruption: 16x16 macro-block scrambling (H.264 artifact).
        channel_shift: R/G/B plane offset (chroma sub-sampling error).
        frame_tear: Horizontal frame tearing (vsync artifact).
        scanline_noise: Periodic horizontal intensity bands (EMI).
        interlace: Alternating row drop + interpolation.
        noise: Per-sample parameter jitter settings.
        seed: Base seed for reproducibility (``None`` = random).
        filename_suffix: Template appended to source basename.
            Supports placeholders: ``{TIMESTAMP}``, ``{DATETIME}``,
            ``{DATE}``, ``{INDEX}``, ``{PROFILE}``, ``{MODE}``.
    """

    name: str = "untitled"
    pixel_sort: ModeConfig = field(default_factory=ModeConfig)
    row_displacement: ModeConfig = field(default_factory=ModeConfig)
    block_corruption: ModeConfig = field(default_factory=ModeConfig)
    channel_shift: ModeConfig = field(default_factory=ModeConfig)
    frame_tear: ModeConfig = field(default_factory=ModeConfig)
    scanline_noise: ModeConfig = field(default_factory=ModeConfig)
    interlace: ModeConfig = field(default_factory=ModeConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int | None = None
    filename_suffix: str = "_glitch_{TIMESTAMP}"

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict for ExecutionStore persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GlitchProfile:
        """Deserialize from a dict, using defaults for any missing fields.

        Nested ``ModeConfig`` / ``NoiseConfig`` sub-dicts are handled
        automatically.  Unknown keys are silently ignored.
        """
        mode_kwargs: dict[str, ModeConfig] = {}
        for mode_name in GLITCH_MODES:
            raw = data.get(mode_name)
            if isinstance(raw, dict):
                mode_kwargs[mode_name] = ModeConfig.from_dict(raw)

        noise_raw = data.get("noise")
        noise = NoiseConfig.from_dict(noise_raw) if isinstance(noise_raw, dict) else NoiseConfig()

        seed_raw = data.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None

        return cls(
            name=str(data.get("name", "untitled")),
            **mode_kwargs,
            noise=noise,
            seed=seed,
            filename_suffix=str(data.get("filename_suffix", "_glitch_{TIMESTAMP}")),
        )

    # -- convenience properties ----------------------------------------------

    @property
    def enabled_modes(self) -> list[str]:
        """Return names of modes that are currently enabled."""
        return [m for m in GLITCH_MODES if getattr(self, m).enabled]

    @property
    def mode_configs(self) -> dict[str, ModeConfig]:
        """Return ``{mode_name: ModeConfig}`` for all 7 modes."""
        return {m: getattr(self, m) for m in GLITCH_MODES}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLITCH_MODES: list[str] = [
    "pixel_sort",
    "row_displacement",
    "block_corruption",
    "channel_shift",
    "frame_tear",
    "scanline_noise",
    "interlace",
]
"""Ordered list of mode field names on :class:`GlitchProfile`."""

MODE_LABELS: dict[str, str] = {
    "pixel_sort": "Pixel Sort (Brightness)",
    "row_displacement": "Row Displacement",
    "block_corruption": "Block Corruption (Macro-block)",
    "channel_shift": "Channel Shift (RGB Offset)",
    "frame_tear": "Frame Tear",
    "scanline_noise": "Scan Line Noise",
    "interlace": "Interlacing",
}
"""Human-readable labels for the panel UI."""

MODE_DESCRIPTIONS: dict[str, str] = {
    "pixel_sort": "Sort pixels within rows by luminance between thresholds",
    "row_displacement": "Shift rows horizontally by random offsets",
    "block_corruption": "Scramble 16x16 pixel blocks to wrong positions",
    "channel_shift": "Offset R/G/B planes by different pixel amounts",
    "frame_tear": "Composite image halves at different horizontal offsets",
    "scanline_noise": "Add periodic horizontal intensity bands",
    "interlace": "Drop alternating rows and interpolate",
}
"""Tooltip descriptions shown next to each mode in the UI."""


# ---------------------------------------------------------------------------
# Filename suffix expansion
# ---------------------------------------------------------------------------

_UNSAFE_FILENAME_RE = re.compile(r"[^\w\-.,]")


def _sanitize_for_filename(value: str) -> str:
    """Replace characters that are unsafe in filenames with underscores."""
    return _UNSAFE_FILENAME_RE.sub("_", value)


def expand_suffix(template: str, profile: GlitchProfile, index: int) -> str:
    """Expand placeholder tokens in a filename suffix template.

    Args:
        template: Suffix string with ``{PLACEHOLDER}`` tokens.
        profile: The glitch profile being applied (for name / mode info).
        index: Zero-based index of this sample within the current batch.

    Returns:
        The expanded suffix string with all placeholders resolved.
    """
    now = datetime.now(tz=timezone.utc)
    enabled = profile.enabled_modes

    replacements: dict[str, str] = {
        "TIMESTAMP": str(int(time.time())),
        "DATETIME": now.strftime("%Y-%m-%dT%H-%M-%S"),
        "DATE": now.strftime("%Y-%m-%d"),
        "INDEX": f"{index:03d}",
        "PROFILE": _sanitize_for_filename(profile.name),
        "MODE": _sanitize_for_filename(",".join(enabled)) if enabled else "none",
    }

    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)
    return result


# ---------------------------------------------------------------------------
# ExecutionStore helpers
# ---------------------------------------------------------------------------

_STORE_VERSION = "v1"


def _store_key(dataset_id: str) -> str:
    """Build a dataset-scoped store key for profile persistence."""
    return f"glitch_panel_{dataset_id}_{_STORE_VERSION}"


def get_dataset_profile(ctx: object) -> GlitchProfile:
    """Load the current glitch profile from dataset-scoped ExecutionStore.

    Returns a default :class:`GlitchProfile` if nothing is stored yet.
    """
    try:
        data = ctx.store(_store_key(ctx.dataset._doc.id)).get("profile")  # type: ignore[attr-defined]
        if isinstance(data, dict):
            return GlitchProfile.from_dict(data)
    except Exception:
        pass
    return GlitchProfile()


def save_dataset_profile(ctx: object, profile: GlitchProfile) -> None:
    """Persist a :class:`GlitchProfile` to the dataset-scoped ExecutionStore."""
    try:
        ctx.store(_store_key(ctx.dataset._doc.id)).set("profile", profile.to_dict())  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Flat-params ↔ GlitchProfile conversion (for operator forms)
# ---------------------------------------------------------------------------


def profile_from_params(params: dict[str, object]) -> GlitchProfile:
    """Build a GlitchProfile from the flat key namespace used by operator forms.

    The operator form stores each mode as ``{mode}_enabled`` and
    ``{mode}_intensity`` at the top level of *params*.  Noise, seed, and
    suffix live under their own top-level keys.

    Falls back to ``GlitchProfile.from_dict(params.get("profile"))`` if
    a nested profile dict is present (backwards-compatible with the panel
    trigger path).
    """
    # If a nested profile dict was passed (panel trigger), prefer it
    nested = params.get("profile")
    if isinstance(nested, dict) and any(m in nested for m in GLITCH_MODES):
        return GlitchProfile.from_dict(nested)

    modes: dict[str, ModeConfig] = {}
    for mode_name in GLITCH_MODES:
        modes[mode_name] = ModeConfig(
            enabled=bool(params.get(f"{mode_name}_enabled", False)),
            intensity=float(params.get(f"{mode_name}_intensity", 50.0)),
        )

    seed_raw = params.get("seed")
    if isinstance(seed_raw, str):
        seed_raw = seed_raw.strip()
        seed_raw = int(seed_raw) if seed_raw else None
    elif isinstance(seed_raw, (int, float)):
        seed_raw = int(seed_raw)
    else:
        seed_raw = None

    return GlitchProfile(
        **modes,
        noise=NoiseConfig(
            enabled=bool(params.get("noise_enabled", False)),
            scale=float(params.get("noise_scale", 10.0)),
        ),
        seed=seed_raw,
        filename_suffix=str(params.get("filename_suffix", "_glitch_{TIMESTAMP}")),
    )
