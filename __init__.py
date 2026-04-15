"""fo-glitch — FiftyOne glitch augmentation plugin.

Registers the Glitch Configurator panel and associated operators for
generating realistic camera corruption artifacts as training data
augmentation.
"""

from __future__ import annotations

import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.operators.panel import Panel, PanelConfig

from .config import (
    GLITCH_MODES,
    MODE_LABELS,
    GlitchProfile,
    ModeConfig,
    NoiseConfig,
    get_dataset_profile,
    save_dataset_profile,
)
from .glitch import generate_preview_base64


class GlitchPanel(Panel):
    """Panel for configuring, previewing, and applying glitch effects.

    The preview is generated as a base64 data URI stored in transient
    panel state.  It is never written to disk or added to the dataset.
    Closing the panel or clicking Apply discards the preview.
    """

    @property
    def config(self) -> PanelConfig:
        """Panel metadata."""
        return PanelConfig(
            name="glitch_panel",
            label="Glitch Configurator",
            icon="blur_on",
            surfaces="grid modal",
            help_markdown=(
                "Configure glitch / corruption effects and apply them to "
                "dataset samples.  Use **Preview** to see the effect on the "
                "current sample, then **Apply Glitch** to generate new "
                "augmented samples."
            ),
        )

    # -- lifecycle -----------------------------------------------------------

    def on_load(self, ctx) -> None:
        """Restore persisted profile when the panel opens."""
        profile = get_dataset_profile(ctx)
        self._set_profile_state(ctx, profile)
        ctx.panel.set_state("preview_data_uri", None)

    def on_unload(self, ctx) -> None:
        """Discard ephemeral preview when the panel closes."""
        ctx.panel.set_state("preview_data_uri", None)

    def on_change_dataset(self, ctx) -> None:
        """Reload profile and clear preview when the dataset changes."""
        profile = get_dataset_profile(ctx)
        self._set_profile_state(ctx, profile)
        ctx.panel.set_state("preview_data_uri", None)

    # -- state helpers -------------------------------------------------------

    def _set_profile_state(self, ctx, profile: GlitchProfile) -> None:
        """Push a GlitchProfile into panel state for the render cycle."""
        for mode_name in GLITCH_MODES:
            mc: ModeConfig = getattr(profile, mode_name)
            ctx.panel.set_state(f"{mode_name}_enabled", mc.enabled)
            ctx.panel.set_state(f"{mode_name}_intensity", mc.intensity)
        ctx.panel.set_state("noise_enabled", profile.noise.enabled)
        ctx.panel.set_state("noise_scale", profile.noise.scale)
        ctx.panel.set_state("seed", profile.seed)
        ctx.panel.set_state("filename_suffix", profile.filename_suffix)

    def _read_profile(self, ctx) -> GlitchProfile:
        """Reconstruct a GlitchProfile from current panel state."""
        modes: dict[str, ModeConfig] = {}
        for mode_name in GLITCH_MODES:
            modes[mode_name] = ModeConfig(
                enabled=bool(ctx.panel.get_state(f"{mode_name}_enabled", False)),
                intensity=float(ctx.panel.get_state(f"{mode_name}_intensity", 50.0)),
            )
        seed_raw = ctx.panel.get_state("seed")
        seed = int(seed_raw) if seed_raw is not None and seed_raw != "" else None

        return GlitchProfile(
            **modes,
            noise=NoiseConfig(
                enabled=bool(ctx.panel.get_state("noise_enabled", False)),
                scale=float(ctx.panel.get_state("noise_scale", 10.0)),
            ),
            seed=seed,
            filename_suffix=str(
                ctx.panel.get_state("filename_suffix", "_glitch_{TIMESTAMP}")
            ),
        )

    def _persist(self, ctx) -> None:
        """Save the current panel state to the dataset ExecutionStore."""
        save_dataset_profile(ctx, self._read_profile(ctx))

    # -- change handlers -----------------------------------------------------

    def on_mode_toggle(self, ctx) -> None:
        """Handle a mode enabled/disabled toggle change."""
        self._persist(ctx)
        # Clear stale preview when config changes
        ctx.panel.set_state("preview_data_uri", None)

    def on_mode_intensity(self, ctx) -> None:
        """Handle a mode intensity value change."""
        self._persist(ctx)
        ctx.panel.set_state("preview_data_uri", None)

    def on_noise_toggle(self, ctx) -> None:
        """Handle the noise-enabled toggle change."""
        self._persist(ctx)

    def on_noise_scale(self, ctx) -> None:
        """Handle the noise scale change."""
        self._persist(ctx)

    def on_seed_change(self, ctx) -> None:
        """Handle seed text field change."""
        self._persist(ctx)
        ctx.panel.set_state("preview_data_uri", None)

    def on_suffix_change(self, ctx) -> None:
        """Handle filename suffix template change."""
        self._persist(ctx)

    # -- action handlers -----------------------------------------------------

    def on_preview(self, ctx) -> None:
        """Generate an ephemeral base64 preview and store it in panel state.

        The preview is never written to disk or added to the dataset.
        It exists only in transient panel state and is discarded when the
        panel closes, the dataset changes, or Apply is clicked.
        """
        sample_id = self._resolve_sample_id(ctx)
        if sample_id is None:
            ctx.panel.set_state("preview_data_uri", None)
            ctx.ops.notify(
                "No sample selected — select a sample first.",
                type="warning",
            )
            return

        sample = ctx.dataset[sample_id]
        profile = self._read_profile(ctx)

        data_uri = generate_preview_base64(
            sample.filepath,
            profile,
            seed=profile.seed,
        )
        ctx.panel.set_state("preview_data_uri", data_uri)

    def on_apply(self, ctx) -> None:
        """Clear the preview, then trigger the apply operator."""
        ctx.panel.set_state("preview_data_uri", None)

        # Build flat params for the operator
        params: dict[str, object] = {}
        for mode_name in GLITCH_MODES:
            params[f"{mode_name}_enabled"] = ctx.panel.get_state(
                f"{mode_name}_enabled", False,
            )
            params[f"{mode_name}_intensity"] = ctx.panel.get_state(
                f"{mode_name}_intensity", 50.0,
            )
        params["noise_enabled"] = ctx.panel.get_state("noise_enabled", False)
        params["noise_scale"] = ctx.panel.get_state("noise_scale", 10.0)
        params["seed"] = ctx.panel.get_state("seed")
        params["filename_suffix"] = ctx.panel.get_state(
            "filename_suffix", "_glitch_{TIMESTAMP}",
        )

        ctx.trigger(
            "@Burhan-Q/fo-glitch/apply_glitch",
            params=params,
        )

    # -- render --------------------------------------------------------------

    def render(self, ctx) -> types.Property:
        """Build the panel UI layout."""
        panel = types.Object()

        panel.str(
            "header",
            view=types.Header(),
            default="Glitch Configurator",
        )

        # -- Corruption modes ------------------------------------------------
        for mode_name in GLITCH_MODES:
            label = MODE_LABELS[mode_name]
            enabled = ctx.panel.get_state(f"{mode_name}_enabled", False)

            panel.bool(
                f"{mode_name}_enabled",
                label=label,
                default=enabled,
                on_change=self.on_mode_toggle,
                view=types.CheckboxView(space=4),
            )
            panel.float(
                f"{mode_name}_intensity",
                label="Intensity (%)",
                default=ctx.panel.get_state(f"{mode_name}_intensity", 50.0),
                min=0.0,
                max=100.0,
                on_change=self.on_mode_intensity,
                view=types.FieldView(space=8, read_only=not enabled),
            )

        # -- Preview ---------------------------------------------------------
        panel.str(
            "preview_header",
            view=types.Header(),
            default="Preview",
        )

        has_enabled = any(
            ctx.panel.get_state(f"{m}_enabled", False) for m in GLITCH_MODES
        )

        panel.btn(
            "preview_btn",
            label="Preview Current Sample",
            icon="visibility",
            on_click=self.on_preview,
            disabled=not has_enabled,
        )

        preview_uri = ctx.panel.get_state("preview_data_uri")
        if preview_uri:
            panel.str(
                "preview_image",
                view=types.MarkdownView(),
                default=f"![Glitch preview]({preview_uri})",
            )

        # -- Settings --------------------------------------------------------
        panel.str(
            "settings_header",
            view=types.Header(),
            default="Settings",
        )

        panel.bool(
            "noise_enabled",
            label="Enable noise injection",
            description="Jitter each mode's intensity per sample so results vary",
            default=ctx.panel.get_state("noise_enabled", False),
            on_change=self.on_noise_toggle,
        )

        if ctx.panel.get_state("noise_enabled", False):
            panel.float(
                "noise_scale",
                label="Noise scale (%)",
                default=ctx.panel.get_state("noise_scale", 10.0),
                min=0.0,
                max=50.0,
                on_change=self.on_noise_scale,
            )

        panel.str(
            "seed",
            label="Seed (blank = random)",
            default=str(ctx.panel.get_state("seed", "") or ""),
            on_change=self.on_seed_change,
        )

        panel.str(
            "filename_suffix",
            label="Filename suffix",
            description=(
                "Appended to source filename. Placeholders: "
                "{TIMESTAMP}, {DATETIME}, {DATE}, {INDEX}, {PROFILE}, {MODE}"
            ),
            default=ctx.panel.get_state("filename_suffix", "_glitch_{TIMESTAMP}"),
            on_change=self.on_suffix_change,
        )

        # -- Apply -----------------------------------------------------------
        panel.str(
            "apply_header",
            view=types.Header(),
            default="Apply",
        )

        panel.btn(
            "apply_btn",
            label="Apply Glitch...",
            icon="auto_fix_high",
            on_click=self.on_apply,
            disabled=not has_enabled,
        )

        return types.Property(panel)

    # -- internal helpers ----------------------------------------------------

    def _resolve_sample_id(self, ctx) -> str | None:
        """Return the ID of the sample to preview.

        Prefers the first selected sample, then the first sample in the
        current view.
        """
        if ctx.selected:
            return ctx.selected[0]
        view = ctx.view
        if view and len(view) > 0:
            return view.first().id
        return None


def register(p: foo.PluginContext) -> None:
    """Register all plugin components with FiftyOne."""
    from .operators import ApplyGlitch, CleanPreviews

    p.register(GlitchPanel)
    p.register(ApplyGlitch)
    p.register(CleanPreviews)
