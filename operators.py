"""Operators for previewing and applying glitch augmentations."""

from __future__ import annotations

import os
from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np

from .config import (
    GLITCH_MODES,
    MODE_LABELS,
    GlitchProfile,
    expand_suffix,
    profile_from_params,
)
from .glitch import apply_profile, load_image, save_image


# ---------------------------------------------------------------------------
# Apply operator (unified: configure + inline preview + apply)
# ---------------------------------------------------------------------------


class ApplyGlitch(foo.Operator):
    """Configure and apply glitch augmentation to dataset samples.

    A single self-contained operator form with inline preview.  The
    preview renders as a base64 image directly inside the dialog — no
    view changes, no dataset modifications, fully ephemeral.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        """Operator metadata."""
        return foo.OperatorConfig(
            name="apply_glitch",
            label="Apply Glitch",
            description=(
                "Configure and apply glitch augmentation to dataset samples"
            ),
            icon="auto_fix_high",
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    # ---------------------------------------------------------------
    # Input form
    # ---------------------------------------------------------------

    def resolve_input(self, ctx) -> types.Property:
        """Build the operator input form.

        Each enabled mode shows a checkbox + a number input for
        intensity (0.00–100.00 %).  Disabled modes show only the
        checkbox.  No sliders — plain number fields are compact and
        give exact precision control.
        """
        inputs = types.Object()

        # -- Corruption modes ----------------------------------------
        # Checkbox at space=6, intensity at space=6 → same row.
        # Disabled modes: checkbox only at space=6, so two fit per row.
        inputs.md("### Corruption Modes")

        for mode_name in GLITCH_MODES:
            enabled = bool(ctx.params.get(f"{mode_name}_enabled", False))

            # Every mode is always: checkbox(4) + intensity(8) = 12.
            # The layout never changes regardless of enabled state,
            # so the grid flow is always predictable.
            inputs.bool(
                f"{mode_name}_enabled",
                label=MODE_LABELS[mode_name],
                default=enabled,
                view=types.CheckboxView(space=4),
            )
            inputs.float(
                f"{mode_name}_intensity",
                label="Intensity (%)",
                default=float(
                    ctx.params.get(f"{mode_name}_intensity", 50.0)
                ),
                min=0.0,
                max=100.0,
                view=types.FieldView(space=8, read_only=not enabled),
            )

        # -- Inline preview ------------------------------------------
        has_enabled = any(
            ctx.params.get(f"{m}_enabled") for m in GLITCH_MODES
        )

        inputs.md("### Preview")
        inputs.bool(
            "show_preview",
            label="Show preview",
            description=(
                "Render a glitched preview of the selected sample"
            ),
            default=bool(ctx.params.get("show_preview", False)),
        )

        if ctx.params.get("show_preview") and has_enabled:
            preview_path = self._generate_preview_file(ctx)
            if preview_path is not None:
                # Serve through FiftyOne's /media endpoint
                from urllib.parse import quote

                media_url = f"/media?filepath={quote(preview_path)}"
                inputs.define_property(
                    "preview_image",
                    types.String(),
                    default=media_url,
                    view=types.ImageView(height=300),
                )
            else:
                inputs.view(
                    "no_sample_warning",
                    types.Warning(
                        label="No sample available",
                        description=(
                            "Select a sample in the grid to preview."
                        ),
                    ),
                )

        # -- Settings ------------------------------------------------
        inputs.md("### Settings")

        inputs.bool(
            "noise_enabled",
            label="Enable noise injection",
            description=(
                "Jitter each mode's intensity per sample so results vary"
            ),
            default=bool(ctx.params.get("noise_enabled", False)),
        )
        if ctx.params.get("noise_enabled"):
            inputs.float(
                "noise_scale",
                label="Noise scale (%)",
                default=float(ctx.params.get("noise_scale", 10.0)),
                min=0.0,
                max=50.0,
            )

        inputs.str(
            "seed",
            label="Seed (blank = random)",
            default=str(ctx.params.get("seed", "") or ""),
        )

        inputs.str(
            "filename_suffix",
            label="Filename suffix",
            description=(
                "Appended to source filename. Placeholders: "
                "{TIMESTAMP}, {DATETIME}, {DATE}, {INDEX}, {PROFILE}, {MODE}"
            ),
            default=str(
                ctx.params.get("filename_suffix", "_glitch_{TIMESTAMP}")
            ),
        )

        # -- Target --------------------------------------------------
        inputs.md("### Target")

        target_choices = types.Dropdown()
        target_choices.add_choice("current_sample", label="Current sample")
        target_choices.add_choice("current_view", label="Current view")
        target_choices.add_choice(
            "random_fraction",
            label="Random fraction of dataset",
        )
        target_choices.add_choice("saved_view", label="Saved view")
        target_choices.add_choice("entire_dataset", label="Entire dataset")

        inputs.enum(
            "target",
            target_choices.values(),
            default=ctx.params.get("target", "current_view"),
            label="Target",
            view=target_choices,
        )

        target = ctx.params.get("target", "current_view")

        if target == "random_fraction":
            inputs.float(
                "fraction",
                label="Fraction (0.01–1.00)",
                default=float(ctx.params.get("fraction", 0.1)),
                min=0.01,
                max=1.0,
            )

        if target == "saved_view":
            saved_views = (
                ctx.dataset.list_saved_views() if ctx.dataset else []
            )
            if saved_views:
                view_choices = types.Dropdown()
                for sv in saved_views:
                    view_choices.add_choice(sv, label=sv)
                inputs.enum(
                    "saved_view_name",
                    view_choices.values(),
                    label="Saved view",
                    view=view_choices,
                )
            else:
                inputs.view(
                    "no_views_warning",
                    types.Warning(
                        label="No saved views",
                        description="This dataset has no saved views.",
                    ),
                )

        # -- Validation / delegation ---------------------------------
        if not has_enabled:
            inputs.view(
                "no_modes_warning",
                types.Warning(
                    label="No modes enabled",
                    description=(
                        "Enable at least one corruption mode above."
                    ),
                ),
            )

        inputs.bool(
            "delegate",
            default=False,
            label="Delegate execution?",
            description=(
                "Run in the background (recommended for large targets)"
            ),
            view=types.CheckboxView(),
        )

        return types.Property(inputs)

    # ---------------------------------------------------------------
    # Delegation
    # ---------------------------------------------------------------

    def resolve_delegation(self, ctx) -> bool:
        """Auto-delegate when the target is large."""
        if ctx.params.get("delegate"):
            return True
        target_view = self._resolve_target(ctx)
        if target_view is not None and len(target_view) > 100:
            return True
        return False

    # ---------------------------------------------------------------
    # Execute
    # ---------------------------------------------------------------

    def execute(self, ctx):
        """Apply glitch augmentation to every sample in the target.

        Yields progress updates and creates new samples with corrupted
        images saved alongside the originals.
        """
        profile = profile_from_params(ctx.params)
        suffix_template = ctx.params.get(
            "filename_suffix",
            profile.filename_suffix,
        )
        dataset: fo.Dataset = ctx.dataset

        target_view = self._resolve_target(ctx)
        if target_view is None or len(target_view) == 0:
            ctx.ops.notify(
                "No samples in target — nothing to do.", type="warning",
            )
            yield {"status": "empty_target", "count": 0}
            return

        total = len(target_view)
        new_samples: list[fo.Sample] = []

        for i, sample in enumerate(target_view.iter_samples()):
            rng = _make_rng(profile.seed, index=i)

            img = load_image(sample.filepath)
            corrupted, applied_params = apply_profile(img, profile, rng)

            src_path = Path(sample.filepath)
            expanded_suffix = expand_suffix(suffix_template, profile, i)
            out_path = str(
                src_path.parent
                / f"{src_path.stem}{expanded_suffix}{src_path.suffix}"
            )
            save_image(corrupted, out_path)

            new_sample = fo.Sample(filepath=out_path)
            for tag in sample.tags:
                if tag != "_glitch_preview":
                    new_sample.tags.append(tag)
            new_sample.tags.append("glitched")
            new_sample["glitch_source_id"] = sample.id
            new_sample["glitch_profile"] = profile.to_dict()
            new_sample["glitch_applied_params"] = applied_params
            if profile.seed is not None:
                new_sample["glitch_seed"] = profile.seed + i
            new_samples.append(new_sample)

            yield ctx.trigger(
                "set_progress",
                dict(
                    progress=(i + 1) / total,
                    label=f"Glitching {i + 1}/{total}",
                ),
            )

        dataset.add_samples(new_samples)
        ctx.ops.notify(f"Created {len(new_samples)} glitched samples")
        ctx.ops.reload_dataset()

        yield {
            "status": "ok",
            "count": len(new_samples),
            "modes": profile.enabled_modes,
        }

    def resolve_output(self, ctx) -> types.Property:
        """Display a summary of what was generated."""
        outputs = types.Object()
        result = ctx.results or {}

        count = result.get("count", 0)
        modes = result.get("modes", [])
        mode_str = (
            ", ".join(MODE_LABELS.get(m, m) for m in modes)
            if modes
            else "none"
        )

        outputs.md(
            f"**Generated {count} glitched sample(s)**\n\n"
            f"**Modes applied:** {mode_str}"
        )
        return types.Property(outputs)

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _generate_preview_file(self, ctx) -> str | None:
        """Generate a glitched preview image and write it to a temp file.

        The file is written to ``/tmp`` and overwritten on each call.
        It is never added to the dataset.  Returns the file path, or
        ``None`` if no sample is available.
        """
        sample_id = _resolve_sample_id(ctx)
        if sample_id is None:
            return None

        sample: fo.Sample = ctx.dataset[sample_id]
        profile = profile_from_params(ctx.params)

        img = load_image(sample.filepath)

        # Downscale for speed
        from PIL import Image as PILImage

        h, w = img.shape[:2]
        max_dim = 512
        ratio = min(max_dim / max(h, w), 1.0)
        if ratio < 1.0:
            new_h, new_w = int(h * ratio), int(w * ratio)
            pil = PILImage.fromarray(img).resize(
                (new_w, new_h), PILImage.LANCZOS,
            )
            img = np.asarray(pil, dtype=np.uint8)

        rng = np.random.default_rng(
            profile.seed if profile.seed is not None else None,
        )
        corrupted, _ = apply_profile(img, profile, rng)

        # Write next to the source file so FiftyOne's media server can
        # serve it.  Uses a fixed name that gets overwritten each time.
        src_dir = str(Path(sample.filepath).parent)
        preview_path = str(Path(src_dir) / ".fo_glitch_preview.jpg")
        save_image(corrupted, preview_path)
        return preview_path

    def _resolve_target(self, ctx) -> fo.DatasetView | None:
        """Map the user's target selection to a FiftyOne view.

        Returns ``None`` if the target cannot be resolved.
        """
        target = ctx.params.get("target", "current_view")
        dataset: fo.Dataset = ctx.dataset

        if target == "current_sample":
            selected = ctx.selected
            if not selected:
                return None
            return dataset.select(selected[:1])

        if target == "current_view":
            return ctx.target_view()

        if target == "random_fraction":
            fraction = float(ctx.params.get("fraction", 0.1))
            count = max(1, int(len(dataset) * fraction))
            return dataset.take(count)

        if target == "saved_view":
            view_name = ctx.params.get("saved_view_name")
            if not view_name:
                return None
            return dataset.load_saved_view(view_name)

        if target == "entire_dataset":
            return dataset.view()

        return None


# ---------------------------------------------------------------------------
# Cleanup operator
# ---------------------------------------------------------------------------


class CleanPreviews(foo.Operator):
    """Delete all preview samples and their files from disk.

    Utility operator for cleaning up ``_glitch_preview`` samples that
    may remain from earlier versions of the plugin.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        """Operator metadata — unlisted utility operator."""
        return foo.OperatorConfig(
            name="clean_previews",
            label="Clean Glitch Previews",
            description=(
                "Remove all temporary glitch preview samples and files"
            ),
            unlisted=True,
        )

    def execute(self, ctx) -> dict[str, int]:
        """Delete preview samples and their image files."""
        dataset: fo.Dataset = ctx.dataset
        preview_view = dataset.match_tags("_glitch_preview")
        count = len(preview_view)
        if count == 0:
            return {"deleted": 0}

        filepaths: list[str] = preview_view.values("filepath")
        dataset.delete_samples(preview_view)
        for fp in filepaths:
            try:
                os.remove(fp)
            except OSError:
                pass

        ctx.ops.reload_dataset()
        return {"deleted": count}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_sample_id(ctx) -> str | None:
    """Return the ID of a sample for preview.

    Prefers the first selected sample, then falls back to the first
    sample in the current view.
    """
    if ctx.selected:
        return ctx.selected[0]
    view = ctx.view
    if view and len(view) > 0:
        return view.first().id
    return None


def _make_rng(seed: int | None, index: int) -> np.random.Generator:
    """Create a per-sample random generator.

    Args:
        seed: Base seed from the profile (``None`` for non-deterministic).
        index: Sample index within the current batch.

    Returns:
        A numpy ``Generator`` instance.
    """
    if seed is not None:
        return np.random.default_rng(seed + index)
    return np.random.default_rng()
