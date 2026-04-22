"""Operators for previewing and applying glitch augmentations."""

from __future__ import annotations

import os
from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np

from .config import (
    BLOCK_PATTERNS,
    DELEGATE_THRESHOLD,
    GLITCH_MODES,
    MODE_DESCRIPTIONS,
    MODE_LABELS,
    TARGET_CHOICES,
    VALID_PLACEHOLDERS_HINT,
    _safe_float,
    _safe_int,
    expand_suffix,
    find_unknown_placeholders,
    profile_from_params,
)
from .glitch import apply_profile, generate_preview_image, load_image, save_image


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
            label="Apply Glitch Augmentations",
            description=("Configure and apply glitch augmentation to dataset samples"),
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
        inputs.md("### Corruption Modes")

        for mode_name in GLITCH_MODES:
            enabled = bool(ctx.params.get(f"{mode_name}_enabled", False))

            inputs.bool(
                f"{mode_name}_enabled",
                label=MODE_LABELS[mode_name],
                description=MODE_DESCRIPTIONS[mode_name],
                default=enabled,
                view=types.CheckboxView(space=4, descriptionView="tooltip"),
            )
            inputs.float(
                f"{mode_name}_intensity",
                label="Intensity (%)",
                default=_safe_float(ctx.params.get(f"{mode_name}_intensity"), default=50.0),
                min=0.0,
                max=100.0,
                view=types.FieldView(space=8, read_only=not enabled),
            )

            # Extra tuning for block corruption — size / pattern / layers
            if mode_name == "block_corruption" and enabled:
                inputs.float(
                    "block_size_pct",
                    label="Block size (% of image)",
                    description=(
                        "Block edge length as a percentage of the image's "
                        "shorter side. 2% ~= 20 px on a 1024-wide image."
                    ),
                    default=_safe_float(ctx.params.get("block_size_pct"), default=2.0),
                    min=0.5,
                    max=10.0,
                    view=types.FieldView(space=4, descriptionView="tooltip"),
                )

                pattern_choices = types.Dropdown()
                for p in BLOCK_PATTERNS:
                    pattern_choices.add_choice(p, label=p.capitalize())
                inputs.enum(
                    "block_pattern",
                    pattern_choices.values(),
                    default=ctx.params.get("block_pattern", "uniform"),
                    label="Pattern",
                    view=types.DropdownView(space=4),
                )

                inputs.int(
                    "block_layers",
                    label="Layers",
                    description=(
                        "Number of multi-pass corruption layers (1–4). "
                        "Each layer uses a different block size around "
                        "the configured one; intensity is split across "
                        "layers to keep total density similar."
                    ),
                    default=_safe_int(ctx.params.get("block_layers"), default=1),
                    min=1,
                    max=4,
                    view=types.FieldView(space=4, descriptionView="tooltip"),
                )

        has_enabled = any(ctx.params.get(f"{m}_enabled") for m in GLITCH_MODES)

        # -- Noise injection (always visible, mirrors mode layout) ---
        inputs.md("### Noise Injection")
        noise_enabled = bool(ctx.params.get("noise_enabled", False))
        inputs.bool(
            "noise_enabled",
            label="Enable noise injection",
            description="Jitter each mode's intensity per sample so results vary",
            default=noise_enabled,
            view=types.CheckboxView(space=4, descriptionView="tooltip"),
        )
        inputs.float(
            "noise_scale",
            label="Noise scale (%)",
            default=_safe_float(ctx.params.get("noise_scale"), default=10.0),
            min=0.0,
            max=50.0,
            view=types.FieldView(space=8, read_only=not noise_enabled),
        )

        # -- Inline preview ------------------------------------------
        inputs.md("### Preview")
        inputs.bool(
            "show_preview",
            label="Show preview",
            description="Render a glitched preview of the selected sample",
            default=bool(ctx.params.get("show_preview", False)),
            view=types.SwitchView(descriptionView="tooltip"),
        )

        if ctx.params.get("show_preview") and has_enabled:
            preview_path = self._generate_preview_file(ctx)
            if preview_path is not None:
                # Serve through FiftyOne's /media endpoint with a
                # cache-busting token based on the file's mtime, so
                # updates bypass the browser image cache.
                from urllib.parse import quote

                mtime = int(os.path.getmtime(preview_path) * 1000)
                media_url = (
                    f"/media?filepath={quote(preview_path)}&v={mtime}"
                )
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
                        description="Select a sample in the grid to preview.",
                    ),
                )

        # -- Settings ------------------------------------------------
        inputs.md("### Settings")

        inputs.str(
            "seed",
            label="Seed (blank = random)",
            default=str(ctx.params.get("seed", "") or ""),
        )

        suffix_value = str(
            ctx.params.get("filename_suffix", "_glitch_{TIMESTAMP}")
        )
        inputs.str(
            "filename_suffix",
            label="Filename suffix",
            description=(
                f"Appended to source filename. Valid placeholders: "
                f"{VALID_PLACEHOLDERS_HINT}. Any other "
                f"{{...}} token will be stripped from the output filename."
            ),
            default=suffix_value,
        )

        unknown_tokens = find_unknown_placeholders(suffix_value)
        if unknown_tokens:
            inputs.view(
                "filename_suffix_warning",
                types.Warning(
                    label=(
                        f"Unknown placeholder(s): {', '.join(unknown_tokens)}"
                    ),
                    description=(
                        f"These tokens are not recognized and will be "
                        f"stripped from the output filename. Valid "
                        f"placeholders: {VALID_PLACEHOLDERS_HINT}."
                    ),
                ),
            )

        # -- Augment Samples ----------------------------------------
        inputs.md("### Augment Samples")

        target_choices = types.Dropdown()
        for key, (tlabel, tdesc) in TARGET_CHOICES.items():
            target_choices.add_choice(key, label=tlabel, description=tdesc)

        inputs.enum(
            "target",
            target_choices.values(),
            default=ctx.params.get("target", "current_view"),
            label="Augment Samples:",
            view=target_choices,
        )

        target = ctx.params.get("target", "current_view")

        if target == "random_fraction":
            inputs.float(
                "fraction",
                label="Fraction (0.01–1.00)",
                default=_safe_float(ctx.params.get("fraction"), default=0.1),
                min=0.01,
                max=1.0,
            )

        if target == "saved_view":
            saved_views = ctx.dataset.list_saved_views() if ctx.dataset else []
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

        if target == "samples_with_tag":
            tags = ctx.dataset.distinct("tags") if ctx.dataset else []
            if tags:
                tag_view = types.Dropdown(multiple=True)
                for t in tags:
                    tag_view.add_choice(t, label=t)
                inputs.list(
                    "target_tags",
                    types.String(),
                    default=list(ctx.params.get("target_tags") or []),
                    label="Tags",
                    view=tag_view,
                )
            else:
                inputs.view(
                    "no_tags_warning",
                    types.Warning(
                        label="No sample tags",
                        description="This dataset has no sample tags to match.",
                    ),
                )

        # -- Delegation recommendation -------------------------------
        try:
            target_view = self._resolve_target(ctx)
            count = len(target_view) if target_view is not None else 0
        except Exception:
            count = 0
        if count > DELEGATE_THRESHOLD:
            inputs.view(
                "delegate_recommendation",
                types.Notice(
                    label=f"Large target ({count} samples)",
                    description=(
                        f"Consider **Schedule** rather than **Execute** "
                        f"for targets larger than {DELEGATE_THRESHOLD} samples."
                    ),
                ),
            )

        # -- Validation ----------------------------------------------
        if not has_enabled:
            inputs.view(
                "no_modes_warning",
                types.Warning(
                    label="No modes enabled",
                    description=("Enable at least one corruption mode above."),
                ),
            )

        return types.Property(inputs)

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
                "No samples in target — nothing to do.",
                type="warning",
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
                src_path.parent / f"{src_path.stem}{expanded_suffix}{src_path.suffix}"
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
        mode_str = ", ".join(MODE_LABELS.get(m, m) for m in modes) if modes else "none"

        outputs.md(
            f"**Generated {count} glitched sample(s)**\n\n**Modes applied:** {mode_str}"
        )
        return types.Property(outputs)

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _generate_preview_file(self, ctx) -> str | None:
        """Generate a glitched preview image and write it next to the source.

        Uses a fixed filename that is overwritten on each call.  Never
        added to the dataset.  Returns the file path, or ``None`` if no
        sample is available.
        """
        sample_id = _resolve_sample_id(ctx)
        if sample_id is None:
            return None

        sample: fo.Sample = ctx.dataset[sample_id]
        profile = profile_from_params(ctx.params)
        corrupted = generate_preview_image(sample.filepath, profile, profile.seed)

        preview_path = str(
            Path(sample.filepath).parent / ".fo_glitch_preview.jpg"
        )
        save_image(corrupted, preview_path)
        return preview_path

    def _resolve_target(self, ctx) -> fo.DatasetView | None:
        """Map the user's target selection to a FiftyOne view.

        Returns ``None`` if the target cannot be resolved (e.g. no
        samples selected, no tags chosen, etc.).
        """
        target = ctx.params.get("target", "current_view")
        dataset: fo.Dataset = ctx.dataset
        if dataset is None:
            return None

        if target == "selected_samples":
            ids = list(ctx.selected) if ctx.selected else []
            # Fall back to the modal's current sample if nothing is
            # highlighted in the grid.
            if not ids and getattr(ctx, "current_sample", None):
                ids = [ctx.current_sample]
            return dataset.select(ids) if ids else None

        if target == "samples_with_tag":
            tags = list(ctx.params.get("target_tags") or [])
            if not tags:
                return None
            return dataset.match_tags(tags)

        if target == "current_view":
            return ctx.target_view()

        if target == "random_fraction":
            fraction = _safe_float(ctx.params.get("fraction"), default=0.1)
            fraction = min(max(fraction, 0.01), 1.0)
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
            description=("Remove all temporary glitch preview samples and files"),
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
