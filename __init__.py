"""fo-glitch — FiftyOne glitch augmentation plugin.

Registers the ``apply_glitch`` operator (and the ``clean_previews``
utility) for generating realistic camera corruption artifacts as
training-data augmentation.  The operator form is the sole UI surface.
"""

from __future__ import annotations

import fiftyone.operators as foo

from .operators import ApplyGlitch, CleanPreviews


def register(p: foo.PluginContext) -> None:
    """Register all plugin components with FiftyOne."""
    p.register(ApplyGlitch)
    p.register(CleanPreviews)
