"""Measurement-only instrumentation. Nothing here influences tracking."""

from mcreid.diagnostics.shadow import ShadowProbe, ShadowRow, summarise

__all__ = ["ShadowProbe", "ShadowRow", "summarise"]
