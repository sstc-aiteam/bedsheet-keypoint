"""
Heatmap decoder backends.

Backends are selected via a single "decoder spec" dict/string, which carries
the backend kind plus any metadata (e.g., gcnn mode: c4 vs so2).
"""

from .types import DecoderSpec, normalize_decoder_spec

__all__ = [
    "DecoderSpec",
    "normalize_decoder_spec",
]


