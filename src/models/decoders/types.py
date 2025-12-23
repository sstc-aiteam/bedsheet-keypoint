from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Union

DecoderKind = Literal["standard", "gcnn"]


@dataclass(frozen=True)
class DecoderSpec:
    """
    Normalized decoder specification for ClipHeatmapHead.

    This is derived from user-friendly inputs (string/dict) and always contains:
    - kind: "standard" | "gcnn"
    - params: backend-specific parameters
    """

    kind: DecoderKind
    params: Dict[str, Any]


def normalize_decoder_spec(
    spec: Optional[Union[str, Mapping[str, Any]]],
    *,
    legacy_use_gcnn: bool,
    legacy_gcnn_hidden: int,
    legacy_gcnn_mode: str,
    legacy_so2_num_angles: int,
    legacy_so2_num_gconvs: int,
) -> DecoderSpec:
    """
    Normalize a decoder spec.

    Accepted inputs:
    - None: use legacy flags (use_gcnn + gcnn settings)
    - "standard" | "gcnn"
    - dict-like: must include {"kind": ...} plus backend parameters.
      For GCNN: include at least {"kind":"gcnn","mode":"c4"|"so2"} and optional params.
    """
    if spec is None:
        if legacy_use_gcnn:
            mode = str(legacy_gcnn_mode).lower()
            params: Dict[str, Any] = {
                "mode": mode,
                "hidden": int(legacy_gcnn_hidden),
            }
            if mode == "so2":
                params["so2_num_angles"] = int(legacy_so2_num_angles)
                params["so2_num_gconvs"] = int(legacy_so2_num_gconvs)
            return DecoderSpec(kind="gcnn", params=params)
        return DecoderSpec(kind="standard", params={})

    if isinstance(spec, str):
        kind = spec.lower()
        if kind == "cnn":
            kind = "standard"
        if kind not in ("standard", "gcnn"):
            raise ValueError("decoder spec string must be one of: 'standard', 'gcnn'")
        return DecoderSpec(kind=kind, params={})

    # Mapping / dict-like
    kind = str(spec.get("kind", "")).lower()
    if kind == "cnn":
        kind = "standard"
    if kind not in ("standard", "gcnn"):
        raise ValueError("decoder spec dict must include kind in: 'standard', 'gcnn'")

    params = dict(spec)
    params.pop("kind", None)
    return DecoderSpec(kind=kind, params=params)


