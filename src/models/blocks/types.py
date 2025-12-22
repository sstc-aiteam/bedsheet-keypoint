from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Keep the public API intentionally tiny and stable.
ConvBlockKind = Literal["standard", "gcnn_lifting_c4", "gcnn_group_c4"]
GcnnGroup = Literal["C4"]


@dataclass(frozen=True)
class GcnnConfig:
    """
    Minimal config for our native discrete GCNN implementation.

    Notes:
    - We start with C4 (90-degree rotations) as in the daily report.
    - Later we can extend to C8 / D4 / D8 without changing the factory signature.
    """

    group: GcnnGroup = "C4"
