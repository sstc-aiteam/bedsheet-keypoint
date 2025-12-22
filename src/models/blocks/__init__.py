"""
Pluggable model building blocks.

This package includes a minimal, dependency-free GCNN (group CNN) skeleton based on
the concepts described in `docs/20251219_daily_report.md`:
- Discrete rotation group C4
- Lifting convolution: R^2 -> G
- Group convolution: G -> G (left-regular representation / twist-shift)
- Group pooling / projection: G -> invariant plane feature

All components are designed to be modular and easy to integrate gradually.
"""

from .types import ConvBlockKind, GcnnGroup
from .conv_factory import make_conv_block

from .gcnn_groups import CyclicGroupC4
from .gcnn_layers import LiftingConvC4, GroupConvC4, GroupPooling
from .so2_layers import LiftingConvSO2, GroupConvSO2

__all__ = [
    "ConvBlockKind",
    "GcnnGroup",
    "make_conv_block",
    "CyclicGroupC4",
    "LiftingConvC4",
    "GroupConvC4",
    "GroupPooling",
    "LiftingConvSO2",
    "GroupConvSO2",
]
