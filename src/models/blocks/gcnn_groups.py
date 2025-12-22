from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class CyclicGroupC4:
    """
    Discrete rotation group C4 = {0, 90, 180, 270} degrees.

    We represent group elements by integer indices i in {0,1,2,3} corresponding to angle i * 90°.
    This matches the daily report's emphasis on discrete groups and left regular representation.
    """

    order: int = 4

    def __post_init__(self) -> None:
        if self.order != 4:
            raise ValueError("CyclicGroupC4 only supports order=4.")

    def elements(self) -> List[int]:
        return [0, 1, 2, 3]

    def product(self, a: int, b: int) -> int:
        # group multiplication corresponds to adding angles mod 4
        return int((a + b) % 4)

    def inverse(self, a: int) -> int:
        return int((-a) % 4)

    def angle_radians(self, a: int) -> torch.Tensor:
        # 0, pi/2, pi, 3pi/2
        return torch.tensor(float(a) * 0.5 * torch.pi)


