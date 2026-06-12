from abc import ABC, abstractmethod
import numpy as np
from datetime import date


class BaseMission(ABC):
    name: str = ""
    weight: float = 0.0

    @abstractmethod
    def basket_size_range(self) -> tuple[int, int]:
        """Min, max items per receipt."""

    @abstractmethod
    def category_mix(self) -> dict[str, float]:
        """Probability of selecting from each category."""

    @abstractmethod
    def qty_per_item_range(self) -> tuple[int, int]:
        """Min, max quantity per line item."""

    @abstractmethod
    def utility_features(self) -> dict[str, float]:
        """Features that influence mission selection probability."""

    def is_available(self, day: date, rng: np.random.Generator) -> bool:
        """Whether this mission is possible on this day. Default: always."""
        return True
