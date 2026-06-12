from abc import ABC, abstractmethod
import importlib
import pkgutil
from pathlib import Path


class BaseSegment(ABC):
    value: str = ""
    weight: float = 0.0

    @abstractmethod
    def behavioral_profile(self) -> dict:
        """Behavioral parameters: basket size, visit freq, price sensitivity, etc."""

    @abstractmethod
    def causal_params(self) -> dict:
        """Taste vector params for MNL: {attribute: (mean, std)}"""

    def receipt_probability_modifier(self, day, rng) -> float:
        """Multiplier on base receipt probability. Default: 1.0"""
        return 1.0


class BaseDimension(ABC):
    dimension_name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_segment_registry'):
            cls._segment_registry = {}

    @classmethod
    def register(cls, segment_cls: type[BaseSegment]):
        if not hasattr(cls, '_segment_registry'):
            cls._segment_registry = {}
        cls._segment_registry[segment_cls.value] = segment_cls
        return segment_cls

    @classmethod
    def get(cls, value: str) -> BaseSegment:
        cls._auto_discover()
        reg = getattr(cls, '_segment_registry', {})
        return reg[value]()

    @classmethod
    def values(cls) -> list[str]:
        cls._auto_discover()
        reg = getattr(cls, '_segment_registry', {})
        return sorted(reg.keys())

    @classmethod
    def weights(cls) -> dict[str, float]:
        cls._auto_discover()
        return {v: cls.get(v).weight for v in cls.values()}

    @classmethod
    def profiles(cls) -> dict[str, dict]:
        cls._auto_discover()
        return {v: cls.get(v).behavioral_profile() for v in cls.values()}

    @classmethod
    def causal_params_map(cls) -> dict[str, dict]:
        cls._auto_discover()
        return {v: cls.get(v).causal_params() for v in cls.values()}

    @classmethod
    def _auto_discover(cls):
        reg = getattr(cls, '_segment_registry', {})
        if reg:
            return
        package_dir = Path(__file__).parent / cls.dimension_name
        if not package_dir.exists():
            return
        for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
            if module_name not in ("__init__", "base"):
                try:
                    importlib.import_module(
                        f"demand_signal_kit.data.synthetic.dimensions.{cls.dimension_name}.{module_name}"
                    )
                except ImportError:
                    pass
