from demand_signal_kit.data.synthetic.dimensions.base import BaseDimension, BaseSegment
import importlib
import pkgutil
from pathlib import Path


_dimension_registry: dict[str, type[BaseDimension]] = {}


def register_dimension(cls: type[BaseDimension]):
    _dimension_registry[cls.dimension_name] = cls
    return cls


def list_dimensions() -> list[str]:
    _auto_discover_dimensions()
    return sorted(_dimension_registry.keys())


def get_dimension(name: str) -> BaseDimension:
    _auto_discover_dimensions()
    return _dimension_registry[name]()


def get_all_dimension_values() -> dict[str, list[str]]:
    _auto_discover_dimensions()
    return {name: dim.values() for name, dim in _dimension_registry.items()}


def _auto_discover_dimensions():
    if _dimension_registry:
        return
    dims_dir = Path(__file__).parent
    for _, dim_name, _ in pkgutil.iter_modules([str(dims_dir)]):
        if dim_name not in ("__init__", "base", "ontology"):
            try:
                mod = importlib.import_module(
                    f"demand_signal_kit.data.synthetic.dimensions.{dim_name}"
                )
                if hasattr(mod, "Dimension") or hasattr(mod, "dimension_name"):
                    pass
            except ImportError:
                pass


__all__ = [
    "BaseDimension", "BaseSegment", "register_dimension",
    "list_dimensions", "get_dimension", "get_all_dimension_values",
]
