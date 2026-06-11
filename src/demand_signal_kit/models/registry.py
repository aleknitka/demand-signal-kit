from __future__ import annotations
from typing import TYPE_CHECKING
import importlib
import pkgutil
from pathlib import Path

if TYPE_CHECKING:
    from demand_signal_kit.models.base import BaseForecaster

_REGISTRY: dict[str, type[BaseForecaster]] = {}


def register_model(name: str):
    def decorator(cls: type[BaseForecaster]):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_model(name: str) -> type[BaseForecaster]:
    _auto_discover()
    if name not in _REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    _auto_discover()
    return sorted(_REGISTRY.keys())


def _auto_discover():
    if _REGISTRY:
        return
    package_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name not in ("base", "registry", "__init__"):
            importlib.import_module(f"demand_signal_kit.models.{module_name}")
