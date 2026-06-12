from demand_signal_kit.data.synthetic.dimensions.base import BaseDimension
from demand_signal_kit.data.synthetic.dimensions import register_dimension
import importlib
import pkgutil
from pathlib import Path


@register_dimension
class GeoSociologicalDimension(BaseDimension):
    dimension_name = "geo_sociological"

    @classmethod
    def _auto_discover(cls):
        reg = getattr(cls, '_segment_registry', {})
        if reg:
            return
        pkg_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(pkg_dir)]):
            if module_name not in ("__init__",):
                importlib.import_module(
                    f"demand_signal_kit.data.synthetic.dimensions.geo_sociological.{module_name}"
                )
