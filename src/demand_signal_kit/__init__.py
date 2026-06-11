"""Mimo Forecasting - Extensible demand forecasting toolkit."""

from demand_signal_kit.models.registry import get_model, list_models
from demand_signal_kit.models.base import BaseForecaster

__version__ = "0.1.0"
__all__ = ["get_model", "list_models", "BaseForecaster"]
