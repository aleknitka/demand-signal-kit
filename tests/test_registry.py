from demand_signal_kit.models.registry import get_model, list_models, _REGISTRY


def test_list_models():
    models = list_models()
    assert "prophet" in models
    assert "lightgbm" in models


def test_get_model():
    cls = get_model("lightgbm")
    assert cls.name == "lightgbm"
    instance = cls()
    assert instance.name == "lightgbm"


def test_get_model_not_found():
    import pytest
    with pytest.raises(KeyError, match="not found"):
        get_model("nonexistent_model")
