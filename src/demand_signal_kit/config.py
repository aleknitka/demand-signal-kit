from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    source_type: str = "csv"  # csv | parquet | postgres
    source_path: str = ""
    date_column: str = "date"
    product_column: str = "product_id"
    target_column: str = "quantity"
    event_columns: list[str] = field(default_factory=list)
    connection_string: str = ""


@dataclass
class FeatureConfig:
    lag_periods: list[int] = field(default_factory=lambda: [7, 14, 28])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 14, 28])
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    include_cyclical: bool = True
    include_holidays: bool = True
    holiday_country: str = "US"
    custom_events: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    models: list[str] = field(default_factory=lambda: ["prophet", "lightgbm"])
    forecast_horizon: int = 14
    test_size: float = 0.2
    date_column: str = "date"
    target_column: str = "quantity"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
