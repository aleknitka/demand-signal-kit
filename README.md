<p align="center">
  <img src="assets/logo.svg" alt="Demand Signal Kit" width="400">
</p>

# Demand Signal Kit

Extensible demand forecasting toolkit with causal modeling, synthetic data generation, and an interactive dashboard. Uses Polars for data processing and supports multiple forecasting models via a plugin registry.

## Features

- **Causal demand modeling** — DAG-based simulator with do-calculus interventions
- **Multi-dimensional segmentation** — Affluence, promo sensitivity, geo-sociological, loyalty engagement
- **Shopping missions** — Weekly stockup, quick topup, special occasion, bulk buy, impulse browse
- **Ontology mappings** — Cross-dimension correlations (e.g., luxury → low promo sensitivity)
- **Synthetic data generation** — Stores, products, customers, promos, receipts with realistic inter-relationships
- **Multiple models** — LightGBM and Prophet with extensible plugin registry
- **Feature engineering** — Lag features, rolling statistics, cyclical encodings, holiday detection
- **Web dashboard** — Interactive Streamlit UI for exploration, training, and forecasting

## Quick Start

```bash
# Install dependencies
uv sync

# Generate synthetic data
uv run demand-signal-kit generate --stores 10 --products 200 --flat

# Run the demo pipeline
uv run demand-signal-kit demo

# Launch the dashboard
uv run streamlit run app.py
```

## CLI Commands

```bash
# Generate synthetic sales data
uv run demand-signal-kit generate --stores 50 --products 1000 --format parquet --flat

# Run demo pipeline with sample data
uv run demand-signal-kit demo

# Train models and compare
uv run demand-signal-kit train --source csv --path data.csv -m lightgbm -m prophet

# Generate forecasts
uv run demand-signal-kit predict --source csv --path data.csv -m lightgbm --horizon 30

# List available models
uv run demand-signal-kit models
```

## Causal Demand Simulator

The DAG-based simulator lets you model demand as a causal graph and intervene on any variable:

```python
from demand_signal_kit.simulation import DagSimulator

sim = DagSimulator(products, n_agents=10000)

# Baseline
result = sim.evaluate()

# What if we shift to more luxury customers?
result = sim.intervene('dimension_weights', {
    'affluence': {'budget': 0.15, 'middle': 0.35, 'luxury': 0.50},
}).evaluate()

# What if we cut prices?
result = sim.intervene('product_catalog', products.with_columns(...)).evaluate()

# Compare multiple scenarios
results = sim.compare_scenarios([
    {'node_name': 'dimension_weights', 'value': luxury_weights},
    {'node_name': 'dimension_weights', 'value': budget_weights},
])
```

### DAG Structure (16 nodes, 23 edges)

```
Layer 0: dimension_weights, ontology_rules, product_catalog, day_context, mission_weights
    ↓
Layer 1: agent_dims (affluence, promo_sens, geo, loyalty — ontology-adjusted)
    ↓
Layer 2: agent_taste_vectors, selected_missions
    ↓
Layer 3: effective_product_attrs (temporal effects applied)
    ↓
Layer 4: utility_matrix → choice_probabilities (MNL softmax)
    ↓
Layer 5: demand_result, revenue_by_product
```

## Multi-Dimensional Segmentation

Each customer is mapped across 4 independent dimensions:

| Dimension | Segments | File |
|-----------|----------|------|
| Affluence | budget, middle, luxury | `dimensions/affluence/` |
| Promo Sensitivity | low, medium, high | `dimensions/promo_sensitivity/` |
| Geo-Sociological | urban_professional, suburban_family, rural_value | `dimensions/geo_sociological/` |
| Loyalty Engagement | dormant, occasional, champion | `dimensions/loyalty_engagement/` |

### Adding a New Segment

Create a file in the dimension directory:

```python
# dimensions/affluence/student.py
from demand_signal_kit.data.synthetic.dimensions.affluence import AffluenceDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment

@AffluenceDimension.register
class StudentSegment(BaseSegment):
    value = "student"
    weight = 0.05

    def behavioral_profile(self):
        return {"avg_basket_size": (1, 3), "visit_frequency_days": (5, 14), ...}

    def causal_params(self):
        return {"price": (-1.5, 0.3), "promo_depth": (0.9, 0.2), ...}
```

### Adding a New Dimension

Create a directory with `__init__.py` and segment files:

```
dimensions/eco_conscious/
├── __init__.py        # Register the dimension
├── low.py             # Low eco-conscious segment
├── medium.py          # Medium eco-conscious segment
└── high.py            # High eco-conscious segment
```

## Synthetic Data Generation

```bash
# Generate with defaults (10 stores, 100 products)
uv run demand-signal-kit generate

# Production scale
uv run demand-signal-kit generate --stores 200 --products 5000 --format parquet

# Include flat forecast file
uv run demand-signal-kit generate --stores 50 --products 500 --flat
```

### Generated Tables

| Table | Description |
|-------|-------------|
| `stores` | Store dimension (type, size, region, traffic, sensitivity) |
| `products` | Product dimension (category, price, elasticity, cannibalization) |
| `customers` | Customer dimension (segments, loyalty tier, enrollment) |
| `promos` | Promotions (type, discount, scope, duration) |
| `receipts` | Transaction headers (customer, store, time, totals) |
| `receipt_items` | Line items (product, qty, price, discount, promo) |

## Adding New Models

Create a new file in `src/demand_signal_kit/models/`:

```python
import polars as pl
from demand_signal_kit.models.base import BaseForecaster
from demand_signal_kit.models.registry import register_model

@register_model("my_model")
class MyForecaster(BaseForecaster):
    def fit(self, train_df, date_column, target_column, product_column, feature_columns):
        # Training logic
        pass

    def predict(self, future_df, date_column, product_column, feature_columns):
        # Prediction logic
        return pl.DataFrame()
```

The model is automatically discovered and available via CLI.

## Project Structure

```
src/demand_signal_kit/
├── cli.py                      # Click CLI interface
├── config.py                   # Dataclass configurations
├── simulation/                 # DAG-based causal simulator
│   ├── dag.py                  # DagNode, Dag, Intervention
│   ├── equations/              # Pure math functions
│   ├── retail_dag.py           # build_retail_dag() factory
│   └── adapter.py              # DagSimulator API
├── data/
│   ├── loader.py               # CSV/Parquet/PostgreSQL loading
│   ├── schema.py               # DataFrame validation
│   └── synthetic/              # Synthetic data engine
│       ├── dimensions/         # Multi-dimensional segmentation
│       │   ├── ontology.py     # Cross-dimension correlations
│       │   ├── affluence/      # Budget, Middle, Luxury
│       │   ├── promo_sensitivity/  # Low, Medium, High
│       │   ├── geo_sociological/   # Urban, Suburban, Rural
│       │   └── loyalty_engagement/ # Dormant, Occasional, Champion
│       ├── missions/           # Shopping mission engine
│       ├── causal/             # MNL choice model + simulator
│       └── generator.py        # SyntheticDataGenerator
├── features/                   # Feature engineering
├── models/                     # Forecasting models (plugin registry)
└── evaluation/                 # MAE, RMSE, MAPE, SMAPE
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Web Dashboard

```bash
uv sync --extra dashboard
uv run streamlit run app.py
```

The dashboard provides:
- **Data tab**: Upload CSV/Parquet, generate sample data
- **Features tab**: Build and inspect features with correlation analysis
- **Train tab**: Train multiple models side-by-side
- **Forecast tab**: Generate future predictions with interactive charts
- **Compare tab**: Model comparison with radar charts and residual analysis
