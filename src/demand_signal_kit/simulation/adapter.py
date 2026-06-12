import polars as pl
import numpy as np
from dataclasses import dataclass

from demand_signal_kit.simulation.dag import Dag
from demand_signal_kit.simulation.retail_dag import build_retail_dag


@dataclass
class DagResult:
    market_shares: pl.DataFrame
    demand_by_product: dict[str, float]
    revenue_by_product: dict[str, float]
    segment_shares: dict[str, dict[str, float]]
    n_agents: int


class DagSimulator:
    """High-level API wrapping the retail demand DAG.

    Provides the same interface as CausalDemandSimulator
    but executes through the DAG framework.
    """

    def __init__(
        self,
        products: pl.DataFrame,
        n_agents: int = 10000,
        seed: int = 42,
    ):
        self.products = products
        self.n_agents = n_agents
        self.seed = seed
        self._dag = build_retail_dag(products, n_agents, seed)

    def evaluate(self, seed: int | None = None) -> DagResult:
        """Run baseline evaluation."""
        context = self._dag.evaluate(seed=seed or self.seed)
        return self._context_to_result(context)

    def intervene(self, node_name: str, value) -> "DagSimulator":
        """Create a new simulator with an intervention applied."""
        new_sim = DagSimulator.__new__(DagSimulator)
        new_sim.products = self.products
        new_sim.n_agents = self.n_agents
        new_sim.seed = self.seed
        new_sim._dag = self._dag.intervene(node_name, value)
        return new_sim

    def set_initial(self, node_name: str, value):
        """Set an initial parameter value."""
        self._dag.set_initial(node_name, value)

    def compare_scenarios(self, scenarios: list[dict]) -> list[DagResult]:
        """Run baseline + multiple scenarios.

        Each scenario dict has 'node_name' and 'value' keys.
        """
        results = [self.evaluate()]
        for scenario in scenarios:
            sim = self.intervene(scenario["node_name"], scenario["value"])
            results.append(sim.evaluate())
        return results

    def _context_to_result(self, context: dict) -> DagResult:
        demand = context.get("demand_result", {})
        revenue = context.get("revenue_by_product", {})
        product_attrs = context.get("product_attrs", pl.DataFrame())

        product_ids = product_attrs["product_id"].to_list() if len(product_attrs) > 0 else []
        shares = demand.get("market_shares", {})

        market_df = pl.DataFrame({
            "product_id": product_ids,
            "market_share": [shares.get(pid, 0) for pid in product_ids],
            "expected_demand": [demand.get("expected_demand", {}).get(pid, 0) for pid in product_ids],
        })

        return DagResult(
            market_shares=market_df,
            demand_by_product=demand.get("expected_demand", {}),
            revenue_by_product=revenue,
            segment_shares=demand.get("segment_shares", {}),
            n_agents=self.n_agents,
        )
