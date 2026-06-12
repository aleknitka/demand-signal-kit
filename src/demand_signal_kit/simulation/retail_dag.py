from demand_signal_kit.simulation.dag import Dag, ParameterNode, StochasticNode, DeterministicNode, VectorizedNode
from demand_signal_kit.simulation.equations import dimensions, taste, missions, choice
import polars as pl
import numpy as np
from datetime import date


ATTRIBUTE_NAMES = ["price", "promo_depth", "quality_score", "brand_strength", "seasonal_relevance"]


def build_retail_dag(
    products: pl.DataFrame,
    n_agents: int = 10000,
    seed: int = 42,
) -> Dag:
    """Assemble the 20-node retail demand DAG."""
    dag = Dag()

    # Layer 0: Exogenous Parameters
    dag.add_node(ParameterNode(
        name="dimension_weights",
        parents=[],
        default_value={
            "affluence": {"budget": 0.30, "middle": 0.45, "luxury": 0.25},
            "promo_sensitivity": {"low": 0.30, "medium": 0.45, "high": 0.25},
            "geo_sociological": {"urban_professional": 0.30, "suburban_family": 0.45, "rural_value": 0.25},
            "loyalty_engagement": {"dormant": 0.20, "occasional": 0.50, "champion": 0.30},
        },
    ))

    dag.add_node(ParameterNode(
        name="ontology_rules",
        parents=[],
        default_value=[],
    ))

    dag.add_node(ParameterNode(
        name="product_catalog",
        parents=[],
        default_value=products,
    ))

    dag.add_node(ParameterNode(
        name="day_context",
        parents=[],
        default_value=date(2024, 6, 15),
    ))

    dag.add_node(ParameterNode(
        name="mission_weights",
        parents=[],
        default_value={"weekly_stockup": 0.30, "quick_topup": 0.25, "special_occasion": 0.15, "bulk_buy": 0.20, "impulse_browse": 0.10},
    ))

    dag.add_node(ParameterNode(
        name="n_agents",
        parents=[],
        default_value=n_agents,
    ))

    dag.add_node(ParameterNode(
        name="attribute_names",
        parents=[],
        default_value=ATTRIBUTE_NAMES,
    ))

    # Layer 1: Agent Dimensions
    def sample_dims_fn(**kwargs):
        dw = kwargs.get("dimension_weights", {})
        ont = kwargs.get("ontology_rules", [])
        n = kwargs.get("n_agents", 1000)
        rng = kwargs.get("_rng", np.random.default_rng(42))
        agents = []
        for _ in range(n):
            dims = dimensions.sample_all_dimensions(dw, ont, rng)
            agents.append(dims)
        return agents

    dag.add_node(StochasticNode(
        name="agent_dims",
        parents=["dimension_weights", "ontology_rules", "n_agents"],
        sample_fn=sample_dims_fn,
    ))

    # Layer 2: Taste Vectors
    def compose_taste_fn(**kwargs):
        agent_dims = kwargs.get("agent_dims", [])
        dw = kwargs.get("dimension_weights", {})
        attr_names = kwargs.get("attribute_names", ATTRIBUTE_NAMES)
        rng = kwargs.get("_rng", np.random.default_rng(42))

        from demand_signal_kit.data.synthetic.dimensions import get_dimension
        dim_segments = {}
        for dim_name in dw.keys():
            try:
                dim = get_dimension(dim_name)
                dim_segments[dim_name] = {v: dim.get(v) for v in dim.values()}
            except (KeyError, ValueError):
                continue

        agents = []
        for dims in agent_dims:
            t = taste.compose_taste_vector(dims, dim_segments, attr_names, rng)
            agents.append({"segments": dims, "taste": t})
        return agents

    dag.add_node(DeterministicNode(
        name="agent_taste_vectors",
        parents=["agent_dims", "dimension_weights", "attribute_names"],
        compute_fn=compose_taste_fn,
    ))

    # Layer 2b: Mission Selection
    def select_missions_fn(**kwargs):
        agent_dims = kwargs.get("agent_dims", [])
        day = kwargs.get("day_context", date(2024, 6, 15))
        mw = kwargs.get("mission_weights", {})
        rng = kwargs.get("_rng", np.random.default_rng(42))

        results = []
        for dims in agent_dims:
            utils = missions.compute_mission_utilities(dims, day, mw, rng)
            name = missions.select_mission_from_utilities(utils, rng)
            params = missions.get_mission_params(name)
            results.append({"mission": name, "params": params})
        return results

    dag.add_node(DeterministicNode(
        name="selected_missions",
        parents=["agent_dims", "day_context", "mission_weights"],
        compute_fn=select_missions_fn,
    ))

    # Layer 3: Product Attributes
    def build_product_attrs_fn(**kwargs):
        products = kwargs.get("product_catalog", pl.DataFrame())
        attr_names = kwargs.get("attribute_names", ATTRIBUTE_NAMES)
        rng = kwargs.get("_rng", np.random.default_rng(42))

        attrs_df = products.select(["product_id", "unit_price", "category"]).rename({"unit_price": "price"})
        for attr in attr_names:
            if attr not in attrs_df.columns:
                if attr == "promo_depth":
                    attrs_df = attrs_df.with_columns(pl.lit(0.0).alias(attr))
                elif attr in ("quality_score", "brand_strength", "seasonal_relevance"):
                    attrs_df = attrs_df.with_columns(pl.lit(float(rng.uniform(0.2, 0.8))).alias(attr))

        return attrs_df

    dag.add_node(DeterministicNode(
        name="product_attrs",
        parents=["product_catalog", "attribute_names"],
        compute_fn=build_product_attrs_fn,
    ))

    def apply_temporal_fn(**kwargs):
        product_attrs = kwargs.get("product_attrs", pl.DataFrame())
        day = kwargs.get("day_context", date(2024, 6, 15))
        attr_names = kwargs.get("attribute_names", ATTRIBUTE_NAMES)

        attrs_np = product_attrs.select(attr_names).to_numpy()
        modified = choice.apply_temporal_effects(attrs_np, day, attr_names)
        return modified

    dag.add_node(DeterministicNode(
        name="effective_product_attrs",
        parents=["product_attrs", "day_context", "attribute_names"],
        compute_fn=apply_temporal_fn,
    ))

    # Layer 4: Choice Model
    def compute_utilities_fn(**kwargs):
        agents = kwargs.get("agent_taste_vectors", [])
        product_attrs = kwargs.get("effective_product_attrs", np.array([]))
        attr_names = kwargs.get("attribute_names", ATTRIBUTE_NAMES)

        taste_matrix = taste.compose_taste_matrix(agents, attr_names)
        utilities = choice.compute_utilities(taste_matrix, product_attrs)
        return utilities

    dag.add_node(VectorizedNode(
        name="utility_matrix",
        parents=["agent_taste_vectors", "effective_product_attrs", "attribute_names"],
        compute_fn=compute_utilities_fn,
    ))

    def compute_probs_fn(**kwargs):
        utilities = kwargs.get("utility_matrix", np.array([]))
        return choice.softmax(utilities)

    dag.add_node(VectorizedNode(
        name="choice_probabilities",
        parents=["utility_matrix"],
        compute_fn=compute_probs_fn,
    ))

    # Layer 5: Outputs
    def compute_demand_fn(**kwargs):
        probs = kwargs.get("choice_probabilities", np.array([]))
        agents = kwargs.get("agent_taste_vectors", [])
        product_attrs = kwargs.get("product_attrs", pl.DataFrame())

        product_ids = product_attrs["product_id"].to_list()
        agent_dims = [a.get("segments", {}) for a in agents]
        return choice.predict_demand(probs, agent_dims, product_ids)

    dag.add_node(VectorizedNode(
        name="demand_result",
        parents=["choice_probabilities", "agent_taste_vectors", "product_attrs"],
        compute_fn=compute_demand_fn,
    ))

    def compute_revenue_fn(**kwargs):
        demand = kwargs.get("demand_result", {})
        product_attrs = kwargs.get("product_attrs", pl.DataFrame())
        prices = dict(zip(product_attrs["product_id"].to_list(), product_attrs["price"].to_list()))
        expected = demand.get("expected_demand", {})
        return {pid: expected.get(pid, 0) * prices.get(pid, 0) for pid in expected}

    dag.add_node(DeterministicNode(
        name="revenue_by_product",
        parents=["demand_result", "product_attrs"],
        compute_fn=compute_revenue_fn,
    ))

    return dag
