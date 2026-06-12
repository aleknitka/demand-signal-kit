from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class DagNode(ABC):
    """Base class for all DAG nodes."""
    name: str = ""
    parents: list[str] = field(default_factory=list)

    @abstractmethod
    def evaluate(self, context: dict[str, Any], rng: np.random.Generator) -> Any:
        """Compute this node's value from parent values in context."""

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', parents={self.parents})"


@dataclass
class ParameterNode(DagNode):
    """Leaf input — value set by user or initial config, no computation."""
    default_value: Any = None

    def evaluate(self, context, rng):
        return context.get(self.name, self.default_value)


@dataclass
class StochasticNode(DagNode):
    """Samples from a distribution using parent values."""
    sample_fn: Any = None

    def evaluate(self, context, rng):
        if self.sample_fn:
            parent_vals = {p: context.get(p) for p in self.parents}
            parent_vals["_rng"] = rng
            return self.sample_fn(**parent_vals)
        return None


@dataclass
class DeterministicNode(DagNode):
    """Pure function of parent values."""
    compute_fn: Any = None

    def evaluate(self, context, rng):
        parent_vals = {p: context.get(p) for p in self.parents}
        parent_vals["_rng"] = rng
        if self.compute_fn:
            return self.compute_fn(**parent_vals)
        return None


@dataclass
class VectorizedNode(DagNode):
    """Optimized numpy/Polars computation."""
    compute_fn: Any = None

    def evaluate(self, context, rng):
        parent_vals = {p: context.get(p) for p in self.parents}
        parent_vals["_rng"] = rng
        if self.compute_fn:
            return self.compute_fn(**parent_vals)
        return None


@dataclass
class Intervention:
    """A do-calculus intervention on a node."""
    node_name: str
    value: Any


class Dag:
    """Directed Acyclic Graph for causal modeling.

    Nodes are evaluated in topological order. Interventions bypass
    a node's structural equation and propagate to descendants.
    """

    def __init__(self):
        self._nodes: dict[str, DagNode] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._interventions: dict[str, Any] = {}

    def add_node(self, node: DagNode):
        self._nodes[node.name] = node
        if node.name not in self._adjacency:
            self._adjacency[node.name] = []
        for parent in node.parents:
            if parent not in self._adjacency:
                self._adjacency[parent] = []
            self._adjacency[parent].append(node.name)

    def nodes(self) -> list[DagNode]:
        return list(self._nodes.values())

    def edges(self) -> list[tuple[str, str]]:
        edges = []
        for parent, children in self._adjacency.items():
            for child in children:
                edges.append((parent, child))
        return edges

    def topological_order(self) -> list[str]:
        in_degree = {n: 0 for n in self._nodes}
        for parent, children in self._adjacency.items():
            for child in children:
                if child in in_degree:
                    in_degree[child] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self._adjacency.get(node, []):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(order) != len(self._nodes):
            raise ValueError("Graph has a cycle!")

        return order

    def intervene(self, node_name: str, value: Any) -> Dag:
        new_dag = Dag()
        new_dag._nodes = dict(self._nodes)
        new_dag._adjacency = {k: list(v) for k, v in self._adjacency.items()}
        new_dag._interventions = dict(self._interventions)
        new_dag._interventions[node_name] = value
        return new_dag

    def set_initial(self, node_name: str, value: Any):
        if node_name not in self._interventions:
            self._interventions[node_name] = value

    def evaluate(self, seed: int = 42) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        context = dict(self._interventions)
        order = self.topological_order()

        for node_name in order:
            if node_name in context:
                continue
            node = self._nodes[node_name]
            value = node.evaluate(context, rng)
            context[node_name] = value

        return context

    def get_node(self, name: str) -> DagNode | None:
        return self._nodes.get(name)

    def summary(self) -> dict:
        return {
            "nodes": len(self._nodes),
            "edges": len(self.edges()),
            "node_names": sorted(self._nodes.keys()),
        }
