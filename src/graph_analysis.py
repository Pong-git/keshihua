from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class GraphIndex:
    outgoing: dict[Any, list[tuple[Any, dict[str, Any]]]]
    incoming: dict[Any, list[tuple[Any, dict[str, Any]]]]
    node_ids: set[Any]


def build_graph_index(nodes: pd.DataFrame, edges: pd.DataFrame) -> GraphIndex:
    outgoing: dict[Any, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
    incoming: dict[Any, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
    node_ids = set(nodes["id"].tolist())

    for edge in edges.to_dict(orient="records"):
        source = edge["source"]
        target = edge["target"]
        outgoing[source].append((target, edge))
        incoming[target].append((source, edge))

    return GraphIndex(dict(outgoing), dict(incoming), node_ids)


def direct_neighbors(index: GraphIndex, entity_id: Any, direction: str = "both") -> set[Any]:
    neighbors: set[Any] = set()
    if direction in {"both", "out"}:
        neighbors.update(target for target, _ in index.outgoing.get(entity_id, []))
    if direction in {"both", "in"}:
        neighbors.update(source for source, _ in index.incoming.get(entity_id, []))
    return neighbors


def ego_subgraph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    center_id: Any,
    depth: int = 1,
    max_nodes: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = build_graph_index(nodes, edges)
    visited = {center_id}
    queue = deque([(center_id, 0)])

    while queue and len(visited) < max_nodes:
        current, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for neighbor in sorted(direct_neighbors(index, current), key=str):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, current_depth + 1))
            if len(visited) >= max_nodes:
                break

    sub_nodes = nodes[nodes["id"].isin(visited)].copy()
    sub_edges = edges[
        edges["source"].isin(visited) & edges["target"].isin(visited)
    ].copy()
    return sub_nodes, sub_edges


def shortest_path(
    index: GraphIndex,
    source: Any,
    targets: Iterable[Any],
    max_depth: int = 4,
) -> list[Any] | None:
    target_set = set(targets)
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        current, path = queue.popleft()
        if current in target_set and current != source:
            return path
        if len(path) - 1 >= max_depth:
            continue
        for neighbor in direct_neighbors(index, current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def common_neighbors(index: GraphIndex, entity_a: Any, entity_b: Any) -> set[Any]:
    return direct_neighbors(index, entity_a) & direct_neighbors(index, entity_b)


def weak_components(nodes: pd.DataFrame, edges: pd.DataFrame) -> list[set[Any]]:
    """Return weakly connected components using both incoming and outgoing links."""
    index = build_graph_index(nodes, edges)
    unvisited = set(index.node_ids)
    components: list[set[Any]] = []

    while unvisited:
        start = next(iter(unvisited))
        component = {start}
        queue = deque([start])
        unvisited.remove(start)

        while queue:
            current = queue.popleft()
            for neighbor in direct_neighbors(index, current):
                if neighbor not in unvisited:
                    continue
                unvisited.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def weak_component_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute component size and vessel ratio for every node."""
    type_lookup = nodes.set_index("id")["type"].to_dict()
    rows: list[dict[str, Any]] = []
    for component_id, component in enumerate(weak_components(nodes, edges)):
        size = len(component)
        vessel_count = sum(1 for node_id in component if type_lookup.get(node_id) == "vessel")
        vessel_ratio = vessel_count / size if size else 0.0
        for node_id in component:
            rows.append(
                {
                    "id": node_id,
                    "component_id": component_id,
                    "component_size": size,
                    "component_vessel_count": vessel_count,
                    "component_vessel_ratio": vessel_ratio,
                }
            )
    return pd.DataFrame(rows)


def directed_cycles_through(
    index: GraphIndex,
    entity_id: Any,
    min_length: int = 2,
    max_length: int = 6,
    allowed_edge_types: set[str] | None = None,
    limit: int = 20,
    max_branch: int = 12,
) -> list[list[Any]]:
    """Find short directed cycles that pass through an entity.

    The graph is small enough for a bounded DFS around one node. Cycles are
    returned as node paths ending at the start node.
    """
    cycles: list[list[Any]] = []
    path = [entity_id]

    def dfs(current: Any) -> None:
        if len(cycles) >= limit:
            return
        if len(path) > max_length:
            return

        candidates = index.outgoing.get(current, [])
        candidates = sorted(candidates, key=lambda item: float(item[1].get("weight", 0)), reverse=True)[
            :max_branch
        ]
        for neighbor, edge in candidates:
            if allowed_edge_types and edge.get("type") not in allowed_edge_types:
                continue
            if neighbor == entity_id and len(path) >= min_length:
                cycles.append(path + [entity_id])
                if len(cycles) >= limit:
                    return
            elif neighbor not in path and len(path) < max_length:
                path.append(neighbor)
                dfs(neighbor)
                path.pop()

    dfs(entity_id)
    return cycles


def summarize_entity(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    entity_id: Any,
) -> dict[str, Any]:
    index = build_graph_index(nodes, edges)
    neighbors = direct_neighbors(index, entity_id)
    entity_edges = edges[(edges["source"] == entity_id) | (edges["target"] == entity_id)]
    neighbor_nodes = nodes[nodes["id"].isin(neighbors)]

    return {
        "entity": entity_id,
        "degree": len(neighbors),
        "edge_count": len(entity_edges),
        "avg_weight": round(float(entity_edges["weight"].mean()), 4)
        if not entity_edges.empty
        else 0.0,
        "neighbor_types": neighbor_nodes["type"].value_counts().to_dict(),
        "edge_types": entity_edges["type"].value_counts().to_dict(),
    }
