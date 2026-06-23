from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ILLEGAL_FISHING_KEYWORDS, SUSPECT_ENTITIES
from .graph_analysis import (
    GraphIndex,
    build_graph_index,
    direct_neighbors,
    directed_cycles_through,
    shortest_path,
    weak_component_features,
)


def find_keyword_nodes(nodes: pd.DataFrame) -> set[Any]:
    pattern = "|".join(ILLEGAL_FISHING_KEYWORDS)
    mask = nodes["label"].str.contains(pattern, case=False, na=False, regex=True)
    return set(nodes.loc[mask, "id"].tolist())


def _edge_types_between(edges: pd.DataFrame, source: Any, target: Any) -> str:
    linked = edges[
        ((edges["source"] == source) & (edges["target"] == target))
        | ((edges["source"] == target) & (edges["target"] == source))
    ]
    if linked.empty:
        return ""
    return ", ".join(sorted(linked["type"].dropna().astype(str).unique()))


def keyword_evidence_paths(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    entity_id: Any,
    max_depth: int = 3,
    limit: int = 12,
) -> pd.DataFrame:
    """Return short paths from an entity to illegal-fishing keyword nodes."""
    index = build_graph_index(nodes, edges)
    keyword_nodes = find_keyword_nodes(nodes)
    label_lookup = nodes.set_index("id")["label"].to_dict()
    type_lookup = nodes.set_index("id")["type"].to_dict()

    rows: list[dict[str, Any]] = []
    for target in sorted(keyword_nodes, key=lambda value: str(label_lookup.get(value, value))):
        path = shortest_path(index, entity_id, [target], max_depth=max_depth)
        if not path:
            continue
        relation_steps = [
            _edge_types_between(edges, path[idx], path[idx + 1])
            for idx in range(len(path) - 1)
        ]
        rows.append(
            {
                "target_keyword_node": label_lookup.get(target, target),
                "target_type": type_lookup.get(target, "unknown"),
                "path_length": len(path) - 1,
                "path": " -> ".join(map(str, path)),
                "relation_steps": " -> ".join(relation_steps),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "target_keyword_node",
                "target_type",
                "path_length",
                "path",
                "relation_steps",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["path_length", "target_keyword_node"])
        .head(limit)
        .reset_index(drop=True)
    )


def official_suspect_paths(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    entity_id: Any,
    max_depth: int = 3,
) -> pd.DataFrame:
    """Return short paths from an entity to official suspicious entities."""
    index = build_graph_index(nodes, edges)
    label_lookup = nodes.set_index("id")["label"].to_dict()
    type_lookup = nodes.set_index("id")["type"].to_dict()

    rows: list[dict[str, Any]] = []
    for suspect in SUSPECT_ENTITIES:
        if suspect == entity_id:
            continue
        path = shortest_path(index, entity_id, [suspect], max_depth=max_depth)
        if not path:
            continue
        relation_steps = [
            _edge_types_between(edges, path[idx], path[idx + 1])
            for idx in range(len(path) - 1)
        ]
        rows.append(
            {
                "target_suspect": label_lookup.get(suspect, suspect),
                "target_type": type_lookup.get(suspect, "unknown"),
                "path_hops": len(path) - 1,
                "path": " -> ".join(map(str, path)),
                "relation_steps": " -> ".join(relation_steps),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "target_suspect",
                "target_type",
                "path_hops",
                "path",
                "relation_steps",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["path_hops", "target_suspect"])
        .reset_index(drop=True)
    )


def common_neighbor_summary(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    entity_ids: list[Any],
    max_examples: int = 8,
) -> pd.DataFrame:
    """Compare direct-neighbor overlap for every pair of selected entities."""
    index = build_graph_index(nodes, edges)
    node_lookup = nodes.set_index("id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []

    for left_idx, left in enumerate(entity_ids):
        for right in entity_ids[left_idx + 1 :]:
            overlap = direct_neighbors(index, left) & direct_neighbors(index, right)
            typed_counts: dict[str, int] = {}
            examples: list[str] = []
            for neighbor in sorted(overlap, key=str):
                node_type = node_lookup.get(neighbor, {}).get("type", "unknown")
                typed_counts[node_type] = typed_counts.get(node_type, 0) + 1
                if len(examples) < max_examples:
                    examples.append(f"{neighbor} ({node_type})")

            rows.append(
                {
                    "entity_a": left,
                    "entity_b": right,
                    "common_neighbor_count": len(overlap),
                    "common_neighbor_types": "; ".join(
                        f"{key}:{value}" for key, value in sorted(typed_counts.items())
                    ),
                    "examples": "; ".join(examples),
                }
            )

    return pd.DataFrame(rows).sort_values("common_neighbor_count", ascending=False)


def suspect_expansion_candidates(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    suspects: list[Any] | None = None,
    max_depth: int = 2,
    limit: int = 80,
) -> pd.DataFrame:
    """Find entities near the official suspects and rank them as investigation leads.

    This is the most task-aligned recommendation table: it starts from the
    official suspect entities and expands outward through the graph.
    """
    suspects = suspects or SUSPECT_ENTITIES
    suspect_set = set(suspects)
    index = build_graph_index(nodes, edges)
    node_lookup = nodes.set_index("id").to_dict(orient="index")
    keyword_nodes = find_keyword_nodes(nodes)

    rows: list[dict[str, Any]] = []
    for entity_id in nodes["id"].tolist():
        if entity_id in suspect_set:
            continue

        paths: list[list[Any]] = []
        connected_suspects: list[Any] = []
        for suspect in suspects:
            path = shortest_path(index, suspect, [entity_id], max_depth=max_depth)
            if path:
                paths.append(path)
                connected_suspects.append(suspect)

        if not paths:
            continue

        neighbors = direct_neighbors(index, entity_id)
        entity_edges = edges[(edges["source"] == entity_id) | (edges["target"] == entity_id)]
        neighbor_nodes = nodes[nodes["id"].isin(neighbors)]
        min_distance = min(len(path) - 1 for path in paths)
        vessel_links = int((neighbor_nodes["type"] == "vessel").sum())
        company_links = int((neighbor_nodes["type"] == "company").sum())
        organization_links = int(
            neighbor_nodes["type"].isin(["organization", "political_organization"]).sum()
        )
        ownership_edges = int((entity_edges["type"] == "ownership").sum())
        membership_edges = int((entity_edges["type"] == "membership").sum())
        high_weight_edges = int((entity_edges["weight"] >= 0.9).sum())
        keyword_neighbors = len(neighbors & keyword_nodes)

        path_examples = []
        for path in sorted(paths, key=lambda value: (len(value), str(value)))[:3]:
            relation_steps = [
                _edge_types_between(edges, path[idx], path[idx + 1])
                for idx in range(len(path) - 1)
            ]
            path_examples.append(
                f"{' -> '.join(map(str, path))} [{ ' -> '.join(relation_steps) }]"
            )

        rows.append(
            {
                "candidate": entity_id,
                "type": node_lookup.get(entity_id, {}).get("type", "unknown"),
                "country": node_lookup.get(entity_id, {}).get("country", ""),
                "connected_suspect_count": len(connected_suspects),
                "connected_suspects": "; ".join(map(str, connected_suspects)),
                "min_distance_to_suspect": min_distance,
                "degree": len(neighbors),
                "edge_count": len(entity_edges),
                "vessel_links": vessel_links,
                "company_links": company_links,
                "organization_links": organization_links,
                "ownership_edges": ownership_edges,
                "membership_edges": membership_edges,
                "high_weight_edges": high_weight_edges,
                "keyword_neighbors": keyword_neighbors,
                "path_examples": "; ".join(path_examples),
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "connected_suspect_count",
                "min_distance_to_suspect",
                "vessel_links",
                "ownership_edges",
                "membership_edges",
            ],
            ascending=[False, True, False, False, False],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def fishhook_feature_table(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    entity_ids: list[Any] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Compute FishHook-inspired anomaly features for entities.

    The original paper uses graph embedding distance and supply-chain cycles.
    This implementation keeps the same analysis spirit with available local
    graph features: weak-component vessel ratio, 1-2 hop family/political
    links, short directed cycles, and graph distance to official suspects.
    """
    index = build_graph_index(nodes, edges)
    component_features = weak_component_features(nodes, edges).set_index("id").to_dict(orient="index")
    avg_vessel_ratio = (
        sum(row["component_vessel_ratio"] for row in component_features.values())
        / len(component_features)
        if component_features
        else 0.0
    )

    type_lookup = nodes.set_index("id")["type"].to_dict()
    entities = entity_ids or nodes["id"].tolist()
    rows: list[dict[str, Any]] = []
    cycle_edge_types = {"ownership", "membership", "partnership"}

    for entity_id in entities:
        one_hop = direct_neighbors(index, entity_id)
        two_hop = set(one_hop)
        for neighbor in one_hop:
            two_hop.update(direct_neighbors(index, neighbor))
        two_hop.discard(entity_id)

        entity_edges = edges[(edges["source"] == entity_id) | (edges["target"] == entity_id)]
        first_second_edges = edges[
            edges["source"].isin(two_hop | {entity_id})
            & edges["target"].isin(two_hop | {entity_id})
        ]
        family_1_2 = int((first_second_edges["type"] == "family_relationship").sum())
        political_1_2 = sum(
            1 for node_id in two_hop if type_lookup.get(node_id) == "political_organization"
        )
        cycles = directed_cycles_through(
            index,
            entity_id,
            min_length=2,
            max_length=4,
            allowed_edge_types=cycle_edge_types,
            limit=12,
            max_branch=8,
        )
        suspect_distances: list[int] = []
        suspect_paths: list[str] = []
        for suspect in SUSPECT_ENTITIES:
            if suspect == entity_id:
                continue
            path = shortest_path(index, entity_id, [suspect], max_depth=4)
            if path:
                suspect_distances.append(len(path) - 1)
                suspect_paths.append(" -> ".join(map(str, path)))

        component = component_features.get(entity_id, {})
        vessel_ratio = float(component.get("component_vessel_ratio", 0.0))
        vessel_ratio_delta = vessel_ratio - avg_vessel_ratio
        min_suspect_distance = min(suspect_distances) if suspect_distances else 99
        connected_suspects = len(suspect_distances)
        vessel_links = sum(1 for node_id in one_hop if type_lookup.get(node_id) == "vessel")
        high_weight_edges = int((entity_edges["weight"] >= 0.9).sum())
        ownership_edges = int((entity_edges["type"] == "ownership").sum())
        membership_edges = int((entity_edges["type"] == "membership").sum())

        anomaly_score = (
            max(vessel_ratio_delta, 0) * 100
            + family_1_2 * 0.1
            + political_1_2 * 1.0
            + len(cycles) * 5.0
            + connected_suspects * 8.0
            + max(0, 5 - min_suspect_distance) * 4.0
            + vessel_links * 2.0
            + ownership_edges * 0.7
            + membership_edges * 0.5
            + high_weight_edges * 0.3
        )

        rows.append(
            {
                "entity": entity_id,
                "type": type_lookup.get(entity_id, "unknown"),
                "fishhook_anomaly_score": round(anomaly_score, 3),
                "component_id": component.get("component_id", ""),
                "component_size": component.get("component_size", 0),
                "component_vessel_ratio": round(vessel_ratio, 4),
                "vessel_ratio_delta": round(vessel_ratio_delta, 4),
                "family_edges_1_2_hop": family_1_2,
                "political_nodes_1_2_hop": political_1_2,
                "short_cycle_count": len(cycles),
                "connected_suspect_count": connected_suspects,
                "min_suspect_distance": min_suspect_distance if suspect_distances else "",
                "vessel_links": vessel_links,
                "ownership_edges": ownership_edges,
                "membership_edges": membership_edges,
                "high_weight_edges": high_weight_edges,
                "cycle_examples": "; ".join(" -> ".join(map(str, cycle)) for cycle in cycles[:3]),
                "suspect_path_examples": "; ".join(suspect_paths[:3]),
            }
        )

    result = pd.DataFrame(rows).sort_values("fishhook_anomaly_score", ascending=False)
    if limit:
        result = result.head(limit)
    return result.reset_index(drop=True)


def fishhook_candidate_ranking(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    limit: int = 80,
    candidates: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidates = candidates if candidates is not None else suspect_expansion_candidates(nodes, edges, max_depth=2, limit=200)
    if candidates.empty:
        return candidates
    features = features if features is not None else fishhook_feature_table(nodes, edges, candidates["candidate"].tolist())
    merged = candidates.merge(
        features,
        left_on="candidate",
        right_on="entity",
        how="left",
        suffixes=("", "_fishhook"),
    )
    merged = merged.drop(columns=[column for column in ["entity", "type_fishhook"] if column in merged.columns])
    return (
        merged.sort_values("fishhook_anomaly_score", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def suspect_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    return fishhook_feature_table(nodes, edges, SUSPECT_ENTITIES)


def recommend_companies(nodes: pd.DataFrame, edges: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    ranking = fishhook_candidate_ranking(nodes, edges, limit=200)
    suspects = set(SUSPECT_ENTITIES)
    companies = ranking[
        (ranking["type"] == "company") & (~ranking["candidate"].isin(suspects))
    ].copy()
    return companies.head(limit).reset_index(drop=True)
