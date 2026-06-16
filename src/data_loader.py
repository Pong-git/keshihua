from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_PATH


def normalize_node_type(value: Any) -> str:
    if value is None or pd.isna(value) or str(value).strip().lower() in {"", "nan", "none"}:
        return "unknown"
    return str(value)


def load_raw_graph(path: Path = DATA_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_nodes_edges(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw_graph(path)
    nodes = pd.DataFrame(raw.get("nodes", []))
    edges = pd.DataFrame(raw.get("links", []))

    if "type" not in nodes.columns:
        nodes["type"] = "unknown"
    if "country" not in nodes.columns:
        nodes["country"] = ""
    if "type" not in edges.columns:
        edges["type"] = "unknown"
    if "weight" not in edges.columns:
        edges["weight"] = 1.0

    nodes["type"] = nodes["type"].map(normalize_node_type)
    edges["type"] = edges["type"].map(normalize_node_type)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0)

    nodes["label"] = nodes["id"].map(str)
    edges["source_label"] = edges["source"].map(str)
    edges["target_label"] = edges["target"].map(str)
    return nodes, edges


def build_node_lookup(nodes: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    return nodes.set_index("id").to_dict(orient="index")
