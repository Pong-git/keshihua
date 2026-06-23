from __future__ import annotations

from difflib import get_close_matches
import math
import re

import pandas as pd

from src.config import EDGE_TYPE_COLORS, NODE_TYPE_COLORS, SUSPECT_ENTITIES
from src.data_loader import load_nodes_edges
from src.graph_analysis import build_graph_index, direct_neighbors, ego_subgraph, summarize_entity
from src.risk_model import (
    common_neighbor_summary,
    fishhook_candidate_ranking,
    fishhook_feature_table,
    keyword_evidence_paths,
    official_suspect_paths,
    recommend_companies,
    suspect_expansion_candidates,
    suspect_summary,
)

try:
    import plotly.graph_objects as go
    import streamlit as st
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"缺少运行可视化界面的依赖：{missing}。请先执行 pip install -r requirements.txt"
    ) from exc


THEME = {
    "bg": "#f6f8fb",
    "surface": "#ffffff",
    "ink": "#1f2937",
    "muted": "#667085",
    "line": "#e5e7eb",
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#475569",
}


@st.cache_data(show_spinner=False)
def cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_nodes_edges()


@st.cache_data(show_spinner=False)
def cached_suspect_expansion(max_depth: int) -> pd.DataFrame:
    nodes, edges = load_nodes_edges()
    return suspect_expansion_candidates(nodes, edges, max_depth=max_depth, limit=120)


@st.cache_data(show_spinner=False)
def cached_fishhook_ranking() -> pd.DataFrame:
    nodes, edges = load_nodes_edges()
    return fishhook_candidate_ranking(nodes, edges, limit=120)


def normalize_search_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def compact_search_text(value: object) -> str:
    return re.sub(r"\s+", "", normalize_search_text(value))


def short_label(value: object, max_len: int = 24) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def page_style() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 18% 0%, rgba(37, 99, 235, 0.09), transparent 28rem),
                linear-gradient(180deg, #f8fafc 0%, {THEME["bg"]} 42%, #ffffff 100%);
            color: {THEME["ink"]};
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #eef4ff 0%, #f8fafc 70%);
            border-right: 1px solid #dbe4f0;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
            color: #111827;
        }}
        .block-container {{
            padding-top: 2rem;
            max-width: 1500px;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.88);
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {THEME["muted"]};
            font-size: 0.95rem;
        }}
        div[data-testid="stMetricValue"] {{
            color: #0f172a;
            font-weight: 760;
        }}
        .viz-card {{
            background: rgba(255,255,255,0.92);
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
            margin-bottom: 16px;
        }}
        .section-note {{
            color: {THEME["muted"]};
            font-size: 0.96rem;
            margin-top: -0.35rem;
            margin-bottom: 0.65rem;
        }}
        .entity-chip {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1d4ed8;
            font-weight: 650;
            margin: 0 8px 8px 0;
        }}
        .small-muted {{
            color: {THEME["muted"]};
            font-size: 0.88rem;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}
        div[data-testid="stButton"] > button {{
            border-radius: 999px;
            min-height: 2.35rem;
            width: 100%;
            padding: 0.35rem 0.55rem;
            font-weight: 650;
            border: 1px solid #d0d7e2;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        div[data-testid="stButton"] > button:hover {{
            border-color: #93c5fd;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.14);
        }}
        div[data-testid="stButton"] > button p {{
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 0.9rem;
        }}
        .edge-control-strip {{
            display: flex;
            flex-wrap: nowrap;
            align-items: center;
            gap: 10px;
            margin: 12px 0 14px 0;
            overflow-x: auto;
            padding-bottom: 2px;
        }}
        .edge-control-strip a {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 118px;
            height: 42px;
            padding: 0 18px;
            border-radius: 999px;
            border: 1px solid #cbd5e1;
            background: #ffffff;
            color: #1f2937;
            font-weight: 720;
            text-decoration: none;
            white-space: nowrap;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
            transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
        }}
        .edge-control-strip a:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
            filter: brightness(0.98);
        }}
        .edge-control-strip a.edge-action {{
            min-width: 106px;
            background: #ffffff;
            color: #334155;
        }}
        .edge-control-strip a.edge-on {{
            background: var(--edge-color);
            border-color: var(--edge-color);
            color: #ffffff;
        }}
        .edge-control-strip a.edge-off {{
            background: #f1f5f9;
            border-color: #cbd5e1;
            color: #64748b;
        }}
        div[data-testid="stElementContainer"]:has(.edge-control-token) {{
            display: none;
            height: 0;
            margin: 0;
            padding: 0;
        }}
        .summary-chart-gap {{
            height: 36px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            display: flex;
            gap: 8px;
            width: 100%;
        }}
        div[data-testid="stTabs"], div[data-testid="stTabs"] > div {{
            width: 100%;
        }}
        .stTabs [data-baseweb="tab"] {{
            flex: 1 1 0;
            justify-content: center;
            min-width: 0;
            background: rgba(255,255,255,0.75);
            border-radius: 999px;
            padding: 8px 10px;
            text-align: center;
        }}
        .stTabs [data-baseweb="tab"] p {{
            width: 100%;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, Arial", size=13, color=THEME["ink"]),
        hoverlabel=dict(bgcolor="#111827", font_color="#ffffff"),
    )
    return fig


def horizontal_bar_figure(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title_text: str,
    color: str = THEME["blue"],
    top_n: int = 10,
) -> go.Figure:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return go.Figure()
    plot_df = df[[label_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0)
    plot_df["label_text"] = plot_df[label_col].map(lambda value: str(short_label(value, 34)))
    plot_df = plot_df.sort_values(value_col, ascending=False).head(top_n).sort_values(value_col)
    fig = go.Figure(
        go.Bar(
            x=plot_df[value_col],
            y=plot_df["label_text"],
            orientation="h",
            marker=dict(color=color, line=dict(color="rgba(255,255,255,0.9)", width=1)),
            text=plot_df[value_col].map(lambda value: f"{value:.1f}" if isinstance(value, float) else str(value)),
            textposition="outside",
            hovertemplate="%{y}<br>数值：%{x}<extra></extra>",
        )
    )
    fig.update_layout(title=dict(text=title_text, x=0.02, font=dict(size=18)))
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(type="category", showgrid=False, categoryorder="array", categoryarray=plot_df["label_text"].tolist())
    return plot_layout(fig, height=max(360, 42 * len(plot_df) + 90))


def donut_figure(counts: dict[str, int], title_text: str, color_map: dict[str, str] | None = None) -> go.Figure:
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [(color_map or {}).get(label, NODE_TYPE_COLORS.get(label, EDGE_TYPE_COLORS.get(label, "#94a3b8"))) for label in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.58,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="label+percent",
            textposition="auto",
            textfont=dict(size=14),
            insidetextorientation="radial",
            automargin=True,
            hovertemplate="%{label}<br>数量：%{value}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title_text, x=0.02, font=dict(size=18)),
        showlegend=False,
        uniformtext=dict(minsize=11, mode="show"),
    )
    fig = plot_layout(fig, height=360)
    fig.update_layout(margin=dict(l=92, r=92, t=68, b=72))
    return fig


def candidate_bubble_figure(candidates: pd.DataFrame) -> go.Figure:
    if candidates.empty:
        return go.Figure()
    plot_df = candidates.copy()
    plot_df["min_distance_to_suspect"] = pd.to_numeric(plot_df["min_distance_to_suspect"], errors="coerce").fillna(0)
    plot_df["connected_suspect_count"] = pd.to_numeric(plot_df["connected_suspect_count"], errors="coerce").fillna(0)
    plot_df["vessel_links"] = pd.to_numeric(plot_df["vessel_links"], errors="coerce").fillna(0)
    plot_df["priority_score"] = (
        plot_df["connected_suspect_count"] * 100
        - plot_df["min_distance_to_suspect"] * 20
        + plot_df["vessel_links"].map(lambda value: math.log1p(max(value, 0)) * 8)
    )
    plot_df = plot_df.sort_values(
        ["connected_suspect_count", "min_distance_to_suspect", "vessel_links", "priority_score"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    distance_values = sorted(plot_df["min_distance_to_suspect"].dropna().unique().tolist())
    sampled_groups = []
    per_distance_limit = max(18, 70 // max(len(distance_values), 1))
    for distance in distance_values:
        sampled_groups.append(
            plot_df[plot_df["min_distance_to_suspect"] == distance].head(per_distance_limit)
        )
    plot_df = pd.concat(sampled_groups, ignore_index=True).head(90)
    offsets = [-0.16, -0.09, -0.03, 0.04, 0.11, 0.17]
    plot_df["x_plot"] = plot_df["min_distance_to_suspect"] + [offsets[idx % len(offsets)] for idx in range(len(plot_df))]
    plot_df["y_plot"] = plot_df["connected_suspect_count"] + [offsets[(idx * 2) % len(offsets)] for idx in range(len(plot_df))]
    plot_df["marker_size"] = plot_df["vessel_links"].map(lambda value: min(52, max(18, 18 + math.log1p(max(value, 0)) * 11)))
    type_values = sorted(plot_df["type"].fillna("unknown").astype(str).unique())
    color_lookup = {value: NODE_TYPE_COLORS.get(value, "#94a3b8") for value in type_values}
    fig = go.Figure()
    for type_name, group in plot_df.groupby(plot_df["type"].fillna("unknown").astype(str)):
        fig.add_trace(
            go.Scatter(
                x=group["x_plot"],
                y=group["y_plot"],
                mode="markers",
                name=type_name,
                marker=dict(
                    size=group["marker_size"],
                    color=color_lookup[type_name],
                    opacity=0.76,
                    line=dict(color="#ffffff", width=2),
                ),
                hovertext=group["candidate"].map(lambda value: short_label(value, 42)),
                customdata=group[["min_distance_to_suspect", "connected_suspect_count", "vessel_links", "path_examples"]].values,
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "最近距离：%{customdata[0]} 跳<br>"
                    "连接可疑实体：%{customdata[1]} 个<br>"
                    "船只连接：%{customdata[2]}<br>"
                    "<extra></extra>"
                ),
            )
        )
    min_distance = float(plot_df["min_distance_to_suspect"].min())
    max_distance = float(plot_df["min_distance_to_suspect"].max())
    min_connected = float(plot_df["connected_suspect_count"].min())
    max_connected = float(plot_df["connected_suspect_count"].max())
    fig.add_annotation(
        x=0.01,
        y=1.15,
        xref="paper",
        yref="paper",
        text="读法：左上角优先；气泡越大，船只连接越多",
        showarrow=False,
        align="left",
        font=dict(size=12, color=THEME["muted"]),
    )
    fig.update_layout(
        title=dict(
            text="扩展候选分布：左上角优先查看",
            x=0.02,
            font=dict(size=18),
        ),
        legend=dict(title="实体类型", orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
    )
    tick_values = sorted(plot_df["min_distance_to_suspect"].dropna().unique().tolist())
    fig.update_xaxes(
        title="到官方可疑实体的最近距离",
        tickmode="array",
        tickvals=tick_values,
        ticktext=[f"{int(value)}跳" if float(value).is_integer() else f"{value:g}跳" for value in tick_values],
        range=[min_distance - 0.58, max_distance + 0.58],
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
    )
    fig.update_yaxes(
        title="连接到的官方可疑实体数量",
        dtick=1,
        range=[max(0, min_connected - 0.62), max_connected + 0.72],
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
    )
    fig = plot_layout(fig, height=520)
    fig.update_layout(margin=dict(l=78, r=72, t=132, b=72))
    return fig


def feature_radar_figure(row: pd.Series) -> go.Figure:
    dimensions = [
        ("short_cycle_count", "短环路"),
        ("connected_suspect_count", "关联嫌疑"),
        ("vessel_links", "船只连接"),
        ("ownership_edges", "所有权"),
        ("membership_edges", "成员关系"),
        ("family_edges_1_2_hop", "家族关系"),
        ("political_nodes_1_2_hop", "政治组织"),
    ]
    values = [float(pd.to_numeric(row.get(col, 0), errors="coerce") or 0) for col, _ in dimensions]
    max_value = max(values) if values else 1
    if max_value > 0:
        scaled = [
            max(12, math.log1p(value) / math.log1p(max_value) * 100) if value > 0 else 0
            for value in values
        ]
    else:
        scaled = [0 for _ in values]
    labels = [label for _, label in dimensions]
    hover_values = [f"{value:.2f}".rstrip("0").rstrip(".") for value in values]
    fig = go.Figure(
        go.Scatterpolar(
            r=scaled + [scaled[0]],
            theta=labels + [labels[0]],
            customdata=hover_values + [hover_values[0]],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.22)",
            line=dict(color=THEME["blue"], width=3),
            marker=dict(size=8, color=THEME["blue"], line=dict(color="#ffffff", width=1.5)),
            hovertemplate="%{theta}<br>原始值：%{customdata}<br>均衡强度：%{r:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "低", "中", "高", "峰值"],
                tickfont=dict(size=10),
                gridcolor="#e5e7eb",
            ),
            angularaxis=dict(gridcolor="#e5e7eb", tickfont=dict(size=13)),
        ),
        showlegend=False,
        title=dict(text="当前实体结构画像", x=0.02, font=dict(size=18)),
    )
    return plot_layout(fig, height=360)


def suspect_compare_figure(suspects: pd.DataFrame) -> go.Figure:
    if suspects.empty:
        return go.Figure()
    plot_df = suspects.copy()
    plot_df["entity_label"] = plot_df["entity"].map(lambda value: str(short_label(value, 26)))
    fig = go.Figure()
    for col_name, label, color in [
        ("fishhook_anomaly_score", "FishHook分", THEME["blue"]),
        ("vessel_links", "一跳船只", THEME["cyan"]),
        ("short_cycle_count", "短环路", THEME["red"]),
        ("ownership_edges", "所有权关系", THEME["amber"]),
    ]:
        if col_name in plot_df.columns:
            fig.add_trace(
                go.Bar(
                    x=plot_df["entity_label"],
                    y=pd.to_numeric(plot_df[col_name], errors="coerce").fillna(0),
                    name=label,
                    marker_color=color,
                    customdata=plot_df[["entity"]].values,
                    hovertemplate="<b>%{customdata[0]}</b><br>" + f"{label}：%{{y}}<extra></extra>",
                )
            )
    fig.update_layout(
        title=dict(text="4个官方可疑实体的关键特征对比", x=0.02, font=dict(size=18)),
        barmode="group",
        legend=dict(title="指标"),
    )
    fig.update_xaxes(
        title="官方可疑实体",
        type="category",
        categoryorder="array",
        categoryarray=plot_df["entity_label"].tolist(),
    )
    fig.update_yaxes(title="数值")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return plot_layout(fig, height=420)


def feature_strip_figure(features: pd.DataFrame) -> go.Figure:
    cols = [
        ("fishhook_anomaly_score", "FishHook分"),
        ("vessel_links", "船只"),
        ("ownership_edges", "所有权"),
        ("membership_edges", "成员"),
        ("short_cycle_count", "短环路"),
        ("connected_suspect_count", "关联嫌疑"),
    ]
    row = features.iloc[0]
    labels = [label for col_name, label in cols if col_name in features.columns]
    values = [float(pd.to_numeric(row.get(col_name, 0), errors="coerce") or 0) for col_name, _ in cols if col_name in features.columns]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=[THEME["blue"], THEME["cyan"], THEME["amber"], THEME["green"], THEME["red"], THEME["purple"]][: len(values)]),
            hovertemplate="%{x}<br>数值：%{y}<extra></extra>",
        )
    )
    fig.update_layout(title=dict(text="当前实体关键特征", x=0.02, font=dict(size=18)))
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return plot_layout(fig, height=320)


def find_entity_matches(query: str, entities: list[object], limit: int = 20) -> list[object]:
    normalized_query = normalize_search_text(query)
    compact_query = compact_search_text(query)
    if not normalized_query:
        return []

    matches = [
        entity
        for entity in entities
        if normalized_query in normalize_search_text(entity)
        or compact_query in compact_search_text(entity)
    ]
    if matches:
        return matches[:limit]

    entity_labels = {str(entity): entity for entity in entities}
    close_labels = get_close_matches(str(query), list(entity_labels), n=limit, cutoff=0.62)
    return [entity_labels[label] for label in close_labels]


def set_edge_type_visibility(state_key: str, edge_types: list[str], visible: bool) -> None:
    current_state = dict(st.session_state.get(state_key, {}))
    for edge_type in edge_types:
        current_state[edge_type] = visible
    st.session_state[state_key] = current_state


def toggle_edge_type_visibility(state_key: str, edge_type: str) -> None:
    current_state = dict(st.session_state.get(state_key, {}))
    current_state[edge_type] = not bool(current_state.get(edge_type, True))
    st.session_state[state_key] = current_state


def relation_summary_figure(nodes: pd.DataFrame, edges: pd.DataFrame, center_id: object) -> go.Figure:
    center_edges = edges[(edges["source"] == center_id) | (edges["target"] == center_id)].copy()
    if center_edges.empty:
        return go.Figure()

    node_types = nodes.set_index("id")["type"].to_dict()
    link_counts: dict[tuple[str, str], int] = {}
    for edge in center_edges.to_dict(orient="records"):
        neighbor = edge["target"] if edge["source"] == center_id else edge["source"]
        edge_type = str(edge.get("type", "unknown"))
        neighbor_type = str(node_types.get(neighbor, "unknown"))
        relation_label = f"关系：{edge_type}"
        type_label = f"实体类型：{neighbor_type}"
        link_counts[("当前实体", relation_label)] = link_counts.get(("当前实体", relation_label), 0) + 1
        link_counts[(relation_label, type_label)] = link_counts.get((relation_label, type_label), 0) + 1

    labels = ["当前实体"]
    labels.extend(sorted({target for source, target in link_counts if source == "当前实体"}))
    labels.extend(sorted({target for source, target in link_counts if source != "当前实体"}))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    colors = []
    for label in labels:
        if label == "当前实体":
            colors.append("#111827")
        elif label.startswith("关系："):
            edge_type = label.replace("关系：", "")
            colors.append(EDGE_TYPE_COLORS.get(edge_type, "#64748b"))
        else:
            node_type = label.replace("实体类型：", "")
            colors.append(NODE_TYPE_COLORS.get(node_type, NODE_TYPE_COLORS["unknown"]))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=22,
                    thickness=18,
                    line=dict(color="#ffffff", width=2),
                    label=labels,
                    color=colors,
                ),
                link=dict(
                    source=[label_to_index[source] for source, _ in link_counts],
                    target=[label_to_index[target] for _, target in link_counts],
                    value=list(link_counts.values()),
                    color="rgba(100, 116, 139, 0.25)",
                    hovertemplate="%{source.label} → %{target.label}<br>数量：%{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(size=15, family="Microsoft YaHei, Arial", color="#111827"),
        paper_bgcolor="#ffffff",
    )
    return fig


def simplified_network_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    center_id: object,
    show_full_edges: bool = False,
    selected_edge_types: list[str] | None = None,
) -> go.Figure:
    if selected_edge_types is None:
        selected_edge_types = sorted(edges["type"].dropna().astype(str).unique().tolist())
    filtered_edges = edges[edges["type"].astype(str).isin(selected_edge_types)].copy()
    center_edge_mask = (edges["source"] == center_id) | (edges["target"] == center_id)
    center_edges = edges[center_edge_mask & edges["type"].astype(str).isin(selected_edge_types)].copy()
    display_edges = filtered_edges if show_full_edges else center_edges
    direct_neighbors_set = set(center_edges["source"].tolist()) | set(center_edges["target"].tolist())
    direct_neighbors_set.discard(center_id)
    visible_nodes = set(display_edges["source"].tolist()) | set(display_edges["target"].tolist()) | {center_id}

    node_types = nodes.set_index("id")["type"].to_dict()
    grouped_nodes: dict[str, list[object]] = {}
    second_hop_nodes = []
    for node_id in sorted(visible_nodes, key=str):
        if node_id == center_id:
            continue
        if node_id in direct_neighbors_set:
            grouped_nodes.setdefault(str(node_types.get(node_id, "unknown")), []).append(node_id)
        else:
            second_hop_nodes.append(node_id)

    positions = {center_id: (0.0, 0.0)}
    type_order = sorted(grouped_nodes, key=lambda item: (-len(grouped_nodes[item]), item))
    group_count = len(type_order)
    total_direct_nodes = sum(len(grouped_nodes[node_type]) for node_type in type_order)
    gap = 0.08 if group_count > 1 else 0
    usable_angle = max(0.1, 2 * math.pi - gap * group_count)
    cursor = -math.pi / 2
    for node_type in type_order:
        members = sorted(grouped_nodes[node_type], key=str)
        member_count = max(len(members), 1)
        spread = usable_angle * (member_count / max(total_direct_nodes, 1))
        start_angle = cursor
        cursor += spread + gap
        ring_count = max(1, math.ceil(math.sqrt(member_count / 2)))
        slots_per_ring = max(1, math.ceil(member_count / ring_count))
        for member_idx, node_id in enumerate(members):
            ring_idx = member_idx % ring_count
            slot_idx = member_idx // ring_count
            angle = start_angle + ((slot_idx + 0.5) / slots_per_ring) * spread
            radius = 1.15 + 0.28 * ring_idx
            positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    golden_angle = math.pi * (3 - math.sqrt(5))
    for idx, node_id in enumerate(sorted(second_hop_nodes, key=str)):
        angle = idx * golden_angle
        radius = 2.35 + 0.12 * (idx % 4)
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    edge_traces = []
    for edge_type, group in display_edges.groupby("type"):
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        for edge in group.to_dict(orient="records"):
            source = edge["source"]
            target = edge["target"]
            if source not in positions or target not in positions:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            x_values.extend([x0, x1, None])
            y_values.extend([y0, y1, None])
        edge_traces.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(
                    width=0.8 if show_full_edges else 1.8,
                    color=EDGE_TYPE_COLORS.get(edge_type, "#9ca3af"),
                ),
                opacity=0.35 if show_full_edges else 0.8,
                hoverinfo="skip",
                name=f"边：{edge_type}",
            )
        )

    node_x = []
    node_y = []
    labels = []
    colors = []
    sizes = []
    line_colors = []
    symbols = []
    text_labels = []
    for node in nodes.to_dict(orient="records"):
        node_id = node["id"]
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        node_x.append(x)
        node_y.append(y)
        suspect_note = "<br>官方可疑实体" if node_id in SUSPECT_ENTITIES else ""
        labels.append(f"{node['label']}<br>类型：{node['type']}{suspect_note}")
        color = NODE_TYPE_COLORS.get(node["type"], NODE_TYPE_COLORS["unknown"])
        colors.append(color if node_id in direct_neighbors_set or node_id == center_id else "#cbd5e1")
        sizes.append(30 if node_id == center_id else 15 if node_id in direct_neighbors_set else 9)
        if node_id == center_id:
            line_colors.append("#111827")
            symbols.append("circle")
            text_labels.append("当前实体")
        elif node_id in SUSPECT_ENTITIES:
            line_colors.append("#dc2626")
            symbols.append("diamond")
            text_labels.append("官方可疑")
        else:
            line_colors.append("#ffffff")
            symbols.append("circle")
            text_labels.append("")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, symbol=symbols, line=dict(width=2, color=line_colors)),
        text=text_labels,
        textposition="top center",
        hovertext=labels,
        hoverinfo="text",
        name="节点",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        legend_itemclick=False,
        legend_itemdoubleclick=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _axis_dimension(df: pd.DataFrame, column: str, label: str) -> dict:
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    minimum = float(values.min())
    maximum = float(values.max())
    median = float(values.median())
    if minimum == maximum:
        maximum = minimum + 1
    tick_values = [minimum, median, maximum]
    tick_text = [f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}" for value in tick_values]
    return {
        "label": label,
        "values": values,
        "range": [minimum, maximum],
        "tickvals": tick_values,
        "ticktext": tick_text,
    }


def parallel_coordinates_figure(features: pd.DataFrame, top_n: int = 20) -> go.Figure:
    axis_map = [
        ("fishhook_anomaly_score", "异常分"),
        ("component_vessel_ratio", "船只比例"),
        ("family_edges_1_2_hop", "家族边"),
        ("political_nodes_1_2_hop", "政治组织"),
        ("short_cycle_count", "短环路"),
        ("connected_suspect_count", "关联嫌疑"),
        ("vessel_links", "船只连接"),
        ("ownership_edges", "所有权边"),
        ("membership_edges", "成员边"),
    ]
    features = features.loc[:, ~features.columns.duplicated()].copy().head(top_n)
    available = [(column, label) for column, label in axis_map if column in features.columns]
    if not available:
        return go.Figure()

    score = pd.to_numeric(features["fishhook_anomaly_score"], errors="coerce").fillna(0)
    dimensions = [_axis_dimension(features, column, label) for column, label in available]
    fig = go.Figure(
        data=[
            go.Parcoords(
                line=dict(
                    color=score,
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="异常分", len=0.78),
                ),
                dimensions=dimensions,
                labelfont=dict(size=14, color="#111827"),
                tickfont=dict(size=11, color="#4b5563"),
            )
        ]
    )
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=70, t=70, b=50),
        font=dict(size=13),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


def evidence_sankey_figure(paths: pd.DataFrame, limit: int = 10) -> go.Figure:
    if paths.empty:
        return go.Figure()

    link_counts: dict[tuple[str, str], int] = {}
    node_depths: dict[str, int] = {}
    max_depth = 1
    for path_text in paths["path"].head(limit):
        parts = [part.strip() for part in str(path_text).split("->") if part.strip()]
        max_depth = max(max_depth, len(parts) - 1)
        for idx, part in enumerate(parts):
            node_depths[part] = min(idx, node_depths.get(part, idx))
        for source, target in zip(parts, parts[1:]):
            link_counts[(source, target)] = link_counts.get((source, target), 0) + 1

    labels = sorted({node for edge in link_counts for node in edge}, key=lambda node: (node_depths.get(node, 0), str(node)))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    layer_counts: dict[int, int] = {}
    for label in labels:
        layer_counts[node_depths.get(label, 0)] = layer_counts.get(node_depths.get(label, 0), 0) + 1
    layer_seen: dict[int, int] = {}
    node_x = []
    node_y = []
    node_colors = []
    suspect_labels = {str(item) for item in SUSPECT_ENTITIES}
    for label in labels:
        depth = node_depths.get(label, 0)
        layer_seen[depth] = layer_seen.get(depth, 0) + 1
        denominator = max(layer_counts.get(depth, 1) - 1, 1)
        node_x.append(depth / max(max_depth, 1))
        node_y.append((layer_seen[depth] - 1) / denominator if layer_counts.get(depth, 1) > 1 else 0.5)
        if str(label) in suspect_labels:
            node_colors.append("#dc2626")
        else:
            node_colors.append("#2563eb")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=24,
                    thickness=18,
                    line=dict(color="#ffffff", width=2),
                    label=[short_label(label, 34) for label in labels],
                    color=node_colors,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=labels,
                ),
                link=dict(
                    source=[label_to_index[source] for source, _ in link_counts],
                    target=[label_to_index[target] for _, target in link_counts],
                    value=list(link_counts.values()),
                    color="rgba(37, 99, 235, 0.28)",
                    hovertemplate="路径段：%{source.label} -> %{target.label}<br>出现次数：%{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        height=560,
        margin=dict(l=30, r=40, t=40, b=30),
        font=dict(size=15, family="Microsoft YaHei, Arial", color="#111827"),
        paper_bgcolor="#ffffff",
    )
    fig.add_annotation(
        x=0,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=14, color="#4b5563"),
        text="红色为官方可疑实体，蓝色为其他实体；流向表示路径方向。",
    )
    return fig


def path_hops_bar_figure(paths: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if paths.empty:
        return go.Figure()
    plot_df = paths.head(top_n).copy()
    plot_df["path_hops"] = pd.to_numeric(plot_df["path_hops"], errors="coerce").fillna(0)
    plot_df["label_text"] = plot_df["target"].map(lambda value: str(short_label(value, 34)))
    plot_df = plot_df.sort_values(["path_hops", "label_text"], ascending=[False, True])
    colors = plot_df["is_official_suspect"].map(lambda value: "#dc2626" if bool(value) else "#2563eb")
    fig = go.Figure(
        go.Bar(
            x=plot_df["path_hops"],
            y=plot_df["label_text"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.9)", width=1)),
            text=plot_df["path_hops"].map(lambda value: f"{int(value)}"),
            textposition="outside",
            customdata=plot_df[["target_kind", "path"]].values,
            hovertemplate=(
                "%{y}<br>"
                "类型：%{customdata[0]}<br>"
                "跳数：%{x}<br>"
                "路径：%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title=dict(text="路径跳数", x=0.02, font=dict(size=18)), showlegend=False)
    fig.update_xaxes(title="跳数", dtick=1, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(type="category", showgrid=False, categoryorder="array", categoryarray=plot_df["label_text"].tolist())
    return plot_layout(fig, height=max(360, 42 * len(plot_df) + 90))


def common_neighbor_heatmap(common: pd.DataFrame) -> go.Figure:
    if common.empty:
        return go.Figure()
    plot_df = common.copy()
    plot_df["common_neighbor_count"] = pd.to_numeric(plot_df["common_neighbor_count"], errors="coerce").fillna(0)
    plot_df = plot_df[plot_df["common_neighbor_count"] > 0].copy()
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="当前选择的实体之间没有共同邻居",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color=THEME["muted"]),
        )
        return plot_layout(fig, height=420)
    plot_df["pair_label"] = plot_df.apply(
        lambda row: f"{short_label(row['entity_a'], 18)}  ↔  {short_label(row['entity_b'], 18)}",
        axis=1,
    )
    plot_df = plot_df.sort_values("common_neighbor_count", ascending=True).tail(10)
    fig = go.Figure(
        go.Bar(
            x=plot_df["common_neighbor_count"],
            y=plot_df["pair_label"],
            orientation="h",
            marker=dict(
                color=plot_df["common_neighbor_count"],
                colorscale=[[0, "#dbeafe"], [1, "#2563eb"]],
                line=dict(color="#ffffff", width=1),
            ),
            text=plot_df["common_neighbor_count"].map(lambda value: f"{int(value)} 个"),
            textposition="outside",
            customdata=plot_df[["common_neighbor_types", "examples"]].values,
            hovertemplate=(
                "%{y}<br>"
                "共同邻居：%{x} 个<br>"
                "类型：%{customdata[0]}<br>"
                "示例：%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(title=dict(text="两两实体的一跳共同邻居数", x=0.02, font=dict(size=18)), showlegend=False)
    fig.update_xaxes(title="共同邻居数量", dtick=1, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(title="", showgrid=False)
    fig = plot_layout(fig, height=440)
    fig.update_layout(margin=dict(l=170, r=70, t=70, b=50))
    return fig


def common_neighbor_network_figure(common: pd.DataFrame) -> go.Figure:
    suspect_nodes = sorted(set(common["entity_a"].astype(str)) | set(common["entity_b"].astype(str)))
    neighbor_to_suspects: dict[str, set[str]] = {}
    neighbor_types: dict[str, str] = {}

    for row in common.to_dict(orient="records"):
        examples = [item.strip() for item in str(row.get("examples", "")).split(";") if item.strip()]
        for example in examples[:5]:
            neighbor = example.rsplit(" (", 1)[0]
            neighbor_type = "unknown"
            if " (" in example and example.endswith(")"):
                neighbor_type = example.rsplit(" (", 1)[1][:-1]
            neighbor_types[neighbor] = neighbor_type
            neighbor_to_suspects.setdefault(neighbor, set()).update([str(row["entity_a"]), str(row["entity_b"])])

    neighbor_nodes = sorted(
        neighbor_to_suspects,
        key=lambda item: (-len(neighbor_to_suspects[item]), str(item)),
    )[:18]

    positions = {}
    suspect_count = max(len(suspect_nodes) - 1, 1)
    for idx, node in enumerate(suspect_nodes):
        positions[node] = (0, 1 - 2 * idx / suspect_count)
    neighbor_count = max(len(neighbor_nodes) - 1, 1)
    for idx, node in enumerate(neighbor_nodes):
        positions[node] = (1.55, 1 - 2 * idx / neighbor_count if len(neighbor_nodes) > 1 else 0)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for target in neighbor_nodes:
        for source in sorted(neighbor_to_suspects.get(target, []), key=str):
            if source not in positions or target not in positions:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    suspect_x = []
    suspect_y = []
    suspect_text = []
    for node in suspect_nodes:
        x0, y0 = positions[node]
        suspect_x.append(x0)
        suspect_y.append(y0)
        suspect_text.append(short_label(node, 22))

    neighbor_x = []
    neighbor_y = []
    neighbor_text = []
    neighbor_hover = []
    neighbor_size = []
    neighbor_color = []
    for node in neighbor_nodes:
        x, y = positions[node]
        linked = sorted(neighbor_to_suspects.get(node, []), key=str)
        neighbor_x.append(x)
        neighbor_y.append(y)
        neighbor_text.append(short_label(node, 22))
        neighbor_hover.append(f"{node}<br>连接实体：{'; '.join(linked)}")
        neighbor_size.append(12 + 4 * len(linked))
        neighbor_color.append(NODE_TYPE_COLORS.get(neighbor_types.get(node, "unknown"), "#60a5fa"))

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#cbd5e1", width=1.4), hoverinfo="skip", name="共享连接"),
            go.Scatter(
                x=suspect_x,
                y=suspect_y,
                mode="markers+text",
                marker=dict(size=20, color="#ef4444", line=dict(color="#ffffff", width=2)),
                text=suspect_text,
                textposition="middle left",
                hovertext=suspect_nodes,
                hoverinfo="text",
                name="参与比较实体",
            ),
            go.Scatter(
                x=neighbor_x,
                y=neighbor_y,
                mode="markers+text",
                marker=dict(size=neighbor_size, color=neighbor_color, line=dict(color="#ffffff", width=2)),
                text=neighbor_text,
                textposition="middle right",
                hovertext=neighbor_hover,
                hoverinfo="text",
                name="共同邻居",
            ),
        ]
    )
    dynamic_height = max(440, 120 + max(len(suspect_nodes), len(neighbor_nodes)) * 34)
    fig.update_layout(
        height=dynamic_height,
        margin=dict(l=120, r=180, t=74, b=34),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=dict(text="共同邻居示例连接图", x=0.02, font=dict(size=18)),
        xaxis=dict(visible=False, range=[-0.42, 2.45]),
        yaxis=dict(visible=False, range=[-1.12, 1.12]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="VAST 2023 MC1 可视分析", layout="wide")
    page_style()
    st.title("VAST Challenge 2023 MC1 非法捕鱼知识图谱可视分析")
    st.markdown(
        '<div class="section-note">围绕官方可疑实体，进行上下文网络、FishHook 异常评分、候选扩展和证据链可视分析。</div>',
        unsafe_allow_html=True,
    )

    nodes, edges = cached_data()
    all_entities = nodes["id"].tolist()

    with st.sidebar:
        st.header("分析设置")
        preset = st.selectbox("官方可疑实体", SUSPECT_ENTITIES)
        manual = st.text_input("搜索其他实体")
        selected = preset
        if manual.strip():
            matches = find_entity_matches(manual, all_entities)
            if matches:
                selected = st.selectbox("选择匹配结果", matches, format_func=str)
            else:
                st.warning("没有找到匹配实体，仍显示官方选择项。")
        depth = st.slider("关系网络深度", min_value=1, max_value=2, value=1)
        max_nodes = st.slider("最多显示节点数", min_value=20, max_value=500, value=180, step=10)
        graph_mode = st.radio(
            "主图显示方式",
            ["简洁圆形图", "完整圆形图"],
            index=0,
            help="简洁圆形图只画当前实体的直接关系；完整圆形图保留当前深度内全部边，适合检查原始关系。",
        )
        evidence_depth = st.slider("证据路径最大深度", min_value=2, max_value=5, value=3)
        expansion_depth = st.slider("可疑实体扩展深度", min_value=1, max_value=3, value=2)

    index = build_graph_index(nodes, edges)
    stats = summarize_entity(nodes, edges, selected)
    fishhook_selected = fishhook_feature_table(nodes, edges, [selected])
    sub_nodes, sub_edges = ego_subgraph(nodes, edges, selected, depth=depth, max_nodes=max_nodes)
    direct_degree = len(direct_neighbors(index, selected))

    metric_cols = st.columns(4)
    metric_cols[0].metric("邻居数量", stats["degree"])
    metric_cols[1].metric("相关边数", stats["edge_count"])
    metric_cols[2].metric("平均边权重", stats["avg_weight"])
    metric_cols[3].metric("FishHook异常分数", fishhook_selected.iloc[0]["fishhook_anomaly_score"])

    if depth == 2 and max_nodes <= direct_degree + 1:
        st.warning(
            f"当前实体的一跳邻居已有 {direct_degree} 个，最多显示节点数为 {max_nodes}，"
            "所以二跳关系可能被截断。请把“最多显示节点数”调高后再观察深度 2。"
        )
    if len(sub_nodes) >= max_nodes:
        st.info(f"当前子图已经达到显示上限：{max_nodes} 个节点。")

    st.markdown(
        f"""
        <div class="viz-card">
          <span class="entity-chip">当前实体：{short_label(selected, 60)}</span>
          <span class="entity-chip">类型：{fishhook_selected.iloc[0].get("type", "unknown")}</span>
          <span class="entity-chip">FishHook：{fishhook_selected.iloc[0]["fishhook_anomaly_score"]}</span>
          <div class="small-muted">分数表示调查优先级，不等于违法定论。建议结合关系网络、路径证据和共同邻居一起判断。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.72, 1.08])
    with left:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.subheader("上下文关系网络")
        edge_type_options = sorted(sub_edges["type"].dropna().astype(str).unique().tolist())
        st.caption("关系类型图层：点击按钮切换显示/隐藏。隐藏某种关系后，对应边和只由这些边连接的节点会一起隐藏。")
        edge_state_key = "graph_edge_type_visible"
        current_edge_state = st.session_state.get(edge_state_key, {})
        edge_label_map = {
            "family_relationship": "family",
            "membership": "member",
            "ownership": "owner",
            "partnership": "partner",
            "unknown": "unknown",
        }
        edge_class_map = {
            "family_relationship": "edge-family",
            "membership": "edge-member",
            "ownership": "edge-owner",
            "partnership": "edge-partner",
            "unknown": "edge-unknown",
        }
        preferred_edge_order = ["family_relationship", "membership", "ownership", "partnership", "unknown"]
        edge_type_options = [edge_type for edge_type in preferred_edge_order if edge_type in edge_type_options] + [
            edge_type for edge_type in edge_type_options if edge_type not in preferred_edge_order
        ]
        edge_state = {
            edge_type: bool(current_edge_state.get(edge_type, True))
            for edge_type in edge_type_options
        }
        st.session_state[edge_state_key] = edge_state

        color_rules = []
        for edge_type in edge_type_options:
            visible = edge_state.get(edge_type, True)
            color = EDGE_TYPE_COLORS.get(edge_type, EDGE_TYPE_COLORS["unknown"])
            background = color if visible else "#f1f5f9"
            border = color if visible else "#cbd5e1"
            text = "#ffffff" if visible else "#64748b"
            key_class = "st-key-edge_toggle_" + re.sub(r"[^a-zA-Z0-9_]+", "_", edge_type)
            color_rules.append(
                f"""
                div.{key_class} div[data-testid="stButton"] > button {{
                    background: {background} !important;
                    border-color: {border} !important;
                    color: {text} !important;
                    min-height: 2.65rem;
                    font-size: 0.96rem;
                    font-weight: 760;
                }}
                div.{key_class} div[data-testid="stButton"] > button p {{
                    color: {text} !important;
                    font-size: 0.96rem;
                    font-weight: 760;
                }}
                """
            )
        st.markdown(
            f"""
            <style>
            div[data-testid="stElementContainer"]:has(.edge-control-anchor)
            + div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] {{
                gap: 0.55rem;
                align-items: center;
                margin: 0.25rem 0 0.5rem 0;
                flex-wrap: nowrap;
                overflow-x: auto;
                padding-bottom: 0.25rem;
            }}
            div[data-testid="stElementContainer"]:has(.edge-control-anchor)
            + div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                flex: 0 0 118px;
                min-width: 118px;
                width: 118px;
            }}
            div[data-testid="stElementContainer"]:has(.edge-control-anchor)
            + div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(n+3) {{
                flex-basis: 132px;
                min-width: 132px;
                width: 132px;
            }}
            div[data-testid="stElementContainer"]:has(.edge-control-anchor)
            + div[data-testid="stLayoutWrapper"] > div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {{
                min-height: 2.65rem;
                border-radius: 999px;
                font-size: 0.96rem;
                font-weight: 720;
            }}
            div.st-key-edge_show_all div[data-testid="stButton"] > button {{
                background: #2563eb !important;
                border-color: #2563eb !important;
                color: #ffffff !important;
            }}
            div.st-key-edge_show_all div[data-testid="stButton"] > button p {{
                color: #ffffff !important;
            }}
            div.st-key-edge_hide_all div[data-testid="stButton"] > button {{
                background: #64748b !important;
                border-color: #64748b !important;
                color: #ffffff !important;
            }}
            div.st-key-edge_hide_all div[data-testid="stButton"] > button p {{
                color: #ffffff !important;
            }}
            {''.join(color_rules)}
            </style>
            <div class="edge-control-anchor"></div>
            """,
            unsafe_allow_html=True,
        )

        control_columns = st.columns([1.05, 1.05] + [1.18] * len(edge_type_options), gap="small")
        with control_columns[0]:
            st.button(
                "全部显示",
                key="edge_show_all",
                use_container_width=True,
                on_click=set_edge_type_visibility,
                args=(edge_state_key, edge_type_options, True),
            )
        with control_columns[1]:
            st.button(
                "全部隐藏",
                key="edge_hide_all",
                use_container_width=True,
                on_click=set_edge_type_visibility,
                args=(edge_state_key, edge_type_options, False),
            )
        for idx, edge_type in enumerate(edge_type_options, start=2):
            label = edge_label_map.get(edge_type, short_label(edge_type, 10))
            marker_class = "edge-control-" + re.sub(r"[^a-zA-Z0-9_]+", "_", edge_type)
            with control_columns[idx]:
                st.markdown(f'<span class="edge-control-token {marker_class}"></span>', unsafe_allow_html=True)
                st.button(
                    str(label),
                    key=f"edge_toggle_{re.sub(r'[^a-zA-Z0-9_]+', '_', edge_type)}",
                    use_container_width=True,
                    on_click=toggle_edge_type_visibility,
                    args=(edge_state_key, edge_type),
                )

        edge_state = {
            edge_type: bool(st.session_state[edge_state_key].get(edge_type, True))
            for edge_type in edge_type_options
        }
        st.session_state[edge_state_key] = edge_state
        selected_graph_edge_types = [
            edge_type
            for edge_type in edge_type_options
            if edge_state.get(edge_type, True)
        ]
        if graph_mode == "简洁圆形图":
            st.caption("该图保留原来的圆形节点网络形式，只显示当前实体的直接关系；二跳节点作为外层背景点，减少交叉线。")
            st.plotly_chart(
                simplified_network_figure(
                    sub_nodes,
                    sub_edges,
                    selected,
                    selected_edge_types=selected_graph_edge_types,
                ),
                use_container_width=True,
            )
        else:
            st.caption("完整圆形图会显示当前深度内的全部边，数据最完整，但节点多时会比较拥挤。")
            st.plotly_chart(
                simplified_network_figure(
                    sub_nodes,
                    sub_edges,
                    selected,
                    show_full_edges=True,
                    selected_edge_types=selected_graph_edge_types,
                ),
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        st.subheader("实体摘要")
        st.plotly_chart(donut_figure(stats["neighbor_types"], "邻居类型分布", NODE_TYPE_COLORS), use_container_width=True)
        st.markdown('<div class="summary-chart-gap"></div>', unsafe_allow_html=True)
        st.plotly_chart(donut_figure(stats["edge_types"], "关系类型分布", EDGE_TYPE_COLORS), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("当前实体 FishHook 特征")
    feature_cols = st.columns([1, 1])
    with feature_cols[0]:
        st.plotly_chart(feature_strip_figure(fishhook_selected), use_container_width=True)
    with feature_cols[1]:
        st.plotly_chart(feature_radar_figure(fishhook_selected.iloc[0]), use_container_width=True)
    with st.expander("查看当前实体特征明细"):
        st.dataframe(fishhook_selected, use_container_width=True)

    (
        tab_suspects,
        tab_evidence,
        tab_neighbors,
        tab_expansion,
        tab_ranking,
        tab_parallel,
        tab_recommend,
        tab_subgraph,
    ) = st.tabs(
        [
            "已知可疑",
            "证据路径",
            "共同邻居",
            "扩展候选",
            "FishHook排名",
            "平行对比",
            "推荐公司",
            "子图数据",
        ]
    )

    with tab_ranking:
        st.caption("综合考虑：靠近官方可疑实体、弱连通分量船只比例、1-2跳家族/政治关系、短环路和结构连接。")
        ranking = cached_fishhook_ranking()
        type_options = sorted(ranking["type"].fillna("unknown").astype(str).unique().tolist())
        selected_types = st.multiselect(
            "候选实体类型",
            type_options,
            default=[item for item in ["company", "vessel", "person", "organization"] if item in type_options],
        )
        show_ranking = ranking
        if selected_types:
            show_ranking = show_ranking[show_ranking["type"].isin(selected_types)]
        display_cols = [
            "candidate",
            "type",
            "fishhook_anomaly_score",
            "connected_suspect_count",
            "min_distance_to_suspect",
            "component_vessel_ratio",
            "short_cycle_count",
            "vessel_links",
            "path_examples",
        ]
        chart_cols = st.columns([1.35, 1])
        with chart_cols[0]:
            st.plotly_chart(
                horizontal_bar_figure(
                    show_ranking,
                    "candidate",
                    "fishhook_anomaly_score",
                    "FishHook 候选排名 Top 10",
                    THEME["blue"],
                    top_n=10,
                ),
                use_container_width=True,
            )
        with chart_cols[1]:
            type_counts = show_ranking["type"].fillna("unknown").value_counts().to_dict()
            st.plotly_chart(donut_figure(type_counts, "候选实体类型构成", NODE_TYPE_COLORS), use_container_width=True)
        with st.expander("查看候选排名明细"):
            st.dataframe(show_ranking[[col for col in display_cols if col in show_ranking.columns]], use_container_width=True)

    with tab_parallel:
        st.caption("每条折线代表一个候选实体。高分实体如果在多个轴上同时偏高，说明它具有多种异常结构特征。")
        ranking = cached_fishhook_ranking()
        top_n = st.slider("平行坐标显示实体数", min_value=8, max_value=40, value=20, step=4)
        features = ranking.rename(columns={"candidate": "entity"}).loc[:, lambda df: ~df.columns.duplicated()].copy()
        st.plotly_chart(parallel_coordinates_figure(features, top_n=top_n), use_container_width=True)
        with st.expander("查看平行坐标对应数据"):
            st.dataframe(
                features[
                    [
                        "entity",
                        "type",
                        "fishhook_anomaly_score",
                        "component_vessel_ratio",
                        "family_edges_1_2_hop",
                        "political_nodes_1_2_hop",
                        "short_cycle_count",
                        "connected_suspect_count",
                        "cycle_examples",
                        "suspect_path_examples",
                    ]
                ].head(top_n),
                use_container_width=True,
            )

    with tab_expansion:
        st.caption("从官方 4 个可疑实体出发，寻找 1-3 跳内的其他实体。")
        candidates = cached_suspect_expansion(expansion_depth)
        type_options = sorted(candidates["type"].dropna().astype(str).unique().tolist())
        selected_types = st.multiselect(
            "扩展候选类型",
            type_options,
            default=[item for item in ["company", "vessel", "person", "organization"] if item in type_options],
            key="expansion_types",
        )
        if selected_types:
            candidates = candidates[candidates["type"].isin(selected_types)]
        if not candidates.empty and "min_distance_to_suspect" in candidates.columns:
            distance_counts = (
                pd.to_numeric(candidates["min_distance_to_suspect"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
            )
            st.caption(
                "当前筛选后候选距离分布："
                + "；".join(f"{distance}跳 {count} 个" for distance, count in distance_counts.items())
            )
        st.plotly_chart(candidate_bubble_figure(candidates), use_container_width=True)
        expand_cols = st.columns([1, 1])
        with expand_cols[0]:
            st.plotly_chart(
                horizontal_bar_figure(
                    candidates,
                    "candidate",
                    "connected_suspect_count",
                    "连接官方可疑实体数量 Top 10",
                    THEME["green"],
                    top_n=10,
                ),
                use_container_width=True,
            )
        with expand_cols[1]:
            st.plotly_chart(donut_figure(candidates["type"].fillna("unknown").value_counts().to_dict(), "扩展候选类型构成", NODE_TYPE_COLORS), use_container_width=True)
        with st.expander("查看扩展候选明细"):
            st.dataframe(candidates, use_container_width=True)

    with tab_evidence:
        st.caption(
            "证据路径来自知识图谱里的最短路径搜索：先显示当前实体到非法捕鱼关键词节点的路径；"
            "如果在同一深度范围内能到达官方可疑实体，也会一起显示。路径只是辅助线索，不是单独证据。"
        )
        keyword_paths = keyword_evidence_paths(nodes, edges, selected, max_depth=evidence_depth, limit=20)
        suspect_paths = official_suspect_paths(nodes, edges, selected, max_depth=evidence_depth)

        combined_parts = []
        if not keyword_paths.empty:
            keyword_display = keyword_paths.rename(
                columns={
                    "target_keyword_node": "target",
                    "path_length": "path_hops",
                }
            ).copy()
            keyword_display["target_kind"] = "关键词节点"
            keyword_display["is_official_suspect"] = False
            combined_parts.append(
                keyword_display[
                    [
                        "target",
                        "target_kind",
                        "target_type",
                        "path_hops",
                        "path",
                        "relation_steps",
                        "is_official_suspect",
                    ]
                ]
            )
        if not suspect_paths.empty:
            suspect_display = suspect_paths.rename(columns={"target_suspect": "target"}).copy()
            suspect_display["target_kind"] = "官方可疑实体"
            suspect_display["is_official_suspect"] = True
            combined_parts.append(
                suspect_display[
                    [
                        "target",
                        "target_kind",
                        "target_type",
                        "path_hops",
                        "path",
                        "relation_steps",
                        "is_official_suspect",
                    ]
                ]
            )

        if not combined_parts:
            st.info("在当前最大深度内没有找到关键词节点或官方可疑实体路径。")
        else:
            combined_paths = (
                pd.concat(combined_parts, ignore_index=True)
                .sort_values(["path_hops", "is_official_suspect", "target"], ascending=[True, False, True])
                .reset_index(drop=True)
            )
            shown_keyword_paths = combined_paths[~combined_paths["is_official_suspect"]].head(10)
            shown_suspect_paths = combined_paths[combined_paths["is_official_suspect"]]
            shown_paths = pd.concat([shown_keyword_paths, shown_suspect_paths], ignore_index=True)
            path_cols = st.columns([1.5, 1])
            with path_cols[0]:
                st.plotly_chart(evidence_sankey_figure(shown_paths, limit=len(shown_paths)), use_container_width=True)
            with path_cols[1]:
                st.plotly_chart(path_hops_bar_figure(shown_paths, top_n=len(shown_paths)), use_container_width=True)
            st.caption("红色节点是官方可疑实体，蓝色节点是其他实体或关键词目标。")
            with st.expander("查看路径明细"):
                st.dataframe(
                    combined_paths.drop(columns=["is_official_suspect"]),
                    use_container_width=True,
                )

    with tab_neighbors:
        if selected in SUSPECT_ENTITIES:
            st.caption(
                "共同邻居=两个实体都直接连接到的同一个一跳节点。"
                "当前选择的是官方可疑实体，因此这里比较官方4个可疑实体之间共享了哪些中介、组织、地点或船只。"
            )
        else:
            st.caption(
                "共同邻居=两个实体都直接连接到的同一个一跳节点。"
                "当前选择的不是官方可疑实体，因此这里只比较当前实体分别与官方4个可疑实体共享了哪些邻居。"
            )
        comparison_entities = list(SUSPECT_ENTITIES)
        if selected not in comparison_entities:
            comparison_entities = [selected] + comparison_entities
        overlap = common_neighbor_summary(nodes, edges, comparison_entities)
        if selected not in SUSPECT_ENTITIES and not overlap.empty:
            overlap = overlap[
                (overlap["entity_a"] == selected) | (overlap["entity_b"] == selected)
            ].copy()
        col1, col2 = st.columns([0.92, 1.18])
        with col1:
            st.plotly_chart(common_neighbor_heatmap(overlap), use_container_width=True)
        with col2:
            st.plotly_chart(common_neighbor_network_figure(overlap), use_container_width=True)
        st.caption(
            "左图显示每一对实体共享的一跳邻居数量；右图左侧是参与比较的实体，右侧是共同邻居示例，连线表示该邻居同时连接到这些实体。"
        )
        with st.expander("查看共同邻居明细"):
            st.dataframe(overlap, use_container_width=True)

    with tab_suspects:
        st.caption(
            "横轴是官方给出的 4 个可疑实体；颜色代表对比指标。"
            "用它快速看出每个实体的异常分、船只连接、短环路和所有权关系差异。"
        )
        suspect_features = suspect_summary(nodes, edges)
        suspect_display_cols = [
            "entity",
            "type",
            "fishhook_anomaly_score",
            "connected_suspect_count",
            "min_suspect_distance",
            "vessel_links",
            "ownership_edges",
            "membership_edges",
            "short_cycle_count",
            "family_edges_1_2_hop",
            "political_nodes_1_2_hop",
        ]
        suspect_column_names = {
            "entity": "实体",
            "type": "类型",
            "fishhook_anomaly_score": "FishHook异常分",
            "connected_suspect_count": "可达其他官方可疑数",
            "min_suspect_distance": "最近官方可疑距离",
            "vessel_links": "一跳船只数",
            "ownership_edges": "所有权边",
            "membership_edges": "成员边",
            "short_cycle_count": "短环路数",
            "family_edges_1_2_hop": "1-2跳家族关系",
            "political_nodes_1_2_hop": "1-2跳政治组织",
        }
        st.plotly_chart(suspect_compare_figure(suspect_features), use_container_width=True)
        st.caption(
            "指标含义：FishHook分=综合调查优先级；一跳船只=直接相连的船只数量；"
            "短环路=关系网络中反复闭合的可疑结构；所有权关系=直接涉及的 ownership 边。"
        )
        with st.expander("查看官方可疑实体特征明细"):
            st.dataframe(
                suspect_features[[col for col in suspect_display_cols if col in suspect_features.columns]].rename(
                    columns=suspect_column_names
                ),
                use_container_width=True,
            )

    with tab_recommend:
        st.caption("该表从 FishHook 候选排名中筛选 company 类型实体，推荐顺序统一由 FishHook 异常分数决定。")
        company_cols = [
            "candidate",
            "type",
            "fishhook_anomaly_score",
            "connected_suspect_count",
            "min_distance_to_suspect",
            "short_cycle_count",
            "vessel_links",
            "path_examples",
        ]
        company_recommendations = recommend_companies(nodes, edges, limit=20)
        rec_cols = st.columns([1.4, 1])
        with rec_cols[0]:
            st.plotly_chart(
                horizontal_bar_figure(
                    company_recommendations,
                    "candidate",
                    "fishhook_anomaly_score",
                    "推荐调查公司 Top 10",
                    THEME["purple"],
                    top_n=10,
                ),
                use_container_width=True,
            )
        with rec_cols[1]:
            st.plotly_chart(
                horizontal_bar_figure(
                    company_recommendations,
                    "candidate",
                    "short_cycle_count",
                    "推荐公司短环路对比",
                    THEME["red"],
                    top_n=10,
                ),
                use_container_width=True,
            )
        with st.expander("查看推荐公司明细"):
            st.dataframe(
                company_recommendations[[col for col in company_cols if col in company_recommendations.columns]],
                use_container_width=True,
        )

    with tab_subgraph:
        st.write("节点")
        st.dataframe(sub_nodes, use_container_width=True)
        st.write("边")
        st.dataframe(sub_edges, use_container_width=True)


if __name__ == "__main__":
    main()
