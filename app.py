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
    target_nodes: set[str] = set()
    for path_text in paths["path"].head(limit):
        parts = [part.strip() for part in str(path_text).split("->") if part.strip()]
        max_depth = max(max_depth, len(parts) - 1)
        if parts:
            target_nodes.add(parts[-1])
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
    for label in labels:
        depth = node_depths.get(label, 0)
        layer_seen[depth] = layer_seen.get(depth, 0) + 1
        denominator = max(layer_counts.get(depth, 1) - 1, 1)
        node_x.append(depth / max(max_depth, 1))
        node_y.append((layer_seen[depth] - 1) / denominator if layer_counts.get(depth, 1) > 1 else 0.5)
        if label in SUSPECT_ENTITIES:
            node_colors.append("#dc2626")
        elif label in target_nodes:
            node_colors.append("#f59e0b")
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
        text="红色为官方可疑实体，蓝色为中间实体，橙色为关键词相关目标节点；流向表示证据路径方向。",
    )
    return fig


def common_neighbor_heatmap(common: pd.DataFrame) -> go.Figure:
    entities = sorted(set(common["entity_a"].astype(str)) | set(common["entity_b"].astype(str)))
    matrix = pd.DataFrame(0, index=entities, columns=entities)
    for row in common.to_dict(orient="records"):
        a = str(row["entity_a"])
        b = str(row["entity_b"])
        count = int(row["common_neighbor_count"])
        matrix.loc[a, b] = count
        matrix.loc[b, a] = count
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[short_label(item, 24) for item in matrix.columns],
            y=[short_label(item, 24) for item in matrix.index],
            colorscale="Reds",
            text=matrix.values,
            texttemplate="%{text}",
            hovertemplate="%{y} 与 %{x}<br>共同邻居：%{z}<extra></extra>",
        )
    )
    fig.update_layout(height=420, margin=dict(l=130, r=30, t=30, b=90))
    return fig


def common_neighbor_network_figure(common: pd.DataFrame) -> go.Figure:
    suspect_nodes = sorted(set(common["entity_a"].astype(str)) | set(common["entity_b"].astype(str)))
    neighbor_nodes: list[str] = []
    edges: list[tuple[str, str]] = []

    for row in common.to_dict(orient="records"):
        examples = [item.strip() for item in str(row.get("examples", "")).split(";") if item.strip()]
        for example in examples[:5]:
            neighbor = example.rsplit(" (", 1)[0]
            neighbor_nodes.append(neighbor)
            edges.append((str(row["entity_a"]), neighbor))
            edges.append((str(row["entity_b"]), neighbor))

    neighbor_nodes = sorted(set(neighbor_nodes), key=str)[:30]
    allowed = set(suspect_nodes) | set(neighbor_nodes)
    edges = [(source, target) for source, target in edges if source in allowed and target in allowed]

    positions = {}
    for idx, node in enumerate(suspect_nodes):
        positions[node] = (0, -idx)
    for idx, node in enumerate(neighbor_nodes):
        positions[node] = (1.6, -idx * max(1, len(suspect_nodes) / max(1, len(neighbor_nodes))))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in edges:
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    for node, (x, y) in positions.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(short_label(node, 22))
        is_suspect = node in suspect_nodes
        node_color.append("#dc2626" if is_suspect else "#60a5fa")
        node_size.append(18 if is_suspect else 12)

    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#cbd5e1", width=1), hoverinfo="skip"),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=node_size, color=node_color, line=dict(color="#ffffff", width=1)),
                text=node_text,
                textposition="middle right",
                hovertext=list(positions.keys()),
                hoverinfo="text",
            ),
        ]
    )
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="VAST 2023 MC1 可视分析", layout="wide")
    st.title("VAST Challenge 2023 MC1 非法捕鱼知识图谱可视分析")

    nodes, edges = cached_data()
    all_entities = nodes["id"].tolist()

    with st.sidebar:
        st.header("分析设置")
        preset = st.selectbox("官方可疑实体", SUSPECT_ENTITIES)
        manual = st.text_input("搜索其他实体")
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

    selected = preset
    if manual.strip():
        matches = find_entity_matches(manual, all_entities)
        if matches:
            selected = st.sidebar.selectbox("选择匹配结果", matches, format_func=str)
        else:
            st.sidebar.warning("没有找到匹配实体，仍显示官方选择项。")

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

    left, right = st.columns([2, 1])
    with left:
        st.subheader("上下文关系网络")
        edge_type_options = sorted(sub_edges["type"].dropna().astype(str).unique().tolist())
        selected_graph_edge_types = st.multiselect(
            "显示关系类型",
            edge_type_options,
            default=edge_type_options,
            help="取消某种关系类型后，对应边和只由这些边连接的节点会一起隐藏。",
        )
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

    with right:
        st.subheader("实体摘要")
        st.write("邻居类型分布")
        st.dataframe(pd.DataFrame(stats["neighbor_types"].items(), columns=["类型", "数量"]))
        st.write("关系类型分布")
        st.dataframe(pd.DataFrame(stats["edge_types"].items(), columns=["关系", "数量"]))

    st.subheader("当前实体 FishHook 特征")
    st.dataframe(fishhook_selected, use_container_width=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "FishHook候选排名",
            "平行坐标对比",
            "从可疑实体扩展",
            "证据路径",
            "共同邻居分析",
            "官方可疑实体",
            "推荐调查公司",
            "当前子图数据",
        ]
    )

    with tab1:
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
        st.dataframe(show_ranking[[col for col in display_cols if col in show_ranking.columns]], use_container_width=True)

    with tab2:
        st.caption("每条折线代表一个候选实体。高分实体如果在多个轴上同时偏高，说明它具有多种异常结构特征。")
        ranking = cached_fishhook_ranking()
        top_n = st.slider("平行坐标显示实体数", min_value=8, max_value=40, value=20, step=4)
        features = ranking.rename(columns={"candidate": "entity"}).loc[:, lambda df: ~df.columns.duplicated()].copy()
        st.plotly_chart(parallel_coordinates_figure(features, top_n=top_n), use_container_width=True)
        with st.expander("这个图怎么看？", expanded=True):
            st.write(
                "平行坐标用于比较多个候选实体的异常模式。横向每一根轴是一个指标，例如异常分、短环路、"
                "关联官方嫌疑实体数量、船只连接、所有权边等；一条线是一名候选实体。"
                "如果某条线在多个关键轴上都靠上，说明它不只是单项指标高，而是在多个结构维度上都可疑。"
                "这适合用来挑选优先调查对象，而不是直接证明违法。"
            )
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

    with tab3:
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
        st.dataframe(candidates, use_container_width=True)

    with tab4:
        st.caption("当前实体到非法捕鱼相关关键词节点的短路径。关键词路径只是辅助线索，不是单独证据。")
        evidence_paths = keyword_evidence_paths(nodes, edges, selected, max_depth=evidence_depth, limit=20)
        if evidence_paths.empty:
            st.info("在当前最大深度内没有找到关键词证据路径。")
        else:
            st.plotly_chart(evidence_sankey_figure(evidence_paths), use_container_width=True)
            with st.expander("这个证据路径图怎么看？", expanded=True):
                st.write(
                    "这个图从左到右读。最左侧通常是当前分析实体或官方可疑实体，中间是路径经过的实体，"
                    "最右侧是命中非法捕鱼相关关键词的目标节点。连线表示路径中的一步关系，线越粗表示同一段路径在多条证据路径中出现次数越多。"
                    "它适合说明“该实体通过哪些中间对象接近非法捕鱼语义线索”，但关键词路径只是辅助证据，"
                    "需要结合 ownership、membership、船只连接、共同邻居和 FishHook 候选排名一起判断。"
                )
            st.dataframe(evidence_paths, use_container_width=True)

    with tab5:
        st.caption("共同邻居用于发现可疑实体是否共享中介、组织或地点。颜色越深，两个实体共享邻居越多。")
        comparison_entities = list(SUSPECT_ENTITIES)
        if selected not in comparison_entities:
            comparison_entities = [selected] + comparison_entities
        overlap = common_neighbor_summary(nodes, edges, comparison_entities)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(common_neighbor_heatmap(overlap), use_container_width=True)
        with col2:
            st.plotly_chart(common_neighbor_network_figure(overlap), use_container_width=True)
        st.dataframe(overlap, use_container_width=True)

    with tab6:
        st.caption(
            "该表用于解释官方给出的 4 个已知可疑实体在 FishHook 特征上的差异。"
            "其中“连通分量”类字段是整片网络组件的属性，不是单个实体独有属性。"
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
            "component_id",
            "component_size",
            "component_vessel_ratio",
            "vessel_ratio_delta",
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
            "component_id": "连通分量编号",
            "component_size": "所在分量节点数",
            "component_vessel_ratio": "分量船只比例",
            "vessel_ratio_delta": "船只比例偏离",
        }
        st.dataframe(
            suspect_features[[col for col in suspect_display_cols if col in suspect_features.columns]].rename(
                columns=suspect_column_names
            ),
            use_container_width=True,
        )
        with st.expander("为什么有些字段四个实体都相同？", expanded=True):
            st.write(
                "因为这些字段描述的是“实体所在的弱连通分量”，不是实体本身。"
                "如果四个官方可疑实体都落在同一个大连通分量里，那么它们的连通分量编号、所在分量节点数、"
                "分量船只比例、船只比例偏离就会完全相同。"
                "真正用于区分四个实体的是 FishHook异常分、一跳船只数、所有权边、成员边、短环路数、"
                "1-2跳家族关系、1-2跳政治组织、到其他官方可疑实体的距离等实体级字段。"
            )

    with tab7:
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
        st.dataframe(
            company_recommendations[[col for col in company_cols if col in company_recommendations.columns]],
            use_container_width=True,
        )

    with tab8:
        st.write("节点")
        st.dataframe(sub_nodes, use_container_width=True)
        st.write("边")
        st.dataframe(sub_edges, use_container_width=True)


if __name__ == "__main__":
    main()
