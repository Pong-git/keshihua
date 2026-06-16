from __future__ import annotations

from .config import OUTPUT_DIR
from .data_loader import load_nodes_edges
from .config import SUSPECT_ENTITIES
from .risk_model import (
    common_neighbor_summary,
    fishhook_candidate_ranking,
    fishhook_feature_table,
    keyword_evidence_paths,
    recommend_companies,
    suspect_expansion_candidates,
    suspect_summary,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_nodes_edges()

    nodes.to_csv(OUTPUT_DIR / "nodes.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(OUTPUT_DIR / "edges.csv", index=False, encoding="utf-8-sig")
    expansion = suspect_expansion_candidates(nodes, edges, max_depth=2, limit=120)
    expansion.to_csv(
        OUTPUT_DIR / "suspect_expansion_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fishhook_features = fishhook_feature_table(nodes, edges, expansion["candidate"].tolist(), limit=120)
    fishhook_features.to_csv(
        OUTPUT_DIR / "fishhook_anomaly_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fishhook_candidate_ranking(nodes, edges, limit=120, candidates=expansion, features=fishhook_features).to_csv(
        OUTPUT_DIR / "fishhook_candidate_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    suspect_summary(nodes, edges).to_csv(
        OUTPUT_DIR / "suspect_summary.csv", index=False, encoding="utf-8-sig"
    )
    recommend_companies(nodes, edges, limit=30).to_csv(
        OUTPUT_DIR / "company_fishhook_ranking.csv", index=False, encoding="utf-8-sig"
    )
    common_neighbor_summary(nodes, edges, SUSPECT_ENTITIES).to_csv(
        OUTPUT_DIR / "suspect_common_neighbors.csv", index=False, encoding="utf-8-sig"
    )
    for entity in SUSPECT_ENTITIES:
        safe_name = str(entity).replace("/", "_").replace("\\", "_").replace(" ", "_")
        keyword_evidence_paths(nodes, edges, entity, max_depth=3, limit=20).to_csv(
            OUTPUT_DIR / f"evidence_paths_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"Exported analysis tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
