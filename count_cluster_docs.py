"""count_cluster_docs.py — count distinct documents per cluster.

For each cluster, counts how many distinct row_ids contain at least one
candidate that belongs to that cluster.

Usage (from keyword_extraction/):
    python count_cluster_docs.py

Reads:
    outputs/candidate_clusters.csv   (from cluster_candidates.py)
    outputs/candidates_mapping.csv   (from candidate_extractor.py)

Writes:
    outputs/cluster_doc_counts.csv

Config:
    ranking_config.json — "clustering.output_path",
                          "clustering.doc_counts_path",
                          "candidates_mapping_path"
"""

if __name__ == "__main__":
    import json
    from pathlib import Path

    import pandas as pd

    with open("ranking_config.json") as f:
        rcfg = json.load(f)

    clust_cfg    = rcfg.get("clustering", {})
    clusters_path = clust_cfg.get("output_path",      "outputs/candidate_clusters.csv")
    counts_path   = clust_cfg.get("doc_counts_path",  "outputs/cluster_doc_counts.csv")
    mapping_path  = rcfg.get("candidates_mapping_path", "outputs/candidates_mapping.csv")

    clusters = pd.read_csv(clusters_path)   # candidate_id, candidate, cluster_id, cluster_label, cluster_size
    mapping  = pd.read_csv(mapping_path)    # candidate_id, row_id

    # Join mapping → clusters, then count distinct row_ids per cluster
    merged = mapping.merge(clusters[["candidate_id", "cluster_id"]], on="candidate_id", how="inner")
    doc_counts = (
        merged.groupby("cluster_id")["row_id"]
        .nunique()
        .rename("doc_count")
        .reset_index()
    )

    # Attach cluster_label and cluster_size for readability
    meta = clusters[["cluster_id", "cluster_label", "cluster_size"]].drop_duplicates("cluster_id")
    result = meta.merge(doc_counts, on="cluster_id").sort_values("doc_count", ascending=False)

    out = Path(counts_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Saved → {out}  ({len(result):,} clusters)")

    print("\n=== TOP 20 CLUSTERS BY DOCUMENT COUNT ===")
    print(result.head(20).to_string(index=False))
