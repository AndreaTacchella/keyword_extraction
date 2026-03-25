"""cluster_candidates.py — run two-stage candidate clustering.

Usage (from keyword_extraction/):
    python cluster_candidates.py

Reads:
    outputs/candidates_list.csv  (from candidate_extractor.py)

Writes:
    outputs/candidate_clusters.csv

Config:
    ranking_config.json — "clustering" section
"""

if __name__ == "__main__":
    import json
    import logging
    import time
    from pathlib import Path

    import pandas as pd
    import spacy

    from ranking.loader import load_candidates_list
    from ranking.embedder import get_embedder
    from ranking.clustering import cluster_candidates

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open("ranking_config.json") as f:
        cfg = json.load(f)

    clust_cfg = cfg.get("clustering", {})
    sem_cfg   = cfg.get("semantic", {})
    col_map   = cfg.get("candidates_list_columns", {
        "candidate_id": "candidate_id",
        "candidate": "candidate",
        "surface_form": "surface_form",
    })

    # ── Load candidates ───────────────────────────────────────────────────────
    clist_path = cfg["candidates_list_path"]
    candidates_list = load_candidates_list(clist_path, col_map)
    print(f"Loaded {len(candidates_list):,} candidates from {clist_path}")

    # ── Load spaCy with lemmatizer enabled ────────────────────────────────────
    spacy_model = clust_cfg.get("spacy_model", "en_core_web_sm")
    print(f"Loading spaCy model: {spacy_model} (lemmatizer enabled)")
    nlp = spacy.load(spacy_model, disable=["ner", "parser"])

    # ── Load embedder ─────────────────────────────────────────────────────────
    embedder = get_embedder(
        model_name=sem_cfg.get("model_name", "all-MiniLM-L6-v2"),
        backend=sem_cfg.get("backend", "sentence_transformers"),
    )

    # ── Run clustering ────────────────────────────────────────────────────────
    threshold      = clust_cfg.get("semantic_threshold", 0.92)
    max_block_size = clust_cfg.get("max_block_size", 2000)
    batch_size     = clust_cfg.get("batch_size", 64)
    print(f"\nRunning clustering (threshold={threshold}, max_block_size={max_block_size}) ...")

    t0 = time.time()
    clustered = cluster_candidates(
        candidates_list,
        embedder=embedder,
        nlp=nlp,
        semantic_threshold=threshold,
        max_block_size=max_block_size,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(clust_cfg.get("output_path", "outputs/candidate_clusters.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clustered[["candidate_id", "candidate", "cluster_id", "cluster_label", "cluster_size"]].to_csv(
        out_path, index=False
    )
    print(f"Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== CLUSTER SIZE DISTRIBUTION ===")
    size_dist = (
        clustered.groupby("cluster_size")
        .size()
        .rename("n_clusters")
        .reset_index()
    )
    size_dist["n_candidates"] = size_dist["cluster_size"] * size_dist["n_clusters"]
    print(size_dist.to_string(index=False))

    print(f"\nTotal clusters : {clustered['cluster_id'].nunique():,}")
    print(f"Singletons     : {(clustered['cluster_size'] == 1).sum():,}")
    print(f"Multi-member   : {(clustered['cluster_size'] > 1).sum():,} candidates in non-singleton clusters")

    # ── Spot-check ────────────────────────────────────────────────────────────
    print("\n=== SAMPLE: 'battery' clusters (cluster_size > 1) ===")
    battery = clustered[
        clustered["candidate"].str.contains(r"^batter", na=False)
        & (clustered["cluster_size"] > 1)
    ].sort_values(["cluster_id", "candidate"])
    if len(battery):
        print(battery[["candidate", "cluster_id", "cluster_label", "cluster_size"]].head(30).to_string(index=False))
    else:
        print("  (no multi-member battery clusters at this threshold)")

    print("\n=== SAMPLE: largest clusters ===")
    largest = (
        clustered[clustered["cluster_size"] > 1]
        .sort_values("cluster_size", ascending=False)
        .drop_duplicates("cluster_id")
        .head(10)
    )
    print(largest[["cluster_label", "cluster_size"]].to_string(index=False))
