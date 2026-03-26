"""run_pipeline.py — run the full keyword extraction pipeline end-to-end.

Usage (from keyword_extraction/):
    python run_pipeline.py

Stages
------
    1. Extract   — candidate_extractor.py  → outputs/candidates_list.csv
                                             outputs/candidates_mapping.csv
    2. Rank      — ranking/ranker.py       → outputs/ranked_candidates.csv
    3. Cluster   — ranking/clustering.py   → outputs/candidate_clusters.csv

Config
------
    config.json         — extractor settings (spaCy model, filters, n_process)
    ranking_config.json — data paths, ranking & clustering settings
"""

if __name__ == "__main__":
    import json
    import logging
    import time
    from pathlib import Path

    import pandas as pd
    import spacy

    from candidate_extractor import CandidateExtractor, save_outputs
    from ranking import CandidateRanker, get_embedder
    from ranking.loader import load_candidates_list
    from ranking.clustering import cluster_candidates

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    t_total = time.time()

    with open("ranking_config.json") as f:
        rcfg = json.load(f)

    # ── Stage 1: Extraction ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 — CANDIDATE EXTRACTION")
    print("=" * 60)

    data_path    = rcfg["data_path"]
    id_col       = rcfg["id_column"]
    title_col    = rcfg["title_column"]
    abstract_col = rcfg["abstract_column"]

    df = pd.read_parquet(data_path)
    df = (
        df[[id_col, title_col, abstract_col]]
        .set_index(id_col)
        .rename(columns={title_col: "title", abstract_col: "abstract"})
    )
    print(f"Loaded {len(df):,} documents from {data_path}")

    t0 = time.time()
    extractor = CandidateExtractor("config.json")
    candidates = extractor.extract(df)
    list_path, mapping_path = save_outputs(candidates, output_dir="outputs/")
    print(f"Extraction done in {time.time() - t0:.1f}s")
    print(f"  Unique candidates : {candidates['candidate'].nunique():,}")
    print(f"  candidates_list   → {list_path}")
    print(f"  candidates_mapping→ {mapping_path}")

    # ── Stage 2: Ranking ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — RANKING (TF-IDF + SEMANTIC)")
    print("=" * 60)

    t0 = time.time()
    ranker = CandidateRanker("ranking_config.json")
    scores = ranker.rank_from_config()
    out_path = ranker.save(scores)
    print(f"Ranking done in {time.time() - t0:.1f}s")
    print(f"  Scored pairs      : {len(scores):,}")
    print(f"  ranked_candidates → {out_path}")

    # ── Stage 3: Clustering ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3 — CLUSTERING (MORPHOLOGICAL + SEMANTIC)")
    print("=" * 60)

    clust_cfg = rcfg.get("clustering", {})
    sem_cfg   = rcfg.get("semantic", {})
    col_map   = rcfg.get("candidates_list_columns", {
        "candidate_id": "candidate_id",
        "candidate": "candidate",
        "surface_form": "surface_form",
    })

    candidates_list = load_candidates_list(rcfg["candidates_list_path"], col_map)

    spacy_model = clust_cfg.get("spacy_model", "en_core_web_sm")
    print(f"Loading spaCy model: {spacy_model}")
    nlp = spacy.load(spacy_model, disable=["ner", "parser"])

    embedder = get_embedder(
        model_name=sem_cfg.get("model_name", "all-MiniLM-L6-v2"),
        backend=sem_cfg.get("backend", "sentence_transformers"),
    )

    t0 = time.time()
    clustered = cluster_candidates(
        candidates_list,
        embedder=embedder,
        nlp=nlp,
        semantic_threshold=clust_cfg.get("semantic_threshold", 0.92),
        max_block_size=clust_cfg.get("max_block_size", 2000),
        batch_size=clust_cfg.get("batch_size", 64),
    )
    clust_out = Path(clust_cfg.get("output_path", "outputs/candidate_clusters.csv"))
    clust_out.parent.mkdir(parents=True, exist_ok=True)
    clustered[["candidate_id", "candidate", "cluster_id", "cluster_label", "cluster_size"]].to_csv(
        clust_out, index=False
    )
    print(f"Clustering done in {time.time() - t0:.1f}s")
    print(f"  Total clusters    : {clustered['cluster_id'].nunique():,}")
    print(f"  Singletons        : {(clustered['cluster_size'] == 1).sum():,}")
    print(f"  Multi-member      : {(clustered['cluster_size'] > 1).sum():,} candidates")
    print(f"  candidate_clusters→ {clust_out}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — total time: {time.time() - t_total:.1f}s")
    print("=" * 60)
