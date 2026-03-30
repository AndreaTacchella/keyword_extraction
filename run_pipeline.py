"""run_pipeline.py — run the full keyword extraction pipeline end-to-end.

Usage (from keyword_extraction/):
    python run_pipeline.py

Stages
------
    1. Extract   — candidate_extractor.py  → outputs/candidates_list.csv
                                             outputs/candidates_mapping.csv
    2. Rank      — ranking/ranker.py       → outputs/ranked_candidates.csv
    3. Cluster   — ranking/clustering.py   → outputs/candidate_clusters.csv

Resume behaviour
----------------
    Each stage checks whether its output file(s) already exist before running.
    If they do, the stage is skipped — so if the pipeline crashes mid-way,
    just re-run the script and it will resume from the first incomplete stage.

    Within Stage 1 (extraction), documents are processed in chunks of
    extract_chunk_size.  Each chunk is saved to a parquet checkpoint
    immediately after processing.  On restart, completed chunks are skipped.

Config
------
    config.json         — extractor settings (spaCy model, filters, n_process)
    ranking_config.json — data paths, ranking, clustering & checkpoint settings
"""

if __name__ == "__main__":
    import json
    import logging
    import math
    import time
    from pathlib import Path

    import pandas as pd
    import spacy

    from candidate_extractor import CandidateExtractor, save_outputs
    from ranking import CandidateRanker, get_embedder
    from ranking.checkpoint import CheckpointManager
    from ranking.loader import load_candidates_list
    from ranking.clustering import cluster_candidates

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    t_total = time.time()

    with open("ranking_config.json") as f:
        rcfg = json.load(f)

    data_path    = rcfg["data_path"]
    id_col       = rcfg["id_column"]
    title_col    = rcfg["title_column"]
    abstract_col = rcfg["abstract_column"]

    list_path    = Path(rcfg.get("candidates_list_path",    "outputs/candidates_list.csv"))
    mapping_path = Path(rcfg.get("candidates_mapping_path", "outputs/candidates_mapping.csv"))
    ranked_path  = Path(rcfg.get("output_path",             "outputs/ranked_candidates.csv"))
    clust_cfg    = rcfg.get("clustering", {})
    clust_path   = Path(clust_cfg.get("output_path",        "outputs/candidate_clusters.csv"))
    counts_path  = Path(clust_cfg.get("doc_counts_path",   "outputs/cluster_doc_counts.csv"))

    # ── Stage 1: Extraction ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 — CANDIDATE EXTRACTION")
    print("=" * 60)

    if list_path.exists() and mapping_path.exists():
        print(f"  Output files already exist — skipping extraction.")
        print(f"  candidates_list    → {list_path}")
        print(f"  candidates_mapping → {mapping_path}")
    else:
        ext_cfg        = rcfg.get("extraction", {})
        chunk_size     = ext_cfg.get("extract_chunk_size", 5000)
        ckpt_dir       = ext_cfg.get("checkpoint_dir", "outputs/checkpoints/extraction")

        df = pd.read_parquet(data_path)
        df = (
            df[[id_col, title_col, abstract_col]]
            .set_index(id_col)
            .rename(columns={title_col: "title", abstract_col: "abstract"})
        )
        print(f"Loaded {len(df):,} documents from {data_path}")

        mgr      = CheckpointManager(ckpt_dir)
        n_chunks = math.ceil(len(df) / chunk_size)
        id_width = len(str(n_chunks - 1))

        if mgr.n_completed():
            print(f"Resuming: {mgr.n_completed()}/{n_chunks} extraction chunks already done.")

        t0        = time.time()
        extractor = CandidateExtractor("config.json")

        for i in range(n_chunks):
            chunk_id = f"chunk_{i:0{id_width}d}"
            if mgr.is_done(chunk_id):
                continue
            chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            print(f"  Chunk {i+1}/{n_chunks}: extracting {len(chunk):,} docs ...")
            candidates_chunk = extractor.extract(chunk)
            mgr.save(chunk_id, candidates_chunk)
            print(f"  Chunk {i+1}/{n_chunks}: {candidates_chunk['candidate'].nunique():,} unique candidates saved.")

        print("Merging extraction chunks ...")
        all_candidates = mgr.load_all()
        list_path, mapping_path = save_outputs(
            all_candidates,
            output_dir=str(list_path.parent),
        )
        print(f"Extraction done in {time.time() - t0:.1f}s")
        print(f"  Unique candidates  : {all_candidates['candidate'].nunique():,}")
        print(f"  candidates_list    → {list_path}")
        print(f"  candidates_mapping → {mapping_path}")

    # ── Stage 2: Ranking ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — RANKING (TF-IDF + SEMANTIC)")
    print("=" * 60)

    if ranked_path.exists():
        print(f"  Output file already exists — skipping ranking.")
        print(f"  ranked_candidates → {ranked_path}")
    else:
        t0     = time.time()
        ranker = CandidateRanker("ranking_config.json")
        scores = ranker.rank_from_config()
        out    = ranker.save(scores)
        print(f"Ranking done in {time.time() - t0:.1f}s")
        print(f"  Scored pairs      : {len(scores):,}")
        print(f"  ranked_candidates → {out}")

    # ── Stage 3: Clustering ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3 — CLUSTERING (MORPHOLOGICAL + SEMANTIC)")
    print("=" * 60)

    if clust_path.exists():
        print(f"  Output file already exists — skipping clustering.")
        print(f"  candidate_clusters → {clust_path}")
    else:
        sem_cfg = rcfg.get("semantic", {})
        col_map = rcfg.get("candidates_list_columns", {
            "candidate_id": "candidate_id",
            "candidate": "candidate",
            "surface_form": "surface_form",
        })

        candidates_list = load_candidates_list(str(list_path), col_map)

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
        clust_path.parent.mkdir(parents=True, exist_ok=True)
        clustered[["candidate_id", "candidate", "cluster_id", "cluster_label", "cluster_size"]].to_csv(
            clust_path, index=False
        )
        print(f"Clustering done in {time.time() - t0:.1f}s")
        print(f"  Total clusters     : {clustered['cluster_id'].nunique():,}")
        print(f"  Singletons         : {(clustered['cluster_size'] == 1).sum():,}")
        print(f"  Multi-member       : {(clustered['cluster_size'] > 1).sum():,} candidates")
        print(f"  candidate_clusters → {clust_path}")

    # ── Stage 4: Cluster document counts ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 4 — CLUSTER DOCUMENT COUNTS")
    print("=" * 60)

    if counts_path.exists():
        print(f"  Output file already exists — skipping.")
        print(f"  cluster_doc_counts → {counts_path}")
    else:
        clusters = pd.read_csv(clust_path)
        mapping  = pd.read_csv(mapping_path)

        merged = mapping.merge(clusters[["candidate_id", "cluster_id"]], on="candidate_id", how="inner")
        doc_counts = (
            merged.groupby("cluster_id")["row_id"]
            .nunique()
            .rename("doc_count")
            .reset_index()
        )
        meta   = clusters[["cluster_id", "cluster_label", "cluster_size"]].drop_duplicates("cluster_id")
        result = meta.merge(doc_counts, on="cluster_id").sort_values("doc_count", ascending=False)

        counts_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(counts_path, index=False)
        print(f"  cluster_doc_counts → {counts_path}  ({len(result):,} clusters)")
        print("\n  Top 10 clusters by document count:")
        print(result.head(10)[["cluster_label", "cluster_size", "doc_count"]].to_string(index=False))

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — total time: {time.time() - t_total:.1f}s")
    print("=" * 60)
