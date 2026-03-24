"""rank_example.py — minimal demonstration of the ranking pipeline.

Runs entirely in-memory: no files needed.
Shows the full API and the resulting scored output columns.

Usage (from keyword_extraction/):
    python rank_example.py
"""

if __name__ == "__main__":
    import pandas as pd

    from ranking.canonical import build_canonical_table
    from ranking.formatting import format_documents
    from ranking.embedder import SentenceTransformerEmbedder
    from ranking.lexical import compute_tfidf
    from ranking.semantic import compute_semantic_scores

    # ── Tiny in-memory source DataFrame ──────────────────────────────────────
    source_df = pd.DataFrame(
        {
            "title": [
                "Lithium-ion battery with solid electrolyte",
                "High energy density electrode materials",
                "Solid-state battery cell manufacturing",
            ],
            "abstract": [
                "A solid-state electrolyte improves safety and energy density in lithium batteries.",
                "Novel electrode materials achieve high energy density and cycle stability.",
                "A method for manufacturing solid-state battery cells at scale.",
            ],
        },
        index=pd.Index([1, 2, 3], name="row_id"),
    )

    # ── Mock extractor outputs ────────────────────────────────────────────────
    candidates_list = pd.DataFrame(
        {
            "candidate_id":    [0,                  1,                2,                    3,                   4],
            "candidate":       ["solid electrolyte", "energy density", "electrode material", "lithium battery",   "battery cell"],
            "surface_form":    ["solid electrolyte", "energy density", "electrode materials","lithium batteries",  "battery cells"],
            "extraction_source": ["noun_chunk"] * 5,
        }
    )

    # Multiple rows for (candidate_id=1, row_id=2) → tf=2 for "energy density" in doc 2
    candidates_mapping = pd.DataFrame(
        {
            "candidate_id": [0, 0, 1, 1, 1, 2, 3, 4],
            "row_id":       [1, 3, 1, 2, 2, 2, 1, 3],
        }
    )

    # ── Step 1: canonical table (tf aggregation) ──────────────────────────────
    canonical = build_canonical_table(candidates_list, candidates_mapping, source_df)
    print("Canonical table (one row per unique (row_id, candidate)):")
    print(canonical[["row_id", "candidate", "tf"]].to_string(index=False))
    print()

    # ── Step 2: lexical scoring ───────────────────────────────────────────────
    scored = compute_tfidf(canonical)
    print("After TF-IDF:")
    print(scored[["row_id", "candidate", "tf", "df", "idf", "tfidf_score"]].to_string(index=False))
    print()

    # ── Step 3: document formatting ───────────────────────────────────────────
    doc_texts = format_documents(source_df, mode="cls_sep")
    print("Formatted document texts:")
    for rid, text in doc_texts.items():
        print(f"  [{rid}] {text[:80]}...")
    print()

    # ── Step 4: semantic scoring ──────────────────────────────────────────────
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    scored = compute_semantic_scores(scored, doc_texts, embedder, batch_size=8)

    # ── Step 5: within-document ranks ────────────────────────────────────────
    scored["rank_tfidf"] = (
        scored.groupby("row_id")["tfidf_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    scored["rank_semantic"] = (
        scored.groupby("row_id")["semantic_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # ── Final output ──────────────────────────────────────────────────────────
    print("Final scored output:")
    cols = ["row_id", "candidate", "tf", "tfidf_score", "semantic_score", "rank_tfidf", "rank_semantic"]
    print(scored[cols].sort_values(["row_id", "rank_tfidf"]).to_string(index=False))
