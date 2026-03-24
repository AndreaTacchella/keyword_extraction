"""semantic.py — transformer-based semantic similarity ranker.

Score: cosine similarity between a document embedding and a candidate embedding.

Because embedder.embed() returns L2-normalized vectors, cosine similarity
equals the dot product, computed here via element-wise multiply + sum.

Embedding reuse
---------------
Each unique candidate string is embedded exactly once, regardless of how many
documents it appears in. Each unique document text is embedded exactly once,
regardless of how many candidates it contains. Embeddings are looked up by
index for the score computation — O(n_pairs) with no redundant model calls.

Scalability note
----------------
The two embedding matrices (doc_embeddings and cand_embeddings) are held in
memory simultaneously. At 384 dims × float32:
    25K docs  × 384 = ~37 MB
    125K cands × 384 = ~190 MB
For corpora > 1M documents, consider chunking by row_id batch and computing
scores incrementally rather than materializing the full doc matrix at once.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .embedder import EmbedderBase

logger = logging.getLogger(__name__)


def compute_semantic_scores(
    canonical: pd.DataFrame,
    doc_texts: pd.Series,
    embedder: EmbedderBase,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Compute cosine-similarity semantic scores for each (row_id, candidate) pair.

    Parameters
    ----------
    canonical : pd.DataFrame
        Canonical table (from compute_tfidf or build_canonical_table).
        Required columns: row_id, candidate.
    doc_texts : pd.Series
        Index: row_id (must cover all row_ids in canonical).
        Values: formatted document strings (output of format_documents()).
        Missing row_ids produce empty strings → semantic_score ≈ 0.
    embedder : EmbedderBase
        Embedding backend. Must return L2-normalized float32 arrays.
    batch_size : int
        Batch size passed to embedder.embed().

    Returns
    -------
    pd.DataFrame
        Input columns plus: semantic_score (float32).
    """
    result = canonical.copy()

    # ── Embed unique candidates ───────────────────────────────────────────────
    # Use the normalized candidate string (not surface_form) for consistency
    # with TF-IDF scoring and blacklist logic.
    unique_candidates = result["candidate"].unique().tolist()
    logger.info("Embedding %d unique candidates ...", len(unique_candidates))
    cand_matrix = embedder.embed(unique_candidates, batch_size=batch_size)
    cand_index = {c: i for i, c in enumerate(unique_candidates)}

    # ── Embed unique documents ────────────────────────────────────────────────
    unique_row_ids = result["row_id"].unique().tolist()
    # Use dict lookup instead of reindex to handle source DataFrames with
    # duplicate index labels (reindex raises on duplicate labels).
    doc_texts_lookup = doc_texts.to_dict()
    unique_doc_texts = [doc_texts_lookup.get(rid, "") for rid in unique_row_ids]
    logger.info("Embedding %d unique documents ...", len(unique_row_ids))
    doc_matrix = embedder.embed(unique_doc_texts, batch_size=batch_size)
    doc_index = {rid: i for i, rid in enumerate(unique_row_ids)}

    # ── Vectorized score lookup ───────────────────────────────────────────────
    # Build parallel integer index arrays for fast matrix row selection.
    row_idx = np.fromiter((doc_index[r]  for r in result["row_id"]),  dtype=np.intp, count=len(result))
    cnd_idx = np.fromiter((cand_index[c] for c in result["candidate"]), dtype=np.intp, count=len(result))

    # dot product of L2-normalized vectors = cosine similarity
    scores = (doc_matrix[row_idx] * cand_matrix[cnd_idx]).sum(axis=1)
    result["semantic_score"] = scores.astype(np.float32)

    logger.info(
        "Semantic scores: range [%.4f, %.4f]",
        float(result["semantic_score"].min()),
        float(result["semantic_score"].max()),
    )
    return result
