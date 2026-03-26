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
For corpora > 1M documents, use compute_semantic_scores_chunked() instead.
It processes documents in batches of doc_chunk_size, writes each batch to
disk via CheckpointManager, and resumes automatically after interruptions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

from .embedder import EmbedderBase
from .checkpoint import CheckpointManager

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


def compute_semantic_scores_chunked(
    canonical: pd.DataFrame,
    doc_texts: pd.Series,
    embedder: EmbedderBase,
    batch_size: int = 64,
    doc_chunk_size: int = 100_000,
    checkpoint_dir: str = "outputs/checkpoints/semantic",
) -> pd.DataFrame:
    """
    Chunked variant of compute_semantic_scores for 10M+ document corpora.

    Documents are processed in chunks of doc_chunk_size.  Each chunk's scored
    rows are written to disk immediately via CheckpointManager.  On restart,
    completed chunks are skipped automatically — processing resumes from the
    first incomplete chunk.

    Candidate embeddings are computed once in full (candidates are few relative
    to documents; 125K × 384 × 4 bytes ≈ 190 MB).  Only the doc embeddings are
    chunked (10M × 384 × 4 bytes = 15 GB unbatched → ≈60 MB per 100K chunk).

    Parameters
    ----------
    canonical : pd.DataFrame
        Required columns: row_id, candidate.
    doc_texts : pd.Series
        Index: row_id. Values: formatted document strings.
    embedder : EmbedderBase
        Must return L2-normalized float32 arrays.
    batch_size : int
        Embedding batch size (passed to embedder.embed()).
    doc_chunk_size : int
        Number of unique row_ids to embed per chunk (default 100,000).
    checkpoint_dir : str
        Directory for chunk parquet files and manifest.json.

    Returns
    -------
    pd.DataFrame
        Same schema as compute_semantic_scores() — input columns plus
        semantic_score (float32), with rows in the same order as canonical.
    """
    mgr = CheckpointManager(checkpoint_dir)

    # ── Embed unique candidates once ──────────────────────────────────────────
    unique_candidates = canonical["candidate"].unique().tolist()
    logger.info("Embedding %d unique candidates ...", len(unique_candidates))
    cand_matrix = embedder.embed(unique_candidates, batch_size=batch_size)
    cand_index = {c: i for i, c in enumerate(unique_candidates)}

    # ── Partition row_ids into chunks ─────────────────────────────────────────
    doc_texts_lookup = doc_texts.to_dict()
    unique_row_ids = canonical["row_id"].unique().tolist()
    n_chunks = (len(unique_row_ids) + doc_chunk_size - 1) // doc_chunk_size
    id_width = len(str(n_chunks - 1))

    already_done = mgr.n_completed()
    if already_done:
        logger.info(
            "Resuming: %d/%d chunks already done, skipping them.",
            already_done, n_chunks,
        )

    for chunk_idx in tqdm(range(n_chunks), desc="doc chunks", unit="chunk"):
        chunk_id = f"chunk_{chunk_idx:0{id_width}d}"

        if mgr.is_done(chunk_id):
            logger.info("Chunk %s: already done, skipping.", chunk_id)
            continue

        chunk_row_ids = unique_row_ids[chunk_idx * doc_chunk_size : (chunk_idx + 1) * doc_chunk_size]
        logger.info(
            "Chunk %s (%d/%d): embedding %d documents ...",
            chunk_id, chunk_idx + 1, n_chunks, len(chunk_row_ids),
        )

        # Embed this chunk's documents
        chunk_doc_texts = [doc_texts_lookup.get(rid, "") for rid in chunk_row_ids]
        chunk_doc_matrix = embedder.embed(chunk_doc_texts, batch_size=batch_size)
        chunk_doc_index = {rid: i for i, rid in enumerate(chunk_row_ids)}

        # Score only the canonical rows whose row_id falls in this chunk
        chunk_row_id_set = set(chunk_row_ids)
        mask = canonical["row_id"].isin(chunk_row_id_set)
        chunk_canonical = canonical[mask].copy()

        row_idx = np.fromiter(
            (chunk_doc_index[r] for r in chunk_canonical["row_id"]),
            dtype=np.intp,
            count=len(chunk_canonical),
        )
        cnd_idx = np.fromiter(
            (cand_index[c] for c in chunk_canonical["candidate"]),
            dtype=np.intp,
            count=len(chunk_canonical),
        )
        scores = (chunk_doc_matrix[row_idx] * cand_matrix[cnd_idx]).sum(axis=1)
        chunk_canonical["semantic_score"] = scores.astype(np.float32)

        mgr.save(chunk_id, chunk_canonical)
        logger.info(
            "Chunk %s: scored %d pairs (range [%.4f, %.4f]).",
            chunk_id,
            len(chunk_canonical),
            float(chunk_canonical["semantic_score"].min()),
            float(chunk_canonical["semantic_score"].max()),
        )

    # ── Reassemble in original row order ─────────────────────────────────────
    logger.info("All %d chunks done. Reassembling ...", n_chunks)
    all_scored = mgr.load_all()

    # Merge back onto canonical to preserve original row order
    result = canonical.merge(
        all_scored[["row_id", "candidate", "semantic_score"]],
        on=["row_id", "candidate"],
        how="left",
    )
    result["semantic_score"] = result["semantic_score"].astype(np.float32)

    logger.info(
        "Semantic scores (chunked): range [%.4f, %.4f]",
        float(result["semantic_score"].min()),
        float(result["semantic_score"].max()),
    )
    return result
