"""canonical.py — build the canonical (row_id, candidate) table.

The canonical table is the shared input to both rankers. It has one row per
unique (row_id, candidate) pair and carries:

    row_id        : original DataFrame index value
    candidate_id  : integer id from the candidates list
    candidate     : normalized candidate phrase
    surface_form  : first-seen surface form for this candidate
    tf            : within-document term frequency (count of occurrences)
    extraction_source : pipe-joined source tags, if available

document-frequency granularity
-------------------------------
row_id values correspond directly to original DataFrame index values.
df(c) = number of distinct row_ids containing candidate c.
If row_ids encode section information (e.g. "{appln_id}_{section}"), df is
section-level, not patent-level. This is intentional for this baseline:
section-level df is simpler and avoids cross-section join complexity.
To switch to patent-level df, strip the section suffix from row_id before
passing to build_canonical_table().
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns from the mapping that uniquely identify a (candidate, document) pair
_PAIR_COLS = ["candidate_id", "row_id"]


def build_canonical_table(
    candidates_list: pd.DataFrame,
    candidates_mapping: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical per-(row_id, candidate) table.

    tf computation
    --------------
    tf is computed from the raw mapping before any deduplication.
    If keep_duplicates=True was set in the extractor, the mapping may contain
    multiple rows for the same (candidate_id, row_id) pair. Each repeat counts
    as one additional occurrence, so groupby.size() gives the correct tf.
    If the mapping only records unique pairs (keep_duplicates=False), tf=1
    everywhere (binary TF). Both are valid inputs.

    Parameters
    ----------
    candidates_list : pd.DataFrame
        Columns: candidate_id, candidate, surface_form [, extraction_source, ...]
        Produced by load_candidates_list().
    candidates_mapping : pd.DataFrame
        Columns: candidate_id, row_id.
        Must NOT be pre-deduplicated — pass the raw mapping so tf is correct.
        Produced by load_candidates_mapping().
    source_df : pd.DataFrame
        Index: row_id.
        Used only to warn about row_ids present in mapping but absent in source.

    Returns
    -------
    pd.DataFrame
        One row per unique (row_id, candidate).
        Columns: row_id, candidate_id, candidate, surface_form, tf,
        plus extraction_source and any other columns from candidates_list.
    """
    # ── tf: count occurrences per (candidate_id, row_id) ─────────────────────
    # Uses the raw mapping (with duplicates) to get true within-doc frequency.
    tf_series = (
        candidates_mapping
        .groupby(_PAIR_COLS, sort=False)
        .size()
        .rename("tf")
    )
    tf_df = tf_series.reset_index()

    binary_tf = candidates_mapping.drop_duplicates(subset=_PAIR_COLS).shape[0] == len(candidates_mapping)
    if binary_tf:
        logger.info("Mapping has no duplicate pairs — tf=1 everywhere (binary TF).")
    else:
        logger.info(
            "Mapping has %d duplicate pairs — using occurrence counts as tf.",
            len(candidates_mapping) - candidates_mapping.drop_duplicates(subset=_PAIR_COLS).shape[0],
        )

    # ── Join candidate metadata ───────────────────────────────────────────────
    canonical = tf_df.merge(candidates_list, on="candidate_id", how="left")

    # ── Drop null/empty candidate strings ────────────────────────────────────
    before = len(canonical)
    canonical = canonical.dropna(subset=["candidate"])
    canonical = canonical[canonical["candidate"].str.strip() != ""]
    dropped = before - len(canonical)
    if dropped:
        logger.warning("Dropped %d rows with null/empty candidate strings.", dropped)

    # ── Warn about row_ids not in source DataFrame ────────────────────────────
    known_ids = set(source_df.index)
    unknown = set(canonical["row_id"]) - known_ids
    if unknown:
        logger.warning(
            "%d row_ids in mapping not found in source DataFrame "
            "(those rows will get empty document text for semantic scoring).",
            len(unknown),
        )

    logger.info(
        "Canonical table: %d unique (row_id, candidate) pairs | "
        "%d unique row_ids | %d unique candidates",
        len(canonical),
        canonical["row_id"].nunique(),
        canonical["candidate"].nunique(),
    )
    return canonical.reset_index(drop=True)
