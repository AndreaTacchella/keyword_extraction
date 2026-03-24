"""ranker.py — orchestrator: loads inputs, runs both rankers, assembles output.

Public API
----------
    ranker = CandidateRanker("ranking_config.json")

    # Option A: pass an already-loaded source DataFrame
    scores = ranker.rank(source_df)

    # Option B: load everything from paths in config
    scores = ranker.rank_from_config()

    ranker.save(scores)

Output schema
-------------
One row per unique (row_id, candidate). Columns:

    row_id          : original DataFrame index value
    candidate       : normalized candidate phrase
    surface_form    : first-seen surface form (from candidates_list)
    extraction_source : pipe-joined extraction method tags (if available)
    tf              : within-document term frequency
    df              : document frequency (# distinct row_ids with this candidate)
    idf             : smoothed IDF = log((N+1)/(df+1)) + 1
    tfidf_score     : tf * idf
    semantic_score  : cosine similarity (document embedding, candidate embedding)
    rank_tfidf      : within-document rank by tfidf_score (1 = best)
    rank_semantic   : within-document rank by semantic_score (1 = best)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .loader import load_candidates_list, load_candidates_mapping, load_source_dataframe
from .canonical import build_canonical_table
from .formatting import format_documents
from .embedder import get_embedder, EmbedderBase
from .lexical import compute_tfidf
from .semantic import compute_semantic_scores

logger = logging.getLogger(__name__)

# Preferred column order in final output
_OUTPUT_COLUMNS = [
    "row_id",
    "candidate",
    "surface_form",
    "extraction_source",
    "tf",
    "df",
    "idf",
    "tfidf_score",
    "semantic_score",
    "rank_tfidf",
    "rank_semantic",
]

_DEFAULT_CLIST_COL_MAP = {
    "candidate_id": "candidate_id",
    "candidate": "candidate",
    "surface_form": "surface_form",
    "extraction_source": "extraction_source",
}

_DEFAULT_CMAPPING_COL_MAP = {
    "candidate_id": "candidate_id",
    "row_id": "row_id",
}


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CandidateRanker:
    """
    Unsupervised keyword candidate ranker.

    Combines a lexical TF-IDF ranker and a semantic cosine-similarity ranker
    to produce a scored DataFrame with one row per (row_id, candidate) pair.

    Parameters
    ----------
    config_path : str
        Path to ranking_config.json.
    """

    def __init__(self, config_path: str):
        self.cfg = _load_config(config_path)
        self._embedder: EmbedderBase | None = None  # lazy-loaded on first use

    @property
    def embedder(self) -> EmbedderBase:
        """Lazy-load the embedding model (expensive first call)."""
        if self._embedder is None:
            sem = self.cfg.get("semantic", {})
            self._embedder = get_embedder(
                model_name=sem.get("model_name", "all-MiniLM-L6-v2"),
                backend=sem.get("backend", "sentence_transformers"),
            )
        return self._embedder

    def rank(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full ranking pipeline on an already-loaded source DataFrame.

        Parameters
        ----------
        source_df : pd.DataFrame
            Must have:
              - 'title' column (str or NaN)
              - 'abstract' column (str or NaN)
              - index = row_id values matching those in the mapping file.
            Use load_source_dataframe() or rank_from_config() to get this
            automatically from a raw parquet/CSV with arbitrary column names.

        Returns
        -------
        pd.DataFrame
            One row per (row_id, candidate). See module docstring for schema.
        """
        cfg = self.cfg

        # ── Load extractor outputs ────────────────────────────────────────────
        clist = load_candidates_list(
            cfg["candidates_list_path"],
            cfg.get("candidates_list_columns", _DEFAULT_CLIST_COL_MAP),
        )
        cmapping = load_candidates_mapping(
            cfg["candidates_mapping_path"],
            cfg.get("candidates_mapping_columns", _DEFAULT_CMAPPING_COL_MAP),
        )

        # ── Canonical table ───────────────────────────────────────────────────
        canonical = build_canonical_table(clist, cmapping, source_df)

        # ── Lexical scoring ───────────────────────────────────────────────────
        tfidf_cfg = cfg.get("tfidf", {})
        canonical = compute_tfidf(canonical, smoothed=tfidf_cfg.get("smoothed_idf", True))

        # ── Format documents (only rows present in canonical) ─────────────────
        fmt_cfg = cfg.get("text_formatting", {})
        relevant_source = source_df.loc[
            source_df.index.isin(canonical["row_id"].unique())
        ]
        doc_texts = format_documents(
            relevant_source,
            mode=fmt_cfg.get("mode", "cls_sep"),
            cls_token=fmt_cfg.get("cls_token", "[CLS]"),
            sep_token=fmt_cfg.get("sep_token", "[SEP]"),
        )

        # ── Semantic scoring ──────────────────────────────────────────────────
        sem_cfg = cfg.get("semantic", {})
        canonical = compute_semantic_scores(
            canonical,
            doc_texts,
            self.embedder,
            batch_size=sem_cfg.get("batch_size", 64),
        )

        # ── Within-document ranks ─────────────────────────────────────────────
        canonical["rank_tfidf"] = (
            canonical
            .groupby("row_id")["tfidf_score"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        canonical["rank_semantic"] = (
            canonical
            .groupby("row_id")["semantic_score"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        # ── Reorder columns ───────────────────────────────────────────────────
        present = [c for c in _OUTPUT_COLUMNS if c in canonical.columns]
        extra   = [c for c in canonical.columns if c not in _OUTPUT_COLUMNS]
        canonical = canonical[present + extra]

        logger.info("Ranking complete: %d scored (row_id, candidate) pairs.", len(canonical))
        return canonical.reset_index(drop=True)

    def rank_from_config(self) -> pd.DataFrame:
        """
        Load the source DataFrame from the path in config, then run rank().

        The id_column, title_column, and abstract_column config keys are used
        to set the index and rename columns before passing to rank().

        Returns
        -------
        pd.DataFrame
        """
        cfg = self.cfg
        source_df = load_source_dataframe(
            cfg["data_path"],
            id_col=cfg.get("id_column", "APPLN_ID"),
            title_col=cfg.get("title_column", "APPLN_TITLE"),
            abstract_col=cfg.get("abstract_column", "APPLN_ABSTR"),
        )
        return self.rank(source_df)

    def save(self, scores: pd.DataFrame, path: str | None = None) -> Path:
        """
        Write the scored DataFrame to CSV.

        Parameters
        ----------
        scores : pd.DataFrame
            Output of rank() or rank_from_config().
        path : str, optional
            Override the output path from config.

        Returns
        -------
        pathlib.Path
            Absolute path of the file written.
        """
        out_path = Path(path or self.cfg.get("output_path", "outputs/ranked_candidates.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_path, index=False)
        logger.info("Saved %d rows to %s", len(scores), out_path)
        return out_path
