"""ranking — unsupervised keyword candidate ranking for patent text."""

from .ranker import CandidateRanker
from .clustering import cluster_candidates, embed_candidates
from .loader import load_candidates_list, load_candidates_mapping, load_source_dataframe
from .canonical import build_canonical_table
from .formatting import format_documents, get_formatter
from .embedder import EmbedderBase, SentenceTransformerEmbedder, get_embedder
from .lexical import compute_tfidf
from .semantic import compute_semantic_scores, compute_semantic_scores_chunked
from .checkpoint import CheckpointManager

__all__ = [
    "CandidateRanker",
    "cluster_candidates",
    "embed_candidates",
    "load_candidates_list",
    "load_candidates_mapping",
    "load_source_dataframe",
    "build_canonical_table",
    "format_documents",
    "get_formatter",
    "EmbedderBase",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "compute_tfidf",
    "compute_semantic_scores",
    "compute_semantic_scores_chunked",
    "CheckpointManager",
]
