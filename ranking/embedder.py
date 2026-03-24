"""embedder.py — modular embedding backend for the semantic ranker.

Swapping the encoder
--------------------
To use a different encoder:
    1. Subclass EmbedderBase and implement embed().
    2. Register it in get_embedder() with a new backend name.
    3. Set "backend": "<your_name>" in ranking_config.json.

The rest of the pipeline (semantic.py, ranker.py) is unaffected.

embed() contract
----------------
- Input:  list of strings (possibly empty strings for missing text)
- Output: np.ndarray of shape (n, dim), float32, L2-normalized row-wise
- Empty strings should produce valid (non-NaN) embedding vectors.
  SentenceTransformerEmbedder returns near-zero vectors for empty strings
  (the model's padding output), which yields cosine score ≈ 0. That is
  the intended behavior: documents with no text get no semantic signal.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EmbedderBase:
    """
    Abstract base for embedding backends.

    Implement embed() in a subclass to plug in a different encoder.
    """

    def embed(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embedding vectors.

        Parameters
        ----------
        texts : list of str
        batch_size : int

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
            Rows are L2-normalized so that dot product equals cosine similarity.
        """
        raise NotImplementedError


class SentenceTransformerEmbedder(EmbedderBase):
    """
    Embedding backend using the sentence-transformers library.

    Default model: all-MiniLM-L6-v2
        ~22M parameters, 384-dimensional embeddings.
        Max input length: 256 tokens (longer inputs are silently truncated).
        Fast on CPU and Apple Silicon. Good general-purpose baseline.

    To use a different model, change "model_name" in ranking_config.json.
    Any model compatible with sentence-transformers SentenceTransformer()
    can be used as a drop-in replacement.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic ranking. "
                "Install with: pip install sentence-transformers"
            )
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """
        Encode texts and return L2-normalized float32 embeddings.

        Note: inputs longer than 256 tokens are silently truncated by the
        all-MiniLM-L6-v2 model. For long abstracts this is expected and
        acceptable for a v1 baseline.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,  # L2 normalize → cosine = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


def get_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    backend: str = "sentence_transformers",
) -> EmbedderBase:
    """
    Factory: return an embedder instance by backend name.

    Parameters
    ----------
    model_name : str
        Model identifier passed to the backend.
    backend : str
        Backend name. Supported: "sentence_transformers".
        To add a new backend, register it here and subclass EmbedderBase.

    Returns
    -------
    EmbedderBase instance
    """
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(
        f"Unknown embedding backend '{backend}'. "
        "Subclass EmbedderBase, implement embed(), and register it in get_embedder()."
    )
