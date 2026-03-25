"""clustering.py — two-stage candidate clustering: morphological + semantic.

Stage 1 — Morphological
    Lemmatize every token in every candidate with spaCy.
    Candidates that share the same lemma form are merged into one group.
    Handles: "batteries" → "battery", "electrode materials" → "electrode material".
    Deterministic; no threshold.

Stage 2 — Semantic (first-token blocking + union-find)
    Computing 124k × 124k cosine similarity is infeasible (~60 GB RAM).
    Instead, candidates are grouped by their first token (blocking), and pairwise
    cosine similarity is computed only within each block. Pairs above the
    configurable threshold are merged using union-find.
    Handles: "battery pack" / "battery module" (same first token, high similarity).

Both stages feed into a final merge step: any two morphological groups that are
joined by a semantic edge are merged into a single cluster.

Each cluster gets:
    cluster_id    — compact integer, stable (order of first appearance)
    cluster_label — the candidate closest to the cluster centroid embedding
    cluster_size  — number of candidates in the cluster (1 = singleton)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .embedder import EmbedderBase

logger = logging.getLogger(__name__)


# ── Union-Find ────────────────────────────────────────────────────────────────

class _UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._size   = [1] * n

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]  # path halving
            i = self._parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self._size[ri] < self._size[rj]:
            ri, rj = rj, ri
        self._parent[rj] = ri
        self._size[ri] += self._size[rj]

    def groups(self) -> dict[int, list[int]]:
        """Return {root: [member_indices]}."""
        g: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            g[self.find(i)].append(i)
        return dict(g)


# ── Stage 1: Morphological ────────────────────────────────────────────────────

def _lemmatize(text: str, nlp) -> str:
    """Return the space-joined lemmas of all tokens in text, lowercased."""
    return " ".join(tok.lemma_.lower() for tok in nlp(text))


def build_morphological_clusters(
    candidates: list,
    nlp,
) -> dict:
    """
    Group candidates by shared spaCy lemma form.

    Parameters
    ----------
    candidates : list of str
    nlp : spaCy Language model (must have the lemmatizer component enabled)

    Returns
    -------
    dict {candidate_str: cluster_id (int)}
        Candidates with the same lemma form share a cluster_id.
    """
    logger.info("Morphological stage: lemmatizing %d candidates ...", len(candidates))

    lemma_to_cluster: dict = {}
    candidate_to_cluster: dict = {}
    next_id = 0

    for cand, doc in zip(candidates, nlp.pipe(candidates, batch_size=256)):
        lemma = " ".join(tok.lemma_.lower() for tok in doc)
        if lemma not in lemma_to_cluster:
            lemma_to_cluster[lemma] = next_id
            next_id += 1
        candidate_to_cluster[cand] = lemma_to_cluster[lemma]

    n_groups = len(lemma_to_cluster)
    n_merged = len(candidates) - n_groups
    logger.info(
        "Morphological stage: %d candidates → %d lemma groups (%d merged)",
        len(candidates), n_groups, n_merged,
    )
    return candidate_to_cluster


# ── Stage 2: Semantic ─────────────────────────────────────────────────────────

def embed_candidates(
    candidates: list,
    embedder: EmbedderBase,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Embed all candidate strings.

    Parameters
    ----------
    candidates : list of str
    embedder : EmbedderBase — must return L2-normalized float32 arrays
    batch_size : int

    Returns
    -------
    np.ndarray of shape (len(candidates), embedding_dim), float32, L2-normalized
    """
    logger.info("Embedding %d candidates ...", len(candidates))
    return embedder.embed(candidates, batch_size=batch_size)


def build_semantic_clusters(
    candidates: list,
    embeddings: np.ndarray,
    threshold: float = 0.85,
    max_block_size: int = 2000,
) -> dict:
    """
    Cluster candidates by semantic similarity using first-token blocking.

    Candidates are grouped by their first token. Within each block, pairwise
    cosine similarity is computed (dot product of L2-normalized vectors) and
    pairs above `threshold` are merged via union-find.

    Cross-block merges are not computed — candidates with different leading
    tokens are assumed to be distinct concepts at this threshold level.

    Parameters
    ----------
    candidates : list of str
    embeddings : np.ndarray, shape (n, dim), L2-normalized
        Row i corresponds to candidates[i].
    threshold : float
        Cosine similarity threshold for merging (default 0.85).

    Returns
    -------
    dict {candidate_str: cluster_id (int)}
    """
    n = len(candidates)
    uf = _UnionFind(n)
    cand_idx = {c: i for i, c in enumerate(candidates)}

    # Block by first token
    blocks: dict = defaultdict(list)
    for i, cand in enumerate(candidates):
        first_tok = cand.split()[0] if cand.strip() else "__empty__"
        blocks[first_tok].append(i)

    n_merges = 0
    large_blocks = sum(1 for idxs in blocks.values() if len(idxs) > 100)
    if large_blocks:
        logger.info(
            "Semantic stage: %d blocks with >100 candidates (largest pairwise "
            "matrices may be slow)", large_blocks
        )

    logger.info(
        "Semantic stage: processing %d first-token blocks (threshold=%.2f) ...",
        len(blocks), threshold,
    )

    for first_tok, idxs in blocks.items():
        if len(idxs) < 2:
            continue
        if len(idxs) > max_block_size:
            logger.debug(
                "Skipping block '%s' (%d candidates > max_block_size=%d)",
                first_tok, len(idxs), max_block_size,
            )
            continue
        block_embs = embeddings[idxs]            # (b, dim)
        sim = block_embs @ block_embs.T          # (b, b) cosine similarity
        rows, cols = np.where(sim >= threshold)
        for r, c in zip(rows, cols):
            if r < c:                            # upper triangle only
                uf.union(idxs[r], idxs[c])
                n_merges += 1

    logger.info("Semantic stage: %d pairs merged above threshold %.2f", n_merges, threshold)

    # Build {candidate: cluster_id}
    groups = uf.groups()
    candidate_to_cluster: dict = {}
    for cluster_id, members in enumerate(groups.values()):
        for idx in members:
            candidate_to_cluster[candidates[idx]] = cluster_id

    n_clusters = len(groups)
    logger.info(
        "Semantic stage: %d candidates → %d clusters (%d singletons)",
        n, n_clusters, sum(1 for m in groups.values() if len(m) == 1),
    )
    return candidate_to_cluster


# ── Merge stages ──────────────────────────────────────────────────────────────

def merge_cluster_maps(
    candidates: list,
    morph_map: dict,
    sem_map: dict,
) -> dict:
    """
    Reconcile morphological and semantic cluster maps into a single assignment.

    A morphological group and a semantic group are merged whenever they share
    at least one candidate (i.e., the semantic stage connected two candidates
    that were already grouped morphologically). This is done by treating morph
    groups as nodes in a graph and adding edges from the semantic stage.

    Parameters
    ----------
    candidates : list of str
    morph_map  : {candidate: morph_cluster_id}
    sem_map    : {candidate: sem_cluster_id}

    Returns
    -------
    dict {candidate: final_cluster_id (int)}
        Compact integer ids in order of first appearance.
    """
    # Unique morph groups as nodes; union across semantic bridges
    morph_ids = sorted(set(morph_map.values()))
    n_morph = max(morph_ids) + 1
    uf = _UnionFind(n_morph)

    # For each semantic cluster, collect the morph_ids of its members and union them
    sem_to_morph: dict = defaultdict(set)
    for cand in candidates:
        sem_to_morph[sem_map[cand]].add(morph_map[cand])

    for morph_ids_in_sem in sem_to_morph.values():
        ids = list(morph_ids_in_sem)
        for i in range(1, len(ids)):
            uf.union(ids[0], ids[i])

    # Assign compact final cluster ids (in order of first seen morph group)
    root_to_final: dict = {}
    next_id = 0
    final_map: dict = {}
    for cand in candidates:
        root = uf.find(morph_map[cand])
        if root not in root_to_final:
            root_to_final[root] = next_id
            next_id += 1
        final_map[cand] = root_to_final[root]

    n_final = len(root_to_final)
    n_merged = len(candidates) - n_final
    logger.info(
        "Merge: %d candidates → %d final clusters (%d merged total)",
        len(candidates), n_final, n_merged,
    )
    return final_map


# ── Cluster labels ────────────────────────────────────────────────────────────

def assign_cluster_labels(
    candidates: list,
    cluster_map: dict,
    embeddings: np.ndarray,
) -> dict:
    """
    For each cluster, select the representative candidate (label).

    The representative is the candidate whose embedding is closest to the
    cluster centroid (mean of member embeddings). For singleton clusters,
    the candidate itself is the label.

    Parameters
    ----------
    candidates : list of str
    cluster_map : {candidate: cluster_id}
    embeddings : np.ndarray, shape (n, dim), corresponds to candidates order

    Returns
    -------
    dict {cluster_id: cluster_label (str)}
    """
    # Collect member indices per cluster
    cluster_members: dict = defaultdict(list)
    for i, cand in enumerate(candidates):
        cluster_members[cluster_map[cand]].append(i)

    labels: dict = {}
    for cid, member_idxs in cluster_members.items():
        if len(member_idxs) == 1:
            labels[cid] = candidates[member_idxs[0]]
            continue
        member_embs = embeddings[member_idxs]          # (k, dim)
        centroid = member_embs.mean(axis=0)            # (dim,)
        # cosine similarity to centroid (embeddings already L2-normalized)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        sims = member_embs @ centroid
        best_idx = member_idxs[int(np.argmax(sims))]
        labels[cid] = candidates[best_idx]

    return labels


# ── Main entry point ──────────────────────────────────────────────────────────

def cluster_candidates(
    candidates_list: pd.DataFrame,
    embedder: EmbedderBase,
    nlp,
    semantic_threshold: float = 0.85,
    max_block_size: int = 2000,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Assign cluster_id, cluster_label, and cluster_size to every candidate.

    Parameters
    ----------
    candidates_list : pd.DataFrame
        Must have columns: candidate_id, candidate.
        Typically loaded from candidates_list.csv via load_candidates_list().
    embedder : EmbedderBase
        Embedding backend (L2-normalized outputs). Reuse the one from ranking.
    nlp : spaCy Language
        Must have the lemmatizer component **enabled**.
        Do NOT pass the extractor's nlp (which disables lemmatizer).
        Load with: spacy.load("en_core_web_sm")
    semantic_threshold : float
        Cosine similarity cutoff for merging candidates in stage 2.
        0.85 targets near-synonyms; lower values produce broader clusters.
    batch_size : int
        Embedding batch size.

    Returns
    -------
    pd.DataFrame
        candidates_list with three new columns appended:
        cluster_id (int), cluster_label (str), cluster_size (int).
    """
    candidates = candidates_list["candidate"].tolist()

    # Stage 1: morphological
    morph_map = build_morphological_clusters(candidates, nlp)

    # Stage 2: semantic
    embeddings = embed_candidates(candidates, embedder, batch_size=batch_size)
    sem_map    = build_semantic_clusters(candidates, embeddings, threshold=semantic_threshold, max_block_size=max_block_size)

    # Merge
    final_map  = merge_cluster_maps(candidates, morph_map, sem_map)

    # Cluster labels
    label_map  = assign_cluster_labels(candidates, final_map, embeddings)

    # Cluster sizes
    from collections import Counter
    size_map = Counter(final_map.values())

    # Attach to DataFrame
    result = candidates_list.copy()
    result["cluster_id"]    = result["candidate"].map(final_map)
    result["cluster_label"] = result["cluster_id"].map(label_map)
    result["cluster_size"]  = result["cluster_id"].map(size_map)

    n_clusters   = result["cluster_id"].nunique()
    n_singletons = (result["cluster_size"] == 1).sum()
    n_merged     = (result["cluster_size"] > 1).sum()
    logger.info(
        "Done: %d candidates → %d clusters (%d singletons, %d in multi-member clusters)",
        len(result), n_clusters, n_singletons, n_merged,
    )
    return result
