"""lexical.py — candidate-level TF-IDF ranker.

TF-IDF formula
--------------
tf(c, d)  = number of times candidate c appears in dataset row d.
            Derived from the raw (possibly duplicate) mapping file.
            Falls back to tf=1 (binary) if the extractor ran with
            keep_duplicates=False.

df(c)     = number of distinct dataset rows containing candidate c.

N         = number of distinct dataset rows in the canonical table
            (not the total source DataFrame size, which may include rows
            with no extracted candidates).

idf(c)    = log((N + 1) / (df(c) + 1)) + 1   [sklearn smooth IDF]

tfidf_score(c, d) = tf(c, d) * idf(c)

The smoothed IDF avoids division by zero and reduces extreme penalties for
very frequent or very rare candidates. The +1 additive smoothing outside
the log keeps IDF ≥ 1 for all candidates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_tfidf(
    canonical: pd.DataFrame,
    smoothed: bool = True,
) -> pd.DataFrame:
    """
    Compute candidate-level TF-IDF scores for each (row_id, candidate) pair.

    Parameters
    ----------
    canonical : pd.DataFrame
        Canonical table from build_canonical_table().
        Required columns: row_id, candidate, tf.
    smoothed : bool
        Use smoothed IDF (sklearn formula). Recommended. If False, uses
        log(N / df) + 1 (standard corpus IDF, breaks if df == N).

    Returns
    -------
    pd.DataFrame
        Input columns plus: df, idf, tfidf_score (all float64 except df int).
    """
    result = canonical.copy()

    # N = distinct row_ids with at least one candidate (canonical table scope)
    N = result["row_id"].nunique()
    if N == 0:
        raise ValueError("Canonical table is empty — no rows to rank.")

    # Document frequency: distinct row_ids per candidate
    df_counts = (
        result.groupby("candidate")["row_id"]
        .nunique()
        .rename("df")
        .reset_index()
    )
    result = result.merge(df_counts, on="candidate", how="left")

    # IDF
    if smoothed:
        result["idf"] = np.log((N + 1) / (result["df"] + 1)) + 1
    else:
        result["idf"] = np.log(N / result["df"]) + 1

    # TF-IDF
    result["tfidf_score"] = result["tf"] * result["idf"]

    logger.info(
        "TF-IDF: N=%d rows | %d unique candidates | "
        "tfidf range [%.4f, %.4f]",
        N,
        result["candidate"].nunique(),
        result["tfidf_score"].min(),
        result["tfidf_score"].max(),
    )
    return result
