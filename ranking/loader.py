"""loader.py — input loading and validation for the ranking stage.

Expected schemas
----------------
candidates_list.csv
    candidate_id  : int
    candidate     : str   — normalized candidate phrase
    surface_form  : str   — first-seen original surface form
    extraction_source : str  — pipe-joined source tags (e.g. "dep_phrase|noun_chunk")

candidates_mapping.csv
    candidate_id  : int
    row_id        : str   — original DataFrame index value

source DataFrame (parquet or CSV)
    One column used as row identifier (set as index)
    One column for title text
    One column for abstract text
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _check_columns(df: pd.DataFrame, required: list, source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


def load_candidates_list(path: str, col_map: dict) -> pd.DataFrame:
    """
    Load the candidate list CSV produced by save_outputs().

    Parameters
    ----------
    path : str
        Path to candidates_list.csv
    col_map : dict
        Maps canonical names to actual column names in the file.
        Required keys: candidate_id, candidate, surface_form
        Optional key:  extraction_source

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id (int), candidate (str), surface_form (str),
        and extraction_source (str) if present in the file.
    """
    df = pd.read_csv(path)

    required_actual = [col_map["candidate_id"], col_map["candidate"]]
    _check_columns(df, required_actual, f"candidates list ({path})")

    # Rename actual column names → canonical names
    rename = {v: k for k, v in col_map.items() if v in df.columns}
    df = df.rename(columns=rename)

    df["candidate_id"] = df["candidate_id"].astype(int)
    df["candidate"] = df["candidate"].astype(str)

    logger.info("Loaded %d unique candidates from %s", len(df), path)
    return df


def load_candidates_mapping(path: str, col_map: dict) -> pd.DataFrame:
    """
    Load the candidate-to-row mapping CSV produced by save_outputs().

    The mapping may contain duplicate (candidate_id, row_id) pairs when
    keep_duplicates=True was set in the extractor config. These duplicates
    represent multiple occurrences of a candidate within the same document
    section and are used to compute term frequency (tf).

    Parameters
    ----------
    path : str
        Path to candidates_mapping.csv
    col_map : dict
        Maps canonical names to actual column names in the file.
        Required keys: candidate_id, row_id

    Returns
    -------
    pd.DataFrame
        Columns: candidate_id (int), row_id.
        Preserves duplicates — do not deduplicate before passing to
        build_canonical_table().
    """
    df = pd.read_csv(path)

    required_actual = [col_map["candidate_id"], col_map["row_id"]]
    _check_columns(df, required_actual, f"candidates mapping ({path})")

    rename = {v: k for k, v in col_map.items() if v in df.columns}
    df = df.rename(columns=rename)

    df["candidate_id"] = df["candidate_id"].astype(int)

    logger.info(
        "Loaded %d occurrence rows (%d unique pairs) from %s",
        len(df),
        df.drop_duplicates().shape[0],
        path,
    )
    return df


def load_source_dataframe(
    path: str,
    id_col: str,
    title_col: str,
    abstract_col: str,
) -> pd.DataFrame:
    """
    Load the source patents DataFrame.

    Supports .parquet and .csv formats.

    Parameters
    ----------
    path : str
        Path to the data file.
    id_col : str
        Column to set as the DataFrame index (row identifier).
    title_col : str
        Column containing document titles.
    abstract_col : str
        Column containing document abstracts.

    Returns
    -------
    pd.DataFrame
        Index: row_id (the values from id_col).
        Columns include 'title' and 'abstract' (renamed from title_col / abstract_col).
        All other columns are preserved as-is.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif p.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .csv")

    required = [id_col, title_col, abstract_col]
    _check_columns(df, required, f"source DataFrame ({path})")

    df = df.set_index(id_col)
    df = df.rename(columns={title_col: "title", abstract_col: "abstract"})

    logger.info("Loaded %d rows from %s", len(df), path)
    return df
