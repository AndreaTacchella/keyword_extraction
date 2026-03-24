"""formatting.py — document text formatting for the ranking stage.

Document representation
-----------------------
Default mode "cls_sep":
    [CLS] <title> [SEP] <abstract>

This is a deliberate design choice: the literal special-token strings are
preserved even when the downstream encoder does not strictly require them,
because they signal the boundary between title and abstract for encoders
that are sensitive to positional structure (e.g. cross-encoder fine-tunes).

To change the format, either:
    - switch to mode="plain" in config for a no-special-token baseline, or
    - register a new formatter with @register_formatter("my_mode") and set
      "mode": "my_mode" in ranking_config.json.

Adding a new formatter requires no changes to any other module.
"""

from __future__ import annotations

import pandas as pd


_FORMATTERS: dict = {}


def register_formatter(name: str):
    """Decorator to register a document formatter function."""
    def decorator(fn):
        _FORMATTERS[name] = fn
        return fn
    return decorator


@register_formatter("cls_sep")
def format_cls_sep(
    title,
    abstract,
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    **kwargs,
) -> str:
    """
    Format a document as: [CLS] <title> [SEP] <abstract>

    Null/empty fields are handled gracefully:
        title only   → "[CLS] <title>"
        abstract only → "[CLS] [SEP] <abstract>"
        both missing  → ""
    """
    title_str    = str(title).strip()    if pd.notna(title)    and str(title).strip()    else ""
    abstract_str = str(abstract).strip() if pd.notna(abstract) and str(abstract).strip() else ""

    if title_str and abstract_str:
        return f"{cls_token} {title_str} {sep_token} {abstract_str}"
    if title_str:
        return f"{cls_token} {title_str}"
    if abstract_str:
        return f"{cls_token} {sep_token} {abstract_str}"
    return ""


@register_formatter("plain")
def format_plain(
    title,
    abstract,
    **kwargs,
) -> str:
    """
    Format a document as: <title>. <abstract>

    No special tokens. Useful for encoders trained without them.
    """
    title_str    = str(title).strip()    if pd.notna(title)    and str(title).strip()    else ""
    abstract_str = str(abstract).strip() if pd.notna(abstract) and str(abstract).strip() else ""

    if title_str and abstract_str:
        return f"{title_str}. {abstract_str}"
    return title_str or abstract_str


def get_formatter(mode: str):
    """Return a registered formatter function by mode name."""
    if mode not in _FORMATTERS:
        raise ValueError(
            f"Unknown formatting mode '{mode}'. Available: {sorted(_FORMATTERS)}"
        )
    return _FORMATTERS[mode]


def format_documents(
    source_df: pd.DataFrame,
    mode: str = "cls_sep",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
) -> pd.Series:
    """
    Apply document formatting to a DataFrame, producing one string per row.

    Parameters
    ----------
    source_df : pd.DataFrame
        Must have 'title' and 'abstract' columns. Index is used as row_id.
        Pass only the rows you need (e.g. filtered to row_ids in canonical)
        to avoid formatting unused rows.
    mode : str
        Formatting mode. One of: "cls_sep" (default), "plain".
    cls_token, sep_token : str
        Special token strings (used by cls_sep mode only).

    Returns
    -------
    pd.Series
        Index matches source_df.index (row_id). Values are formatted strings.
        Rows where both title and abstract are missing produce empty strings.
    """
    formatter = get_formatter(mode)
    return source_df.apply(
        lambda row: formatter(
            row.get("title"),
            row.get("abstract"),
            cls_token=cls_token,
            sep_token=sep_token,
        ),
        axis=1,
    )
