"""candidate_extractor.py

Lightweight, configurable baseline for extracting keyword candidate phrases
from patent title and abstract text using spaCy.

Public API
----------
    extractor = CandidateExtractor("config.json")
    candidates_df = extractor.extract(df)
    save_outputs(candidates_df, output_dir="outputs/")

Input DataFrame columns
-----------------------
    title    : str or NaN
    abstract : str or NaN

extract() output DataFrame columns
-----------------------------------
    row_id         : original index from input DataFrame
    candidate      : normalized candidate string
    surface_form   : original text (pre-normalization, post-trimming)
    source_section : "title" or "abstract"

save_outputs() writes two files
--------------------------------
    candidates_list.csv    — one row per unique candidate
                             columns: candidate_id, candidate, surface_form
    candidates_mapping.csv — one row per (candidate, document) occurrence
                             columns: candidate_id, row_id
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_outputs(
    candidates_df: pd.DataFrame,
    output_dir: str = ".",
    list_filename: str = "candidates_list.csv",
    mapping_filename: str = "candidates_mapping.csv",
) -> tuple:
    """
    Split the extraction output into two files and write them to disk.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Output of ``CandidateExtractor.extract()``.
        Required columns: candidate, surface_form, row_id.
    output_dir : str
        Directory where both files are written (created if absent).
    list_filename : str
        Filename for the unique-candidates file.
    mapping_filename : str
        Filename for the occurrence-mapping file.

    Returns
    -------
    (list_path, mapping_path) : tuple of pathlib.Path
        Absolute paths of the two files written.

    Files produced
    --------------
    candidates_list.csv
        One row per unique normalized candidate.
        Columns: candidate_id (int), candidate (str), surface_form (str).
        surface_form is the first-seen surface form for that candidate.

    candidates_mapping.csv
        One row per (candidate, document) occurrence — indexes only.
        Columns: candidate_id (int), row_id (original DataFrame index).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Assign a stable integer id to each unique normalized candidate,
    # in order of first appearance in candidates_df.
    # extraction_source is aggregated across all rows (union of sources).
    source_union = (
        candidates_df.groupby("candidate")["extraction_source"]
        .apply(lambda vals: "|".join(sorted({s for v in vals for s in v.split("|")})))
        .reset_index()
        .rename(columns={"extraction_source": "extraction_source_all"})
    )
    unique_candidates = (
        candidates_df[["candidate", "surface_form"]]
        .drop_duplicates(subset="candidate", keep="first")
        .reset_index(drop=True)
        .merge(source_union, on="candidate")
        .rename(columns={"extraction_source_all": "extraction_source"})
    )
    unique_candidates.index.name = "candidate_id"
    unique_candidates = unique_candidates.reset_index()  # candidate_id becomes a column

    # Build id lookup: normalized candidate string → integer id
    id_lookup = unique_candidates.set_index("candidate")["candidate_id"]

    # ── File 1: unique candidates list ──────────────────────────────────────
    list_path = out / list_filename
    unique_candidates.to_csv(list_path, index=False)

    # ── File 2: occurrence mapping (indexes only) ────────────────────────────
    mapping = candidates_df[["candidate", "row_id"]].copy()
    mapping["candidate_id"] = mapping["candidate"].map(id_lookup)
    mapping = mapping[["candidate_id", "row_id"]]
    mapping["candidate_id"] = mapping["candidate_id"].astype(int)
    mapping_path = out / mapping_filename
    mapping.to_csv(mapping_path, index=False)

    logger.info("Wrote %d unique candidates to %s", len(unique_candidates), list_path)
    logger.info("Wrote %d occurrence rows to %s", len(mapping), mapping_path)

    return list_path, mapping_path


# ---------------------------------------------------------------------------
# Config & blacklist I/O
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load and return the JSON configuration file as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_blacklist(blacklist_path: str) -> set:
    """
    Load blacklist from a plain-text file.

    Lines starting with '#' are comments and are ignored.
    Returns a set of lowercased, stripped phrases for O(1) lookup.
    """
    path = Path(blacklist_path)
    if not path.exists():
        logger.warning("Blacklist file not found at '%s'; proceeding without it.", blacklist_path)
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        line.strip().lower()
        for line in lines
        if line.strip() and not line.startswith("#")
    }


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class CandidateExtractor:
    """
    Rule-based candidate keyword phrase extractor for English patent text.

    Candidate generation uses three complementary sources, each togglable
    via config:
        1. spaCy noun chunks
        2. Dependency-tree compound phrase expansion
        3. POS-filtered n-gram backoff

    Candidates are then filtered (rule-based), edge-trimmed, normalized,
    and deduplicated within each (row, section) pair.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.
    """

    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.nlp = spacy.load(self.cfg["spacy_model"])
        self.blacklist = load_blacklist(self.cfg.get("blacklist_path", "blacklist.txt"))
        self._ws_re = re.compile(r"\s+")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract candidate phrases from a DataFrame with title/abstract columns.

        Missing or empty cells are silently skipped.  If a candidate appears
        in both title and abstract of the same row, both occurrences are kept
        (configurable via ``keep_duplicates``).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least one of: 'title', 'abstract'.

        Returns
        -------
        pd.DataFrame
            One row per (row_id, candidate, source_section) occurrence.
            Columns: row_id, candidate, surface_form, source_section.
        """
        sections = [s for s in ("title", "abstract") if s in df.columns]
        if not sections:
            raise ValueError("DataFrame must contain at least one of: 'title', 'abstract'.")

        all_records = []
        for section in sections:
            all_records.extend(self._extract_section(df, section))

        if not all_records:
            return pd.DataFrame(columns=["row_id", "candidate", "surface_form", "source_section", "extraction_source"])

        result = pd.DataFrame(all_records)

        # Drop within-section duplicates if requested
        if not self.cfg.get("keep_duplicates", True):
            result = result.drop_duplicates(subset=["row_id", "candidate", "source_section"])

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Section-level processing
    # ------------------------------------------------------------------

    def _extract_section(self, df: pd.DataFrame, section: str) -> list:
        """
        Extract candidates for one section across all rows, using nlp.pipe
        for efficient batch processing.
        """
        batch_size = self.cfg.get("batch_size", 64)

        # Collect non-empty (index, text) pairs
        pairs = []
        for idx, row in df.iterrows():
            text = row.get(section)
            if pd.isna(text) or not str(text).strip():
                continue
            pairs.append((idx, str(text).strip()))

        if not pairs:
            return []

        row_ids = [p[0] for p in pairs]
        texts   = [p[1] for p in pairs]

        records = []
        for row_id, doc in zip(row_ids, self.nlp.pipe(texts, batch_size=batch_size)):
            for surface, candidate, extraction_source in self._extract_from_doc(doc):
                records.append(
                    {
                        "row_id": row_id,
                        "candidate": candidate,
                        "surface_form": surface,
                        "source_section": section,
                        "extraction_source": extraction_source,
                    }
                )

        return records

    # ------------------------------------------------------------------
    # Document-level candidate extraction
    # ------------------------------------------------------------------

    def _extract_from_doc(self, doc: Doc) -> list:
        """
        Extract (surface_form, normalized_candidate, extraction_source) triples
        from one Doc.

        Pipeline per span:
            collect tagged spans → trim edges → filter → normalize → dedup

        extraction_source records which method(s) produced each candidate.
        When multiple sources yield the same normalized form, their tags are
        joined with '|' (e.g. "dep_phrase|noun_chunk").
        Possible tags: noun_chunk, dep_phrase, ngram.
        """
        cfg = self.cfg
        # (span, source_tag) pairs
        tagged_spans = []

        if cfg.get("use_noun_chunks", True):
            for span in doc.noun_chunks:
                tagged_spans.append((span, "noun_chunk"))

        if cfg.get("use_dependency_phrases", True):
            for span in self._dependency_phrases(doc):
                tagged_spans.append((span, "dep_phrase"))

        if cfg.get("use_ngram_backoff", True):
            for span in self._ngram_spans(doc):
                tagged_spans.append((span, "ngram"))

        # First pass: collect all valid (surface, normalized, source_tag) triples.
        # Multiple sources can produce the same normalized form — accumulate them.
        # seen maps normalized → (surface_form_of_first_occurrence, set_of_sources)
        seen: dict = {}

        for span, tag in tagged_spans:
            tokens = self._trim_edges(list(span))
            if not tokens:
                continue

            if not self._passes_filter(tokens):
                continue

            surface    = " ".join(t.text for t in tokens)
            normalized = self._normalize(tokens)

            if not normalized or normalized in self.blacklist:
                continue

            if normalized not in seen:
                seen[normalized] = (surface, {tag})
            else:
                seen[normalized][1].add(tag)

        # Second pass: build output list with aggregated sources.
        # Source tags are sorted for deterministic output.
        results = []
        for normalized, (surface, sources) in seen.items():
            extraction_source = "|".join(sorted(sources))
            results.append((surface, normalized, extraction_source))

        return results

    # ------------------------------------------------------------------
    # Span sources
    # ------------------------------------------------------------------

    def _dependency_phrases(self, doc: Doc) -> list:
        """
        Build compound noun phrases from the dependency tree.

        For each NOUN/PROPN head, collect contiguous left-side
        compound, amod, and nmod modifiers.  This complements noun
        chunks, particularly for patterns like "solid-state electrolyte"
        where spaCy may chunk incompletely.
        """
        max_len = self.cfg.get("max_candidate_length", 4)
        spans = []

        for token in doc:
            if token.pos_ not in ("NOUN", "PROPN"):
                continue

            left_mods = [
                child for child in token.lefts
                if child.dep_ in ("compound", "amod", "nmod")
                and child.pos_ in ("NOUN", "PROPN", "ADJ")
            ]

            if not left_mods:
                continue

            start  = min(t.i for t in left_mods)
            end    = token.i + 1
            length = end - start

            if 1 < length <= max_len:
                spans.append(doc[start:end])

        return spans

    def _ngram_spans(self, doc: Doc) -> list:
        """
        Generate token n-gram spans as a lightweight recall backoff.

        Only spans that contain at least one NOUN or PROPN are kept,
        which keeps the candidate set manageable.  Use ngram_max ≤ 3
        for good speed/recall balance.
        """
        cfg   = self.cfg
        n_min = cfg.get("ngram_min", 2)
        n_max = min(cfg.get("ngram_max", 3), cfg.get("max_candidate_length", 4))
        n_tok = len(doc)

        spans = []
        for n in range(n_min, n_max + 1):
            for i in range(n_tok - n + 1):
                span = doc[i : i + n]
                if any(t.pos_ in ("NOUN", "PROPN") for t in span):
                    spans.append(span)

        return spans

    # ------------------------------------------------------------------
    # Filtering & normalization helpers
    # ------------------------------------------------------------------

    def _trim_edges(self, tokens: list) -> list:
        """
        Strip stopwords, determiners, adpositions, conjunctions, and
        punctuation from both ends of a token list.
        """
        trim_pos = {"DET", "ADP", "CCONJ", "SCONJ", "PUNCT", "SPACE"}

        while tokens and (tokens[0].is_stop or tokens[0].pos_ in trim_pos):
            tokens = tokens[1:]

        while tokens and (tokens[-1].is_stop or tokens[-1].pos_ in trim_pos):
            tokens = tokens[:-1]

        return tokens

    def _passes_filter(self, tokens: list) -> bool:
        """
        Apply rule-based quality filters to an already-trimmed token list.

        Checks (all configurable):
        - token-count within [min_candidate_length, max_candidate_length]
        - presence of at least one NOUN or PROPN
        - no internal punctuation tokens
        - not a purely numeric phrase
        - boundary tokens are not stopwords or determiners
        """
        cfg     = self.cfg
        max_len = cfg.get("max_candidate_length", 4)
        min_len = cfg.get("min_candidate_length", 1)

        if not (min_len <= len(tokens) <= max_len):
            return False

        if not any(t.pos_ in ("NOUN", "PROPN") for t in tokens):
            return False

        if any(t.is_punct or t.pos_ in ("PUNCT", "SPACE") for t in tokens):
            return False

        if all(t.like_num or t.is_digit for t in tokens):
            return False

        if tokens[0].is_stop or tokens[0].pos_ == "DET":
            return False

        if tokens[-1].is_stop or tokens[-1].pos_ == "DET":
            return False

        return True

    def _normalize(self, tokens: list) -> str:
        """
        Produce a canonical string from a (trimmed) token list.

        Normalization strength is set by ``normalization_mode`` in config:

        ``"conservative"`` (default)
            Lowercase + whitespace normalization only.
            Preserves original word forms — safe for technical terminology.

        ``"lemma"``
            Additionally lemmatizes NOUN and ADJ tokens using spaCy.
            Useful for collapsing plurals (e.g. "batteries" → "battery").
            Requires ``apply_light_lemmatization: true`` in config.
        """
        cfg       = self.cfg
        norm_mode = cfg.get("normalization_mode", "conservative")
        lemmatize = (
            norm_mode == "lemma"
            and cfg.get("apply_light_lemmatization", False)
        )

        parts = []
        for t in tokens:
            if lemmatize and t.pos_ in ("NOUN", "ADJ"):
                parts.append(t.lemma_)
            else:
                parts.append(t.text)

        text = self._ws_re.sub(" ", " ".join(parts)).strip()

        if cfg.get("lowercase", True):
            text = text.lower()

        return text

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def debug_text(self, text: str) -> None:
        """
        Trace the full extraction pipeline for a single text string.

        Prints, for every candidate span attempted:
          - the raw span text
          - which source produced it (noun_chunk / dep_phrase / ngram)
          - the POS tag and dependency label of each token
          - the outcome: KEPT, or the reason it was dropped

        Useful for diagnosing why an expected phrase is absent from results.

        Parameters
        ----------
        text : str
            A single title or abstract string.
        """
        doc = self.nlp(text)

        print(f"\n{'─'*70}")
        print(f"TEXT: {text}")
        print(f"{'─'*70}")
        print("TOKEN-LEVEL PARSE:")
        print(f"  {'token':<20} {'POS':<8} {'DEP':<12} {'stop'}")
        print(f"  {'─'*20} {'─'*8} {'─'*12} {'─'*5}")
        for t in doc:
            print(f"  {t.text:<20} {t.pos_:<8} {t.dep_:<12} {t.is_stop}")

        tagged_spans = []
        if self.cfg.get("use_noun_chunks", True):
            for span in doc.noun_chunks:
                tagged_spans.append((span, "noun_chunk"))
        if self.cfg.get("use_dependency_phrases", True):
            for span in self._dependency_phrases(doc):
                tagged_spans.append((span, "dep_phrase"))
        if self.cfg.get("use_ngram_backoff", True):
            for span in self._ngram_spans(doc):
                tagged_spans.append((span, "ngram"))

        print(f"\nSPAN PIPELINE ({len(tagged_spans)} raw spans):")
        print(f"  {'span':<30} {'source':<12} outcome")
        print(f"  {'─'*30} {'─'*12} {'─'*40}")

        seen: dict = {}
        for span, tag in tagged_spans:
            raw_text = span.text.strip()
            tokens = self._trim_edges(list(span))

            if not tokens:
                print(f"  {raw_text:<30} {tag:<12} DROPPED — empty after edge trimming")
                continue

            trimmed_text = " ".join(t.text for t in tokens)

            if not self._passes_filter(tokens):
                # Report the specific reason
                cfg = self.cfg
                max_len = cfg.get("max_candidate_length", 4)
                min_len = cfg.get("min_candidate_length", 1)
                if not (min_len <= len(tokens) <= max_len):
                    reason = f"DROPPED — length {len(tokens)} outside [{min_len}, {max_len}]"
                elif not any(t.pos_ in ("NOUN", "PROPN") for t in tokens):
                    reason = "DROPPED — no NOUN/PROPN"
                elif any(t.is_punct or t.pos_ in ("PUNCT", "SPACE") for t in tokens):
                    reason = "DROPPED — contains punctuation"
                elif all(t.like_num or t.is_digit for t in tokens):
                    reason = "DROPPED — purely numeric"
                elif tokens[0].is_stop or tokens[0].pos_ == "DET":
                    reason = f"DROPPED — starts with stopword/DET '{tokens[0].text}'"
                elif tokens[-1].is_stop or tokens[-1].pos_ == "DET":
                    reason = f"DROPPED — ends with stopword/DET '{tokens[-1].text}'"
                else:
                    reason = "DROPPED — failed filter (unknown reason)"
                display = raw_text if raw_text == trimmed_text else f"{raw_text} → '{trimmed_text}'"
                print(f"  {display:<30} {tag:<12} {reason}")
                continue

            normalized = self._normalize(tokens)

            if not normalized:
                print(f"  {raw_text:<30} {tag:<12} DROPPED — empty after normalization")
                continue

            if normalized in self.blacklist:
                print(f"  {raw_text:<30} {tag:<12} DROPPED — blacklisted ('{normalized}')")
                continue

            if normalized in seen:
                existing_tag = seen[normalized][1]
                print(f"  {raw_text:<30} {tag:<12} DUPLICATE of '{normalized}' (already from {existing_tag})")
                seen[normalized][1].add(tag)
                continue

            seen[normalized] = [trimmed_text, {tag}]
            display = raw_text if raw_text == trimmed_text else f"{raw_text} → '{trimmed_text}'"
            print(f"  {display:<30} {tag:<12} KEPT → '{normalized}'")

        print(f"\nFINAL CANDIDATES ({len(seen)}):")
        for normalized, (surface, sources) in seen.items():
            print(f"  [{('|'.join(sorted(sources))):<25}]  {normalized}")
