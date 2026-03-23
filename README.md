# Keyword Candidate Extractor — Usage Notes

Lightweight, rule-based baseline for extracting keyword candidate phrases from
English patent `title` and `abstract` text using spaCy.

---

## What it does

Reads a pandas DataFrame with `title` and `abstract` columns and returns a
**candidate-level DataFrame** — one row per extracted phrase occurrence — with:

| Column | Description |
|---|---|
| `row_id` | Original index from the input DataFrame |
| `candidate` | Normalized candidate string |
| `surface_form` | Original text (pre-normalization, post-trimming) |
| `source_section` | `"title"` or `"abstract"` |

Extraction uses three complementary span sources (each individually togglable):

1. **spaCy noun chunks** — high-precision base
2. **Dependency-tree compound phrases** — catches compound+amod patterns spaCy
   may chunk incompletely (e.g. *solid-state electrolyte*)
3. **POS-filtered n-gram backoff** — lightweight recall boost for 2–3 token
   phrases containing at least one NOUN/PROPN

Candidates then pass through:
- Edge trimming (stopwords, determiners, punctuation stripped from boundaries)
- Rule-based quality filters (length, POS content, punctuation, numerics)
- Blacklist check (exact normalized-form match)
- Within-(row, section) deduplication on normalized form

---

## Quick start

```python
import pandas as pd
from candidate_extractor import CandidateExtractor

df = pd.DataFrame({
    "title":    ["Solid-state lithium-ion battery with ceramic electrolyte"],
    "abstract": ["A battery comprising a lithium anode and ceramic electrolyte layer..."],
})

extractor  = CandidateExtractor("config.json")
candidates = extractor.extract(df)
print(candidates)
```

Run the bundled example:

```bash
python example.py
```

---

## Requirements

```bash
pip install spacy pandas
python -m spacy download en_core_web_sm
```

Use `en_core_web_lg` or `en_core_web_trf` (transformer) for better accuracy on
noisy or short texts — just update `spacy_model` in `config.json`.

---

## How to edit the config (`config.json`)

All parameters live in one flat JSON file.  The most important ones:

| Key | Default | Effect |
|---|---|---|
| `spacy_model` | `"en_core_web_sm"` | spaCy model to load |
| `max_candidate_length` | `4` | Maximum tokens per candidate phrase |
| `min_candidate_length` | `1` | Minimum tokens (set to 2 to drop single-token candidates) |
| `use_noun_chunks` | `true` | Enable noun chunk extraction |
| `use_dependency_phrases` | `true` | Enable dependency tree expansion |
| `use_ngram_backoff` | `true` | Enable n-gram backoff |
| `ngram_min` / `ngram_max` | `2` / `3` | N-gram size range (backoff only) |
| `normalization_mode` | `"conservative"` | `"conservative"` = lowercase only; `"lemma"` = also lemmatize NOUN+ADJ |
| `apply_light_lemmatization` | `false` | Must be `true` for lemma mode to activate |
| `lowercase` | `true` | Lowercase normalized form |
| `trim_stopword_edges` | `true` | Strip stopwords/determiners from phrase boundaries |
| `keep_duplicates` | `true` | Keep multiple occurrences of same candidate within a section |
| `blacklist_path` | `"blacklist.txt"` | Path to blacklist file |
| `title_weight` / `abstract_weight` | `2.0` / `1.0` | Stored metadata for future ranking; not used during extraction |
| `batch_size` | `64` | Number of texts processed per `nlp.pipe` batch |

---

## How to edit the blacklist (`blacklist.txt`)

Plain-text file, one phrase per line.

- Lines starting with `#` are comments.
- Matching is **exact** on the normalized (lowercased) form.
- A phrase is suppressed only when it matches **exactly** — compounds are safe.
  Blacklisting `"system"` will drop the lone word *system* but will **not**
  affect *battery management system*.

To add entries, just append lines:

```
fuel cell
active material     # too generic on its own
nanostructured surface
```

---

## Input/output contract

**Input**

- pandas DataFrame
- Must have at least one of `title`, `abstract` columns
- Missing / empty cells are silently skipped
- Row index is preserved as `row_id`

**Output**

- pandas DataFrame at the **candidate occurrence** level
- If a phrase appears in both title and abstract of the same row, both
  occurrences are retained (set `keep_duplicates: false` in config to
  deduplicate within each section)
- Normalized candidates are safe for direct groupby/aggregation

---

## Limitations

- English only (spaCy model must be an English pipeline).
- Noun-chunk quality depends on spaCy model size: `sm` is fast but may miss
  chunking in complex technical sentences; `lg` / `trf` will improve precision.
- N-gram backoff generates many candidates on long abstracts; increasing
  `ngram_max` beyond 3 increases recall but also noise.
- Lemmatization via spaCy is morphological, not domain-aware — unusual
  technical terms may be incorrectly lemmatized; use `"conservative"` mode
  if this is a concern.
- No ranking is implemented. The output is intended as input to a ranking
  stage (TF-IDF, TextRank, SIFRank, etc.).

---

## Extending the extractor

The class is structured so that later additions are isolated:

| What to plug in | Where |
|---|---|
| Candidate ranking / scoring | After `extract()` — add a `rank(candidates_df, df)` function |
| MMR diversification | Post-ranking step on the candidates DataFrame |
| Section-based score priors | Use `title_weight` / `abstract_weight` from config in ranking |
| Alternative span sources | Add a method `_my_spans(doc)` and call it in `_extract_from_doc` |
| Domain-specific filters | Extend `_passes_filter()` |
