# Ranking stage — usage notes

Unsupervised keyword candidate ranker for English patent text.
Produces a scored DataFrame with one row per `(row_id, candidate)` pair.

---

## Inputs

| Source | What it is |
|--------|-----------|
| Source DataFrame | Original patents with `title` and `abstract` |
| `candidates_list.csv` | Unique candidates from the extraction stage |
| `candidates_mapping.csv` | Candidate-to-document occurrence mapping |

The extractor (`candidate_extractor.py`) produces the last two files via `save_outputs()`.

---

## Quick start

```python
from ranking import CandidateRanker

ranker = CandidateRanker("ranking_config.json")
scores = ranker.rank_from_config()   # loads data from paths in config
ranker.save(scores)                  # writes outputs/ranked_candidates.csv
```

Or pass a pre-loaded DataFrame directly:

```python
import pandas as pd
df = pd.read_parquet("data/df_batt.parquet")
df = df.set_index("APPLN_ID").rename(columns={"APPLN_TITLE": "title", "APPLN_ABSTR": "abstract"})
scores = ranker.rank(df)
```

Demo (in-memory, no files required):

```bash
python rank_example.py
```

---

## Canonical internal representation

Before scoring, all inputs are merged into a canonical table with one row per unique `(row_id, candidate)`:

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | str | Original DataFrame index value |
| `candidate_id` | int | Integer id from candidates list |
| `candidate` | str | Normalized candidate phrase |
| `surface_form` | str | First-seen surface form |
| `extraction_source` | str | Pipe-joined extraction method tags |
| `tf` | int | Within-document occurrence count |

**tf computation:** `tf` is derived from the raw mapping file before any deduplication. When `keep_duplicates: true` is set in the extractor config, the mapping may contain repeated `(candidate_id, row_id)` rows; each repeat counts as one additional occurrence. If the mapping only records unique pairs, `tf = 1` everywhere (binary TF).

---

## Output schema

One row per unique `(row_id, candidate)`.

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | str | Original DataFrame index value |
| `candidate` | str | Normalized candidate phrase |
| `surface_form` | str | First-seen surface form |
| `extraction_source` | str | Pipe-joined source tags |
| `tf` | int | Within-document term frequency |
| `df` | int | Document frequency (# distinct row_ids with this candidate) |
| `idf` | float | Smoothed IDF score |
| `tfidf_score` | float | `tf × idf` |
| `semantic_score` | float32 | Cosine similarity (document, candidate) |
| `rank_tfidf` | int | Within-document rank by TF-IDF (1 = best) |
| `rank_semantic` | int | Within-document rank by semantic score (1 = best) |

---

## TF-IDF

**Formula (sklearn smooth IDF):**

```
idf(c) = log((N + 1) / (df(c) + 1)) + 1
tfidf_score(c, d) = tf(c, d) × idf(c)
```

where:
- `N` = number of distinct `row_id` values in the canonical table (documents with at least one candidate)
- `df(c)` = number of distinct `row_id` values containing candidate `c`
- Smoothing prevents division by zero and keeps `idf ≥ 1`

**Document-frequency granularity:** `df` is computed at the `row_id` level. Since `row_id` values in this pipeline are the original DataFrame index values (which may encode section information), `df` reflects section-level frequency, not patent-level frequency.

---

## Semantic scoring

**Document representation:** `[CLS] <title> [SEP] <abstract>` (configurable via `text_formatting.mode`).

**Default encoder:** `all-MiniLM-L6-v2` — 22M parameters, 384-dimensional embeddings, fast on CPU and Apple Silicon. Max input: **256 tokens** (longer abstracts are silently truncated).

**Score:** cosine similarity between the document embedding and the candidate phrase embedding. Embeddings are L2-normalized, so cosine similarity = dot product.

**Efficiency:** each unique candidate and each unique document is embedded exactly once, then reused across all pairs.

---

## Swapping the encoder

1. Subclass `EmbedderBase` in `ranking/embedder.py` and implement `embed()`.
2. Register your backend in `get_embedder()`.
3. Set `"backend": "<your_name>"` and `"model_name": "..."` in `ranking_config.json`.

No other files need to change.

To use a different sentence-transformers model without subclassing, just change `"model_name"` in the config.

---

## Changing the document format

Register a new formatter in `ranking/formatting.py`:

```python
@register_formatter("my_format")
def format_my(title, abstract, **kwargs) -> str:
    ...
```

Set `"text_formatting": {"mode": "my_format"}` in config. No other files need to change.

---

## Configuration reference (`ranking_config.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `data_path` | `"data/df_batt.parquet"` | Source data file (.parquet or .csv) |
| `id_column` | `"APPLN_ID"` | Column to use as row identifier |
| `title_column` | `"APPLN_TITLE"` | Title column name |
| `abstract_column` | `"APPLN_ABSTR"` | Abstract column name |
| `candidates_list_path` | `"outputs/candidates_list.csv"` | Extractor candidates list |
| `candidates_mapping_path` | `"outputs/candidates_mapping.csv"` | Extractor mapping file |
| `candidates_list_columns` | `{...}` | Column name mapping for candidates list |
| `candidates_mapping_columns` | `{...}` | Column name mapping for mapping file |
| `text_formatting.mode` | `"cls_sep"` | Document format: `"cls_sep"` or `"plain"` |
| `text_formatting.cls_token` | `"[CLS]"` | CLS special token string |
| `text_formatting.sep_token` | `"[SEP]"` | SEP special token string |
| `semantic.model_name` | `"all-MiniLM-L6-v2"` | sentence-transformers model |
| `semantic.batch_size` | `64` | Embedding batch size |
| `tfidf.smoothed_idf` | `true` | Use sklearn smooth IDF formula |
| `output_path` | `"outputs/ranked_candidates.csv"` | Output CSV path |

---

## Limitations and next steps

- **Score fusion:** TF-IDF and semantic scores are on different scales; not yet fused. Planned for a later stage.
- **Diversity reranking (MMR):** not implemented.
- **Supervision:** no fine-tuning or gold labels; this is a fully unsupervised baseline.
- **Token limit:** abstracts longer than 256 tokens are truncated by `all-MiniLM-L6-v2`. Longer-context models (e.g. `all-mpnet-base-v2`, 384-token limit) can be dropped in via config.
- **Scalability:** for corpora > 1M documents, the document embedding matrix (~1.5 GB at 384-dim float32 × 1M) may exceed RAM. Chunk by `row_id` batch in that case.
- **Section-level IDF:** `df` is currently section-level (one `row_id` per section). To get patent-level IDF, strip the section suffix from `row_id` before building the canonical table.
