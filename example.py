"""example.py

Run candidate extraction on data/df_batt.parquet.

Usage (from keyword_extraction/):
    python example.py
"""

import pandas as pd
from candidate_extractor import CandidateExtractor, save_outputs


# ── Load data ────────────────────────────────────────────────────────────────

df = pd.read_parquet("data/df_batt.parquet")

# Set APPLN_ID as index; rename columns to the names the extractor expects
df = (
    df[["APPLN_ID", "APPLN_TITLE", "APPLN_ABSTR"]]
    .set_index("APPLN_ID")
    .rename(columns={"APPLN_TITLE": "title", "APPLN_ABSTR": "abstract"})
)

print(f"Loaded {len(df):,} patents")
print(df.head(3).to_string(max_colwidth=80))
print()

# ── Run extraction ───────────────────────────────────────────────────────────

extractor = CandidateExtractor("config.json")
candidates = extractor.extract(df)

print(f"Extracted {len(candidates):,} candidate occurrences")
print(f"Unique candidates: {candidates['candidate'].nunique():,}")
print()

# ── Save outputs ─────────────────────────────────────────────────────────────

list_path, mapping_path = save_outputs(candidates, output_dir="outputs/")
print(f"candidates_list.csv    → {list_path}")
print(f"candidates_mapping.csv → {mapping_path}")
