"""example.py

Minimal usage example for CandidateExtractor.
Run from the keyword_extraction/ directory:

    python example.py
"""

import pandas as pd
from candidate_extractor import CandidateExtractor, save_outputs


# ── Tiny in-memory patent dataset ───────────────────────────────────────────

data = {
    "title": [
        "Solid-state lithium-ion battery with ceramic electrolyte separator",
        "Method for electrochemical reduction of carbon dioxide to formic acid",
        "Deep learning system for real-time anomaly detection in industrial sensors",
        "Photovoltaic module with anti-reflective coating and improved light trapping",
        None,                                        # missing title — handled gracefully
    ],
    "abstract": [
        (
            "A solid-state battery comprising a lithium metal anode, a ceramic "
            "electrolyte layer, and a cathode active material. The ceramic electrolyte "
            "exhibits high ionic conductivity and suppresses lithium dendrite growth, "
            "improving cycle stability and energy density."
        ),
        (
            "Disclosed is an electrochemical cell for converting carbon dioxide into "
            "formic acid using a bismuth catalyst supported on a carbon substrate. "
            "The faradaic efficiency exceeds 90% at low overpotential in aqueous "
            "potassium bicarbonate electrolyte solution."
        ),
        (
            "A convolutional neural network trained on multivariate time-series data "
            "from industrial sensors detects anomalies in real time with sub-second "
            "latency. The model architecture includes attention mechanisms and "
            "residual connections to improve robustness under sensor noise."
        ),
        (
            "A photovoltaic module comprising silicon solar cells coated with a "
            "titanium dioxide anti-reflective layer. The nanostructured surface "
            "increases light absorption by reducing reflection losses across the "
            "visible and near-infrared spectrum."
        ),
        (
            "This embodiment relates to a system for wireless power transfer using "
            "resonant inductive coupling between a transmitter coil and a receiver coil."
        ),
    ],
}

df = pd.DataFrame(data)

print("─" * 60)
print("Input DataFrame")
print("─" * 60)
print(df[["title", "abstract"]].to_string(max_colwidth=55))
print()

# ── Run extraction ───────────────────────────────────────────────────────────

extractor = CandidateExtractor("config.json")
candidates = extractor.extract(df)

print("─" * 60)
print(f"Output: {len(candidates)} candidate occurrences")
print(f"Columns: {list(candidates.columns)}")
print("─" * 60)
print(candidates.to_string(index=False))
print()

# ── Save outputs ─────────────────────────────────────────────────────────────

list_path, mapping_path = save_outputs(candidates, output_dir="outputs/")
print(f"candidates_list.csv    → {list_path}  ({len(candidates['candidate'].unique())} unique candidates)")
print(f"candidates_mapping.csv → {mapping_path}  ({len(candidates)} occurrences)")
print()

# ── Aggregation examples useful for downstream ranking ───────────────────────

print("─" * 60)
print("Candidates per source section")
print("─" * 60)
print(candidates.groupby("source_section")["candidate"].count())
print()

print("─" * 60)
print("Top candidates by row_id (title + abstract combined)")
print("─" * 60)
summary = (
    candidates
    .groupby(["row_id", "candidate"])["source_section"]
    .apply(list)
    .reset_index()
    .rename(columns={"source_section": "found_in"})
)
print(summary.to_string(index=False))
