# AAMAS supplementary reproducibility material

This package supports **Language-Conditioned Rank Reversal in Agentic LLM Judges**.
The manuscript is submitted separately. This archive contains the saved inputs,
analysis code, judge prompt sources, AI-assistance disclosure and available
revision-prompt excerpts, and supporting records for auditing its results.

## Layout

```text
agent_as_a_judge/          Language registry and localized judge prompt sources
paper/code/analysis/      CBC, numerical verifier, anchor and ranking audits
paper/analysis/           DevAI scores, bootstrap records and ablation summaries
paper/analysis/aamas_2027/ Preference inputs, human labels and replication records
paper/figures/            Editable illustrative TikZ source; plots regenerate here
scripts/                  Saved-score runners and figure-generation code
docs/                     AI disclosure and detailed reproduction instructions
requirements.txt          Pinned CPU analysis dependencies
MANIFEST.json             Included-file SHA-256 inventory
```

## Setup

Extract the ZIP and run commands from its root with Python 3.11 or newer:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default commands need no API keys, provider calls, GPU, model downloads,
or second repository checkout.

## Reproduction steps

```bash
# 1. Listed numerical claims and corrected preference-anchor audit
python paper/code/analysis/verify_claims.py
python paper/code/analysis/audit_mrewardbench_anchor.py

# 2. Repeat-call, missingness and score-transform diagnostics
python scripts/analyze_replication.py
python scripts/analyze_panel_sensitivity.py
python scripts/audit_localization_and_transforms.py --bounded-link-only

# 3. Held-out ranking comparisons (1,000 bootstrap draws each)
python paper/code/analysis/recompute_devai_rankings.py
python paper/code/analysis/recompute_preference_rankings.py

# 4. Regenerate the three data plots
python scripts/generate_aamas_figures.py
```

Expected: 90 listed checks with zero mismatches; raw/CBC preference-anchor
agreement 68.7%/68.7%, zero true flips; repeat-call SD 4.08 and ratio 0.20.
Held-out raw/CBC mean Kendall tau is 0.650/0.902 for DevAI and 0.430/0.900 for
the preference subset. The ranking runners may take several minutes.

## Scope

Original and corrected anchor samples have different roles: the original
preserves the floating-point tie error for audit, while the corrected sample
contains current verdict flags. Neither is a new label collection.
The reference module `scripts/compute_meb_foundations.py` is included because
the DevAI saved-score runner imports its pure statistical functions. Its full
provider/workspace entry point is not a supported command in this archive.
Full parser/translation auditing likewise needs the original workspace tree;
use `--bounded-link-only` for the self-contained transform analysis.

The archive excludes the manuscript PDF, TeX compile assets, generated plots,
unrelated legacy analyses, raw provider responses, credentials, private reviews,
and meeting material. Original provider snapshots and retry histories are
incomplete; this package supports saved-score recomputation, not exact replay of
provider calls. Fixed-code localization still mixes translation quality with
judge behavior. Optional LaBSE generation requires extra dependencies and
weights; exact original embedding replay is not established.

See [detailed instructions](REPRODUCE.md) and
[AI assistance disclosure](AI_ASSISTANCE.md). The manuscript-compilation
instructions in the full guide apply to the complete repository, not this ZIP.

Code is under the included MIT license. Upstream data retain their original terms.
