# Included files and selection rationale

## Current release

| Path | Purpose |
|---|---|
| `paper/main.tex`, `references.bib`, `main.pdf` | Audited AAMAS manuscript and 27 cited references. |
| `paper/aamas.cls`, `ACM-Reference-Format.bst`, `by.pdf` | Unmodified official AAMAS 2027 template assets. |
| `paper/figures/` | Editable TikZ overview and three plots generated from saved scores. |
| `paper/code/analysis/` | CBC estimators, 90-check verifier, preference-anchor audit, and saved-score ranking runners. |
| `paper/analysis/` | Canonical task/run scores, ablations, intervals, and original saved analysis records. |
| `paper/analysis/aamas_2027/` | Replication scores/metadata, canonical human labels, workspace hashes, preference subset, and follow-up diagnostics. |
| `scripts/generate_aamas_figures.py`, replication/follow-up scripts | Current saved-data figure and analysis entry points. |
| `agent_as_a_judge/languages.py`, `agent_as_a_judge/module/prompt/` | Original registry and localized prompt sources for inspection and optional embedding generation. No provider credentials are shipped. |
| `docs/` | Current reproduction, scope, corrections, and general AI-assistance disclosures. |
| `scripts/build_supplementary_archive.py`, `docs/README_SUPPLEMENT.md` | Build a local supplementary ZIP outside the public checkout, with dedicated instructions and a SHA-256 inventory. Generated ZIPs are uploaded manually and are not tracked. |

## Historical data and optional collection

The previously tracked curated `data/internal_benchmark/judgments/` and
`data/external_validation/mrewardbench_panel/collection_logs/` remain available.
The older standalone internal analysis and plotting entry points are historical
tools, not the current AAMAS figure instructions. The legacy external analysis
entry point now omits missing margins from weighted means and pair comparisons
and uses the same `1e-9` gold-margin tolerance as the current audit. Its canonical
summaries and anchor outputs are corrected; output metadata uses relative paths.

The compact archive excludes manuscript duplicates, compile assets, generated
plots, packaging-only code, and unrelated information-theoretic/requirement-type
outputs. It retains the statistical reference modules imported by the ranking
runners, the original and corrected anchor samples for distinct audit purposes,
and the available repeat-call metadata files, which contain different records.
It does not include full raw judgment/workspace trees, provider
responses, or collection credentials. Full parser/translation/workspace auditing
requires original benchmark trees beyond the compact data. The embedding record
supports aggregate checks; exact model/runtime and prompt-variant metadata are
incomplete. The legacy Gaussian `theoretical_bound` field is not the current
paper's Hoeffding proposition and supports no reported claim.

## Material kept local

Private reviewer screenshots/rebuttals, advisor meeting material, the old local
preprint-template draft, obsolete supplement planning notes, generated LaTeX
logs and auxiliaries, scratch/backup files, and duplicate root CSV exports are
excluded from publication. The root `data/human_gold.csv` is an obsolete
1,056-label export; the release's canonical human export has 1,098 labels.
These local files are preserved, not erased. The publication uses an explicit
release file selection rather than adding every untracked file.

The prior human-anchor sample is preserved as `human_anchor_sample_original.csv`
for the tie-artifact audit. `human_anchor_sample.csv` now contains corrected flags;
its original sample IDs and margins are unchanged within the audit tolerance.
