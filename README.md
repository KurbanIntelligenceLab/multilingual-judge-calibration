# Language-Conditioned Rank Reversal in Agentic LLM Judges

This repository accompanies the current AAMAS manuscript of this study. It
contains the paper, saved judging scores, analysis code, and supporting records.
The manuscript is a revision of the same preprint study, not a new experiment
claimed to extend that preprint.

The question is whether localized judging inputs change scores and quality gates
when agent-produced code is held fixed. **Evaluator severity is not evaluator
quality, and consistency improvement is not an accuracy gain.**

## Current paper and release

- [Paper PDF](paper/main.pdf): eight pages total, with the main sections through
  Ethics ending on page seven; official AAMAS 2027 template, submission 1655.
- [LaTeX source](paper/main.tex), [bibliography](paper/references.bib), and
  [editable TikZ overview](paper/figures/fig_overview.tex).
- [Overleaf upload](releases/aamas2027_overleaf_upload.zip).
- [Compact reproducibility archive](releases/aamas2027_reproducibility.zip).
- [Reproduction guide](docs/REPRODUCE.md) and [release audit](docs/RELEASE_AUDIT.md).

## Setup

Use Python 3.11 or newer. The saved-score analyses require no provider calls or GPU.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

The core dependencies are pinned to the audited analysis versions. Optional
collection dependencies are in the `collection` extra of `pyproject.toml`;
embedding generation uses the `embeddings` extra and separately downloaded LaBSE
weights. Neither is required to verify the current saved-score results.

## Reproduce the current results

Run from the repository root, or the root of the extracted compact archive:

```bash
python paper/code/analysis/verify_claims.py
python paper/code/analysis/audit_mrewardbench_anchor.py
python scripts/analyze_replication.py
python scripts/analyze_panel_sensitivity.py
python scripts/audit_localization_and_transforms.py --bounded-link-only
python scripts/generate_aamas_figures.py
```

The verifier covers **90 listed numerical claims**, not every manuscript claim.
Separate ranking runners recompute the Table 2 comparisons:

```bash
python paper/code/analysis/recompute_devai_rankings.py
python paper/code/analysis/recompute_preference_rankings.py
```

Each ranking runner uses 1,000 task/item bootstrap replicates and may take a few
minutes. For the PDF, use pdfLaTeX, BibTeX, then pdfLaTeX twice in `paper/`, or
upload the Overleaf ZIP. No template dimensions have been modified.

## What the saved evidence shows

- DevAI: 55 tasks x eight languages x six evaluators x three developer-agent
  frameworks = 7,920 original judge runs, averaged to 2,640 task-level cells.
- Seven of fifteen evaluator pairs reverse observed mean severity order. The
  global residual-permutation test gives p = 0.001; it is not a separate
  significance test for every selected witness pair.
- Held-out cross-language ranking agreement: DevAI mean Kendall tau 0.650 to
  0.902; the M-RewardBench preference subset 0.430 to 0.900 after CBC.
- Gate-25 language spread: 27.9 to 16.4 percentage points on the fitted panel;
  omitted-task mean reduction 8.7 points, 95% percentile interval [3.8,14.0].
- The 1,440-call, ten-task MetaGPT repeat slice has mean SD 4.08 versus largest
  interaction 20.61, ratio 0.20 under the recorded decision rule.

### Correction to the previous accuracy claim

The former preference-panel gold-agreement increase from 68.7% to 76.6% was a
floating-point tie artifact. With a common absolute-margin tolerance of `1e-9`,
the 700-instance sample has **68.7% agreement both before and after CBC**.
The corrected audit is retained. CBC preserves the evaluator-panel mean exactly
on a complete balanced panel; no human-accuracy gain is claimed.

## Data and scope

`paper/analysis/` contains the canonical saved scores, bootstrap records, and
ablation summaries. `paper/analysis/aamas_2027/` contains the replication,
structural/translation diagnostics, canonical 1,098-label human export, and
separate preference panel. The preference subset uses the first 1,500 dataset
IDs (805 alpacaeval-easy and 695 alpacaeval-hard) across seven languages and five
evaluators; 35 of 52,500 margins are unavailable.

The full repository also retains the original curated judgment JSONs and
preference collection logs under `data/`. The compact archive includes the saved
score matrices and necessary audit records, not the full raw workspaces or
provider responses. Earlier EMNLP plotting/analysis outputs remain historical
records; they are not the current paper's figure-generation workflow.

Original provider snapshots, retry logs, and hardware metadata are incomplete.
Saved-score recomputation is supported, not exact provider-level replay. Holding
code fixed does not separate judge behavior from translation quality. The planned
requirement-level human test is blocked by a schema mismatch and binary-only
saved judgments. The complete-panel theorem does not establish general transfer
to unseen languages or selectively missing data. See the manuscript's limitations.

## Further documentation

- [Project explainer](docs/PROJECT_EXPLAINER.md)
- [Included-file rationale](docs/FILES_RATIONALE.md)
- [AI assistance](docs/AI_ASSISTANCE.md)
- [License](LICENSE)
