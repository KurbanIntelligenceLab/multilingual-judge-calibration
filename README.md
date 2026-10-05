# Language-Conditioned Rank Reversal in Agentic LLM Judges

Reproducibility repository for the AAMAS manuscript **Language-Conditioned Rank
Reversal in Agentic LLM Judges**. It contains the manuscript, localized judge
prompts, saved score matrices, replication records, and analysis code.

The study holds agent-produced code fixed while changing the language of judging
inputs. It measures evaluator severity reversals and changes in quality-gate
decisions, then evaluates label-free consensus-based calibration (CBC).
**Greater consistency does not establish greater accuracy.**

## Repository layout

```text
.
├── paper/
│   ├── main.tex, main.pdf          # Current AAMAS manuscript
│   ├── references.bib             # Bibliography
│   ├── figures/                   # TikZ overview and three data plots
│   ├── code/analysis/             # CBC, numerical checks, ranking and anchor audits
│   └── analysis/
│       ├── task_level_scores.csv  # Canonical DevAI task means
│       ├── run_level_scores.csv   # Framework-level saved scores
│       └── aamas_2027/
│           ├── data/              # Human labels, workspace hashes, preference panel
│           └── results/           # Repeat scores, metadata, audits and diagnostics
├── agent_as_a_judge/
│   ├── languages.py               # Language and framework registry
│   └── module/prompt/             # Copies of localized judge prompt sources
├── scripts/                       # Analysis, figure and packaging entry points
├── src/                           # Original standalone calibration utilities
├── data/                          # Curated judgments and preference collection logs
├── docs/                          # Reproduction, scope and disclosure documentation
├── requirements.txt               # Pinned dependencies for saved-score analyses
└── pyproject.toml                 # Package metadata and optional dependencies
```

The `agent_as_a_judge/` directory is part of **this repository**. Inspecting its
prompt sources and running the saved-score analyses require no second checkout.
It contains selected sources, not the complete provider-execution framework.
Earlier analysis artifacts remain as historical records; the commands below
identify the current manuscript workflow. See [file rationale](docs/FILES_RATIONALE.md).

## Setup

Use Python 3.11 or newer. The default analyses run on CPU using saved records;
they require no API credentials, provider calls, or model downloads.

```bash
git clone https://github.com/KurbanIntelligenceLab/multilingual-judge-calibration.git
cd multilingual-judge-calibration
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional collection dependencies are in the `collection` extra of
`pyproject.toml`. Optional embedding generation requires the `embeddings` extra
and LaBSE weights. Neither is needed for the saved-score checks below.

## Reproduce the results

Run commands from the repository root. The supplementary ZIP uses the same paths;
after extracting it, start with environment setup and run these analyses.

### 1. Numerical and accuracy checks

```bash
python paper/code/analysis/verify_claims.py
python paper/code/analysis/audit_mrewardbench_anchor.py
```

Expected: **90 listed numerical checks, zero mismatches**. The corrected
preference-anchor audit reports 68.7% agreement under both raw scores and CBC,
with zero true decision flips. The former accuracy gain was a floating-point
tie artifact; the original sample is retained for audit provenance.

### 2. Replication and sensitivity diagnostics

```bash
python scripts/analyze_replication.py
python scripts/analyze_panel_sensitivity.py
python scripts/audit_localization_and_transforms.py --bounded-link-only
```

The 1,440-call repeat slice covers ten MetaGPT tasks. Its mean repeat-call SD is
4.08 versus largest estimated interaction 20.61, ratio 0.20 under the recorded
decision rule. The sensitivity runner checks random missingness, task variation,
and repeat SD by language. The bounded-link command uses saved scores without
requiring the original code-workspace tree.

### 3. Held-out ranking comparisons

```bash
python paper/code/analysis/recompute_devai_rankings.py
python paper/code/analysis/recompute_preference_rankings.py
```

These runners use 1,000 bootstrap draws and may take several minutes. They
reproduce the printed Table 2 estimates and intervals. Three DevAI method/draw
differences near numerical rank ties are documented; printed results match.

| Panel | Raw mean Kendall tau | CBC mean Kendall tau |
|---|---:|---:|
| DevAI | 0.650 | 0.902 |
| M-RewardBench subset | 0.430 | 0.900 |

### 4. Regenerate figures

```bash
python scripts/generate_aamas_figures.py
```

This generates the three data plots under `paper/figures/`. The explanatory
overview is editable TikZ source in [fig_overview.tex](paper/figures/fig_overview.tex).

### 5. Compile the manuscript

Run from the full repository, with pdfLaTeX and BibTeX installed:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: eight pages total, five tables, and four figures. The main sections
through Ethics end on page seven. The official AAMAS 2027 template dimensions
are unchanged. The supplementary ZIP omits the separately submitted manuscript
and its compile assets.

## Evidence and reproduction limits

- DevAI contains 55 tasks across eight languages, six evaluators, and three
  developer-agent frameworks: 7,920 original judging runs and 2,640 task means.
- Seven of fifteen evaluator pairs reverse observed mean severity order. The
  permutation p = 0.001 tests the total reversal count, not each witness pair.
- At gate 25, mean language spread falls from 27.9 to 16.4 percentage points on
  the fitted panel. The omitted-task mean reduction is 8.7 points, 95% interval
  [3.8, 14.0].
- The preference subset contains 1,500 items across seven languages and five
  evaluators, with 35 unavailable margins among 52,500 cells. Missing margins
  are excluded from means and pairwise comparisons.

Saved-score recomputation is supported. Exact provider-level replay is not
established because original provider snapshots, retry histories, and hardware
metadata are incomplete. Fixed code still leaves judge behavior mixed with
translation quality. Requirement-level human accuracy remains untested because
of a schema mismatch and binary-only saved judgments. See the
[reproduction guide](docs/REPRODUCE.md) for assumptions and detailed limits.

## Prepare supplementary material

The submission ZIP is generated locally and uploaded manually to OpenReview;
generated ZIPs are not kept in the public repository. From the full checkout:

```bash
python scripts/build_supplementary_archive.py --output ../aamas2027_reproducibility.zip
```

Choose a destination outside the checkout. The archive includes a dedicated
README, disclosure, prompt sources, analysis dependencies, scripts, score
matrices, and supporting records. It omits manuscript duplicates, raw provider
responses, old plots, unrelated analyses, private notes, and credentials.
Its `MANIFEST.json` lists every included file with its SHA-256 digest.

## Paper and documentation

- [Paper PDF](paper/main.pdf), [LaTeX source](paper/main.tex), and [bibliography](paper/references.bib)
- [Reproduction guide](docs/REPRODUCE.md)
- [Project explainer](docs/PROJECT_EXPLAINER.md)
- [File rationale](docs/FILES_RATIONALE.md)
- [AI assistance and available revision-prompt excerpts](docs/AI_ASSISTANCE.md)
- [Release audit](docs/RELEASE_AUDIT.md)

## License

Code is provided under the [MIT license](LICENSE). Benchmark data and other
upstream material retain their original licenses and terms. No model weights
or API credentials are included.
