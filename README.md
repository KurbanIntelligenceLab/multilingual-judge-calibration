# CBC Multilingual Judge Calibration

This repository is the standalone reproducibility package for the paper **"Rank Reversal in Multilingual LLM Judges: A Label-Free Double-Centering Calibrator."** It contains the manuscript source, the curated data needed for the second paper, and the minimal code required to reproduce the reported analyses.

## Repository layout

```text
cbc_multilingual_judge_calibration/
├── README.md
├── pyproject.toml
├── requirements.txt
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── acl.sty
│   ├── acl_natbib.bst
│   └── analysis/
├── scripts/
│   ├── run_internal_benchmark_analysis.py
│   ├── run_external_mrewardbench_analysis.py
│   ├── collect_mrewardbench_panel_scores.py
│   └── render_paper_figures.py
├── src/
│   ├── languages.py
│   ├── requirement_taxonomy.py
│   └── llm/
├── data/
│   ├── internal_benchmark/
│   │   └── judgments/
│   └── external_validation/
│       └── mrewardbench_panel/
│           ├── collection_logs/
│           └── analysis_1500_item/
└── docs/
    ├── FILES_RATIONALE.md
    └── PROJECT_EXPLAINER.md
```

## What is included

The repository already ships everything needed to reproduce the paper without making new API calls:

- the full internal eight-language judgment matrix used by CBC;
- the canonical five-evaluator, seven-language M-RewardBench collection logs, trimmed to the first 1,500 items per language;
- the saved internal and external analysis outputs used in the manuscript;
- the paper source and bibliography.

The external collection script is kept for completeness, but rerunning the paper does **not** require recollecting M-RewardBench scores.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Reproduction workflow

Run all commands from the repository root.

### 1. Internal benchmark analysis

```bash
python scripts/run_internal_benchmark_analysis.py
```

This regenerates the main benchmark outputs in `paper/analysis/`, including:

- `calibration_results.json`
- `rank_reversal_delta.csv`
- `beta_hat.csv`
- `leave_one_language_out.json`
- `decision_backbone_selection.json`

### 2. External M-RewardBench analysis

```bash
python scripts/run_external_mrewardbench_analysis.py --max-items-per-language 1500
```

This consumes the saved evaluator logs in `data/external_validation/mrewardbench_panel/collection_logs/` and rewrites the canonical external outputs in `data/external_validation/mrewardbench_panel/analysis_1500_item/`.

The most important external outputs are:

- `raw_vs_cbc_summary.json`
- `raw_vs_cbc_bootstrap_replicates.csv`
- `xu_judge_aware_btl_matrix.csv`
- `human_anchor_validation.json`

### 3. Regenerate the paper figures

```bash
python scripts/render_paper_figures.py
```

This refreshes the figure assets referenced by `paper/main.tex`, including:

- `paper/analysis/beta_heatmap.png`
- `paper/analysis/cbc_before_after.png`
- `paper/analysis/external_cbc_before_after.png`
- `paper/analysis/convergence_curve.png`
- `paper/analysis/rank_reversal_strength.png`

### 4. Compile the paper

If a LaTeX toolchain is available locally:

```bash
cd paper
latexmk -pdf main.tex
```

If `latexmk` is unavailable, standard BibTeX compilation also works:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Optional: recollect a new M-RewardBench panel

The paper already includes saved collection logs. This step is only needed if you want to rebuild the panel from the public benchmark instances.

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

Example command for one evaluator:

```bash
python scripts/collect_mrewardbench_panel_scores.py ^
  --models openrouter/openai/gpt-4o-2024-08-06 ^
  --languages eng_Latn arb_Arab tur_Latn zho_Hans hin_Deva jpn_Jpan spa_Latn ^
  --max-items 1500 ^
  --output-dir data/external_validation/mrewardbench_panel/collection_logs/gpt4o_2024_08_06 ^
  --output-stem run ^
  --resume
```

Each collection run writes:

- `<output-stem>_raw_scores.jsonl`
- `<output-stem>_pair_margins.csv`
- `<output-stem>_metadata.json`

After collecting all evaluators, rerun:

```bash
python scripts/run_external_mrewardbench_analysis.py --max-items-per-language 1500
```

## Data provenance

- The internal benchmark data comes from the multilingual Agent-as-a-Judge benchmark plus the three-language extension used in the paper.
- The external task instances come from the public M-RewardBench release.
- The evaluator-by-language matrices in this repository are derived from self-collected model outputs and are retained under `data/external_validation/mrewardbench_panel/`.

## Additional documentation

- `docs/FILES_RATIONALE.md` explains why each file group is included.
- `docs/PROJECT_EXPLAINER.md` gives a short plain-language overview of the method and experiments.