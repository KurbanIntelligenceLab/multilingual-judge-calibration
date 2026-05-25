# Included Files and Selection Rationale

This repository was carved out of a larger monorepo to isolate the second paper's reproducibility package. Every included path below is required either to rerun the reported analyses, regenerate paper figures/tables, verify reported costs, or inspect the exact manuscript state used for submission.

## Included directories

`paper/`
Contains the submission manuscript, bibliography, and the internal analysis artifacts consumed directly by the paper.

`paper/analysis/`
Stores the internal benchmark outputs used throughout the manuscript, including the recovered bias matrix, bootstrap summaries, leave-one-language-out diagnostics, ablations, decision-level evaluation, and generated figure assets. Obsolete extras were removed; the retained files are either cited directly, consumed by the figure-generation script, or useful for auditability.

`src/`
Provides the minimal Python modules needed by the analysis and collection scripts:
- `languages.py` defines the benchmark language and framework inventory.
- `requirement_taxonomy.py` reconstructs the operational/semantic requirement taxonomy used in the requirement-type analysis.
- `llm/provider.py` and `llm/cost.py` support the optional M-RewardBench collection pipeline and cost accounting.

`scripts/`
Contains the executable entry points for the standalone submission repository:
- `run_internal_benchmark_analysis.py` reproduces the internal eight-language benchmark analyses.
- `run_external_mrewardbench_analysis.py` reproduces the external M-RewardBench panel analysis, human-anchor check, and Xu et al. pairwise baseline.
- `collect_mrewardbench_panel_scores.py` is the optional evaluator-collection pipeline used to build new external panels from public M-RewardBench instances.
- `render_paper_figures.py` regenerates the paper figures from the saved analysis artifacts.

`data/internal_benchmark/judgments/`
Contains the curated subset of the original benchmark outputs needed for the second paper: the `gray_box` judgment JSON files for every model, language, and framework cell. These files are sufficient to recompute the score matrix and all internal analyses without shipping unrelated assets from the first paper.

`data/external_validation/mrewardbench_panel/collection_logs/`
Contains the self-collected evaluator outputs for the five-model, seven-language M-RewardBench panel used in the paper. The logs were normalized to the canonical first 1,500 items per language so they match the retained external analysis exactly.

`data/external_validation/mrewardbench_panel/analysis_1500_item/`
Contains the canonical 1,500-item external analysis outputs used in the manuscript: merged panel tables, CBC bias estimates, bootstrap summaries, the adapted judge-aware BTL baseline, and the human-anchor validation summary.

`docs/`
Contains human-readable support material specific to this paper: a plain-language explainer and this selection-rationale document.

## Included root files

`README.md`
Provides reviewer-facing setup, structure, and reproduction instructions.

`pyproject.toml` and `requirements.txt`
Provide a minimal standalone Python environment specification for this repo rather than the original monorepo.

`.env.example`
Documents the only environment variables needed for the optional external-collection pipeline.

`LICENSE`
Preserves the upstream repository license.

## Artifact decisions

`data/external_validation/mrewardbench_panel/analysis_1500_item/`
All files in this directory are worth keeping. They are the canonical external results actually used in the paper, and each one supports either a reported table value, a figure, the Xu baseline, or the human-anchor validation.

`paper/analysis/`
Most retained files in this directory are worth keeping, but only because they serve one of three purposes:
- direct manuscript support, such as `beta_hat.csv`, `rank_reversal_delta.csv`, and the generated figure PNGs;
- reproducibility of reported uncertainty estimates, such as the bootstrap replicate CSVs;
- inputs consumed by the shipped scripts, such as `task_level_scores.csv`.

Artifacts that were clearly redundant or superseded were removed from the submission copy.