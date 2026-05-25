# Project Explainer

This repository accompanies the paper **"Rank Reversal in Multilingual LLM Judges: A Label-Free Double-Centering Calibrator."** It provides the paper source, the retained data artifacts, and the code needed to reproduce the main results.

## What the paper studies

The paper asks whether multilingual LLM judges behave consistently across prompt languages.

The motivating observation is that evaluator rankings can change with language. A backbone that looks strongest in one language may no longer be strongest in another. That makes multilingual evaluation unstable and can weaken claims about a universally best evaluator.

## What rank reversal means

Rank reversal means that two evaluator backbones change order across languages.

For example, one backbone may outrank another in English, while the ordering flips in Arabic or Spanish. This is the empirical signal that motivates the paper: if such reversals occur, multilingual judge comparisons are not language-neutral.

## Main idea

The paper models judge scores as:

$$
S(t,\ell,b) = \mu(t) + \alpha(b) + \beta(\ell,b) + \gamma(t,\ell) + \epsilon
$$

where:

- $\mu(t)$ is task difficulty;
- $\alpha(b)$ is overall evaluator strength;
- $\beta(\ell,b)$ is the language-backbone interaction term;
- $\gamma(t,\ell)$ is a task-language effect shared across backbones;
- $\epsilon$ is residual noise.

The key quantity is $\beta(\ell,b)$. It captures whether a specific evaluator backbone behaves unusually high or low in a specific language.

## What CBC does

CBC stands for **Consensus-Based Calibration**.

It estimates the interaction term $\beta(\ell,b)$ from the observed language-by-backbone score matrix using double centering, then subtracts that estimated interaction from the scores.

In practical terms, CBC:

1. averages scores over tasks;
2. estimates language-backbone interaction bias;
3. subtracts that bias;
4. compares rankings before and after calibration.

CBC is label-free for calibration: it does not require human annotations to estimate the interaction term.

## What CBC does not do

CBC removes the language-backbone interaction term only.

It does **not** remove a language-wide bias shared by all backbones equally. Correcting that kind of shared shift would require an external anchor such as human judgments or a trusted reference evaluator.

## Experimental settings

### Internal benchmark

- 8 languages
- 6 evaluator backbones
- 55 tasks
- 3 judge frameworks

This benchmark is used for the main rank-reversal analysis, CBC evaluation, leave-one-language-out diagnostics, ablations, and decision-level evaluation.

### External M-RewardBench panel

- 7 languages
- 5 evaluator backbones
- 1,500 aligned items per language

The public benchmark provides the multilingual instances and gold preferences. The evaluator score matrix was collected separately and is retained in this repository.

## Main results

On the internal benchmark:

- raw mean pairwise Kendall $\tau$ is `0.650`;
- CBC raises it to `0.902`;
- 7 of 15 backbone pairs show significant rank reversal.

On the external M-RewardBench panel:

- raw mean pairwise Kendall $\tau$ is `0.430`;
- CBC raises it to `0.900`;
- agreement with public gold preferences improves from `68.7%` to `76.6%`.

The paper also compares CBC with post-hoc baselines including quantile normalization, ComBat-style correction, and an adapted judge-aware Bradley-Terry-Luce baseline.

## Scope and limitations

The repository and paper focus on calibration for **observed** language-backbone cells in a shared evaluation panel.

Important scope boundaries are:

- CBC targets the interaction term $\beta(\ell,b)$, not every possible source of multilingual bias;
- shared language-wide bias across all backbones is not identifiable by double centering alone;
- the main guarantees apply to the observed panel rather than to arbitrary unseen languages;
- the external M-RewardBench setting uses public task instances but self-collected evaluator scores.

## Repository contents

The repository keeps only the assets needed for this paper:

- `paper/` for the manuscript and retained analysis artifacts;
- `scripts/` for the internal analysis, external analysis, collection, and figure-generation entry points;
- `src/` for the minimal helper modules used by those scripts;
- `data/internal_benchmark/` for the internal judgment JSON files;
- `data/external_validation/mrewardbench_panel/` for the retained external logs and canonical 1,500-item external analysis outputs.

## Most useful files

For the shortest path to the main quantitative results, start with:

- `paper/analysis/calibration_results.json`
- `paper/analysis/rank_reversal_delta.csv`
- `paper/analysis/beta_hat.csv`
- `data/external_validation/mrewardbench_panel/analysis_1500_item/raw_vs_cbc_summary.json`
- `data/external_validation/mrewardbench_panel/analysis_1500_item/human_anchor_validation.json`

## Takeaway

The main message of the paper is:

> multilingual judge disagreement is not just noise; a meaningful part of it can be modeled as a language-backbone interaction, and that interaction can be estimated and removed with a simple label-free post-hoc calibrator.
