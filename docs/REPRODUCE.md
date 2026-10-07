# Reproduce the current AAMAS release

## Environment and commands

Use Python 3.11 or newer and install the root `requirements.txt`. Run these
commands from the repository root or extracted compact archive:

```bash
python paper/code/analysis/verify_claims.py
python paper/code/analysis/audit_mrewardbench_anchor.py
python scripts/analyze_replication.py
python scripts/analyze_panel_sensitivity.py
python scripts/audit_localization_and_transforms.py --bounded-link-only
python scripts/generate_aamas_figures.py
python paper/code/analysis/recompute_devai_rankings.py
python paper/code/analysis/recompute_preference_rankings.py
```

The first command verifies 90 listed quantities; it is not a certificate for
every claim. The ranking runners recompute all printed Table 2 means/intervals.
DevAI has three of 5,000 selected method/draw differences near numerical rank
ties, with no printed-value change. Original full-method records are retained.
Resampling preserves all languages and evaluators of each task/item. Seeds are
recorded in the scripts. The preference bootstrap seed is 7.

CBC fits in-bag items and evaluates omitted items. The adapted BTL comparator
estimates rankings from omitted-item comparisons. Unavailable margins are omitted
from weighted score denominators and pairwise comparisons, not treated as ties.
The 1,500 preference IDs include 805 easy and 695 hard AlpacaEval items; 35 of
52,500 margins are unavailable. The historical filename `pilot_complete_panel.csv`
denotes aligned rows, not a matrix with all finite margins.

### Comparator implementations

The ComBat-inspired comparator uses one residual location and scale per
language, shared across evaluators, with shrinkage toward zero and one; it
restores evaluator main effects. This is a pooled location/scale adaptation,
not the standard feature-specific ComBat implementation. Its historical script
function and result label are `combat_eb_adjust` and `ComBat-EB`. Quantile normalization
sorts each evaluator column across languages, averages the in-bag sorted columns
into a target distribution, and maps omitted-task column ranks to that target;
exact ties receive averaged target values. CBC removes the language-by-evaluator
interaction and retains shared language and evaluator main effects.

The adapted BTL procedure turns differences between evaluator preference margins
into pairwise wins, with exact ties counted as half a win. It fits one latent
ranking per language on omitted-item comparisons, using the easy/hard dataset
strata as discrimination groups. The original cited model uses actual judge
identities for these parameters. This adaptation compares ranking procedures;
its fitting data and target differ from CBC's in-bag interaction correction.

### Table 3 sampling

The task-count ablation uses 100 replicates per count (10, 20, 30, 40, 55),
with one continuing NumPy RNG stream seeded at 107. Each replicate selects tasks
uniformly without replacement, then draws a bootstrap with replacement of the
same size within that subset. Fitting uses in-bag multiplicities; evaluation
uses omitted tasks from the selected subset. Sampling has no category strata
or category quotas. The three framework arms are averaged before sampling.

The evaluator ablation enumerates every subset at each size: 15, 20, 15, 6,
and 1 subsets for two through six evaluators. Each subset uses 100 task-bootstrap
replicates, with seed `7 + evaluator_count`. The reported across-subset SD is
the SD of subset-specific mean agreement, not the SD of all bootstrap draws.

## Expected results

| Quantity | Value |
|---|---|
| Reversing pairs | 7/15; global residual-permutation p = 0.001 |
| Largest absolute interaction | 20.6 score points |
| GPT-4o Spanish interaction | -20.6, pointwise 95% interval [-22.5,-18.7] |
| GPT-4o Hindi/Spanish gate-25 pass rates | 90.9% / 10.9% |
| Gate-25 fitted language spread, raw/CBC | 27.9 / 16.4 percentage points |
| Omitted-task mean gate-25 spread reduction | 8.7 points, 95% interval [3.8,14.0] |
| DevAI held-out ranking tau, raw/CBC | 0.650 / 0.902 |
| Preference held-out ranking tau, raw/CBC | 0.430 / 0.900 |
| Preference sample panel-gold agreement, raw/CBC | 68.7% / 68.7% |
| Repeat-call mean SD / largest interaction | 4.08 / 20.61, ratio 0.20, recorded verdict PASS |
| Human export | 1,098 requirement labels, 36.5% satisfied |
| Random 5%/10% missing-cell interaction MAE | available means 0.34/0.51; task-adjusted means 0.23/0.33 points |

The original seven reversal witnesses are descriptive; p is for the total
reversal count. Interaction intervals are pointwise and not multiplicity-adjusted.
The residual permutation assumes exchangeability across permuted language
positions; the task bootstrap and Hoeffding bound allow within-task dependence.

## Figures and PDF

Figure regeneration works in both the repository and supplementary archive.
Manuscript compilation below applies only to the full repository; the ZIP
omits the separately submitted manuscript and its compile assets.

The editable overview is `paper/figures/fig_overview.tex`. Three data figures
are generated by `scripts/generate_aamas_figures.py`. The overview grid is
schematic; its numbers match the saved-score results. The official class and
bibliography style match their locked template hashes.

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: eight pages total, all five tables and four figures, no unresolved
references or overfull boxes. The unchanged class/runtime can emit nonfatal
conditional/column-balancing diagnostics. The rendered pages were inspected.

## Human-anchor correction

The earlier accuracy gain counted tiny positive corrected residuals as decisions
while raw margins were exactly zero. The audit uses a common absolute tolerance
of `1e-9`: 55 apparent flips become ties; zero true decisions change. Three
sampled rows have a missing margin. Corrected panel-gold agreement is 68.7% under
both methods. The planned DevAI requirement-level accuracy test is blocked by a
five-versus-four requirement mismatch and binary-only evaluator verdicts.

## Full-tree and embedding limits

The compact archive reproduces saved-score analyses, not original provider calls.
It lacks raw agent workspaces and provider responses. Original snapshots, retry
histories, and hardware metadata are incomplete. The ten-task reruns retain
route/decoding metadata for 1,000 of 1,440 saved score records: 194 DeepSeek,
237 Qwen, 3 GPT-5.4, and 6 Gemini score records lack matching metadata.
All six evaluators have route and decoding settings represented, but complete
per-call replay is not established. `compute_meb_foundations.py` and the full local parser/
translation entry point require the original `benchmark_tests` tree and some
additional original analysis helpers. The public standalone internal analysis
has its own curated judgment input; its historical extras are not current claims.

The `calibration` subsection of `paper/analysis/meb_foundations.json` is a
historical snapshot. Its quantile and ensemble results are superseded by
`paper/analysis/calibration_results.json` and the corresponding
`calibration_bootstrap_replicates.csv`; use those files for current ranking
results. The aggregate's other sections supply the saved model checks.
Rerun temperature 0 and top-p 0.9 describe text judging requests; they do not
establish those settings for every auxiliary module call or historical
provider snapshot.

The bounded-link-only runner avoids the unavailable benchmark tree and writes
`bounded_link_rerun.json` separately. The arcsin inverse clips corrected angles
to [0,pi/2] before sin-squared. The paper's linear CBC scores remain un-clipped.
The saved local audit includes parser/translation results for inspection.

The LaBSE record contains 35 similarities, its generator, prompt sources, and
English variants. Optional embedding generation additionally needs
`sentence-transformers` and LaBSE weights. Model revision, embedding runtime, and
original English-variant environment were not retained, so exact replay is not
established. The generator writes to root `analysis/`; the manuscript uses the
preserved record in `paper/analysis/aamas_2027/results/`.

The `theoretical_bound` field retained in an old ablation JSON is an unreported
Gaussian/independence heuristic. It is not Proposition 2's task-level bound.

The prior human-anchor sample is preserved as `human_anchor_sample_original.csv`
for the tie-artifact audit. `human_anchor_sample.csv` now contains corrected flags;
its original sample IDs and margins are unchanged within the audit tolerance.

## Supplementary packaging

The public repository contains the supporting material directly. Generate a
ZIP outside the checkout for manual OpenReview upload:

```bash
python scripts/build_supplementary_archive.py --output ../aamas2027_reproducibility.zip
```

The archive has its own README and SHA-256 file inventory. The public repository
does not store generated ZIPs or a releases directory. Judge prompt sources
are included under this repository's `agent_as_a_judge/module/prompt/` directory;
that path names a local folder, not a required second checkout.
