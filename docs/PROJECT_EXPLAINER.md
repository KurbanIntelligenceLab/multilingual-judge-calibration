# Project explainer

The paper asks how localized judging inputs affect an agentic code evaluator when
the code is identical. A score can control a quality gate; language-conditioned
score changes can change whether the same artifact passes.

## Severity and quality

Greater satisfaction scores mean greater evaluator leniency. They do not imply
better judging. A severity rank reversal means two evaluators change order in
their mean scores across languages. Seven of fifteen pairs exhibit such a
reversal; p = 0.001 is a global reversal-count diagnostic, not seven pairwise tests.

## Model and correction

The score model is

```
S(t,l,b) = mu(t) + alpha(b) + g(l) + beta(l,b) + gamma(t,l) + epsilon(t,l,b)
```

Here mu is a task effect, alpha evaluator severity, g the language shift shared
across evaluators, beta the language-by-evaluator interaction, and gamma a
task-language shift shared by evaluators. Under the stated complete balanced
model and mean-zero assumptions, standard two-way ANOVA double centering
identifies the normalized interaction:

```
beta_hat(l,b) = mean(l,b) - mean(all,b) - mean(l,all) + grand_mean
corrected_score = score - beta_hat(l,b)
```

CBC does not require human labels to estimate the interaction. It leaves the
shared language shift intact. It preserves the mean over the complete evaluator
pool for each task-language instance. The paper states a conservative task-level
Hoeffding bound that permits within-task dependence but requires independent tasks.
Linear corrected scores are not clipped; clipping would break the invariance.

## Evidence

DevAI contains 55 tasks, eight languages, six evaluator backbones, and three
developer-agent framework arms (MetaGPT, GPT-Pilot, OpenHands). The same code is
used across language conditions. The backbones are GPT-4o, GPT-5.4, Claude Sonnet
4.6, Gemini 3 Flash Preview, DeepSeek-V3.2, and Qwen3.5-9B. Provider snapshots for
the original panel were not retained.

On omitted tasks, mean cross-language Kendall tau rises from 0.650 to 0.902.
Mean gate-25 language spread falls from 27.9 to 16.4 percentage points on the
fitted panel; the omitted-task mean reduction is 8.7 points [3.8,14.0]. These are
consistency results, not accuracy tests. Peak effects depend substantially on
GPT-4o; residual effects remain without it.

A separately collected M-RewardBench subset has 1,500 aligned items, seven
languages, five evaluators, and 35 missing margins. Its held-out ranking agreement
rises from 0.430 to 0.900. Its former 68.7% to 76.6% human-agreement gain was a
numerical tie artifact. Corrected agreement is 68.7% under both methods.

The repeat-call slice has 1,440 calls on ten MetaGPT tasks. Its mean SD 4.08 is
smaller than the largest interaction 20.61 under the preset ratio rule. This
does not establish stability for every task, framework, provider version, or time.

## Limits

Translation quality remains mixed with evaluator behavior. The planned human
requirement-level test is unsupported by a schema mismatch and binary-only
saved verdicts. A coarser task-score comparison remains untested. The theorem's
balanced-panel assumptions do not establish incomplete-panel identification or
unseen-language transfer. Existing matrices support recomputation; original
provider state and exact LaBSE embedding replay are not fully preserved.

Start with [the paper](../paper/main.pdf), [reproduction instructions](REPRODUCE.md),
and [the release audit](RELEASE_AUDIT.md). The repository contains this same
preprint study's current manuscript revision.
