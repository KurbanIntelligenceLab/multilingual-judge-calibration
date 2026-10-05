# AAMAS release audit, 5 October 2026

This release replaces the old short draft with the audited AAMAS manuscript.
It preserves submission 1655 and the current title. The paper is eight pages
total, with substantive sections through Ethics ending on page seven.

## Manuscript and references

The complete manuscript, all equations, five tables, and four figures were read
and reviewed. Related Work has four focused paragraphs and 27 cited records;
seven references were added from relevant prior material after primary-source
verification. The Bavaresco author list was corrected using the
[published ACL record](https://aclanthology.org/2025.acl-short.20/). The class and
bibliography style match the official template hashes, without dimension changes.

## Corrections retained in this release

- The estimator is standard two-way ANOVA double centering; the paper states its
  complete balanced-panel assumptions and distinguishes severity from quality.
- Missing preference margins are excluded from weighted means and BTL comparisons.
- The former panel-gold accuracy gain is retracted: a common `1e-9` margin
  tolerance yields 68.7% agreement before and after CBC. Fifty-five apparent
  changes were floating-point tie artifacts; none is a true decision flip.
- Framework comparisons are descriptive. The untraceable framework p-value and
  marginal SEM inference treating repeated observations as independent are removed.
- The arcsin inverse constrains corrected angles before mapping back to scores.
  Reported gate results remain unchanged. Linear CBC scores are not clipped.
- The human-label limitation is specific to the planned requirement-level test.
  A coarser task-score test is possible in principle but remains untested.
- Canonical human labels number 1,098, not the obsolete local 1,056-row export.

## Verification

The 90-check numerical verifier passes with zero mismatches. Separate ranking
scripts reproduce printed Table 2 values, with three DevAI method/draw rank-tie
differences documented at full precision. Replication recomputes SD 4.08,
interaction 20.61, ratio 0.20 under its recorded decision rule. The preference
anchor and bounded-link diagnostics are recomputed without provider calls.
Saved LaBSE similarities reproduce the reported mean/range; embeddings were not
regenerated. The three data figures were generated from saved scores and the
TikZ overview was inspected. Compilation and compact-archive extraction are
checked before the public commit. No unresolved citations/references or overfull
boxes remain. Nonfatal class/runtime conditional and balancing diagnostics remain.

The current audit is a scoped inspection of available records, not a certificate
that every claim is correct and not an independent blind review.

## Remaining limits

Fixed code controls the artifact but leaves judge behavior mixed with translation
quality. Peak effects depend substantially on GPT-4o. The repeat slice covers ten
MetaGPT tasks, not the full panel. Theorems require balance and stated error/task
assumptions; random-missingness diagnostics do not cover selective missing calls
or unseen languages. Pointwise intervals are not multiplicity-adjusted. Original
provider snapshots, retry histories, hardware, and exact LaBSE replay metadata
are incomplete. Requirement-level human accuracy remains untested. These limits
are disclosed in the manuscript and reproduction guide.

## Public selection

Private reviewer/rebuttal images, advisor meeting material, old local template
drafts and planning notes, scratch files, build auxiliaries, local credentials,
and duplicate root exports are not included. They are kept locally. The release
is selected explicitly; the existing public raw data is retained. Metadata paths
for newly published analysis outputs are repository-relative. Current instructions
replace obsolete unfinished-experiment and accuracy-gain statements.

## Supplementary package cleanup

The public ZIP copies and releases directory were removed after the release.
The manual-upload supplement is generated outside the checkout. It omits the
separately submitted manuscript and compile assets, generated plots, packaging
code, redundant project documentation, and two unrelated legacy analysis outputs.
It retains the inputs and imported statistical code needed by the documented
saved-score runners. Its dedicated README and SHA-256 manifest describe the
package independently of the full repository.
