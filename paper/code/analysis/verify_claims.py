#!/usr/bin/env python3
"""Recompute selected DevAI, gate, and metadata claims from released
matrices and fail loudly on any mismatch.

    python code/analysis/verify_claims.py

Exit code 0 means the listed checks reproduce within their stated precision.
Ranking/bootstrap ablations and preference-panel estimates have separate
analysis scripts and records; this verifier does not check every paper claim.
"""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import cbc
from cbc import LANGS, EVALUATORS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "analysis")
TOL = 0.05000001                 # half a unit at the reported decimal place

T, tasks = cbc.load_panel(os.path.join(DATA, "task_level_scores.csv"))
rl = pd.read_csv(os.path.join(DATA, "run_level_scores.csv"))
tl = pd.read_csv(os.path.join(DATA, "task_level_scores.csv"))
beta = cbc.beta_hat(T)
n = T.shape[0]

results = []
def check(claim, paper_value, computed, tol=TOL):
    ok = abs(float(paper_value) - float(computed)) <= tol
    results.append((claim, paper_value, round(float(computed), 4), ok))

# --- panel description -----------------------------------------------------
check("run-level observations",            7920, len(rl), 0)
check("task-level cells",                  2640, len(tl), 0)
check("Gemini mean severity",              42.9, tl[tl.backbone == "Gemini"].score.mean())
check("Qwen mean severity",                 3.6, tl[tl.backbone == "Qwen"].score.mean())
check("severity ratio",                    11.9, tl[tl.backbone == "Gemini"].score.mean()
                                                  / tl[tl.backbone == "Qwen"].score.mean())
check("Qwen share of exact zeros (%)",     82.0, (tl[tl.backbone == "Qwen"].score == 0).mean() * 100)

# --- interaction -----------------------------------------------------------
i_es, i_en, j = LANGS.index("Spanish"), LANGS.index("English"), EVALUATORS.index("GPT-4o")
check("max |beta|",                        20.6, np.abs(beta).max())
check("beta GPT-4o Spanish",              -20.6, beta[i_es, j])
check("beta GPT-4o English",               11.8, beta[i_en, j])
draws = cbc.bootstrap(T, lambda X: cbc.double_center(X.mean(0))[i_es, j], 4000, seed=21)
lo, hi = cbc.ci(draws)
check("beta GPT-4o Spanish CI low",       -22.5, lo, 0.25)
check("beta GPT-4o Spanish CI high",      -18.7, hi, 0.25)
check("language effect g range",            8.6, np.ptp(cbc.language_effect(T)))

# --- reversal --------------------------------------------------------------
rev = cbc.reversal_table(T)
check("reversing evaluator pairs",            7, int(rev.reversal.sum()), 0)
check("reversals involving GPT-4o",           4,
      int(rev[rev.reversal].apply(lambda r: "GPT-4o" in (r.i, r.j), axis=1).sum()), 0)
for first, second, language, gap in [
    ("GPT-4o", "GPT-5.4", "English", 35.6),
    ("GPT-4o", "GPT-5.4", "Spanish", -11.2),
    ("GPT-4o", "Gemini", "English", 9.5),
    ("GPT-4o", "Gemini", "Spanish", -32.8),
    ("GPT-4o", "Sonnet", "Hindi", 30.7),
    ("GPT-4o", "Sonnet", "Spanish", -7.2),
    ("GPT-4o", "DeepSeek", "English", 32.4),
    ("GPT-4o", "DeepSeek", "Spanish", -4.3),
    ("GPT-5.4", "Sonnet", "Japanese", -9.0),
    ("GPT-5.4", "Sonnet", "Swahili", 10.7),
    ("GPT-5.4", "DeepSeek", "Chinese", -5.0),
    ("GPT-5.4", "DeepSeek", "Swahili", 14.3),
    ("Sonnet", "DeepSeek", "Arabic", 8.5),
    ("Sonnet", "DeepSeek", "Hindi", -4.5),
]:
    gap_computed = (T[:, LANGS.index(language), EVALUATORS.index(first)]
                    - T[:, LANGS.index(language), EVALUATORS.index(second)]).mean()
    check(f"{first}/{second} gap in {language}", gap, gap_computed)

# --- gate ------------------------------------------------------------------
pr = cbc.pass_rate(T, 25)
check("GPT-4o pass rate, Hindi rubric",    90.9, pr[LANGS.index("Hindi"), j])
check("GPT-4o pass rate, Spanish rubric",  10.9, pr[i_es, j])
for gate, raw, cal in [(10, 34.8, 21.5), (25, 27.9, 16.4), (50, 12.1, 7.6)]:
    check(f"raw gate spread @{gate}",        raw, cbc.gate_spread(T, gate))
    check(f"corrected gate spread @{gate}",  cal, cbc.gate_spread(T - beta[None], gate))
gate_rng = np.random.default_rng(7)
gate_oob = []
for _ in range(1000):
    fit = gate_rng.integers(0, n, n)
    held_out = np.setdiff1d(np.arange(n), np.unique(fit))
    fitted_beta = cbc.beta_hat(T[fit])
    gate_oob.append(cbc.gate_spread(T[held_out], 25) -
                    cbc.gate_spread(T[held_out] - fitted_beta[None], 25))
gate_oob = np.asarray(gate_oob)
check("gate-25 out-of-bag mean reduction", 8.7, gate_oob.mean())
gate_lo, gate_hi = np.percentile(gate_oob, [2.5, 97.5])
check("gate-25 out-of-bag CI low", 3.8, gate_lo)
check("gate-25 out-of-bag CI high", 14.0, gate_hi)
check("gate-25 positive out-of-bag draws", 998, np.count_nonzero(gate_oob > 0), 0)
lang_off = T.mean(0).mean(1) - T.mean(0).mean()
for gate, v in [(10, 39.4), (25, 28.5), (50, 12.7)]:
    check(f"per-language centring @{gate}",   v,
          cbc.gate_spread(T - lang_off[None, :, None], gate))

# --- English default contrast ---------------------------------------------
B = cbc.bootstrap(T, lambda X: cbc.pass_rate(X, 25), 4000, seed=77)
for ev, mean_claim in [("GPT-4o", -27.9), ("Gemini", 13.9), ("GPT-5.4", 15.6)]:
    jj = EVALUATORS.index(ev)
    d = B[:, 1:, jj].mean(1) - B[:, 0, jj]
    check(f"English-vs-localized contrast, {ev}", mean_claim, d.mean(), 0.5)
    low_claim, high_claim = {"GPT-4o": (-37.7, -18.2),
                             "Gemini": (7.0, 21.3),
                             "GPT-5.4": (9.4, 22.3)}[ev]
    low, high = np.percentile(d, [2.5, 97.5])
    check(f"English contrast CI low, {ev}", low_claim, low)
    check(f"English contrast CI high, {ev}", high_claim, high)

# --- leave-one-evaluator-out ----------------------------------------------
for ev, v in [("GPT-4o", 8.0), ("Gemini", 20.1), ("GPT-5.4", 19.2),
              ("Sonnet", 19.7), ("DeepSeek", 19.7), ("Qwen", 20.2)]:
    keep = [q for q in range(6) if q != EVALUATORS.index(ev)]
    check(f"max |beta| without {ev}", v, np.abs(cbc.double_center(T[:, :, keep].mean(0))).max())

for ev, mean, zero_share, interaction, reversals_without in [
    ("Gemini", 42.9, 2.7, 9.3, 6),
    ("GPT-4o", 33.2, 6.1, 20.6, 3),
    ("GPT-5.4", 16.8, 22.7, 8.4, 4),
    ("Sonnet", 15.4, 23.4, 7.4, 4),
    ("DeepSeek", 12.2, 33.0, 5.7, 4),
    ("Qwen", 3.6, 82.0, 4.8, 7),
]:
    sub = tl[tl.backbone == ev]
    j = EVALUATORS.index(ev)
    keep = [q for q in range(6) if q != j]
    check(f"{ev} table mean", mean, sub.score.mean())
    check(f"{ev} table zero share", zero_share, 100 * (sub.score == 0).mean())
    check(f"{ev} table max |beta|", interaction, np.abs(beta[:, j]).max())
    check(f"{ev} reversals without it", reversals_without,
          cbc.n_reversals(T[:, :, keep]), 0)

parse = pd.read_csv(os.path.join(DATA, "aamas_2027", "results", "parse_failure_cells.csv"))
check("requirement judgments", 52560, parse.n_requirements.sum(), 0)
check("noncompliant judgments", 2410,
      (parse.noncompliant_rate * parse.n_requirements).sum(), 0.01)
check("Qwen empty judgments", 2301, parse.loc[parse.backbone == "Qwen", "empty"].sum(), 0)
without_qwen = parse[parse.backbone != "Qwen"]
check("noncompliance without Qwen (%)", 0.12,
      100 * (without_qwen.noncompliant_rate * without_qwen.n_requirements).sum()
      / without_qwen.n_requirements.sum(), 0.01)

# --- agent-invariance ------------------------------------------------------
p3 = rl.pivot_table(index="framework", columns="language", values="score", aggfunc="mean")
check("largest across-agent gap",           1.50, (p3.max(axis=0) - p3.min(axis=0)).max(), 0.02)

# --- human anchor description ---------------------------------------------
hg = pd.read_csv(os.path.join(DATA, "aamas_2027", "data", "human_gold.csv"))
check("human-label judgments",             1098, len(hg), 0)
check("human base rate satisfied (%)",      36.5, hg.satisfied.mean() * 100)
check("majority-class floor (%)",           63.5, (1 - hg.satisfied.mean()) * 100)

# --- report ----------------------------------------------------------------
w = max(len(r[0]) for r in results)
print(f"{'claim':{w}s} {'paper':>9s} {'computed':>10s}   ")
for claim, paper, comp, ok in results:
    print(f"{claim:{w}s} {paper:>9} {comp:>10}   {'ok' if ok else 'MISMATCH'}")
bad = [r for r in results if not r[3]]
print(f"\n{len(results)} claims checked, {len(bad)} mismatches")
if bad:
    for r in bad:
        print(f"  MISMATCH: {r[0]}  paper={r[1]}  computed={r[2]}")
sys.exit(1 if bad else 0)
