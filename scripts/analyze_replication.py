#!/usr/bin/env python3
"""Analyse the replication study and decide whether the headline survives.

    python analyze_replication.py

Reports, per (language, evaluator) cell, the test-retest standard deviation of
repeated judgments on the SAME artifact under the SAME rubric. Then compares it
against the language-conditioned shift the paper reports.

The decision rule is stated here so it cannot be chosen after seeing the number:

  sd_retest < 0.25 * max|beta|   ->  PASS under the preset ratio rule.
  0.25 - 0.60 * max|beta|        ->  PARTIAL; qualify the magnitude claim.
  >= 0.60 * max|beta|            ->  FAIL; reframe or withdraw it.

This rule compares a mean cell-level sd in a 10-task slice with the largest
interaction in the full panel. PASS does not say every cell exceeds its own
repeat-run variation or provide an uncertainty interval for beta.

Writes results/replication_summary.csv and prints the verdict. It computes; it
does not assert an outcome.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
AAMAS_ROOT = REPO_ROOT / "paper" / "analysis" / "aamas_2027"
RES = AAMAS_ROOT / "results"
BASELINE_PANEL = (
    REPO_ROOT / "paper" / "analysis" / "task_level_scores.csv"
)
SCORES = RES / "replication_scores.csv"

if not SCORES.exists():
    sys.exit(f"{SCORES} not found. Run run_replication.py --execute first.\n"
             "This experiment has not been run; no numbers can be reported.")

df = pd.read_csv(SCORES)
if "status" in df.columns:
    df = df[df["status"] == "ok"].copy()
need = {"task", "language", "evaluator", "repeat", "score"}
missing = need - set(df.columns)
if missing:
    sys.exit(f"missing columns: {sorted(missing)}")

g = df.groupby(["language", "evaluator", "task"]).score
per_cell = g.std(ddof=1).rename("sd").reset_index()
counts = g.count()
if counts.min() < 2:
    sys.exit("at least two repeats per (task, language, evaluator) are required")

summary = (per_cell.groupby(["language", "evaluator"]).sd
           .agg(["mean", "median", "max", "count"]).reset_index()
           .rename(columns={"mean": "sd_retest_mean"}))
RES.mkdir(parents=True, exist_ok=True)
summary.to_csv(RES / "replication_summary.csv", index=False)

if not BASELINE_PANEL.exists():
    raise SystemExit(f"baseline panel not found: {BASELINE_PANEL}")
panel = pd.read_csv(BASELINE_PANEL)
cell = panel.pivot_table(
    index="language", columns="backbone", values="score", aggfunc="mean"
)
beta = cell.sub(cell.mean(axis=0), axis=1)
beta = beta.sub(cell.mean(axis=1), axis=0) + cell.values.mean()
max_beta = float(np.abs(beta.to_numpy()).max())
sd = float(summary.sd_retest_mean.mean())
ratio = sd / max_beta

print(summary.round(2).to_string(index=False))
print(f"\nmean test-retest sd            : {sd:.2f} points")
print(f"largest language-conditioned shift: {max_beta:.2f} points")
print(f"ratio                           : {ratio:.2f}")

if ratio < 0.25:
    v = ("PASS under the preset mean-sd/max-interaction ratio rule. "
         "Report the 10-task slice and its limits; do not treat mean sd "
         "as an error floor for each interaction cell.")
elif ratio < 0.60:
    v = ("PARTIAL under the preset ratio rule. Qualify the magnitude claim "
         "and report cell-level variation from the repeated slice.")
else:
    v = ("FAIL under the preset ratio rule. Reframe or withdraw the "
         "magnitude claim.")
decision = {
    "n_scores": int(len(df)),
    "n_task_language_evaluator_cells": int(len(per_cell)),
    "mean_test_retest_sd": sd,
    "largest_language_conditioned_shift": max_beta,
    "ratio": ratio,
    "verdict": v,
}
(RES / "replication_decision.json").write_text(
    json.dumps(decision, indent=2), encoding="utf-8"
)
print("\nVERDICT (rule fixed before the data existed):\n  " + v)
