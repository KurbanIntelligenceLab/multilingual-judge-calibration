"""Recompute DevAI ranking comparisons from saved scores, without LLM calls.

Run from the repository (or compact archive) root:
    python paper/code/analysis/recompute_devai_rankings.py

Random control is evaluated in its original relative position to preserve the
RNG draws used by the saved bootstrap protocol. Original full-method outputs
are not overwritten; the result is a separate verification audit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[2]
ROOT = PAPER.parent
ANALYSIS = PAPER / "analysis"
OUTPUT = ANALYSIS / "aamas_2027" / "results" / "devai_ranking_audit.json"
METHODS = [
    "Raw (no calibration)",
    "Random control",
    "Quantile normalization",
    "ComBat-EB",
    "CBC",
]


def load_reference():
    sys.path.insert(0, str(ROOT))
    # Only the pure language registry is needed. The full repository package's
    # __init__ imports provider classes that are outside this offline archive.
    if "agent_as_a_judge" not in sys.modules:
        namespace = types.ModuleType("agent_as_a_judge")
        namespace.__path__ = [str(ROOT / "agent_as_a_judge")]
        sys.modules["agent_as_a_judge"] = namespace
    source = ROOT / "scripts" / "compute_meb_foundations.py"
    spec = importlib.util.spec_from_file_location("devai_ranking_reference", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load saved-panel reference analysis: {source}")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference


def diagnose_ties(reference, task_scores, mismatches):
    """Inspect mismatched draws without changing the analysis's tie rule."""
    targets = set(mismatches["replicate"].astype(int))
    if not targets:
        return []
    tasks = sorted(task_scores["task"].unique(), key=reference.task_key)
    rng = np.random.default_rng(7)
    diagnostics = []
    for replicate in range(max(targets) + 1):
        fit, omitted = reference.bootstrap_task_split(tasks, rng)
        if replicate not in targets:
            # Identical random-control consumption, even when no matrices are needed.
            for _ in reference.LANGUAGES:
                rng.permutation(len(reference.BACKBONE_ORDER))
            continue
        eval_matrix = reference.build_score_matrix(task_scores[task_scores["task"].isin(omitted)])
        train_matrix = reference.build_score_matrix(task_scores, task_weights=fit)
        random_matrix = reference.random_control(eval_matrix, rng)
        tiny_gaps = []
        for scope, matrix in [("omitted-item means", eval_matrix), ("in-bag means", train_matrix)]:
            for language, values in matrix.iterrows():
                for i, j in combinations(range(len(values)), 2):
                    difference = float(values.iloc[i] - values.iloc[j])
                    if 0 < abs(difference) < 1e-12:
                        tiny_gaps.append({
                            "scope": scope, "language": language,
                            "left": values.index[i], "right": values.index[j],
                            "difference": difference,
                        })
        for record in mismatches[mismatches["replicate"].eq(replicate)].to_dict("records"):
            method = record["method"]
            if method == "Raw (no calibration)":
                diagnostic_matrix = eval_matrix.round(12)
            elif method == "Random control":
                diagnostic_matrix = random_matrix.round(12)
            elif method == "Quantile normalization":
                diagnostic_matrix = reference.col_quantile(eval_matrix.round(12), train_matrix.round(12))
            else:
                raise AssertionError(f"Unexpected tie-sensitive method: {method}")
            diagnostic_tau = reference.mean_pairwise_kendall_tau(diagnostic_matrix)
            confirmed = bool(abs(diagnostic_tau - record["mean_pairwise_tau_saved"]) <= 1e-12)
            if not confirmed:
                raise AssertionError(f"Unexplained saved-record discrepancy for {method} draw {replicate}")
            diagnostics.append({
                "replicate": replicate, "method": method,
                "saved_tau": record["mean_pairwise_tau_saved"],
                "recomputed_tau": record["mean_pairwise_tau_recomputed"],
                "diagnostic_rounded_input_tau": diagnostic_tau,
                "numerical_tie_behavior_confirmed": confirmed,
                "tiny_input_mean_differences": tiny_gaps,
                "note": "Rounding is used only to diagnose numerical ties; no new tie rule is applied to the recomputed results",
            })
    return diagnostics


def main() -> None:
    started = time.perf_counter()
    reference = load_reference()
    task_scores = pd.read_csv(ANALYSIS / "task_level_scores.csv")
    run_scores = pd.read_csv(ANALYSIS / "run_level_scores.csv")
    previous = pd.read_csv(ANALYSIS / "calibration_bootstrap_replicates.csv")
    stored = json.loads((ANALYSIS / "calibration_results.json").read_text(encoding="utf-8"))
    print("Recomputing 1,000 saved-score DevAI bootstrap comparisons (seed 7)...", flush=True)
    replicates, computed = reference.evaluate_calibration_methods(
        task_scores, run_scores, methods=METHODS, bootstrap_reps=1000, seed=7)
    keys = ["replicate", "method"]
    selected_previous = previous[previous["method"].isin(METHODS)]
    comparison = replicates.merge(selected_previous, on=keys, how="outer",
                                  suffixes=("_recomputed", "_saved"), validate="one_to_one", indicator=True)
    if not comparison["_merge"].eq("both").all():
        raise AssertionError("Recomputed bootstrap records do not match the saved method/draw layout")
    difference = (comparison["mean_pairwise_tau_recomputed"]
                  - comparison["mean_pairwise_tau_saved"]).abs()
    mismatches = comparison[difference > 1e-12]
    for column in ("n_eval_tasks", "n_backbones"):
        if not np.array_equal(comparison[f"{column}_recomputed"], comparison[f"{column}_saved"]):
            raise AssertionError(f"Saved bootstrap {column} does not reproduce")
    stored_results = {row["method"]: row for row in stored["results"]}
    for result in computed["results"]:
        for column in ("mean_pairwise_tau", "ci_low", "ci_high"):
            if f"{result[column]:.3f}" != f"{stored_results[result['method']][column]:.3f}":
                raise AssertionError(f"Saved {result['method']} {column} does not reproduce at the paper's three-decimal precision")
    tie_diagnostics = diagnose_ties(reference, task_scores, mismatches)
    audit = {
        "protocol": computed["protocol"],
        "methods": METHODS,
        "rng_note": "Random control retains original method order and RNG consumption; omitted methods do not consume RNG draws",
        "input_scope": "Saved DevAI task-mean and framework-run score matrices; no provider calls",
        "verification": {
            "selected_method_replicates": len(comparison),
            "maximum_absolute_replicate_tau_difference": float(difference.max()),
            "replicate_tolerance": 1e-12,
            "exact_replicate_agreement": len(mismatches) == 0,
            "matching_replicates_within_tolerance": int((difference <= 1e-12).sum()),
            "numerical_tie_sensitive_replicates": len(mismatches),
            "paper_summary_decimal_places": 3,
            "summary_mean_and_interval_match_at_paper_precision": True,
            "omitted_task_counts_match": True,
            "evaluator_counts_match": True,
            "all_discrepancies_explained_as_numerical_ties": True,
        },
        "numerical_tie_diagnostics": tie_diagnostics,
        "saved_results": [row for row in stored["results"] if row["method"] in METHODS],
        "recomputed_results": computed["results"],
        "recomputed_full_panel": computed["full_panel"],
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
