"""Recompute the preference-panel ranking table from saved scores only.

Run from the repository (or compact archive) root:
    python paper/code/analysis/recompute_preference_rankings.py

CBC is fitted on bootstrap in-bag items and evaluated on omitted items.
The adapted BTL comparator estimates rankings from omitted-item comparisons.
Missing margins are omitted from both score means and pairwise comparisons.
No provider calls are made. Precomputed pair outcomes avoid repeated item-level
DataFrame loops; their sufficient statistics are checked against the reference
implementation before the full calculation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[2]
ROOT = PAPER.parent
DATA = PAPER / "analysis" / "aamas_2027" / "data" / "mrewardbench"
SOURCE = ROOT / "scripts" / "analyze_mrewardbench_external_pilot.py"


def load_reference():
    spec = importlib.util.spec_from_file_location("preference_reference", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the reference analysis: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("bootstrap-reps must be positive")
    started = time.perf_counter()
    reference = load_reference()
    summary_path = DATA / "raw_vs_cbc_summary.json"
    replicates_path = DATA / "raw_vs_cbc_bootstrap_replicates.csv"
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    old_replicates = pd.read_csv(replicates_path)
    panel = pd.read_csv(DATA / "pilot_complete_panel.csv")
    task_df = panel.rename(columns={
        "item_id": "task", "evaluator_model": "backbone", "margin": "score",
    })
    languages = old["full_panel"]["languages"]
    backbones = old["full_panel"]["backbones"]
    tasks = sorted(task_df["task"].unique())
    judge_labels = sorted(task_df["subset"].astype(str).unique())
    if task_df.duplicated(["task", "language", "backbone"]).any():
        raise ValueError("Duplicate saved preference cells")
    if len(panel) != len(tasks) * len(languages) * len(backbones):
        raise ValueError("Saved preference rows do not form the expected layout")
    if task_df.groupby("task")["subset"].nunique().max() != 1:
        raise ValueError("An item has inconsistent subset labels")
    item_subsets = (task_df.drop_duplicates("task").set_index("task")
                    ["subset"].reindex(tasks).to_numpy())
    pivot = task_df.pivot(index="task", columns=["language", "backbone"], values="score")
    scores = np.stack([
        np.stack([pivot[(language, backbone)].reindex(tasks).to_numpy(float)
                  for backbone in backbones], axis=1)
        for language in languages
    ], axis=1)
    finite = np.isfinite(scores)
    values = np.where(finite, scores, 0.0)
    pairs = list(combinations(range(len(backbones)), 2))
    left = scores[:, :, [i for i, _ in pairs]]
    right = scores[:, :, [j for _, j in pairs]]
    pair_valid = np.isfinite(left) & np.isfinite(right)
    pair_outcomes = np.where(left > right, 1.0, np.where(left < right, 0.0, 0.5))
    pair_outcomes = np.where(pair_valid, pair_outcomes, 0.0)

    def mean_matrix(weights: np.ndarray) -> pd.DataFrame:
        denominator = np.einsum("t,tlb->lb", weights, finite)
        if np.any(denominator <= 0):
            raise ValueError("No observed margins for a score-mean cell")
        mean = np.einsum("t,tlb->lb", weights, values) / denominator
        return pd.DataFrame(mean, index=languages, columns=backbones)

    def comparison_tables(weights: np.ndarray) -> dict[str, pd.DataFrame]:
        counts = []
        wins = []
        for label in judge_labels:
            subset_weights = weights * (item_subsets == label)
            counts.append(np.einsum("t,tlp->lp", subset_weights, pair_valid))
            wins.append(np.einsum("t,tlp->lp", subset_weights, pair_outcomes))
        result = {}
        for language_index, language in enumerate(languages):
            rows = []
            for pair_index, (i, j) in enumerate(pairs):
                for judge_index in range(len(judge_labels)):
                    count = int(counts[judge_index][language_index, pair_index])
                    if count:
                        rows.append((i, j, judge_index, count,
                                     wins[judge_index][language_index, pair_index] / count))
            result[language] = pd.DataFrame(rows, columns=["i", "j", "k", "n", "ybar"])
        return result

    def btl_matrix(weights: np.ndarray):
        tables = comparison_tables(weights)
        rows = []
        judge_weights = {}
        for language in languages:
            fitted, gammas = reference.fit_judge_aware_btl(
                tables[language], len(backbones), len(judge_labels))
            rows.append(fitted)
            judge_weights[language] = dict(zip(judge_labels, map(float, gammas), strict=True))
        return pd.DataFrame(rows, index=languages, columns=backbones), judge_weights

    # Validate the fast sufficient statistics on the first actual bootstrap split.
    rng = np.random.default_rng(args.seed)
    first_fit, first_omitted = reference.bootstrap_task_split(tasks, rng)
    fit_weights = np.array([first_fit[task] for task in tasks], dtype=float)
    first_omitted_set = set(first_omitted)
    omitted_weights = np.array([task in first_omitted_set for task in tasks], dtype=float)
    omitted_df = task_df[task_df["task"].isin(first_omitted)]
    np.testing.assert_allclose(mean_matrix(fit_weights), reference.build_score_matrix(
        task_df, languages, backbones, first_fit), rtol=0, atol=1e-12)
    np.testing.assert_allclose(mean_matrix(omitted_weights), reference.build_score_matrix(
        omitted_df, languages, backbones), rtol=0, atol=1e-12)
    actual_tables = comparison_tables(omitted_weights)
    expected_tables = reference.build_pairwise_evaluator_comparisons(omitted_df, languages, backbones)
    for language in languages:
        pd.testing.assert_frame_equal(actual_tables[language], expected_tables[language], check_dtype=False)

    # Reset the seed: validation must not consume any analysis draws.
    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = 0
    for replicate in range(args.bootstrap_reps):
        fit, omitted = reference.bootstrap_task_split(tasks, rng)
        if not omitted:
            skipped += 1
            continue
        train_weights = np.array([fit[task] for task in tasks], dtype=float)
        omitted_set = set(omitted)
        eval_weights = np.array([task in omitted_set for task in tasks], dtype=float)
        raw_matrix = mean_matrix(eval_weights)
        beta = reference.compute_beta_from_matrix(mean_matrix(train_weights))
        corrected = raw_matrix - beta
        adapted, _ = btl_matrix(eval_weights)
        raw_tau = reference.mean_pairwise_kendall_tau(raw_matrix)
        cbc_tau = reference.mean_pairwise_kendall_tau(corrected)
        rows.append({
            "replicate": replicate, "raw_tau": raw_tau, "cbc_tau": cbc_tau,
            "xu_tau": reference.mean_pairwise_kendall_tau(adapted),
            "tau_gain": cbc_tau - raw_tau, "n_eval_tasks": len(omitted),
        })
        if (replicate + 1) % 100 == 0:
            print(f"Completed {replicate + 1}/{args.bootstrap_reps} saved-score replicates "
                  f"in {time.perf_counter() - started:.1f} s", flush=True)
    replicates = pd.DataFrame(rows)
    results = {
        label: reference.summarize_metric(replicates[column].to_numpy(float))
        for label, column in [
            ("Raw", "raw_tau"), ("CBC", "cbc_tau"),
            ("JudgeAwareBTL_Xu2026_adapted", "xu_tau"), ("CBC_minus_Raw", "tau_gain"),
        ]
    }
    if args.seed == old["protocol"]["seed"] and args.bootstrap_reps == len(old_replicates):
        np.testing.assert_allclose(replicates["raw_tau"], old_replicates["raw_tau"], rtol=0, atol=1e-12)
        if not replicates["n_eval_tasks"].equals(old_replicates["n_eval_tasks"]):
            raise AssertionError("Omitted-item splits differ from the saved protocol")
    audit_differences = {}
    if len(replicates) == len(old_replicates):
        for column in ("raw_tau", "cbc_tau", "xu_tau"):
            difference = replicates[column] - old_replicates[column]
            audit_differences[column] = {
                "changed_replicates": int((difference.abs() > 1e-12).sum()),
                "maximum_absolute_difference": float(difference.abs().max()),
                "mean_difference": float(difference.mean()),
            }
    full_matrix = mean_matrix(np.ones(len(tasks)))
    full_beta = reference.compute_beta_from_matrix(full_matrix)
    full_btl, full_judge_weights = btl_matrix(np.ones(len(tasks)))
    full_corrected = full_matrix - full_beta
    payload = dict(old)
    payload["protocol"] = {
        **old["protocol"], "seed": args.seed, "bootstrap_reps": args.bootstrap_reps,
        "skipped_replicates": skipped, "n_tasks_complete_panel": len(tasks),
        "n_aligned_items": len(tasks), "observed_margins": int(finite.sum()),
        "missing_margins": int((~finite).sum()),
        "item_selection": "First 1500 dataset item IDs, 0 through 1499",
        "subset_counts": panel.drop_duplicates("item_id")["subset"].value_counts().to_dict(),
        "missingness": "Nonfinite margins omitted from both weighted numerator and denominator and from pairwise comparisons",
        "cbc_fitting": "Bootstrap in-bag items; margin means and their ordering evaluated on omitted items",
        "btl_fitting": "Adapted BTL ranking estimated from omitted-item comparisons, with source subset as judge stratum",
        "interval": "2.5th and 97.5th percentiles of omitted-item bootstrap evaluation statistics",
        "reference_input_equivalence_checked": True,
    }
    payload["results"] = results
    payload["full_panel"] = {
        **old["full_panel"],
        "raw_tau_full_panel": reference.mean_pairwise_kendall_tau(full_matrix),
        "cbc_tau_full_panel": reference.mean_pairwise_kendall_tau(full_corrected),
        "xu_tau_full_panel": reference.mean_pairwise_kendall_tau(full_btl),
        "backbone_rankings_raw": reference.rank_backbones(full_matrix),
        "backbone_rankings_cbc": reference.rank_backbones(full_corrected),
        "backbone_rankings_xu": reference.rank_backbones(full_btl),
        "xu_judge_weights": full_judge_weights,
    }
    prior_audit = old.get("missing_margin_audit", {})
    payload["missing_margin_audit"] = {
        "corrections": [
            "Omit missing margins from weighted score-mean denominators",
            "Omit unavailable pairwise comparisons instead of counting them as ties",
        ],
        "previous_results": prior_audit.get("previous_results", old["results"]),
        "current_results": results,
        "replicate_comparison_to_previous_file": audit_differences,
        "runtime_seconds": time.perf_counter() - started,
    }
    replicates.to_csv(replicates_path, index=False)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "audit": payload["missing_margin_audit"]}, indent=2))


if __name__ == "__main__":
    main()
