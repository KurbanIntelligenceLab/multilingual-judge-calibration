"""Saved-panel diagnostics for missingness, task interactions, and repeat noise.

The missingness study masks complete task-language-evaluator cells at random.
It measures recovery of the interaction estimated from the observed complete
panel, not recovery of an unobserved true judge effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "analysis" / "aamas_2027" / "results"
LANGUAGES = ["English", "Arabic", "Turkish", "Chinese", "Hindi", "Japanese", "Spanish", "Swahili"]
EVALUATORS = ["GPT-4o", "GPT-5.4", "Sonnet", "Gemini", "DeepSeek", "Qwen"]


def double_center(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=-1, keepdims=True) - a.mean(axis=-2, keepdims=True) + a.mean(axis=(-2, -1), keepdims=True)


def task_adjusted_cell_means(y: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Least squares task plus cell fixed effects, by alternating centering."""
    observed = np.where(seen, y, np.nan)
    cells = np.nanmean(observed, axis=0)
    assert np.isfinite(cells).all() and np.isfinite(np.nanmean(observed, axis=(1, 2))).all()
    for _ in range(100):
        task_effect = np.nanmean(observed - cells[None, :, :], axis=(1, 2))
        task_effect -= task_effect.mean()
        updated = np.nanmean(observed - task_effect[:, None, None], axis=0)
        if np.max(np.abs(updated - cells)) < 1e-10:
            return updated
        cells = updated
    raise RuntimeError("Fixed-effect updates did not converge")


def missingness_study(y: np.ndarray, full_beta: np.ndarray) -> dict:
    rng = np.random.default_rng(20270930)
    output = {"seed": 20270930, "replicates": 500, "target": "complete-panel estimated beta", "missingness": "independent MCAR task-language-evaluator cells", "rows": []}
    for fraction in (0.05, 0.10):
        records = []
        for _ in range(500):
            seen = rng.random(y.shape) >= fraction
            observed = np.where(seen, y, np.nan)
            if np.isnan(np.nanmean(observed, axis=0)).any():
                raise AssertionError("An entire language-evaluator cell was missing")
            estimates = {
                "available_cell_means": double_center(np.nanmean(observed, axis=0)),
                "task_fixed_effects": double_center(task_adjusted_cell_means(y, seen)),
            }
            for method, beta in estimates.items():
                err = beta - full_beta
                records.append({
                    "method": method,
                    "mae": float(np.mean(np.abs(err))),
                    "max_abs_error": float(np.max(np.abs(err))),
                    "sign_errors_abs_beta_at_least_5": int(np.sum((np.sign(beta) != np.sign(full_beta)) & (np.abs(full_beta) >= 5))),
                })
        frame = pd.DataFrame(records)
        for method, subset in frame.groupby("method"):
            output["rows"].append({
                "fraction_masked": fraction,
                "method": method,
                "mean_mae": float(subset.mae.mean()),
                "p95_mae": float(subset.mae.quantile(.95)),
                "mean_max_abs_error": float(subset.max_abs_error.mean()),
                "p95_max_abs_error": float(subset.max_abs_error.quantile(.95)),
                "replicates_with_sign_error_abs_beta_at_least_5": int((subset.sign_errors_abs_beta_at_least_5 > 0).sum()),
            })
    return output


def task_heterogeneity(y: np.ndarray, full_beta: np.ndarray) -> dict:
    per_task_beta = double_center(y)
    residual = per_task_beta - full_beta
    selected = per_task_beta[:, LANGUAGES.index("Spanish"), EVALUATORS.index("GPT-4o")]
    return {
        "pooled_beta_rms": float(np.sqrt(np.mean(full_beta ** 2))),
        "task_specific_deviation_rms": float(np.sqrt(np.mean(residual ** 2))),
        "task_specific_deviation_mean_absolute": float(np.mean(np.abs(residual))),
        "gpt4o_spanish_beta_task_mean": float(selected.mean()),
        "gpt4o_spanish_beta_task_sd": float(selected.std(ddof=1)),
        "gpt4o_spanish_negative_tasks": int((selected < 0).sum()),
        "n_tasks": int(len(selected)),
        "interpretation": "Task-specific residual includes repeat-call noise and task-by-language-by-evaluator variation; these cannot be separated in the original panel.",
    }


def repeat_sd_by_language() -> list[dict]:
    frame = pd.read_csv(OUT / "replication_scores.csv")
    assert frame.groupby(["task", "language", "evaluator"]).repeat.nunique().eq(3).all()
    cell_sd = frame.groupby(["task", "language", "evaluator"]).score.std(ddof=1).reset_index(name="sd")
    return [
        {"language": language, "mean_sd": float(subset.sd.mean()), "median_sd": float(subset.sd.median()), "n_task_evaluator_cells": len(subset)}
        for language, subset in cell_sd.groupby("language")
    ]


def main() -> None:
    frame = pd.read_csv(PAPER / "analysis" / "task_level_scores.csv")
    tasks = sorted(frame.task.unique())
    index = pd.MultiIndex.from_product([tasks, LANGUAGES, EVALUATORS], names=["task", "language", "backbone"])
    y = frame.set_index(["task", "language", "backbone"]).reindex(index).score.to_numpy().reshape(len(tasks), len(LANGUAGES), len(EVALUATORS))
    assert np.isfinite(y).all() and y.shape == (55, 8, 6)
    full_beta = double_center(y.mean(axis=0))
    assert abs(full_beta[LANGUAGES.index("Spanish"), EVALUATORS.index("GPT-4o")] + 20.605825116241785) < 1e-9
    report = {
        "missingness_study": missingness_study(y, full_beta),
        "task_heterogeneity": task_heterogeneity(y, full_beta),
        "repeat_sd_by_language": repeat_sd_by_language(),
    }
    destination = OUT / "panel_sensitivity.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
