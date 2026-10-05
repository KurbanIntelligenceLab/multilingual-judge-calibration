"""Recheck the M-RewardBench panel-mean gold-agreement claim.

Run from the repository root:
    python paper/code/analysis/audit_mrewardbench_anchor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[2]
DATA = PAPER / "analysis" / "aamas_2027" / "data" / "mrewardbench"
RESULTS = PAPER / "analysis" / "aamas_2027" / "results"
TOLERANCE = 1e-9
KEYS = ["item_id", "item_numeric_id", "subset", "language"]


def main() -> None:
    panel = pd.read_csv(DATA / "pilot_complete_panel.csv")
    previous = pd.read_csv(DATA / "human_anchor_sample_original.csv")
    beta = pd.read_csv(DATA / "beta_hat.csv").set_index("language")
    beta_long = beta.stack().rename("beta_hat").reset_index()
    beta_long.columns = ["language", "evaluator_model", "beta_hat"]
    scored = panel.merge(beta_long, on=["language", "evaluator_model"], validate="many_to_one")
    scored["cbc_margin"] = scored["margin"] - scored["beta_hat"]
    item = (
        scored.groupby(KEYS, as_index=False)
        .agg(
            raw_margin=("margin", "mean"),
            cbc_margin=("cbc_margin", "mean"),
            evaluators=("margin", "count"),
        )
    )
    checked = previous[KEYS].merge(item, on=KEYS, validate="one_to_one")
    if len(checked) != len(previous):
        raise AssertionError("The saved sample does not match the score panel.")
    if not np.allclose(checked["raw_margin"], previous["raw_margin"], atol=TOLERANCE):
        raise AssertionError("Raw margins differ from the saved sample.")
    if not np.allclose(checked["cbc_margin"], previous["cbc_margin"], atol=TOLERANCE):
        raise AssertionError("CBC margins differ from the saved sample.")

    checked["raw_correct"] = (checked["raw_margin"] > TOLERANCE).astype(int)
    checked["cbc_correct"] = (checked["cbc_margin"] > TOLERANCE).astype(int)
    previous_flips = int((previous["raw_correct"] != previous["cbc_correct"]).sum())
    corrected_flips = int((checked["raw_correct"] != checked["cbc_correct"]).sum())
    if corrected_flips:
        raise AssertionError("Panel-mean decisions changed after tie correction.")
    complete = checked["evaluators"].eq(beta.shape[1])
    max_complete_difference = float(
        (checked.loc[complete, "raw_margin"] - checked.loc[complete, "cbc_margin"]).abs().max()
    )
    if max_complete_difference > TOLERANCE:
        raise AssertionError("CBC changed a complete evaluator-panel mean.")

    summary = {
        "sampled_item_language_instances": len(checked),
        "evaluator_backbones": int(beta.shape[1]),
        "incomplete_sampled_instances": int((~complete).sum()),
        "previous_apparent_flips": previous_flips,
        "corrected_flips": corrected_flips,
        "tie_tolerance": TOLERANCE,
        "raw_agreement": float(checked["raw_correct"].mean()),
        "corrected_agreement": float(checked["cbc_correct"].mean()),
        "max_complete_panel_mean_difference": max_complete_difference,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    checked.to_csv(RESULTS / "mrewardbench_human_anchor_corrected.csv", index=False)
    (RESULTS / "mrewardbench_human_anchor_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
