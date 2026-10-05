#!/usr/bin/env python3
"""Audit saved localization records and score-transform diagnostics.

No new API calls. Streams existing panel JSON and the released score matrix.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ANALYSIS = REPO / "paper" / "analysis"
OUT = ANALYSIS / "aamas_2027" / "results"
BENCH = REPO / "benchmark_tests"

LANGUAGES = [
    "English",
    "Arabic",
    "Turkish",
    "Chinese",
    "Hindi",
    "Japanese",
    "Spanish",
    "Swahili",
]
BACKBONES = {
    "gpt-4o": "GPT-4o",
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4.6": "Sonnet",
    "gemini-3-flash-preview": "Gemini",
    "deepseek-v3.2": "DeepSeek",
    "qwen3.5-9b": "Qwen",
}
EVALUATORS = list(BACKBONES.values())
FRAMEWORKS = ["MetaGPT", "GPT-Pilot", "OpenHands"]
BACKTICK_RE = re.compile(r"`[^`]+`")


def double_center(matrix: pd.DataFrame) -> pd.DataFrame:
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    grand = float(matrix.values.mean())
    return matrix.sub(row_means, axis=0).sub(col_means, axis=1) + grand


def score_matrix(task_df: pd.DataFrame, column: str = "score") -> pd.DataFrame:
    grouped = (
        task_df.groupby(["language", "backbone"], as_index=False)[column]
        .mean()
        .pivot(index="language", columns="backbone", values=column)
        .reindex(index=LANGUAGES, columns=EVALUATORS)
    )
    return grouped


def reversal_count(means: pd.DataFrame) -> int:
    n = 0
    for left, right in combinations(list(means.columns), 2):
        diffs = {
            language: float(means.loc[language, left] - means.loc[language, right])
            for language in means.index
        }
        product = min(da * db for da in diffs.values() for db in diffs.values())
        if product < 0:
            n += 1
    return n


def gate_spread(panel: pd.DataFrame, column: str, gate: float) -> float:
    rates = (
        panel.assign(passed=panel[column] > gate)
        .groupby(["backbone", "language"], as_index=False)
        .passed.mean()
    )
    spreads = rates.groupby("backbone").passed.agg(lambda s: float(s.max() - s.min()))
    return float(100.0 * spreads.mean())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def inv_logit(y: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-y))


def arcsin_sqrt(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 1.0)
    return np.arcsin(np.sqrt(p))


def inv_arcsin_sqrt(y: np.ndarray) -> np.ndarray:
    # The inverse is monotone only on the transformed proportion's domain.
    # Clipping the angle prevents negative/over-range corrections folding back.
    return np.sin(np.clip(y, 0.0, np.pi / 2.0)) ** 2


def correct_on_scale(
    panel: pd.DataFrame,
    forward,
    inverse,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    work["y"] = forward(work["score"].to_numpy(dtype=float) / 100.0)
    means = score_matrix(work, "y")
    beta = double_center(means)
    work["y_corr"] = work.apply(
        lambda row: row.y - float(beta.loc[row.language, row.backbone]), axis=1
    )
    work["score_corr"] = 100.0 * inverse(work["y_corr"].to_numpy(dtype=float))
    work["score_corr"] = work["score_corr"].clip(0.0, 100.0)
    return work, beta


def bounded_link_sensitivity(task_df: pd.DataFrame) -> dict:
    linear_means = score_matrix(task_df)
    linear_beta = double_center(linear_means)
    linear_vec = linear_beta.to_numpy(dtype=float).ravel()
    panel = task_df.copy()
    panel["score_corr_linear"] = panel.apply(
        lambda row: row.score - float(linear_beta.loc[row.language, row.backbone]),
        axis=1,
    )

    methods = {
        "linear": (
            panel,
            linear_beta,
            "score_corr_linear",
        ),
    }
    logit_panel, logit_beta = correct_on_scale(task_df, logit, inv_logit)
    methods["logit"] = (logit_panel, logit_beta, "score_corr")
    arcsin_panel, arcsin_beta = correct_on_scale(task_df, arcsin_sqrt, inv_arcsin_sqrt)
    methods["arcsin_sqrt"] = (arcsin_panel, arcsin_beta, "score_corr")

    # Rank transform of the 0-100 scores, then inverse via the original values'
    # order statistics is not unique; report beta agreement only.
    rank_panel = task_df.copy()
    rank_panel["y"] = rank_panel["score"].rank(method="average", pct=True)
    rank_means = score_matrix(rank_panel, "y")
    rank_beta = double_center(rank_means)

    report = {}
    for name, (work, beta, corr_col) in methods.items():
        means = score_matrix(work if name == "linear" else task_df.assign(score=work[corr_col]))
        beta_orig_units = double_center(score_matrix(work, "score" if name == "linear" else corr_col))
        vec = beta.to_numpy(dtype=float).ravel()
        if name == "linear":
            spearman = 1.0
            kendall = 1.0
            sign_flips = 0
        else:
            spearman = float(spearmanr(linear_vec, vec).statistic)
            kendall = float(kendalltau(linear_vec, vec).statistic)
            sign_flips = int(
                np.sum(
                    (np.sign(linear_vec) != np.sign(vec))
                    & (np.abs(linear_vec) >= 1.0)
                    & (np.abs(vec) >= 1e-12)
                )
            )
        report[name] = {
            "max_abs_beta_on_fit_scale": float(np.nanmax(np.abs(beta.to_numpy()))),
            "max_abs_beta_on_0_100_after_correction": float(
                np.nanmax(np.abs(beta_orig_units.to_numpy()))
            ),
            "spearman_vs_linear_beta": spearman,
            "kendall_vs_linear_beta": kendall,
            "sign_flips_vs_linear_cells_abs_ge_1": sign_flips,
            "reversals_on_raw_means": reversal_count(score_matrix(task_df)),
            "gate10_raw": gate_spread(work, "score", 10),
            "gate10_corr": gate_spread(work, corr_col, 10),
            "gate25_raw": gate_spread(work, "score", 25),
            "gate25_corr": gate_spread(work, corr_col, 25),
            "gate50_raw": gate_spread(work, "score", 50),
            "gate50_corr": gate_spread(work, corr_col, 50),
        }
        report[name]["gate10_reduction"] = (
            report[name]["gate10_raw"] - report[name]["gate10_corr"]
        )
        report[name]["gate25_reduction"] = (
            report[name]["gate25_raw"] - report[name]["gate25_corr"]
        )
        report[name]["gate50_reduction"] = (
            report[name]["gate50_raw"] - report[name]["gate50_corr"]
        )

    rank_vec = rank_beta.to_numpy(dtype=float).ravel()
    report["rank_pct"] = {
        "max_abs_beta_on_fit_scale": float(np.nanmax(np.abs(rank_beta.to_numpy()))),
        "spearman_vs_linear_beta": float(spearmanr(linear_vec, rank_vec).statistic),
        "kendall_vs_linear_beta": float(kendalltau(linear_vec, rank_vec).statistic),
        "sign_flips_vs_linear_cells_abs_ge_1": int(
            np.sum(
                (np.sign(linear_vec) != np.sign(rank_vec))
                & (np.abs(linear_vec) >= 1.0)
            )
        ),
        "note": "Rank transform has no unique inverse to 0-100, so gate spreads are not reported.",
    }
    report["linear_max_abs_beta"] = float(np.nanmax(np.abs(linear_beta.to_numpy())))
    report["linear_gpt4o_spanish"] = float(linear_beta.loc["Spanish", "GPT-4o"])
    return report


def classify_reason(text: str) -> str:
    has_sat = "<SATISFIED>" in text
    has_unsat = "<UNSATISFIED>" in text
    if "Final judgment step failed" in text:
        return "failed_step"
    if not text.strip():
        return "empty"
    if has_sat and has_unsat:
        return "both_tags"
    if has_sat:
        return "satisfied_tag"
    if has_unsat:
        return "unsatisfied_tag"
    return "missing_tag"


def parse_failure_audit() -> dict:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    files_seen = 0
    for model_dir, backbone in BACKBONES.items():
        for language in LANGUAGES:
            for framework in FRAMEWORKS:
                gray_dir = (
                    BENCH
                    / model_dir
                    / language
                    / framework
                    / "judgment"
                    / framework
                    / "agent_as_a_judge"
                    / "gray_box"
                )
                if not gray_dir.exists():
                    continue
                for path in gray_dir.glob("*.json"):
                    files_seen += 1
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    for stat in obj.get("judge_stats") or []:
                        reason = (stat.get("llm_stats") or {}).get("reason", [])
                        if isinstance(reason, list):
                            text = "\n".join(str(x) for x in reason)
                        else:
                            text = str(reason or "")
                        bucket = classify_reason(text)
                        counts[(backbone, language, bucket)] += 1
                        totals[(backbone, language)] += 1
    rows = []
    for (backbone, language), n in sorted(totals.items()):
        row = {
            "backbone": backbone,
            "language": language,
            "n_requirements": int(n),
        }
        for bucket in (
            "satisfied_tag",
            "unsatisfied_tag",
            "missing_tag",
            "empty",
            "failed_step",
            "both_tags",
        ):
            row[bucket] = int(counts[(backbone, language, bucket)])
            row[f"{bucket}_rate"] = float(counts[(backbone, language, bucket)] / n)
        row["noncompliant_rate"] = float(
            (row["missing_tag"] + row["empty"] + row["failed_step"]) / n
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    by_eval = df.groupby("backbone", as_index=False).agg(
        n_requirements=("n_requirements", "sum"),
        missing_tag=("missing_tag", "sum"),
        empty_resp=("empty", "sum"),
        failed_step=("failed_step", "sum"),
        both_tags=("both_tags", "sum"),
    )
    by_eval["noncompliant_rate"] = (
        by_eval["missing_tag"] + by_eval["empty_resp"] + by_eval["failed_step"]
    ) / by_eval["n_requirements"]
    by_lang = df.groupby("language", as_index=False).agg(
        n_requirements=("n_requirements", "sum"),
        missing_tag=("missing_tag", "sum"),
        empty_resp=("empty", "sum"),
        failed_step=("failed_step", "sum"),
    )
    by_lang["noncompliant_rate"] = (
        by_lang["missing_tag"] + by_lang["empty_resp"] + by_lang["failed_step"]
    ) / by_lang["n_requirements"]
    n_all = int(df.n_requirements.sum())
    n_bad = int((df.missing_tag + df["empty"] + df.failed_step).sum())
    n_no_qwen = int(df.loc[df.backbone != "Qwen", "n_requirements"].sum())
    n_bad_no_qwen = int(
        (
            df.loc[df.backbone != "Qwen", "missing_tag"]
            + df.loc[df.backbone != "Qwen", "empty"]
            + df.loc[df.backbone != "Qwen", "failed_step"]
        ).sum()
    )
    df.to_csv(OUT / "parse_failure_cells.csv", index=False)
    return {
        "files_seen": files_seen,
        "n_requirement_judgments": n_all,
        "overall_noncompliant_rate": float(n_bad / n_all) if n_all else None,
        "noncompliant_rate_excluding_qwen": float(n_bad_no_qwen / n_no_qwen)
        if n_no_qwen
        else None,
        "by_evaluator": by_eval.to_dict(orient="records"),
        "by_language": by_lang.to_dict(orient="records"),
        "max_cell_noncompliant": df.loc[df.noncompliant_rate.idxmax()].to_dict()
        if not df.empty
        else {},
        "qwen_by_language": df[df.backbone == "Qwen"][
            ["language", "n_requirements", "missing_tag", "empty", "noncompliant_rate"]
        ].to_dict(orient="records"),
        "gpt4o_by_language": df[df.backbone == "GPT-4o"][
            ["language", "n_requirements", "missing_tag", "empty", "noncompliant_rate"]
        ].to_dict(orient="records"),
    }


def _load_prompt_module(name: str, relative: str):
    import importlib.util

    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def translation_token_audit() -> dict:
    source_dir = BENCH / "gpt-4o" / "English" / "MetaGPT" / "devai" / "instances"
    source_files = {p.name: p for p in source_dir.glob("*.json")}
    lang_rows = []
    for language in LANGUAGES:
        if language == "English":
            continue
        target_dir = BENCH / "gpt-4o" / language / "MetaGPT" / "devai" / "instances"
        n = 0
        backtick_mismatch = 0
        req_len_mismatch = 0
        missing_files = 0
        for name, src_path in source_files.items():
            tgt_path = target_dir / name
            if not tgt_path.exists():
                missing_files += 1
                continue
            n += 1
            src = json.loads(src_path.read_text(encoding="utf-8"))
            tgt = json.loads(tgt_path.read_text(encoding="utf-8"))
            src_reqs = src.get("requirements") or []
            tgt_reqs = tgt.get("requirements") or []
            if len(src_reqs) != len(tgt_reqs):
                req_len_mismatch += 1
            src_text = "\n".join(
                [src.get("query", "")]
                + [r.get("criteria", "") for r in src_reqs]
            )
            tgt_text = "\n".join(
                [tgt.get("query", "")]
                + [r.get("criteria", "") for r in tgt_reqs]
            )
            if Counter(BACKTICK_RE.findall(src_text)) != Counter(
                BACKTICK_RE.findall(tgt_text)
            ):
                backtick_mismatch += 1
        lang_rows.append(
            {
                "language": language,
                "n_instances_compared": n,
                "missing_files": missing_files,
                "requirement_length_mismatch": req_len_mismatch,
                "backtick_mismatch_instances": backtick_mismatch,
                "backtick_mismatch_rate": float(backtick_mismatch / n) if n else None,
            }
        )

    judge_sys = _load_prompt_module(
        "judge_sys",
        "agent_as_a_judge/module/prompt/system_prompt_judge.py",
    )
    judge_user = _load_prompt_module(
        "judge_user",
        "agent_as_a_judge/module/prompt/prompt_judge.py",
    )

    prompt_rows = []
    for language in LANGUAGES:
        system = judge_sys.get_judge_system_prompt(language=language)
        user = judge_user.get_judge_prompt(
            criteria="__CRITERIA__", evidence="__EVIDENCE__", language=language
        )
        prompt_rows.append(
            {
                "language": language,
                "system_has_satisfied": "<SATISFIED>" in system,
                "system_has_unsatisfied": "<UNSATISFIED>" in system,
                "user_has_satisfied": "<SATISFIED>" in user,
                "user_has_unsatisfied": "<UNSATISFIED>" in user,
            }
        )
    return {
        "translator": "LLM via translate_instances_fields.py; default model gpt-4o-2024-08-06",
        "protected_token_rule": "Keep identifiers, paths, and backtick spans unchanged",
        "instance_audit": lang_rows,
        "judge_prompt_tag_preservation": prompt_rows,
        "professional_translators": False,
        "back_translation_documented": False,
    }


def score_mapping() -> dict:
    return {
        "requirement_parser": (
            "A response containing the substring <SATISFIED> is satisfied; "
            "empty, failed, missing-tag, and <UNSATISFIED> responses are unsatisfied."
        ),
        "task_score": "100 * (number of satisfied requirements) / (number of requirements)",
        "weights": "none; each requirement is unweighted",
        "prerequisites": "recorded in the instance schema but not used in the score",
        "majority_vote_default": 1,
        "task_level_panel": "mean of the three framework runs",
        "gate": "pass if task score > threshold (strict greater-than)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-link-only", action="store_true",
                        help="Recompute saved-score transforms without original benchmark workspaces.")
    args = parser.parse_args()
    if args.bounded_link_only:
        task_df = pd.read_csv(ANALYSIS / "task_level_scores.csv")
        payload = bounded_link_sensitivity(task_df)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "bounded_link_rerun.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return
    if not BENCH.is_dir():
        raise SystemExit("Full parser/translation audit needs original benchmark_tests; use --bounded-link-only for saved-score transforms.")
    OUT.mkdir(parents=True, exist_ok=True)
    task_df = pd.read_csv(ANALYSIS / "task_level_scores.csv")
    task_df = task_df[
        task_df.language.isin(LANGUAGES) & task_df.backbone.isin(EVALUATORS)
    ].copy()

    payload = {
        "score_mapping": score_mapping(),
        "translation": translation_token_audit(),
        "parse_failures": parse_failure_audit(),
        "bounded_link": bounded_link_sensitivity(task_df),
    }
    out_path = OUT / "localization_and_transform_audit.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "wrote": str(out_path),
        "files_seen": payload["parse_failures"]["files_seen"],
        "n_requirement_judgments": payload["parse_failures"]["n_requirement_judgments"],
        "overall_noncompliant_rate": payload["parse_failures"]["overall_noncompliant_rate"],
        "linear_max_abs_beta": payload["bounded_link"]["linear_max_abs_beta"],
        "logit_spearman": payload["bounded_link"]["logit"]["spearman_vs_linear_beta"],
        "arcsin_spearman": payload["bounded_link"]["arcsin_sqrt"]["spearman_vs_linear_beta"],
        "logit_gate25_reduction": payload["bounded_link"]["logit"]["gate25_reduction"],
        "arcsin_gate25_reduction": payload["bounded_link"]["arcsin_sqrt"]["gate25_reduction"],
        "linear_gate25_reduction": payload["bounded_link"]["linear"]["gate25_reduction"],
    }, indent=2))


if __name__ == "__main__":
    main()
