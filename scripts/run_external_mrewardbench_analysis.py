from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    ROOT
    / "data"
    / "external_validation"
    / "mrewardbench_panel"
    / "collection_logs"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "external_validation"
    / "mrewardbench_panel"
    / "analysis_1500_item"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing per-model M-RewardBench collection logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store merged panel-analysis outputs.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
        help="Number of bootstrap train/OOB replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--max-items-per-language",
        type=int,
        default=None,
        help="Optional cap on aligned item_numeric_id values (keeps rows with item_numeric_id < cap).",
    )
    parser.add_argument(
        "--anchor-sample-per-language",
        type=int,
        default=100,
        help="Number of stratified gold-anchor items to sample per language.",
    )
    return parser.parse_args()


def bootstrap_task_split(tasks: list[str], rng: np.random.Generator) -> tuple[dict[str, int], list[str]]:
    sampled = rng.choice(tasks, size=len(tasks), replace=True)
    train_weights = {task: 0 for task in tasks}
    for task in sampled:
        train_weights[str(task)] += 1
    eval_tasks = [task for task in tasks if train_weights[task] == 0]
    return train_weights, eval_tasks


def summarize_metric(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_replicates": int(len(values)),
    }


def mean_pairwise_kendall_tau(score_matrix: pd.DataFrame) -> float:
    taus: list[float] = []
    for left, right in combinations(list(score_matrix.index), 2):
        tau, _ = kendalltau(score_matrix.loc[left].to_numpy(), score_matrix.loc[right].to_numpy())
        if not np.isnan(tau):
            taus.append(float(tau))
    if not taus:
        return 0.0
    return float(np.mean(taus))


def build_score_matrix(
    task_df: pd.DataFrame,
    languages: list[str],
    backbones: list[str],
    task_weights: dict[str, int] | None = None,
) -> pd.DataFrame:
    # Missing margins are unavailable observations, not zero-valued scores.
    # Drop them before both the weighted numerator and its denominator.
    matrix_df = task_df[np.isfinite(task_df["score"])].copy()
    if task_weights is None:
        grouped = matrix_df.groupby(["language", "backbone"], as_index=False)["score"].mean()
    else:
        matrix_df["weight"] = matrix_df["task"].map(task_weights).fillna(0).astype(int)
        matrix_df = matrix_df[matrix_df["weight"] > 0].copy()
        matrix_df["weighted_score"] = matrix_df["score"] * matrix_df["weight"]
        grouped = (
            matrix_df.groupby(["language", "backbone"], as_index=False)[["weighted_score", "weight"]].sum()
        )
        grouped["score"] = grouped["weighted_score"] / grouped["weight"]
        grouped = grouped[["language", "backbone", "score"]]

    matrix = grouped.pivot(index="language", columns="backbone", values="score").reindex(
        index=languages,
        columns=backbones,
    )
    if matrix.isna().any().any():
        raise ValueError("Some language/evaluator mean scores have no observed margins after aligned-item filtering.")
    return matrix


def compute_beta_from_matrix(score_means: pd.DataFrame) -> pd.DataFrame:
    row_means = score_means.mean(axis=1)
    col_means = score_means.mean(axis=0)
    grand_mean = float(score_means.to_numpy(dtype=float).mean())
    return score_means.sub(row_means, axis=0).sub(col_means, axis=1) + grand_mean


def build_pairwise_evaluator_comparisons(
    task_df: pd.DataFrame,
    languages: list[str],
    backbones: list[str],
) -> dict[str, pd.DataFrame]:
    backbone_to_idx = {str(backbone): idx for idx, backbone in enumerate(backbones)}
    judge_labels = sorted(task_df["subset"].astype(str).unique())
    judge_to_idx = {label: idx for idx, label in enumerate(judge_labels)}
    pairwise_by_language: dict[str, pd.DataFrame] = {}

    for language in languages:
        language_df = task_df[task_df["language"] == language]
        rows: list[tuple[int, int, int, float]] = []
        for _, item_df in language_df.groupby("task", sort=False):
            judge_label = str(item_df["subset"].iloc[0])
            score_map = {
                str(row.backbone): float(row.score)
                for row in item_df[["backbone", "score"]].itertuples(index=False)
            }
            for left, right in combinations(backbones, 2):
                left_score = score_map[str(left)]
                right_score = score_map[str(right)]
                if not (np.isfinite(left_score) and np.isfinite(right_score)):
                    continue
                if left_score > right_score:
                    outcome = 1.0
                elif left_score < right_score:
                    outcome = 0.0
                else:
                    outcome = 0.5
                rows.append((backbone_to_idx[str(left)], backbone_to_idx[str(right)], judge_to_idx[judge_label], outcome))

        pairwise_df = pd.DataFrame(rows, columns=["i", "j", "k", "y"])
        pairwise_df = (
            pairwise_df.groupby(["i", "j", "k"], as_index=False)
            .agg(n=("y", "size"), ybar=("y", "mean"))
            .sort_values(["i", "j", "k"], ignore_index=True)
        )
        pairwise_by_language[str(language)] = pairwise_df

    return pairwise_by_language


def fit_judge_aware_btl(
    pairwise_df: pd.DataFrame,
    n_models: int,
    n_judges: int,
) -> tuple[np.ndarray, np.ndarray]:
    i_idx = pairwise_df["i"].to_numpy(dtype=int)
    j_idx = pairwise_df["j"].to_numpy(dtype=int)
    k_idx = pairwise_df["k"].to_numpy(dtype=int)
    weights = pairwise_df["n"].to_numpy(dtype=float)
    ybar = pairwise_df["ybar"].to_numpy(dtype=float)

    n_score_params = n_models - 1
    n_alpha_params = n_judges - 1

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_free = params[:n_score_params]
        alpha_free = params[n_score_params:]
        scores = np.concatenate([score_free, [-float(np.sum(score_free))]])
        alphas = np.concatenate([alpha_free, [-float(np.sum(alpha_free))]])
        return scores, alphas

    def objective_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        scores, alphas = unpack(params)
        gammas = np.exp(alphas)
        score_diff = scores[i_idx] - scores[j_idx]
        logits = gammas[k_idx] * score_diff
        probs = np.clip(expit(logits), 1e-9, 1.0 - 1e-9)
        residual = weights * (probs - ybar)

        loss = -float(np.sum(weights * (ybar * np.log(probs) + (1.0 - ybar) * np.log(1.0 - probs))))

        grad_scores = np.zeros(n_models, dtype=float)
        np.add.at(grad_scores, i_idx, residual * gammas[k_idx])
        np.add.at(grad_scores, j_idx, -residual * gammas[k_idx])
        score_grad = grad_scores[:n_score_params] - grad_scores[-1]

        grad_alphas = np.zeros(n_judges, dtype=float)
        np.add.at(grad_alphas, k_idx, residual * logits)
        alpha_grad = grad_alphas[:n_alpha_params] - grad_alphas[-1]

        return loss, np.concatenate([score_grad, alpha_grad])

    init = np.zeros(n_score_params + n_alpha_params, dtype=float)
    result = minimize(
        fun=lambda x: objective_and_grad(x)[0],
        x0=init,
        jac=lambda x: objective_and_grad(x)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Judge-aware BTL optimization failed: {result.message}")
    scores, alphas = unpack(result.x)
    return scores, np.exp(alphas)


def build_xu_score_matrix(
    task_df: pd.DataFrame,
    languages: list[str],
    backbones: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    pairwise_by_language = build_pairwise_evaluator_comparisons(task_df, languages=languages, backbones=backbones)
    judge_labels = sorted(task_df["subset"].astype(str).unique())
    judge_weights: dict[str, dict[str, float]] = {}
    rows: list[list[float]] = []

    for language in languages:
        pairwise_df = pairwise_by_language[str(language)]
        scores, gammas = fit_judge_aware_btl(pairwise_df, n_models=len(backbones), n_judges=len(judge_labels))
        rows.append(list(scores))
        judge_weights[str(language)] = {
            str(label): float(gamma) for label, gamma in zip(judge_labels, gammas, strict=True)
        }

    matrix = pd.DataFrame(rows, index=languages, columns=backbones, dtype=float)
    return matrix, judge_weights


def load_model_outputs(input_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    metadata_paths = sorted(input_dir.glob("*/run_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No run_metadata.json files found under {input_dir}")

    margin_frames: list[pd.DataFrame] = []
    languages: list[str] | None = None
    model_order: list[str] = []

    for meta_path in metadata_paths:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if languages is None:
            languages = list(metadata["languages"])
        elif list(metadata["languages"]) != languages:
            raise ValueError(f"Language mismatch in {meta_path}")

        model_name = str(metadata["models"][0])
        model_order.append(model_name)

        margin_path = meta_path.with_name("run_pair_margins.csv")
        margin_df = pd.read_csv(margin_path)
        margin_df["evaluator_model"] = margin_df["evaluator_model"].astype(str)
        margin_df["language"] = margin_df["language"].astype(str)
        margin_df["item_id"] = margin_df["item_id"].astype(str)
        margin_df["margin"] = margin_df["margin"].astype(float)
        margin_frames.append(margin_df)

    assert languages is not None
    merged = pd.concat(margin_frames, ignore_index=True)
    return merged, languages, model_order


def restrict_to_complete_tasks(
    merged_df: pd.DataFrame,
    languages: list[str],
    backbones: list[str],
) -> pd.DataFrame:
    expected = len(languages) * len(backbones)
    per_task = (
        merged_df.groupby("item_id")
        .agg(
            n_rows=("margin", "size"),
            n_languages=("language", "nunique"),
            n_backbones=("evaluator_model", "nunique"),
        )
        .reset_index()
    )
    keep_tasks = per_task[
        (per_task["n_rows"] == expected)
        & (per_task["n_languages"] == len(languages))
        & (per_task["n_backbones"] == len(backbones))
    ]["item_id"]

    filtered = merged_df[merged_df["item_id"].isin(set(keep_tasks))].copy()
    dupes = filtered.duplicated(subset=["item_id", "language", "evaluator_model"])
    if dupes.any():
        raise ValueError("Duplicate item/language/model rows found after merging.")
    return filtered


def run_bootstrap_comparison(
    task_df: pd.DataFrame,
    languages: list[str],
    backbones: list[str],
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    tasks = sorted(task_df["task"].unique())
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict] = []
    skipped = 0

    for replicate in range(bootstrap_reps):
        train_weights, eval_tasks = bootstrap_task_split(tasks, rng)
        if not eval_tasks:
            skipped += 1
            continue

        eval_df = task_df[task_df["task"].isin(eval_tasks)].copy()
        eval_matrix = build_score_matrix(eval_df, languages=languages, backbones=backbones)
        beta_hat = compute_beta_from_matrix(
            build_score_matrix(task_df, languages=languages, backbones=backbones, task_weights=train_weights)
        )
        cbc_matrix = eval_matrix - beta_hat
        xu_matrix, _ = build_xu_score_matrix(eval_df, languages=languages, backbones=backbones)

        raw_tau = mean_pairwise_kendall_tau(eval_matrix)
        cbc_tau = mean_pairwise_kendall_tau(cbc_matrix)
        xu_tau = mean_pairwise_kendall_tau(xu_matrix)
        replicate_rows.append(
            {
                "replicate": replicate,
                "raw_tau": raw_tau,
                "cbc_tau": cbc_tau,
                "xu_tau": xu_tau,
                "tau_gain": cbc_tau - raw_tau,
                "n_eval_tasks": len(eval_tasks),
            }
        )

    replicate_df = pd.DataFrame(replicate_rows)
    raw_summary = summarize_metric(replicate_df["raw_tau"].to_numpy(dtype=float))
    cbc_summary = summarize_metric(replicate_df["cbc_tau"].to_numpy(dtype=float))
    xu_summary = summarize_metric(replicate_df["xu_tau"].to_numpy(dtype=float))
    gain_summary = summarize_metric(replicate_df["tau_gain"].to_numpy(dtype=float))

    payload = {
        "protocol": {
            "resampling": "bootstrap_train_oob_eval",
            "bootstrap_reps": bootstrap_reps,
            "seed": seed,
            "skipped_replicates": skipped,
            "metric": "mean_pairwise_kendall_tau_across_languages",
            "n_tasks_complete_panel": int(len(tasks)),
            "n_languages": len(languages),
            "n_backbones": len(backbones),
        },
        "results": {
            "Raw": raw_summary,
            "CBC": cbc_summary,
            "JudgeAwareBTL_Xu2026_adapted": xu_summary,
            "CBC_minus_Raw": gain_summary,
        },
    }
    return replicate_df, payload


def rank_backbones(matrix: pd.DataFrame) -> dict[str, list[str]]:
    ranking = {}
    for language in matrix.index:
        order = matrix.loc[language].sort_values(ascending=False, kind="stable").index.tolist()
        ranking[str(language)] = [str(backbone) for backbone in order]
    return ranking


def allocate_stratified_sample(counts: pd.Series, target_n: int) -> dict[str, int]:
    proportions = counts / float(counts.sum())
    raw_targets = proportions * float(target_n)
    base = raw_targets.apply(np.floor).astype(int)
    remainder = int(target_n - base.sum())
    if remainder > 0:
        fractional = (raw_targets - base).sort_values(ascending=False)
        for subset in fractional.index[:remainder]:
            base.loc[subset] += 1
    return {str(subset): int(value) for subset, value in base.items()}


def run_human_anchor_validation(
    complete_df: pd.DataFrame,
    beta_hat: pd.DataFrame,
    sample_per_language: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    if sample_per_language <= 0:
        raise ValueError("sample_per_language must be positive.")

    scored = complete_df.copy()
    scored["cbc_margin"] = scored.apply(
        lambda row: float(row["margin"] - beta_hat.loc[row["language"], row["evaluator_model"]]),
        axis=1,
    )
    item_df = (
        scored.groupby(["item_id", "item_numeric_id", "subset", "language"], as_index=False)
        .agg(raw_margin=("margin", "mean"), cbc_margin=("cbc_margin", "mean"))
    )

    rng = np.random.default_rng(seed)
    sampled_parts: list[pd.DataFrame] = []
    allocation_rows: list[dict] = []
    for language, language_df in item_df.groupby("language", sort=True):
        counts = language_df["subset"].value_counts().sort_index()
        targets = allocate_stratified_sample(counts, sample_per_language)
        for subset, subset_n in targets.items():
            subset_df = language_df[language_df["subset"] == subset].copy()
            chosen_idx = rng.choice(subset_df.index.to_numpy(), size=subset_n, replace=False)
            sampled_parts.append(subset_df.loc[chosen_idx])
            allocation_rows.append(
                {
                    "language": str(language),
                    "subset": str(subset),
                    "sample_n": int(subset_n),
                    "population_n": int(len(subset_df)),
                }
            )

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df["raw_correct"] = (sampled_df["raw_margin"] > 1e-9).astype(float)
    sampled_df["cbc_correct"] = (sampled_df["cbc_margin"] > 1e-9).astype(float)

    bootstrap_rng = np.random.default_rng(seed + 1)
    raw_rates: list[float] = []
    cbc_rates: list[float] = []
    for _ in range(bootstrap_reps):
        boot_parts: list[pd.DataFrame] = []
        for (language, subset), stratum_df in sampled_df.groupby(["language", "subset"], sort=True):
            chosen = bootstrap_rng.choice(len(stratum_df), size=len(stratum_df), replace=True)
            boot_parts.append(stratum_df.iloc[chosen])
        boot_df = pd.concat(boot_parts, ignore_index=True)
        raw_rates.append(float(boot_df["raw_correct"].mean()))
        cbc_rates.append(float(boot_df["cbc_correct"].mean()))

    raw_array = np.asarray(raw_rates, dtype=float)
    cbc_array = np.asarray(cbc_rates, dtype=float)
    gain_array = cbc_array - raw_array

    payload = {
        "protocol": {
            "evaluation": "human_anchor_preference_agreement",
            "anchor": "public_M-RewardBench_chosen_rejected_gold_preferences",
            "sample_per_language": int(sample_per_language),
            "sampling": "stratified_without_replacement_by_subset_within_language",
            "bootstrap_reps": int(bootstrap_reps),
            "seed": int(seed),
            "tie_handling": "mean_panel_margin <= 1e-9 counted as non-agreement; common absolute tie tolerance 1e-9",
        },
        "sampling_allocation": allocation_rows,
        "overall": {
            "raw_agreement": float(sampled_df["raw_correct"].mean()),
            "raw_ci_low": float(np.percentile(raw_array, 2.5)),
            "raw_ci_high": float(np.percentile(raw_array, 97.5)),
            "cbc_agreement": float(sampled_df["cbc_correct"].mean()),
            "cbc_ci_low": float(np.percentile(cbc_array, 2.5)),
            "cbc_ci_high": float(np.percentile(cbc_array, 97.5)),
            "gain": float(sampled_df["cbc_correct"].mean() - sampled_df["raw_correct"].mean()),
            "gain_ci_low": float(np.percentile(gain_array, 2.5)),
            "gain_ci_high": float(np.percentile(gain_array, 97.5)),
        },
        "by_language": [
            {
                "language": str(language),
                "raw_agreement": float(group_df["raw_correct"].mean()),
                "cbc_agreement": float(group_df["cbc_correct"].mean()),
            }
            for language, group_df in sampled_df.groupby("language", sort=True)
        ],
    }
    return sampled_df, payload


def portable_output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_df, languages, backbones = load_model_outputs(input_dir)
    if args.max_items_per_language is not None:
        merged_df = merged_df[merged_df["item_numeric_id"] < int(args.max_items_per_language)].copy()
    complete_df = restrict_to_complete_tasks(merged_df, languages=languages, backbones=backbones)
    task_df = complete_df.rename(columns={"item_id": "task", "evaluator_model": "backbone", "margin": "score"})[
        ["task", "language", "backbone", "score", "subset", "chosen_model", "rejected_model", "pref_correct"]
    ].copy()

    full_raw_matrix = build_score_matrix(task_df, languages=languages, backbones=backbones)
    beta_hat = compute_beta_from_matrix(full_raw_matrix)
    full_cbc_matrix = full_raw_matrix - beta_hat
    full_xu_matrix, xu_judge_weights = build_xu_score_matrix(task_df, languages=languages, backbones=backbones)
    replicate_df, payload = run_bootstrap_comparison(
        task_df=task_df,
        languages=languages,
        backbones=backbones,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    merged_path = output_dir / "pilot_complete_panel.csv"
    raw_matrix_path = output_dir / "raw_mean_matrix.csv"
    beta_path = output_dir / "beta_hat.csv"
    cbc_matrix_path = output_dir / "cbc_mean_matrix.csv"
    xu_matrix_path = output_dir / "xu_judge_aware_btl_matrix.csv"
    replicate_path = output_dir / "raw_vs_cbc_bootstrap_replicates.csv"
    summary_path = output_dir / "raw_vs_cbc_summary.json"
    anchor_sample_path = output_dir / "human_anchor_sample.csv"
    anchor_summary_path = output_dir / "human_anchor_validation.json"

    complete_df.to_csv(merged_path, index=False)
    full_raw_matrix.to_csv(raw_matrix_path)
    beta_hat.to_csv(beta_path)
    full_cbc_matrix.to_csv(cbc_matrix_path)
    full_xu_matrix.to_csv(xu_matrix_path)
    replicate_df.to_csv(replicate_path, index=False)
    anchor_sample_df, anchor_payload = run_human_anchor_validation(
        complete_df=complete_df,
        beta_hat=beta_hat,
        sample_per_language=args.anchor_sample_per_language,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed + 100,
    )
    anchor_sample_df.to_csv(anchor_sample_path, index=False)

    summary_payload = {
        **payload,
        "full_panel": {
            "languages": languages,
            "backbones": backbones,
            "n_complete_tasks": int(task_df["task"].nunique()),  # historical aligned-row field
            "n_aligned_items": int(task_df["task"].nunique()),
            "observed_margins": int(task_df["score"].notna().sum()),
            "missing_margins": int(task_df["score"].isna().sum()),
            "raw_tau_full_panel": mean_pairwise_kendall_tau(full_raw_matrix),
            "cbc_tau_full_panel": mean_pairwise_kendall_tau(full_cbc_matrix),
            "xu_tau_full_panel": mean_pairwise_kendall_tau(full_xu_matrix),
            "backbone_rankings_raw": rank_backbones(full_raw_matrix),
            "backbone_rankings_cbc": rank_backbones(full_cbc_matrix),
            "backbone_rankings_xu": rank_backbones(full_xu_matrix),
            "xu_judge_weights": xu_judge_weights,
        },
        "outputs": {
            "complete_panel_csv": portable_output_path(merged_path),
            "raw_mean_matrix_csv": portable_output_path(raw_matrix_path),
            "beta_hat_csv": portable_output_path(beta_path),
            "cbc_mean_matrix_csv": portable_output_path(cbc_matrix_path),
            "xu_matrix_csv": portable_output_path(xu_matrix_path),
            "replicates_csv": portable_output_path(replicate_path),
            "human_anchor_sample_csv": portable_output_path(anchor_sample_path),
            "human_anchor_validation_json": portable_output_path(anchor_summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    anchor_summary_path.write_text(json.dumps(anchor_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    print(json.dumps(anchor_payload, indent=2))


if __name__ == "__main__":
    main()
