from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import beta as beta_distribution, kendalltau, ttest_1samp

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from languages import (
    ALL_LANGUAGES,
    FRAMEWORKS as DEFAULT_FRAMEWORKS,
)
from requirement_taxonomy import (
    build_requirement_type_map,
)


INTERNAL_BENCHMARK_DIR = ROOT / "data" / "internal_benchmark" / "judgments"
OUTPUT_DIR = ROOT / "paper" / "analysis"

BACKBONES = {
    "gpt-4o": "GPT-4o",
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4.6": "Sonnet",
    "gemini-3-flash-preview": "Gemini",
    "deepseek-v3.2": "DeepSeek",
    "qwen3.5-9b": "Qwen",
}
BACKBONE_ORDER = list(BACKBONES.values())
LANGUAGES = list(ALL_LANGUAGES)
FRAMEWORKS = list(DEFAULT_FRAMEWORKS)
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 7
ABLATION_BOOTSTRAP_REPS = 100
TASK_SUBSAMPLE_REPS = 100
REQUIREMENT_BOOTSTRAP_REPS = 200
INFORMATION_KNN_K = 2
INFORMATION_PERMUTATION_REPS = 500
TASK_ABLATION_SIZES = [10, 20, 30, 40, 55]
TABLE2_METHODS = [
    "Raw (no calibration)",
    "Random control",
    "Per-language norm.",
    "Backbone-only norm.",
    "Z-score",
    "Quantile normalization",
    "ComBat-EB",
    "Dawid-Skene EM",
    "Ensemble",
    "CBC",
    "CBC-Weighted",
    "Oracle (eval-task beta)",
]


def task_key(name: str) -> tuple[int, str]:
    prefix = name.split("_", 1)[0]
    try:
        return (int(prefix), name)
    except ValueError:
        return (9999, name)


def compute_task_satisfaction_rate(obj: dict) -> float:
    stats = obj.get("judge_stats") or []
    if not stats:
        raise ValueError("Missing judge_stats")
    num_satisfied = sum(1 for stat in stats if bool(stat.get("satisfied", False)))
    return 100.0 * num_satisfied / len(stats)


def gray_box_dir(model_dir: str, language: str, framework: str) -> Path:
    return INTERNAL_BENCHMARK_DIR / model_dir / language / framework / "gray_box"


def load_run_level_scores() -> pd.DataFrame:
    rows: list[dict] = []
    for model_dir, backbone in BACKBONES.items():
        for language in LANGUAGES:
            for framework in FRAMEWORKS:
                gray_dir = gray_box_dir(model_dir, language, framework)
                if not gray_dir.exists():
                    continue
                for path in sorted(gray_dir.glob("*.json"), key=lambda p: task_key(p.stem)):
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    score = compute_task_satisfaction_rate(obj)
                    rows.append(
                        {
                            "backbone": backbone,
                            "language": language,
                            "framework": framework,
                            "task": obj.get("name", path.stem),
                            "score": score,
                            "task_solved": int(score == 100.0),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No benchmark rows were loaded.")
    return df


def aggregate_task_level(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["task", "language", "backbone"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "score"})
    )
    return grouped


def expand_rows_by_task_weights(
    df: pd.DataFrame,
    task_weights: dict[str, int],
) -> pd.DataFrame:
    weighted = df.copy()
    weighted["weight"] = weighted["task"].map(task_weights).fillna(0).astype(int)
    weighted = weighted[weighted["weight"] > 0].copy()
    if weighted.empty:
        raise ValueError("No rows remain after applying task weights.")
    return weighted.loc[weighted.index.repeat(weighted["weight"])].drop(columns=["weight"]).reset_index(drop=True)


def build_score_matrix(
    task_df: pd.DataFrame,
    task_weights: dict[str, int] | None = None,
    backbones: list[str] | None = None,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    backbone_order = BACKBONE_ORDER if backbones is None else list(backbones)
    language_order = LANGUAGES if languages is None else list(languages)
    matrix_df = task_df.copy()
    if task_weights is None:
        grouped = matrix_df.groupby(["language", "backbone"], as_index=False)["score"].mean()
    else:
        matrix_df["weight"] = matrix_df["task"].map(task_weights).fillna(0).astype(int)
        matrix_df = matrix_df[matrix_df["weight"] > 0].copy()
        if matrix_df.empty:
            raise ValueError("No rows remain after applying task weights.")
        matrix_df["weighted_score"] = matrix_df["score"] * matrix_df["weight"]
        grouped = (
            matrix_df.groupby(["language", "backbone"], as_index=False)[["weighted_score", "weight"]]
            .sum()
        )
        grouped["score"] = grouped["weighted_score"] / grouped["weight"]
        grouped = grouped[["language", "backbone", "score"]]

    return (
        grouped.pivot(index="language", columns="backbone", values="score")
        .reindex(index=language_order, columns=backbone_order)
    )


def compute_beta_from_matrix(score_means: pd.DataFrame) -> pd.DataFrame:
    row_means = score_means.mean(axis=1)
    col_means = score_means.mean(axis=0)
    grand_mean = float(score_means.values.mean())
    return score_means.sub(row_means, axis=0).sub(col_means, axis=1) + grand_mean


def compute_beta(
    task_df: pd.DataFrame,
    task_weights: dict[str, int] | None = None,
    backbones: list[str] | None = None,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    score_means = build_score_matrix(task_df, task_weights=task_weights, backbones=backbones, languages=languages)
    return compute_beta_from_matrix(score_means)


def compute_weighted_backbone_weights(score_means: pd.DataFrame) -> pd.Series:
    variances = score_means.var(axis=0, ddof=0)
    variances = variances.replace(0.0, 1e-8)
    weights = 1.0 / variances
    weights = weights / weights.sum()
    return weights.reindex(BACKBONE_ORDER)


def compute_weighted_beta(
    task_df: pd.DataFrame,
    task_weights: dict[str, int] | None = None,
    backbones: list[str] | None = None,
    languages: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    score_means = build_score_matrix(task_df, task_weights=task_weights, backbones=backbones, languages=languages)
    weights = compute_weighted_backbone_weights(score_means)
    language_weighted = score_means.mul(weights, axis=1).sum(axis=1)
    backbone_means = score_means.mean(axis=0)
    grand_weighted = float((backbone_means * weights).sum())
    beta = score_means.sub(language_weighted, axis=0).sub(backbone_means, axis=1) + grand_weighted
    return beta, weights


def compute_rank_reversal(task_df: pd.DataFrame) -> pd.DataFrame:
    score_means = build_score_matrix(task_df)
    rows: list[dict] = []
    for left, right in combinations(BACKBONE_ORDER, 2):
        diffs = {language: float(score_means.loc[language, left] - score_means.loc[language, right]) for language in LANGUAGES}
        best = None
        for lang_a, lang_b in combinations(LANGUAGES, 2):
            product = diffs[lang_a] * diffs[lang_b]
            if best is None or product < best["product"]:
                best = {
                    "backbone_i": left,
                    "backbone_j": right,
                    "language_a": lang_a,
                    "language_b": lang_b,
                    "d_a": diffs[lang_a],
                    "d_b": diffs[lang_b],
                    "product": product,
                    "delta": max(0.0, -product),
                }
        assert best is not None
        best["rank_reversal"] = bool(best["delta"] > 0.0)
        rows.append(best)
    delta_df = pd.DataFrame(rows)

    def one_sided_p(values: np.ndarray, positive: bool) -> float:
        result = ttest_1samp(values, 0.0, alternative="greater" if positive else "less")
        return float(result.pvalue)

    raw_p_values: list[float] = []
    for row in delta_df.to_dict("records"):
        if float(row["delta"]) <= 0.0:
            raw_p_values.append(1.0)
            continue

        left = str(row["backbone_i"])
        right = str(row["backbone_j"])
        language_a = str(row["language_a"])
        language_b = str(row["language_b"])

        lang_a_df = (
            task_df[(task_df["backbone"] == left) & (task_df["language"] == language_a)][["task", "score"]]
            .rename(columns={"score": "left_score"})
            .merge(
                task_df[(task_df["backbone"] == right) & (task_df["language"] == language_a)][["task", "score"]]
                .rename(columns={"score": "right_score"}),
                on="task",
            )
        )
        lang_b_df = (
            task_df[(task_df["backbone"] == left) & (task_df["language"] == language_b)][["task", "score"]]
            .rename(columns={"score": "left_score"})
            .merge(
                task_df[(task_df["backbone"] == right) & (task_df["language"] == language_b)][["task", "score"]]
                .rename(columns={"score": "right_score"}),
                on="task",
            )
        )
        diff_a = (lang_a_df["left_score"] - lang_a_df["right_score"]).to_numpy(dtype=float)
        diff_b = (lang_b_df["left_score"] - lang_b_df["right_score"]).to_numpy(dtype=float)

        p_a = one_sided_p(diff_a, positive=float(row["d_a"]) > 0.0)
        p_b = one_sided_p(diff_b, positive=float(row["d_b"]) > 0.0)
        raw_p_values.append(max(p_a, p_b))

    raw_p = np.asarray(raw_p_values, dtype=float)
    order = np.argsort(raw_p)
    adjusted = np.empty_like(raw_p)
    previous = 1.0
    m_tests = len(raw_p)
    for rank in range(m_tests - 1, -1, -1):
        idx = int(order[rank])
        adjusted[idx] = min(previous, raw_p[idx] * m_tests / float(rank + 1))
        previous = adjusted[idx]

    delta_df["p_value_raw"] = raw_p
    delta_df["p_value_bh"] = adjusted
    delta_df["bh_significant_fdr_005"] = delta_df["p_value_bh"] <= 0.05
    return delta_df


def dummy_block(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(df[column].astype(str), prefix=prefix, dtype=float)


def interaction_block(df: pd.DataFrame, left: str, right: str, prefix: str) -> pd.DataFrame:
    combo = df[left].astype(str) + "__" + df[right].astype(str)
    return pd.get_dummies(combo, prefix=prefix, dtype=float)


def fit_ols_r2(
    df: pd.DataFrame,
    include_framework: bool,
    include_backbone_language: bool,
    include_task_language: bool,
) -> dict:
    blocks = [pd.DataFrame({"intercept": np.ones(len(df), dtype=float)})]
    blocks.append(dummy_block(df, "task", "task"))
    blocks.append(dummy_block(df, "backbone", "backbone"))
    blocks.append(dummy_block(df, "language", "language"))
    if include_framework:
        blocks.append(dummy_block(df, "framework", "framework"))
    if include_backbone_language:
        blocks.append(interaction_block(df, "backbone", "language", "bl"))
    if include_task_language:
        blocks.append(interaction_block(df, "task", "language", "tl"))

    X = pd.concat(blocks, axis=1).to_numpy(dtype=float)
    y = df["score"].to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst
    return {
        "n_obs": int(len(y)),
        "n_params": int(X.shape[1]),
        "sse": sse,
        "sst": sst,
        "r2": r2,
    }


def save_heatmap(beta_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    vmax = float(np.abs(beta_df.to_numpy()).max())
    image = ax.imshow(beta_df.to_numpy(dtype=float), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(beta_df.columns)), labels=list(beta_df.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(beta_df.index)), labels=list(beta_df.index))
    for row_idx, language in enumerate(beta_df.index):
        for col_idx, backbone in enumerate(beta_df.columns):
            value = float(beta_df.loc[language, backbone])
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(r"$\hat{\beta}(\ell,b)$")
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Language")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "beta_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def bootstrap_task_split(
    tasks: list[str],
    rng: np.random.Generator,
) -> tuple[dict[str, int], list[str]]:
    sampled = rng.choice(tasks, size=len(tasks), replace=True)
    train_weights = {task: 0 for task in tasks}
    for task in sampled:
        train_weights[str(task)] += 1
    eval_tasks = [task for task in tasks if train_weights[task] == 0]
    return train_weights, eval_tasks


def mean_pairwise_kendall_tau(score_matrix: pd.DataFrame) -> float:
    taus: list[float] = []
    language_order = list(score_matrix.index)
    for left, right in combinations(language_order, 2):
        tau, _ = kendalltau(score_matrix.loc[left].to_numpy(), score_matrix.loc[right].to_numpy())
        if not np.isnan(tau):
            taus.append(float(tau))
    if not taus:
        return 0.0
    return float(np.mean(taus))


def mean_tau_to_heldout_language(
    score_matrix: pd.DataFrame,
    heldout_language: str,
) -> float:
    taus: list[float] = []
    heldout_scores = score_matrix.loc[heldout_language].to_numpy(dtype=float)
    for language in score_matrix.index:
        if language == heldout_language:
            continue
        tau, _ = kendalltau(heldout_scores, score_matrix.loc[language].to_numpy(dtype=float))
        if not np.isnan(tau):
            taus.append(float(tau))
    if not taus:
        return 0.0
    return float(np.mean(taus))


def exact_binomial_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    lower = 0.0 if successes == 0 else float(beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta_distribution.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def estimate_mixed_mi_ksg(x: np.ndarray, y: np.ndarray, k: int = INFORMATION_KNN_K) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    n = len(x)
    if n <= 1:
        return float("nan")

    label_counts = pd.Series(y).value_counts()
    if (label_counts <= k).any():
        return float("nan")

    m_vals: list[int] = []
    n_y: list[int] = []
    for i in range(n):
        same_label_idx = np.where(y == y[i])[0]
        same_label_idx = same_label_idx[same_label_idx != i]
        if len(same_label_idx) < k:
            return float("nan")
        dists_same = np.abs(x[same_label_idx] - x[i])
        eps = float(np.partition(dists_same, k - 1)[k - 1])
        m_i = int(np.sum(np.abs(x - x[i]) <= eps)) - 1
        m_vals.append(max(m_i, 1))
        n_y.append(int(label_counts[y[i]]))

    return float(digamma(n) - np.mean(digamma(np.asarray(n_y))) + digamma(k) - np.mean(digamma(np.asarray(m_vals))))


def row_zscore(matrix: pd.DataFrame) -> pd.DataFrame:
    row_std = matrix.std(axis=1, ddof=0).replace(0.0, 1.0)
    return matrix.sub(matrix.mean(axis=1), axis=0).div(row_std, axis=0)


def col_center(matrix: pd.DataFrame, train_matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.sub(train_matrix.mean(axis=0), axis=1)


def col_zscore(matrix: pd.DataFrame, train_matrix: pd.DataFrame) -> pd.DataFrame:
    col_std = train_matrix.std(axis=0, ddof=0).replace(0.0, 1.0)
    return matrix.sub(train_matrix.mean(axis=0), axis=1).div(col_std, axis=1)


def col_quantile(matrix: pd.DataFrame, train_matrix: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalize backbone columns to the train-panel target distribution.

    This follows the standard preprocessCore::normalize.quantiles logic: sort each
    train column, average across columns at each rank to obtain a shared target
    distribution, then replace each eval column by that target according to its
    within-column ordering. Ties receive the average target value across their
    occupied rank interval.
    """

    sorted_train = np.sort(train_matrix.to_numpy(dtype=float), axis=0)
    target = sorted_train.mean(axis=1)

    quantile_df = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)
    for backbone in matrix.columns:
        values = matrix[backbone].to_numpy(dtype=float)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        normalized_sorted = target.copy()

        start = 0
        while start < len(sorted_values):
            end = start + 1
            while end < len(sorted_values) and (
                sorted_values[end] == sorted_values[start]
            ):
                end += 1
            if end - start > 1:
                normalized_sorted[start:end] = float(np.mean(normalized_sorted[start:end]))
            start = end

        normalized_values = np.empty_like(values, dtype=float)
        normalized_values[order] = normalized_sorted
        quantile_df[backbone] = normalized_values
    return quantile_df


def combat_eb_adjust(
    eval_matrix: pd.DataFrame,
    train_task_df: pd.DataFrame,
) -> pd.DataFrame:
    train_df = train_task_df.copy()
    global_mean = float(train_df["score"].mean())
    backbone_effect = train_df.groupby("backbone")["score"].mean().reindex(eval_matrix.columns) - global_mean
    train_df["residual"] = train_df.apply(
        lambda row: row["score"] - global_mean - float(backbone_effect[row["backbone"]]),
        axis=1,
    )
    language_mean = train_df.groupby("language")["residual"].mean().reindex(eval_matrix.index).fillna(0.0)
    centered = train_df["residual"] - train_df["language"].map(language_mean)
    pooled_var = float(np.var(centered.to_numpy(dtype=float), ddof=0))
    pooled_std = float(np.sqrt(pooled_var)) if pooled_var > 0 else 1.0
    language_counts = train_df.groupby("language").size().reindex(eval_matrix.index).fillna(0.0)
    tau2 = float(np.var(language_mean.to_numpy(dtype=float), ddof=0))
    gamma_shrunk = {}
    delta_shrunk = {}
    lang_scale = {}
    for language in eval_matrix.index:
        n_l = float(language_counts[language])
        gamma_l = float(language_mean[language])
        if n_l <= 0:
            gamma_shrunk[language] = 0.0
            delta_shrunk[language] = 1.0
            continue
        lambda_mean = tau2 / (tau2 + pooled_var / n_l) if (tau2 + pooled_var / n_l) > 0 else 0.0
        gamma_shrunk[language] = lambda_mean * gamma_l
        lang_resid = train_df.loc[train_df["language"] == language, "residual"].to_numpy(dtype=float) - gamma_l
        if len(lang_resid) > 1:
            delta_l = float(np.std(lang_resid, ddof=0) / pooled_std) if pooled_std > 0 else 1.0
        else:
            delta_l = 1.0
        lang_scale[language] = delta_l
    scale_values = np.array(list(lang_scale.values()), dtype=float)
    scale_var = float(np.var(scale_values, ddof=0))
    for language in eval_matrix.index:
        n_l = float(language_counts[language])
        delta_l = float(lang_scale.get(language, 1.0))
        lambda_scale = scale_var / (scale_var + 1.0 / max(n_l, 1.0)) if (scale_var + 1.0 / max(n_l, 1.0)) > 0 else 0.0
        delta_shrunk[language] = max(1e-6, 1.0 + lambda_scale * (delta_l - 1.0))

    adjusted = eval_matrix.copy()
    for language in adjusted.index:
        for backbone in adjusted.columns:
            alpha_b = float(backbone_effect[backbone])
            gamma_l = float(gamma_shrunk[language])
            delta_l = float(delta_shrunk[language])
            value = float(adjusted.loc[language, backbone])
            adjusted.loc[language, backbone] = global_mean + alpha_b + (value - global_mean - alpha_b - gamma_l) / delta_l
    return adjusted


def fit_dawid_skene(
    train_run_df: pd.DataFrame,
    task_weights: dict[str, int],
    max_iter: int = 25,
    smoothing: float = 1.0,
) -> dict:
    train_df = expand_rows_by_task_weights(train_run_df, task_weights)
    train_df = train_df.copy()
    train_df["item_id"] = train_df["task"] + "::" + train_df["framework"]
    train_df["annotator_id"] = train_df["language"] + "::" + train_df["backbone"]
    pivot = (
        train_df.pivot_table(index="item_id", columns="annotator_id", values="task_solved", aggfunc="mean")
        .sort_index(axis=1)
    )
    annotations = pivot.to_numpy(dtype=float)
    posterior = np.clip(np.nanmean(annotations, axis=1), 1e-3, 1 - 1e-3)
    prior = float(np.mean(posterior))
    annotators = list(pivot.columns)
    confusion = {annotator: {"sens": 0.7, "spec": 0.7} for annotator in annotators}

    for _ in range(max_iter):
        prior = float(np.clip(np.mean(posterior), 1e-3, 1 - 1e-3))
        for j, annotator in enumerate(annotators):
            obs = annotations[:, j]
            tp = float(np.sum(posterior * (obs == 1.0)))
            fn = float(np.sum(posterior * (obs == 0.0)))
            fp = float(np.sum((1.0 - posterior) * (obs == 1.0)))
            tn = float(np.sum((1.0 - posterior) * (obs == 0.0)))
            sens = (tp + smoothing) / (tp + fn + 2 * smoothing)
            spec = (tn + smoothing) / (tn + fp + 2 * smoothing)
            confusion[annotator] = {"sens": float(np.clip(sens, 1e-4, 1 - 1e-4)), "spec": float(np.clip(spec, 1e-4, 1 - 1e-4))}

        new_posterior = np.zeros_like(posterior)
        for i in range(len(posterior)):
            log_p1 = np.log(prior)
            log_p0 = np.log(1.0 - prior)
            for j, annotator in enumerate(annotators):
                obs = annotations[i, j]
                sens = confusion[annotator]["sens"]
                spec = confusion[annotator]["spec"]
                if obs == 1.0:
                    log_p1 += np.log(sens)
                    log_p0 += np.log(1.0 - spec)
                else:
                    log_p1 += np.log(1.0 - sens)
                    log_p0 += np.log(spec)
            max_log = max(log_p1, log_p0)
            p1 = np.exp(log_p1 - max_log)
            p0 = np.exp(log_p0 - max_log)
            new_posterior[i] = p1 / (p1 + p0)
        if np.max(np.abs(new_posterior - posterior)) < 1e-6:
            posterior = new_posterior
            break
        posterior = new_posterior

    return {"prior": prior, "confusion": confusion}


def dawid_skene_eval_matrix(
    eval_run_df: pd.DataFrame,
    ds_params: dict,
    backbones: list[str],
) -> pd.DataFrame:
    eval_df = eval_run_df.copy()
    eval_df["item_id"] = eval_df["task"] + "::" + eval_df["framework"]
    eval_df["annotator_id"] = eval_df["language"] + "::" + eval_df["backbone"]
    pivot = eval_df.pivot_table(index="item_id", columns="annotator_id", values="task_solved", aggfunc="mean")
    annotators = list(pivot.columns)
    posterior = {}
    prior = float(ds_params["prior"])
    confusion = ds_params["confusion"]
    for item_id, row in pivot.iterrows():
        log_p1 = np.log(np.clip(prior, 1e-4, 1 - 1e-4))
        log_p0 = np.log(np.clip(1.0 - prior, 1e-4, 1 - 1e-4))
        for annotator in annotators:
            obs = float(row[annotator])
            params = confusion[annotator]
            sens = params["sens"]
            spec = params["spec"]
            if obs == 1.0:
                log_p1 += np.log(sens)
                log_p0 += np.log(1.0 - spec)
            else:
                log_p1 += np.log(1.0 - sens)
                log_p0 += np.log(spec)
        max_log = max(log_p1, log_p0)
        p1 = np.exp(log_p1 - max_log)
        p0 = np.exp(log_p0 - max_log)
        posterior[item_id] = p1 / (p1 + p0)

    rows = []
    for annotator in annotators:
        language, backbone = annotator.split("::")
        obs = pivot[annotator].to_numpy(dtype=float)
        item_ids = list(pivot.index)
        accuracy = float(
            np.mean([
                posterior[item_id] if obs_val == 1.0 else 1.0 - posterior[item_id]
                for item_id, obs_val in zip(item_ids, obs)
            ])
        )
        rows.append({"language": language, "backbone": backbone, "score": 100.0 * accuracy})
    matrix_df = pd.DataFrame(rows)
    return matrix_df.pivot(index="language", columns="backbone", values="score").reindex(index=LANGUAGES, columns=backbones)


def random_control(matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    shuffled = matrix.copy()
    for language in shuffled.index:
        shuffled.loc[language] = rng.permutation(shuffled.loc[language].to_numpy(dtype=float))
    return shuffled


def ensemble_baseline(matrix: pd.DataFrame, train_matrix: pd.DataFrame) -> pd.DataFrame:
    centered_rank = col_center(matrix, train_matrix).rank(axis=1, method="average", pct=True)
    z_rank = col_zscore(matrix, train_matrix).rank(axis=1, method="average", pct=True)
    q_rank = col_quantile(matrix, train_matrix).rank(axis=1, method="average", pct=True)
    return (centered_rank + z_rank + q_rank) / 3.0


def evaluate_method_matrix(
    method: str,
    full_task_df: pd.DataFrame,
    full_run_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_run_df: pd.DataFrame,
    eval_matrix: pd.DataFrame,
    train_matrix: pd.DataFrame,
    train_weights: dict[str, int],
    backbones: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if method == "Raw (no calibration)":
        return eval_matrix
    if method == "Random control":
        return random_control(eval_matrix, rng)
    if method == "Per-language norm.":
        return row_zscore(eval_matrix)
    if method == "Backbone-only norm.":
        return col_center(eval_matrix, train_matrix)
    if method == "Z-score":
        return col_zscore(eval_matrix, train_matrix)
    if method == "Quantile normalization":
        return col_quantile(eval_matrix, train_matrix)
    if method == "ComBat-EB":
        train_task_df = expand_rows_by_task_weights(full_task_df, train_weights)
        return combat_eb_adjust(eval_matrix, train_task_df)
    if method == "Dawid-Skene EM":
        ds_params = fit_dawid_skene(
            full_run_df[full_run_df["backbone"].isin(backbones)].copy(),
            task_weights=train_weights,
        )
        return dawid_skene_eval_matrix(eval_run_df[eval_run_df["backbone"].isin(backbones)].copy(), ds_params, backbones)
    if method == "Ensemble":
        return ensemble_baseline(eval_matrix, train_matrix)
    if method == "CBC":
        return eval_matrix - compute_beta(full_task_df, task_weights=train_weights, backbones=backbones)
    if method == "CBC-Weighted":
        beta_weighted, _ = compute_weighted_beta(
            full_task_df,
            task_weights=train_weights,
            backbones=backbones,
        )
        return eval_matrix - beta_weighted
    if method == "Oracle (eval-task beta)":
        return eval_matrix - compute_beta(eval_df, backbones=backbones)
    raise ValueError(f"Unknown method: {method}")


def summarize_replicates(replicate_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method, method_df in replicate_df.groupby("method"):
        taus = method_df["mean_pairwise_tau"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": method,
                "mean_pairwise_tau": float(np.mean(taus)),
                "ci_low": float(np.percentile(taus, 2.5)),
                "ci_high": float(np.percentile(taus, 97.5)),
                "n_replicates": int(len(taus)),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("mean_pairwise_tau", ascending=False)


def evaluate_calibration_methods(
    task_df: pd.DataFrame,
    run_df: pd.DataFrame,
    methods: list[str] | None = None,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    backbones: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    methods = TABLE2_METHODS if methods is None else list(methods)
    backbone_order = BACKBONE_ORDER if backbones is None else list(backbones)
    subset_df = task_df[task_df["backbone"].isin(backbone_order)].copy()
    subset_run_df = run_df[run_df["backbone"].isin(backbone_order)].copy()
    tasks = sorted(subset_df["task"].unique(), key=task_key)
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict] = []
    skipped = 0

    for replicate in range(bootstrap_reps):
        train_weights, eval_tasks = bootstrap_task_split(tasks, rng)
        if not eval_tasks:
            skipped += 1
            continue

        eval_df = subset_df[subset_df["task"].isin(eval_tasks)].copy()
        eval_run_df = subset_run_df[subset_run_df["task"].isin(eval_tasks)].copy()
        eval_matrix = build_score_matrix(eval_df, backbones=backbone_order)
        train_matrix = build_score_matrix(subset_df, task_weights=train_weights, backbones=backbone_order)

        for method in methods:
            method_matrix = evaluate_method_matrix(
                method=method,
                full_task_df=subset_df,
                full_run_df=subset_run_df,
                eval_df=eval_df,
                eval_run_df=eval_run_df,
                eval_matrix=eval_matrix,
                train_matrix=train_matrix,
                train_weights=train_weights,
                backbones=backbone_order,
                rng=rng,
            )
            replicate_rows.append(
                {
                    "replicate": replicate,
                    "method": method,
                    "mean_pairwise_tau": mean_pairwise_kendall_tau(method_matrix),
                    "n_eval_tasks": len(eval_tasks),
                    "n_backbones": len(backbone_order),
                }
            )

    replicate_df = pd.DataFrame(replicate_rows)
    summary_df = summarize_replicates(replicate_df)
    full_matrix = build_score_matrix(subset_df, backbones=backbone_order)
    full_task_weights = {str(task): 1 for task in tasks}
    full_panel_rows: list[dict] = []
    for method in methods:
        method_matrix = evaluate_method_matrix(
            method=method,
            full_task_df=subset_df,
            full_run_df=subset_run_df,
            eval_df=subset_df,
            eval_run_df=subset_run_df,
            eval_matrix=full_matrix,
            train_matrix=full_matrix,
            train_weights=full_task_weights,
            backbones=backbone_order,
            rng=np.random.default_rng(seed),
        )
        full_panel_rows.append(
            {
                "method": method,
                "mean_pairwise_tau_full_panel": round(
                    float(mean_pairwise_kendall_tau(method_matrix)),
                    6,
                ),
            }
        )
    payload = {
        "protocol": {
            "resampling": "bootstrap_train_oob_eval",
            "bootstrap_reps": bootstrap_reps,
            "seed": seed,
            "metric": "mean_pairwise_kendall_tau_across_languages",
            "skipped_replicates": skipped,
            "n_backbones": len(backbone_order),
        },
        "results": [
            {
                "method": row["method"],
                "mean_pairwise_tau": round(float(row["mean_pairwise_tau"]), 6),
                "ci_low": round(float(row["ci_low"]), 6),
                "ci_high": round(float(row["ci_high"]), 6),
                "n_replicates": int(row["n_replicates"]),
            }
            for row in summary_df.to_dict("records")
        ],
        "full_panel": full_panel_rows,
    }
    return replicate_df, payload


def run_backbone_ablation(task_df: pd.DataFrame, run_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    subset_rows: list[dict] = []
    for m_size in range(2, len(BACKBONE_ORDER) + 1):
        for subset in combinations(BACKBONE_ORDER, m_size):
            _, payload = evaluate_calibration_methods(
                task_df,
                run_df,
                methods=["CBC"],
                bootstrap_reps=ABLATION_BOOTSTRAP_REPS,
                seed=BOOTSTRAP_SEED + m_size,
                backbones=list(subset),
            )
            tau = payload["results"][0]["mean_pairwise_tau"]
            subset_rows.append(
                {
                    "m": m_size,
                    "backbones": ",".join(subset),
                    "mean_pairwise_tau": tau,
                }
            )
    subset_df = pd.DataFrame(subset_rows)
    summary_rows = []
    for m_size, group_df in subset_df.groupby("m"):
        taus = group_df["mean_pairwise_tau"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "m": int(m_size),
                "mean_pairwise_tau": float(np.mean(taus)),
                "std": float(np.std(taus, ddof=0)) if len(taus) > 1 else 0.0,
                "n_subsets": int(len(taus)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("m")
    payload = {
        "protocol": {
            "bootstrap_reps_per_subset": ABLATION_BOOTSTRAP_REPS,
            "metric": "mean_pairwise_kendall_tau_across_languages",
        },
        "summary": [
            {
                "m": int(row["m"]),
                "mean_pairwise_tau": round(float(row["mean_pairwise_tau"]), 6),
                "std": round(float(row["std"]), 6),
                "n_subsets": int(row["n_subsets"]),
            }
            for row in summary_df.to_dict("records")
        ],
    }
    return subset_df, payload


def run_task_ablation(task_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    full_beta = compute_beta(task_df)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 100)
    rows: list[dict] = []
    tasks = sorted(task_df["task"].unique(), key=task_key)
    for n_tasks in TASK_ABLATION_SIZES:
        for replicate in range(TASK_SUBSAMPLE_REPS):
            sampled_tasks = rng.choice(tasks, size=n_tasks, replace=False)
            subset_df = task_df[task_df["task"].isin(sampled_tasks)].copy()
            train_weights, eval_tasks = bootstrap_task_split(list(sampled_tasks), rng)
            if not eval_tasks:
                continue
            eval_df = subset_df[subset_df["task"].isin(eval_tasks)].copy()
            eval_matrix = build_score_matrix(eval_df)
            beta_hat = compute_beta(subset_df, task_weights=train_weights)
            calibrated = eval_matrix - beta_hat
            rows.append(
                {
                    "n_tasks": n_tasks,
                    "replicate": replicate,
                    "mean_pairwise_tau": mean_pairwise_kendall_tau(calibrated),
                    "mean_abs_beta_error": float((beta_hat - full_beta).abs().to_numpy().mean()),
                }
            )
    ablation_df = pd.DataFrame(rows)
    sigma_hat = float(np.sqrt(np.var(task_df["score"].to_numpy(dtype=float), ddof=0)))
    summary_rows = []
    for n_tasks, group_df in ablation_df.groupby("n_tasks"):
        summary_rows.append(
            {
                "n_tasks": int(n_tasks),
                "mean_pairwise_tau": float(group_df["mean_pairwise_tau"].mean()),
                "tau_std": float(group_df["mean_pairwise_tau"].std(ddof=0)),
                "mean_abs_beta_error": float(group_df["mean_abs_beta_error"].mean()),
                "error_std": float(group_df["mean_abs_beta_error"].std(ddof=0)),
                "theoretical_bound": float(
                    sigma_hat * np.sqrt(4.0 * np.log(60.0 / 0.05) / (3.0 * float(n_tasks)))
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("n_tasks")
    payload = {
        "protocol": {
            "task_subsample_reps": TASK_SUBSAMPLE_REPS,
            "metric": "mean_pairwise_kendall_tau_across_languages",
            "error_metric": "mean_abs_beta_error_vs_full_data_beta",
            "sigma_hat": round(sigma_hat, 6),
        },
        "summary": [
            {
                "n_tasks": int(row["n_tasks"]),
                "mean_pairwise_tau": round(float(row["mean_pairwise_tau"]), 6),
                "tau_std": round(float(row["tau_std"]), 6),
                "mean_abs_beta_error": round(float(row["mean_abs_beta_error"]), 6),
                "error_std": round(float(row["error_std"]), 6),
                "theoretical_bound": round(float(row["theoretical_bound"]), 6),
            }
            for row in summary_df.to_dict("records")
        ],
    }
    return ablation_df, payload


def run_leave_one_language_out(task_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tasks = sorted(task_df["task"].unique(), key=task_key)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 200)
    rows: list[dict] = []
    skipped = 0

    for heldout_language in LANGUAGES:
        train_languages = [language for language in LANGUAGES if language != heldout_language]
        subset_df = task_df[task_df["language"].isin(LANGUAGES)].copy()
        train_lang_df = subset_df[subset_df["language"].isin(train_languages)].copy()

        for replicate in range(BOOTSTRAP_REPS):
            train_weights, eval_tasks = bootstrap_task_split(tasks, rng)
            if not eval_tasks:
                skipped += 1
                continue

            eval_df = subset_df[subset_df["task"].isin(eval_tasks)].copy()
            eval_matrix = build_score_matrix(eval_df)

            train_beta = compute_beta(train_lang_df, task_weights=train_weights, languages=train_languages)

            raw_matrix = eval_matrix.copy()
            zero_shot_beta = pd.DataFrame(0.0, index=LANGUAGES, columns=BACKBONE_ORDER)
            zero_shot_beta.loc[train_languages, :] = train_beta.loc[train_languages, :]
            zero_shot_matrix = eval_matrix - zero_shot_beta

            imputed_row = train_beta.mean(axis=0)
            imputation_beta = zero_shot_beta.copy()
            imputation_beta.loc[heldout_language, :] = imputed_row
            imputation_matrix = eval_matrix - imputation_beta

            for method, matrix in [
                ("Raw", raw_matrix),
                ("LOLO zero-shot", zero_shot_matrix),
                ("LOLO imputation", imputation_matrix),
            ]:
                rows.append(
                    {
                        "heldout_language": heldout_language,
                        "replicate": replicate,
                        "method": method,
                        "mean_tau_to_training_languages": mean_tau_to_heldout_language(matrix, heldout_language),
                        "n_eval_tasks": len(eval_tasks),
                    }
                )

    replicate_df = pd.DataFrame(rows)
    summary_rows = []
    for (heldout_language, method), group_df in replicate_df.groupby(["heldout_language", "method"]):
        taus = group_df["mean_tau_to_training_languages"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "heldout_language": heldout_language,
                "method": method,
                "mean_tau_to_training_languages": float(np.mean(taus)),
                "ci_low": float(np.percentile(taus, 2.5)),
                "ci_high": float(np.percentile(taus, 97.5)),
                "n_replicates": int(len(taus)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["heldout_language", "method"])

    overall_rows = []
    for method, group_df in summary_df.groupby("method"):
        taus = group_df["mean_tau_to_training_languages"].to_numpy(dtype=float)
        overall_rows.append(
            {
                "method": method,
                "mean_tau_to_training_languages": float(np.mean(taus)),
                "min_language_tau": float(np.min(taus)),
                "max_language_tau": float(np.max(taus)),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("mean_tau_to_training_languages", ascending=False)

    payload = {
        "protocol": {
            "evaluation": "leave_one_language_out_task_bootstrap_oob",
            "bootstrap_reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED + 200,
            "metric": "mean_kendall_tau_between_heldout_language_and_each_training_language",
            "skipped_replicates": skipped,
            "n_languages": len(LANGUAGES),
            "n_backbones": len(BACKBONE_ORDER),
        },
        "note": (
            "Under CBC's column-wise sum-to-zero normalization, the mean of beta_hat over the retained "
            "training languages is exactly zero for each backbone. Therefore the requested mean-imputation "
            "variant is algebraically identical to the LOLO zero-shot variant."
        ),
        "by_language": [
            {
                "heldout_language": row["heldout_language"],
                "method": row["method"],
                "mean_tau_to_training_languages": round(float(row["mean_tau_to_training_languages"]), 6),
                "ci_low": round(float(row["ci_low"]), 6),
                "ci_high": round(float(row["ci_high"]), 6),
                "n_replicates": int(row["n_replicates"]),
            }
            for row in summary_df.to_dict("records")
        ],
        "overall": [
            {
                "method": row["method"],
                "mean_tau_to_training_languages": round(float(row["mean_tau_to_training_languages"]), 6),
                "min_language_tau": round(float(row["min_language_tau"]), 6),
                "max_language_tau": round(float(row["max_language_tau"]), 6),
            }
            for row in overall_df.to_dict("records")
        ],
    }
    return replicate_df, payload


def run_backbone_selection_decision_eval(task_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tasks = sorted(task_df["task"].unique(), key=task_key)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 300)
    rows: list[dict] = []
    skipped = 0

    for replicate in range(BOOTSTRAP_REPS):
        train_weights, eval_tasks = bootstrap_task_split(tasks, rng)
        if not eval_tasks:
            skipped += 1
            continue

        train_matrix = build_score_matrix(task_df, task_weights=train_weights)
        train_cbc = train_matrix - compute_beta(task_df, task_weights=train_weights)

        eval_df = task_df[task_df["task"].isin(eval_tasks)].copy()
        oracle_eval = build_score_matrix(eval_df) - compute_beta(eval_df)

        for language in LANGUAGES:
            oracle_scores = oracle_eval.loc[language]
            oracle_best_backbone = str(oracle_scores.idxmax())
            oracle_best_score = float(oracle_scores.max())
            choices = {
                "Raw (no calibration)": str(train_matrix.loc[language].idxmax()),
                "CBC": str(train_cbc.loc[language].idxmax()),
            }
            for method, selected_backbone in choices.items():
                selected_score = float(oracle_scores[selected_backbone])
                rows.append(
                    {
                        "replicate": replicate,
                        "language": language,
                        "method": method,
                        "selected_backbone": selected_backbone,
                        "oracle_best_backbone": oracle_best_backbone,
                        "selection_correct": float(selected_backbone == oracle_best_backbone),
                        "oracle_regret": oracle_best_score - selected_score,
                        "n_eval_tasks": len(eval_tasks),
                    }
                )

    replicate_df = pd.DataFrame(rows)
    per_replicate_df = (
        replicate_df.groupby(["replicate", "method"], as_index=False)
        .agg(
            selection_accuracy=("selection_correct", "mean"),
            mean_oracle_regret=("oracle_regret", "mean"),
        )
    )

    summary_rows = []
    for method, group_df in per_replicate_df.groupby("method"):
        accuracies = group_df["selection_accuracy"].to_numpy(dtype=float)
        regrets = group_df["mean_oracle_regret"].to_numpy(dtype=float)
        method_rows = replicate_df[replicate_df["method"] == method]
        successes = int(method_rows["selection_correct"].sum())
        trials = int(len(method_rows))
        acc_ci_low, acc_ci_high = exact_binomial_ci(successes, trials)
        summary_rows.append(
            {
                "method": method,
                "selection_accuracy": float(np.mean(accuracies)),
                "accuracy_ci_low": acc_ci_low,
                "accuracy_ci_high": acc_ci_high,
                "mean_oracle_regret": float(np.mean(regrets)),
                "regret_ci_low": float(np.percentile(regrets, 2.5)),
                "regret_ci_high": float(np.percentile(regrets, 97.5)),
                "n_replicates": int(len(accuracies)),
                "n_decisions": trials,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("selection_accuracy", ascending=False)

    by_language_rows = []
    for (language, method), group_df in replicate_df.groupby(["language", "method"]):
        by_language_rows.append(
            {
                "language": language,
                "method": method,
                "selection_accuracy": float(group_df["selection_correct"].mean()),
                "mean_oracle_regret": float(group_df["oracle_regret"].mean()),
                "most_common_selection": str(group_df["selected_backbone"].mode().iloc[0]),
                "most_common_oracle": str(group_df["oracle_best_backbone"].mode().iloc[0]),
            }
        )
    by_language_df = pd.DataFrame(by_language_rows).sort_values(["language", "method"])

    payload = {
        "protocol": {
            "evaluation": "per_language_backbone_selection_on_heldout_tasks",
            "bootstrap_reps": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED + 300,
            "selection_rule": "pick top-ranked backbone per language from training tasks",
            "target": "heldout_oracle_eval_task_beta_winner",
            "target_note": (
                "Held-out utility is defined by the backbone with highest held-out CBC-oracle score "
                "(eval matrix minus beta estimated on the held-out tasks), which converts rankings into "
                "a discrete deployment decision without requiring new human labels."
            ),
            "skipped_replicates": skipped,
            "n_languages": len(LANGUAGES),
            "n_backbones": len(BACKBONE_ORDER),
        },
        "overall": [
            {
                "method": row["method"],
                "selection_accuracy": round(float(row["selection_accuracy"]), 6),
                "accuracy_ci_low": round(float(row["accuracy_ci_low"]), 6),
                "accuracy_ci_high": round(float(row["accuracy_ci_high"]), 6),
                "mean_oracle_regret": round(float(row["mean_oracle_regret"]), 6),
                "regret_ci_low": round(float(row["regret_ci_low"]), 6),
                "regret_ci_high": round(float(row["regret_ci_high"]), 6),
                "n_replicates": int(row["n_replicates"]),
                "n_decisions": int(row["n_decisions"]),
            }
            for row in summary_df.to_dict("records")
        ],
        "by_language": [
            {
                "language": row["language"],
                "method": row["method"],
                "selection_accuracy": round(float(row["selection_accuracy"]), 6),
                "mean_oracle_regret": round(float(row["mean_oracle_regret"]), 6),
                "most_common_selection": row["most_common_selection"],
                "most_common_oracle": row["most_common_oracle"],
            }
            for row in by_language_df.to_dict("records")
        ],
    }
    return replicate_df, payload


def run_information_theoretic_measurement(
    run_df: pd.DataFrame,
    k: int = INFORMATION_KNN_K,
    permutation_reps: int = INFORMATION_PERMUTATION_REPS,
    seed: int = BOOTSTRAP_SEED + 400,
) -> dict:
    grouped = list(run_df.groupby(["task", "backbone"], sort=True))
    observed_rows: list[dict] = []
    for (task, backbone), group_df in grouped:
        estimate = estimate_mixed_mi_ksg(group_df["score"].to_numpy(dtype=float), group_df["language"].to_numpy(), k=k)
        observed_rows.append(
            {
                "task": task,
                "backbone": backbone,
                "n_rows": int(len(group_df)),
                "n_languages": int(group_df["language"].nunique()),
                "estimate_nats": estimate,
            }
        )

    observed_df = pd.DataFrame(observed_rows)
    observed_mean = float(observed_df["estimate_nats"].mean())

    rng = np.random.default_rng(seed)
    permutation_means: list[float] = []
    for _ in range(permutation_reps):
        permuted_vals = []
        for _, group_df in grouped:
            permuted_vals.append(
                estimate_mixed_mi_ksg(
                    group_df["score"].to_numpy(dtype=float),
                    rng.permutation(group_df["language"].to_numpy()),
                    k=k,
                )
            )
        permutation_means.append(float(np.mean(permuted_vals)))

    permutation_null = np.asarray(permutation_means, dtype=float)
    debiased = observed_mean - permutation_null
    one_sided_p = float((1 + np.sum(permutation_null >= observed_mean)) / (len(permutation_null) + 1))

    return {
        "protocol": {
            "target": "I(score; language | backbone, task)",
            "estimator": "KSG_style_mixed_continuous_discrete_knn",
            "conditioning": ["backbone", "task"],
            "slice_variable": "framework_level_scores_within_each_task_backbone_slice",
            "k": k,
            "permutation_reps": permutation_reps,
            "seed": seed,
            "n_slices": int(len(observed_df)),
            "slice_size": int(observed_df["n_rows"].iloc[0]) if not observed_df.empty else 0,
        },
        "observed_mean_nats": round(observed_mean, 6),
        "permutation_null_mean_nats": round(float(permutation_null.mean()), 6),
        "debiased_mean_nats": round(float(debiased.mean()), 6),
        "debiased_ci_low_nats": round(float(np.percentile(debiased, 2.5)), 6),
        "debiased_ci_high_nats": round(float(np.percentile(debiased, 97.5)), 6),
        "debiased_mean_bits": round(float(debiased.mean() / np.log(2.0)), 6),
        "debiased_ci_low_bits": round(float(np.percentile(debiased, 2.5) / np.log(2.0)), 6),
        "debiased_ci_high_bits": round(float(np.percentile(debiased, 97.5) / np.log(2.0)), 6),
        "one_sided_permutation_p": round(one_sided_p, 6),
        "by_slice_summary": {
            "mean_nats": round(float(observed_df["estimate_nats"].mean()), 6),
            "median_nats": round(float(observed_df["estimate_nats"].median()), 6),
            "min_nats": round(float(observed_df["estimate_nats"].min()), 6),
            "max_nats": round(float(observed_df["estimate_nats"].max()), 6),
        },
    }


def load_requirement_level_scores() -> pd.DataFrame:
    requirement_type, _, _ = build_requirement_type_map()
    rows: list[dict] = []
    for model_dir, backbone in BACKBONES.items():
        for language in LANGUAGES:
            for framework in FRAMEWORKS:
                gray_dir = gray_box_dir(model_dir, language, framework)
                if not gray_dir.exists():
                    continue
                for path in sorted(gray_dir.glob("*.json"), key=lambda p: task_key(p.stem)):
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    task = obj.get("name", path.stem)
                    for stat in obj.get("judge_stats", []):
                        req_id = int(stat["requirement_index"])
                        rows.append(
                            {
                                "backbone": backbone,
                                "language": language,
                                "framework": framework,
                                "task": task,
                                "requirement_type": requirement_type[(task, req_id)],
                                "score": 100.0 * float(bool(stat.get("satisfied", False))),
                            }
                        )
    return pd.DataFrame(rows)


def aggregate_requirement_task_level(
    req_df: pd.DataFrame,
    included_types: set[str],
) -> pd.DataFrame:
    filtered = req_df[req_df["requirement_type"].isin(included_types)].copy()
    return (
        filtered.groupby(["task", "language", "backbone"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "score"})
    )


def run_requirement_type_decomposition(run_df: pd.DataFrame) -> dict:
    req_df = load_requirement_level_scores()
    type_sets = {
        "operational": {"Data Loading", "Training"},
        "semantic": {"Model Construction", "Evaluation Metrics"},
    }
    payload = {"protocol": {"bootstrap_reps": REQUIREMENT_BOOTSTRAP_REPS}, "splits": {}}
    for split_name, included_types in type_sets.items():
        split_task_df = aggregate_requirement_task_level(req_df, included_types)
        _, split_payload = evaluate_calibration_methods(
            split_task_df,
            run_df,
            methods=["Raw (no calibration)", "CBC", "CBC-Weighted"],
            bootstrap_reps=REQUIREMENT_BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED + len(included_types),
        )
        payload["splits"][split_name] = {
            "included_types": sorted(included_types),
            "n_tasks": int(split_task_df["task"].nunique()),
            "results": split_payload["results"],
        }
    return payload


def classify_cbc_status(
    calibration_payload: dict,
    backbone_payload: dict,
    task_payload: dict,
    requirement_payload: dict,
) -> dict:
    result_map = {
        row["method"]: row["mean_pairwise_tau"]
        for row in calibration_payload["results"]
    }
    simple_baselines = [
        "Random control",
        "Per-language norm.",
        "Backbone-only norm.",
        "Z-score",
        "Quantile normalization",
        "ComBat-EB",
        "Dawid-Skene EM",
        "Ensemble",
    ]
    cbc_tau = float(result_map["CBC"])
    raw_tau = float(result_map["Raw (no calibration)"])
    baseline_best = max(float(result_map[method]) for method in simple_baselines)
    m_summary = {row["m"]: row["mean_pairwise_tau"] for row in backbone_payload["summary"]}
    n_summary = {row["n_tasks"]: row["mean_pairwise_tau"] for row in task_payload["summary"]}
    split_map = {
        split: {
            row["method"]: row["mean_pairwise_tau"]
            for row in split_payload["results"]
        }
        for split, split_payload in requirement_payload["splits"].items()
    }
    strong = (
        cbc_tau >= 0.8
        and cbc_tau - raw_tau >= 0.1
        and cbc_tau - baseline_best >= 0.05
        and m_summary.get(3, 0.0) >= 0.7
        and n_summary.get(20, 0.0) >= 0.7
    )
    partial = (
        cbc_tau > raw_tau
        and cbc_tau > baseline_best
    )
    if strong:
        verdict = "strong_success"
        rationale = "CBC clearly beats raw scores and simple baselines, and the gain persists under backbone and task ablations."
    elif partial:
        verdict = "partial_success"
        rationale = "CBC helps relative to raw scores, but the improvement is limited or brittle under at least one ablation."
    else:
        verdict = "failure"
        rationale = "CBC does not consistently outperform the strongest simple baselines or remain stable under ablation."
    return {
        "verdict": verdict,
        "rationale": rationale,
        "cbc_tau": round(cbc_tau, 6),
        "raw_tau": round(raw_tau, 6),
        "best_simple_baseline_tau": round(baseline_best, 6),
        "backbone_m3_tau": round(float(m_summary.get(3, float("nan"))), 6),
        "task_n20_tau": round(float(n_summary.get(20, float("nan"))), 6),
        "requirement_splits": split_map,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_df = load_run_level_scores()
    task_df = aggregate_task_level(run_df)
    beta_df = compute_beta(task_df)
    delta_df = compute_rank_reversal(task_df)
    calibration_replicates_df, calibration_payload = evaluate_calibration_methods(task_df, run_df)
    backbone_ablation_df, backbone_ablation_payload = run_backbone_ablation(task_df, run_df)
    task_ablation_df, task_ablation_payload = run_task_ablation(task_df)
    lolo_replicates_df, lolo_payload = run_leave_one_language_out(task_df)
    decision_replicates_df, decision_payload = run_backbone_selection_decision_eval(task_df)
    information_payload = run_information_theoretic_measurement(run_df)
    requirement_payload = run_requirement_type_decomposition(run_df)
    cbc_status_payload = classify_cbc_status(
        calibration_payload,
        backbone_ablation_payload,
        task_ablation_payload,
        requirement_payload,
    )

    simplified = fit_ols_r2(
        run_df,
        include_framework=True,
        include_backbone_language=True,
        include_task_language=False,
    )
    full = fit_ols_r2(
        run_df,
        include_framework=True,
        include_backbone_language=True,
        include_task_language=True,
    )

    run_df.to_csv(OUTPUT_DIR / "run_level_scores.csv", index=False)
    task_df.to_csv(OUTPUT_DIR / "task_level_scores.csv", index=False)
    beta_df.to_csv(OUTPUT_DIR / "beta_hat.csv")
    delta_df.to_csv(OUTPUT_DIR / "rank_reversal_delta.csv", index=False)
    calibration_replicates_df.to_csv(OUTPUT_DIR / "calibration_bootstrap_replicates.csv", index=False)
    backbone_ablation_df.to_csv(OUTPUT_DIR / "ablation_backbones.csv", index=False)
    task_ablation_df.to_csv(OUTPUT_DIR / "ablation_tasks.csv", index=False)
    lolo_replicates_df.to_csv(OUTPUT_DIR / "leave_one_language_out_replicates.csv", index=False)
    decision_replicates_df.to_csv(OUTPUT_DIR / "decision_backbone_selection_replicates.csv", index=False)
    save_heatmap(beta_df)

    payload = {
        "dataset": {
            "n_run_rows": int(len(run_df)),
            "n_task_rows": int(len(task_df)),
            "languages": LANGUAGES,
            "backbones": BACKBONE_ORDER,
            "frameworks": FRAMEWORKS,
        },
        "beta_hat": {
            language: {
                backbone: round(float(beta_df.loc[language, backbone]), 6)
                for backbone in BACKBONE_ORDER
            }
            for language in LANGUAGES
        },
        "rank_reversal_delta": [
            {
                "backbone_i": row["backbone_i"],
                "backbone_j": row["backbone_j"],
                "language_a": row["language_a"],
                "language_b": row["language_b"],
                "d_a": round(float(row["d_a"]), 6),
                "d_b": round(float(row["d_b"]), 6),
                "product": round(float(row["product"]), 6),
                "delta": round(float(row["delta"]), 6),
                "rank_reversal": bool(row["rank_reversal"]),
            }
            for row in delta_df.to_dict("records")
        ],
        "ols": {
            "simplified": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in simplified.items()},
            "full": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in full.items()},
        },
        "calibration": calibration_payload,
        "ablation_backbones": backbone_ablation_payload,
        "ablation_tasks": task_ablation_payload,
        "leave_one_language_out": lolo_payload,
        "decision_backbone_selection": decision_payload,
        "information_theoretic_measurement": information_payload,
        "requirement_type_decomposition": requirement_payload,
        "cbc_status": cbc_status_payload,
    }
    (OUTPUT_DIR / "meb_foundations.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "calibration_results.json").write_text(
        json.dumps(calibration_payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ablation_backbones.json").write_text(
        json.dumps(backbone_ablation_payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "ablation_tasks.json").write_text(
        json.dumps(task_ablation_payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "leave_one_language_out.json").write_text(
        json.dumps(lolo_payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "decision_backbone_selection.json").write_text(
        json.dumps(decision_payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "information_theoretic_measurement.json").write_text(
        json.dumps(information_payload, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "requirement_type_cbc.json").write_text(
        json.dumps(requirement_payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["ols"], indent=2))
    print(json.dumps(calibration_payload, indent=2))
    print(json.dumps(cbc_status_payload, indent=2))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
