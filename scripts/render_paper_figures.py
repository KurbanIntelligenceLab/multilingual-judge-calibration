"""Legacy EMNLP plot entry point. For AAMAS use generate_aamas_figures.py.

These plots are retained for historical analysis and are not current paper figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "paper" / "analysis"
EXTERNAL_DIR = (
    ROOT
    / "data"
    / "external_validation"
    / "mrewardbench_panel"
    / "analysis_1500_item"
)

MAIN_BACKBONE_ORDER = ["GPT-4o", "GPT-5.4", "Sonnet", "Gemini", "DeepSeek", "Qwen"]
MAIN_LANGUAGE_ORDER = [
    "English",
    "Arabic",
    "Turkish",
    "Chinese",
    "Hindi",
    "Japanese",
    "Spanish",
    "Swahili",
]
EXTERNAL_BACKBONE_LABELS = {
    "openrouter/anthropic/claude-sonnet-4.6": "Sonnet",
    "deepseek/deepseek-v3.2": "DeepSeek",
    "openrouter/google/gemini-3-flash-preview": "Gemini",
    "openrouter/openai/gpt-4o-2024-08-06": "GPT-4o",
    "openrouter/openai/gpt-5.4": "GPT-5.4",
}
EXTERNAL_LANGUAGE_LABELS = {
    "eng_Latn": "English",
    "arb_Arab": "Arabic",
    "tur_Latn": "Turkish",
    "zho_Hans": "Chinese",
    "hin_Deva": "Hindi",
    "jpn_Jpan": "Japanese",
    "spa_Latn": "Spanish",
}


def rank_matrix(score_matrix: pd.DataFrame) -> pd.DataFrame:
    rank_rows = {}
    for language, row in score_matrix.iterrows():
        order = row.sort_values(ascending=False, kind="stable").index.tolist()
        rank_rows[language] = {backbone: rank + 1 for rank, backbone in enumerate(order)}
    return pd.DataFrame.from_dict(rank_rows, orient="index")[list(score_matrix.columns)]


def plot_beta_heatmap(beta_df: pd.DataFrame, output_path: Path) -> None:
    values = beta_df.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values)))
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(r"Estimated language-backbone bias $\hat{\beta}(\ell, b)$")
    ax.set_xticks(np.arange(beta_df.shape[1]))
    ax.set_xticklabels(beta_df.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(beta_df.shape[0]))
    ax.set_yticklabels(beta_df.index)
    for row_idx in range(beta_df.shape[0]):
        for col_idx in range(beta_df.shape[1]):
            ax.text(col_idx, row_idx, f"{beta_df.iat[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, shrink=0.95)
    cbar.set_label("Bias points")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_heatmaps(
    raw_matrix: pd.DataFrame,
    calibrated_matrix: pd.DataFrame,
    output_path: Path,
    language_labels: dict[str, str] | None = None,
    backbone_labels: dict[str, str] | None = None,
) -> None:
    raw_ranks = rank_matrix(raw_matrix)
    calibrated_ranks = rank_matrix(calibrated_matrix)
    n_backbones = len(raw_matrix.columns)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, matrix, title in [
        (axes[0], raw_ranks, "Raw rankings"),
        (axes[1], calibrated_ranks, "CBC rankings"),
    ]:
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu_r", vmin=1, vmax=n_backbones, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(
            [backbone_labels.get(col, col) if backbone_labels else col for col in matrix.columns],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([language_labels.get(idx, idx) if language_labels else idx for idx in matrix.index])
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, int(matrix.iat[row_idx, col_idx]), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=axes, shrink=0.95)
    cbar.set_label("Rank (1 = best)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bar(delta_df: pd.DataFrame, output_path: Path) -> None:
    fig_df = delta_df.copy()
    fig_df["pair"] = fig_df["backbone_i"] + " vs " + fig_df["backbone_j"]
    fig_df = fig_df.sort_values("delta", ascending=True)
    colors = ["#C44E52" if val > 0 else "#9E9E9E" for val in fig_df["delta"]]

    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    ax.barh(fig_df["pair"], fig_df["delta"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(r"Rank-reversal strength $\delta$")
    ax.set_ylabel("Backbone pair")
    ax.set_title("Observed rank-reversal strength")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(ablation_payload: dict, output_path: Path) -> None:
    summary_df = pd.DataFrame(ablation_payload["summary"])
    x = summary_df["n_tasks"].to_numpy(dtype=float)
    empirical = summary_df["mean_abs_beta_error"].to_numpy(dtype=float)
    empirical_std = summary_df["error_std"].to_numpy(dtype=float)
    theoretical = summary_df["theoretical_bound"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(x, empirical, marker="o", color="#4C72B0", label=r"Empirical $|\hat{\beta} - \beta_{\mathrm{oracle}}|$")
    ax.fill_between(x, empirical - empirical_std, empirical + empirical_std, color="#4C72B0", alpha=0.18)
    ax.plot(x, theoretical, linestyle="--", color="#C44E52", label=r"Theoretical $O(1/\sqrt{n})$ bound")
    ax.set_xlabel("Number of tasks n")
    ax.set_ylabel(r"Mean absolute bias error")
    ax.set_title("CBC convergence with more tasks")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_main_score_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_df = pd.read_csv(ANALYSIS_DIR / "task_level_scores.csv")
    raw_matrix = (
        task_df.groupby(["language", "backbone"], as_index=False)["score"]
        .mean()
        .pivot(index="language", columns="backbone", values="score")
        .reindex(index=MAIN_LANGUAGE_ORDER, columns=MAIN_BACKBONE_ORDER)
    )
    beta_df = pd.read_csv(ANALYSIS_DIR / "beta_hat.csv", index_col=0).reindex(
        index=MAIN_LANGUAGE_ORDER,
        columns=MAIN_BACKBONE_ORDER,
    )
    calibrated_matrix = raw_matrix - beta_df
    return raw_matrix, calibrated_matrix


def load_external_score_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_matrix = pd.read_csv(EXTERNAL_DIR / "raw_mean_matrix.csv", index_col=0)
    calibrated_matrix = pd.read_csv(EXTERNAL_DIR / "cbc_mean_matrix.csv", index_col=0)
    raw_matrix = raw_matrix.rename(index=EXTERNAL_LANGUAGE_LABELS).rename(columns=EXTERNAL_BACKBONE_LABELS)
    calibrated_matrix = calibrated_matrix.rename(index=EXTERNAL_LANGUAGE_LABELS).rename(columns=EXTERNAL_BACKBONE_LABELS)
    desired_cols = ["Sonnet", "DeepSeek", "Gemini", "GPT-4o", "GPT-5.4"]
    desired_rows = ["English", "Arabic", "Turkish", "Chinese", "Hindi", "Japanese", "Spanish"]
    return (
        raw_matrix.reindex(index=desired_rows, columns=desired_cols),
        calibrated_matrix.reindex(index=desired_rows, columns=desired_cols),
    )


def main() -> None:
    raw_main, cbc_main = load_main_score_matrices()
    beta_df = pd.read_csv(ANALYSIS_DIR / "beta_hat.csv", index_col=0).reindex(
        index=MAIN_LANGUAGE_ORDER,
        columns=MAIN_BACKBONE_ORDER,
    )
    plot_beta_heatmap(beta_df, ANALYSIS_DIR / "beta_heatmap.png")
    plot_rank_heatmaps(raw_main, cbc_main, ANALYSIS_DIR / "cbc_before_after.png")

    delta_df = pd.read_csv(ANALYSIS_DIR / "rank_reversal_delta.csv")
    plot_delta_bar(delta_df, ANALYSIS_DIR / "rank_reversal_strength.png")

    ablation_payload = json.loads((ANALYSIS_DIR / "ablation_tasks.json").read_text(encoding="utf-8"))
    plot_convergence(ablation_payload, ANALYSIS_DIR / "convergence_curve.png")

    raw_ext, cbc_ext = load_external_score_matrices()
    plot_rank_heatmaps(raw_ext, cbc_ext, ANALYSIS_DIR / "external_cbc_before_after.png")


if __name__ == "__main__":
    main()
