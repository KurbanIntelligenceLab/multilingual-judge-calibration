"""Regenerate AAMAS figures directly from the saved score panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / "paper" / "analysis"
FIGURES = REPO_ROOT / "paper" / "figures"
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
EVALUATORS = ["GPT-4o", "GPT-5.4", "Sonnet", "Gemini", "DeepSeek", "Qwen"]
DISPLAY_NAMES = {
    "GPT-4o": "GPT-4o",
    "GPT-5.4": "GPT-5.4",
    "Sonnet": "Sonnet 4.6",
    "Gemini": "Gemini 3 Flash",
    "DeepSeek": "DeepSeek-V3.2",
    "Qwen": "Qwen3.5-9B",
}


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(ANALYSIS / "task_level_scores.csv")
    beta = pd.read_csv(ANALYSIS / "beta_hat.csv").set_index("language")
    panel = panel[
        panel.language.isin(LANGUAGES) & panel.backbone.isin(EVALUATORS)
    ].copy()
    panel["corrected_score"] = panel.apply(
        lambda row: row.score - beta.loc[row.language, row.backbone], axis=1
    )
    return panel, beta.reindex(index=LANGUAGES, columns=EVALUATORS)


def figure_gate_by_language(panel: pd.DataFrame) -> None:
    gate = 25
    rates = (
        panel.assign(passed=panel.score > gate)
        .groupby(["backbone", "language"], as_index=False)
        .passed.mean()
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    fig.subplots_adjust(left=0.33, right=0.95, top=0.82, bottom=0.22)
    order = ["GPT-4o", "GPT-5.4", "Gemini", "Sonnet", "DeepSeek", "Qwen"]
    for row, evaluator in enumerate(order):
        subset = rates[rates.backbone == evaluator].set_index("language").reindex(LANGUAGES)
        values = 100 * subset.passed.to_numpy()
        assert np.isfinite(values).all() and len(values) == 8
        ax.plot([values.min(), values.max()], [row, row], color="#A8B1BC",
                lw=1.8, solid_capstyle="round", zorder=1)
        others = [1, 2, 3, 5, 7]
        ax.scatter(values[others], np.full(len(others), row), s=15,
                   color="#8998AA", zorder=2)
        ax.scatter([values[0]], [row], s=29, facecolor="white",
                   edgecolor="#233C58", lw=1.1, zorder=4)
        ax.scatter([values[4]], [row], s=30, marker="^", color="#277D55",
                   zorder=5)
        ax.scatter([values[6]], [row], s=31, marker="s", color="#B24C42",
                   zorder=5)
    assert abs(rates[(rates.backbone == "GPT-4o") &
                     (rates.language == "Hindi")].passed.iloc[0] * 100 - 90.9091) < 0.01
    ax.set_yticks(range(len(order)), [DISPLAY_NAMES[name] for name in order])
    ax.set_ylim(len(order) - 0.45, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Tasks passing gate 25 (%)", fontsize=7.7)
    ax.grid(axis="x", color="#E5E5E5", lw=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.tick_params(axis="y", length=0, labelsize=7.3)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.scatter([], [], s=29, facecolor="white", edgecolor="#233C58",
               label="English")
    ax.scatter([], [], s=30, marker="^", color="#277D55", label="Hindi")
    ax.scatter([], [], s=31, marker="s", color="#B24C42", label="Spanish")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              frameon=False, fontsize=6.6, handletextpad=0.2,
              columnspacing=0.65)
    fig.savefig(FIGURES / "fig1_gate_by_language.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def figure_interaction(beta: pd.DataFrame) -> None:
    short_languages = ["En", "Ar", "Tr", "Zh", "Hi", "Ja", "Es", "Sw"]
    values = beta.to_numpy().T
    bound = float(np.abs(values).max())
    fig, ax = plt.subplots(figsize=(3.35, 2.20))
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(LANGUAGES)), short_languages)
    ax.set_yticks(np.arange(len(EVALUATORS)), [DISPLAY_NAMES[name] for name in EVALUATORS])
    ax.tick_params(length=0, labelsize=7.0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if abs(value) > 0.55 * bound else "#222222",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.037, pad=0.02)
    bar.set_label("Score points", fontsize=7.0)
    bar.ax.tick_params(labelsize=6.5, length=2)
    fig.subplots_adjust(left=0.33, right=0.91, top=0.95, bottom=0.14)
    fig.savefig(FIGURES / "fig2_interaction.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def figure_audit_matrices(panel: pd.DataFrame, beta: pd.DataFrame) -> None:
    """Show every interaction and gate rate with direct, readable labels."""
    order = ["GPT-4o", "GPT-5.4", "Sonnet", "Gemini", "DeepSeek", "Qwen"]
    interaction = beta.reindex(index=LANGUAGES, columns=order).to_numpy().T
    gate = (
        panel.assign(passed=panel.score > 25)
        .groupby(["backbone", "language"]).passed.mean()
        .unstack("language")
        .reindex(index=order, columns=LANGUAGES)
        .to_numpy() * 100
    )
    assert np.isfinite(interaction).all() and np.isfinite(gate).all()
    assert abs(gate[0, 4] - 90.9090909) < 0.001
    assert abs(gate[0, 6] - 10.9090909) < 0.001

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(7.05, 3.55))
    fig.subplots_adjust(left=0.145, right=0.91, top=0.91, bottom=0.10,
                        hspace=0.42)
    bound = float(np.abs(interaction).max())
    for ax, values, cmap, vmin, vmax, title, suffix in [
        (axes[0], interaction, "RdBu_r", -bound, bound,
         "A. Language-by-evaluator interaction (score points)", ""),
        (axes[1], gate, "Blues", 0, 100,
         "B. Tasks passing gate 25 (%)", "%"),
    ]:
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect="auto")
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=5)
        ax.set_xticks(range(len(LANGUAGES)), LANGUAGES)
        ax.set_yticks(range(len(order)), order)
        ax.tick_params(length=0, labelsize=7.7)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                label = f"{value:+.1f}" if suffix == "" else f"{value:.0f}{suffix}"
                dark = (abs(value) > 0.52 * bound) if suffix == "" else (value > 52)
                ax.text(j, i, label, ha="center", va="center", fontsize=7.6,
                        color="white" if dark else "#17212B")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.015)
        cb.ax.tick_params(labelsize=7, length=2)
    fig.savefig(FIGURES / "fig_audit_matrices.pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def gate_spread(panel: pd.DataFrame, score_column: str, gate: float) -> float:
    rates = (
        panel.assign(passed=panel[score_column] > gate)
        .groupby(["backbone", "language"], as_index=False)
        .passed.mean()
    )
    per_evaluator = rates.groupby("backbone").passed.agg(lambda values: values.max() - values.min())
    return float(100 * per_evaluator.mean())


def figure_gate_sweep(panel: pd.DataFrame) -> None:
    gates = np.arange(10, 75, 5)
    raw = [gate_spread(panel, "score", gate) for gate in gates]
    corrected = [gate_spread(panel, "corrected_score", gate) for gate in gates]
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    ax.axvspan(10, 30, color="#E9E9E9", zorder=0)
    ax.plot(gates, raw, marker="o", markersize=3.2,
            label="Raw", color="#B24C42", zorder=2)
    ax.plot(
        gates,
        corrected,
        marker="o", markersize=3.2,
        label="Interaction removed",
        color="#2865A8",
        zorder=2,
    )
    ax.set_xlim(10, 70)
    ax.set_ylim(0, max(raw) + 4)
    ax.set_xlabel("Satisfaction gate", fontsize=7.7)
    ax.set_ylabel("Language spread (pp)", fontsize=7.7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.0)
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.03), fontsize=6.4,
              handlelength=1.2, columnspacing=0.7)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.19, top=0.80)
    fig.savefig(FIGURES / "fig3_gate_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel, beta = load_panel()
    figure_gate_by_language(panel)
    figure_interaction(beta)
    figure_gate_sweep(panel)
    print(f"wrote figures under {FIGURES}")


if __name__ == "__main__":
    main()
