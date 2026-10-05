from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/LaBSE"
OUTPUT_JSON = REPO_ROOT / "analysis" / "system_prompt_semantic_similarity.json"
OUTPUT_MD = REPO_ROOT / "analysis" / "system_prompt_semantic_similarity.md"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


languages_module = _load_module(
    "analysis_languages",
    REPO_ROOT / "agent_as_a_judge" / "languages.py",
)
TRANSLATED_LANGUAGES = languages_module.TRANSLATED_LANGUAGES

_ask_module = _load_module(
    "analysis_system_prompt_ask",
    REPO_ROOT / "agent_as_a_judge" / "module" / "prompt" / "system_prompt_ask.py",
)
_judge_module = _load_module(
    "analysis_system_prompt_judge",
    REPO_ROOT / "agent_as_a_judge" / "module" / "prompt" / "system_prompt_judge.py",
)
_locate_module = _load_module(
    "analysis_system_prompt_locate",
    REPO_ROOT / "agent_as_a_judge" / "module" / "prompt" / "system_prompt_locate.py",
)
_planning_module = _load_module(
    "analysis_system_prompt_planning",
    REPO_ROOT / "agent_as_a_judge" / "module" / "prompt" / "system_prompt_planning.py",
)
_retrieve_module = _load_module(
    "analysis_system_prompt_retrieve",
    REPO_ROOT / "agent_as_a_judge" / "module" / "prompt" / "system_prompt_retrieve.py",
)

PROMPT_ROWS = [
    ("system_prompt_ask", _ask_module.get_ask_system_prompt),
    ("system_prompt_judge", _judge_module.get_judge_system_prompt),
    ("system_prompt_locate", _locate_module.get_system_prompt_locate),
    ("system_prompt_planning", _planning_module.get_planning_system_prompt),
    ("system_prompt_retrieve", _retrieve_module.get_retrieve_system_prompt),
]


def _cosine_similarity(model: SentenceTransformer, text_a: str, text_b: str) -> float:
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])


def _round4(value: float) -> float:
    return round(value, 4)


def _build_report() -> dict:
    model = SentenceTransformer(MODEL_NAME)
    prompt_reports: list[dict] = []
    all_scores: list[float] = []

    for prompt_name, getter in PROMPT_ROWS:
        english_text = getter("English")
        per_language_scores: dict[str, float] = {}
        for language in TRANSLATED_LANGUAGES:
            translated_text = getter(language)
            score = _cosine_similarity(model, english_text, translated_text)
            per_language_scores[language] = _round4(score)
            all_scores.append(score)

        prompt_reports.append(
            {
                "prompt_name": prompt_name,
                "scores": per_language_scores,
                "mean_score": _round4(sum(per_language_scores.values()) / len(per_language_scores)),
                "min_score": _round4(min(per_language_scores.values())),
                "max_score": _round4(max(per_language_scores.values())),
            }
        )

    language_scores = {
        language: _round4(
            sum(
                report["scores"][language]
                for report in prompt_reports
            )
            / len(prompt_reports)
        )
        for language in TRANSLATED_LANGUAGES
    }

    return {
        "model": MODEL_NAME,
        "prompt_count": len(PROMPT_ROWS),
        "language_count": len(TRANSLATED_LANGUAGES),
        "pair_count": len(all_scores),
        "overall_mean_score": _round4(sum(all_scores) / len(all_scores)),
        "overall_min_score": _round4(min(all_scores)),
        "overall_max_score": _round4(max(all_scores)),
        "scores_by_language": language_scores,
        "scores_by_prompt": prompt_reports,
    }


def _to_markdown(report: dict) -> str:
    lines = [
        "# System Prompt Semantic Similarity",
        "",
        f"- Model: `{report['model']}`",
        f"- Prompt-language pairs: {report['pair_count']} "
        f"({report['prompt_count']} prompts x {report['language_count']} translated languages)",
        f"- Overall mean cosine similarity: {report['overall_mean_score']:.4f}",
        f"- Overall min cosine similarity: {report['overall_min_score']:.4f}",
        f"- Overall max cosine similarity: {report['overall_max_score']:.4f}",
        "",
        "## Mean by language",
        "",
    ]

    for language, score in report["scores_by_language"].items():
        lines.append(f"- {language}: {score:.4f}")

    lines.extend(["", "## Scores by prompt", ""])
    for prompt_report in report["scores_by_prompt"]:
        lines.append(
            f"### {prompt_report['prompt_name']} "
            f"(mean {prompt_report['mean_score']:.4f}, "
            f"range {prompt_report['min_score']:.4f}-{prompt_report['max_score']:.4f})"
        )
        for language, score in prompt_report["scores"].items():
            lines.append(f"- {language}: {score:.4f}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    report = _build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(
        "Summary: "
        f"mean={report['overall_mean_score']:.4f}, "
        f"min={report['overall_min_score']:.4f}, "
        f"max={report['overall_max_score']:.4f}"
    )


if __name__ == "__main__":
    main()
