import json
from collections import Counter, defaultdict
from pathlib import Path

from languages import (
    ALL_LANGUAGES,
    FRAMEWORKS as DEFAULT_FRAMEWORKS,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TESTS = ROOT / "data" / "internal_benchmark" / "judgments"
OUTPUT_PATH = ROOT / "paper" / "analysis" / "requirement_type_sensitivity.json"

BACKBONES = {
    "gpt-4o": "GPT-4o-2024-08-06",
    "gpt-5.4": "GPT-5.4",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
}
LANGUAGES = list(ALL_LANGUAGES)
FRAMEWORKS = list(DEFAULT_FRAMEWORKS)
TYPE_ORDER = [
    "Data Loading",
    "Preprocessing",
    "Model Construction",
    "Training",
    "Evaluation Metrics",
    "Visualization",
]


def source_task_path(task_name: str) -> Path:
    return (
        BENCHMARK_TESTS
        / "gpt-4o"
        / "English"
        / "MetaGPT"
        / "gray_box"
        / f"{task_name}.json"
    )


def classify_requirement(raw_category: str | None, criteria: str) -> str:
    category = (raw_category or "").lower()
    text = (criteria or "").lower()

    if "dataset or environment" in category:
        return "Data Loading"
    if "data preprocessing and postprocessing" in category:
        return "Preprocessing"
    if "machine learning method" in category:
        return "Model Construction"
    if "performance metrics" in category or "performence metrics" in category:
        return "Evaluation Metrics"
    if "visualization" in category or "human computer interaction" in category:
        return "Visualization"
    if "save trained model" in category:
        return "Training"

    if any(
        key in text
        for key in [
            "dataset",
            "loaded in",
            "is loaded",
            "is used in `src/data_loader.py`",
            "obtained in `src/data_loader.py`",
            "environment is defined",
            "environment is instantiated",
            "simulator is used",
            "prompts are read from a text file",
            "database is uploaded",
            "stored for future use",
            "retrieve all development steps",
            "development_planning table",
        ]
    ):
        return "Data Loading"

    if any(
        key in text
        for key in [
            "preprocessing",
            "post-processing",
            "clean up the text",
            "dataset is cleaned",
            "data augmentation",
            "tf-idf",
            "word embeddings",
            "resizing and normalization",
            "alignment and standardization",
            "split into training and testing",
            "normalization",
            "standardization",
        ]
    ):
        return "Preprocessing"

    if any(
        key in text
        for key in [
            "model is imported",
            "algorithm is used",
            "algorithm is implemented",
            "classifier is implemented",
            "pre-trained",
            "model is used in `src/model.py`",
            "model is used in `src/main.py`",
            "baseline are saved",
            "comparison with another model",
        ]
    ):
        return "Model Construction"

    if any(
        key in text
        for key in [
            "trained model is saved",
            "learned model is saved",
            "training progress",
            "generation length is limited",
            "generated text is saved",
            "prediction results are saved",
            "final position is printed",
            "recommendations for a test user",
            "generated descriptions",
            "cluster centers are saved",
            "models/saved_models/",
        ]
    ):
        return "Training"

    if any(
        key in text
        for key in [
            "performance report",
            "accuracy",
            "precision",
            "recall",
            "f1-score",
            "rmse",
            "mae",
            "mse",
            "r^2",
            "cross-validation",
            "evaluated",
            "evaluation metrics",
            "tracking performance",
            "node classification performance",
            "classification report",
            "prediction results",
            "feature importances",
        ]
    ):
        return "Evaluation Metrics"

    if any(
        key in text
        for key in [
            "learning curves",
            "return over time curve",
            "plotted",
            "visualized",
            "results/figures/",
            "saved as `results/figures/",
            "saved to the specified folder `results/figures/`",
            "stylized images are saved",
            "model architecture visualization",
            "prediction results are created and saved as pdf report",
            "results/report.pdf",
            "results/report.md",
            "markdown document containing the model architecture",
            "training process, and performance analysis is generated",
            "interactive elements",
            "interactive web application",
            "interactive web page",
            "interactive dashboard",
            "streamlit",
            "dash",
            "flask",
            "users can select",
            "view its development tasks",
            "submit button",
            "numeric input field",
            "mock llm",
            "frontend",
            "streamed",
            "real-time",
            "displayed token-by-token",
            "confusion_matrix.png",
            "word clouds",
            "spectrograms",
            "motion is visualized",
        ]
    ):
        return "Visualization"

    if "results/" in text or "models/" in text or "data/" in text:
        if "results/figures/" in text or "report" in text:
            return "Visualization"
        if "models/saved_models/" in text or "predictions.txt" in text or "generated_text" in text:
            return "Training"
        return "Evaluation Metrics"

    raise ValueError(f"Unmapped requirement category={raw_category!r} criteria={criteria!r}")


def build_requirement_type_map():
    task_root = (
        BENCHMARK_TESTS
        / "gpt-4o"
        / "English"
        / "MetaGPT"
        / "gray_box"
    )
    requirement_type = {}
    counts = Counter()
    samples = defaultdict(list)

    for path in sorted(task_root.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        task = obj["name"]
        for req in obj.get("requirements", []):
            req_id = int(req["requirement_id"])
            req_type = classify_requirement(req.get("category"), req.get("criteria", ""))
            requirement_type[(task, req_id)] = req_type
            counts[req_type] += 1
            if len(samples[req_type]) < 3:
                samples[req_type].append(req.get("criteria", ""))

    return requirement_type, counts, samples


def compute_metrics(requirement_type):
    sat_counts = defaultdict(lambda: [0, 0])

    for model_dir, backbone in BACKBONES.items():
        for language in LANGUAGES:
            for framework in FRAMEWORKS:
                gray_dir = (
                    BENCHMARK_TESTS
                    / model_dir
                    / language
                    / framework
                    / "gray_box"
                )
                for path in sorted(gray_dir.glob("*.json")):
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    task = obj["name"]
                    for stat in obj.get("judge_stats", []):
                        req_id = int(stat["requirement_index"])
                        req_type = requirement_type[(task, req_id)]
                        key = (req_type, language, backbone)
                        sat_counts[key][1] += 1
                        if bool(stat.get("satisfied", False)):
                            sat_counts[key][0] += 1

    result = defaultdict(dict)
    for req_type in TYPE_ORDER:
        for language in LANGUAGES:
            for backbone in BACKBONES.values():
                sat, total = sat_counts[(req_type, language, backbone)]
                result[req_type][(language, backbone)] = 100.0 * sat / total if total else 0.0
    return result


def compute_language_gaps(metrics):
    gap_by_type = {}
    for req_type in TYPE_ORDER:
        values = []
        for language in LANGUAGES:
            per_backbone = [metrics[req_type][(language, backbone)] for backbone in BACKBONES.values()]
            values.append(sum(per_backbone) / len(per_backbone))
        gap_by_type[req_type] = max(values) - min(values)
    return gap_by_type


def main():
    requirement_type, counts, samples = build_requirement_type_map()
    metrics = compute_metrics(requirement_type)
    gaps = compute_language_gaps(metrics)

    payload = {
        "counts": dict(counts),
        "metrics": {
            req_type: {
                language: {
                    backbone: round(metrics[req_type][(language, backbone)], 2)
                    for backbone in BACKBONES.values()
                }
                for language in LANGUAGES
            }
            for req_type in TYPE_ORDER
        },
        "language_gaps": {k: round(v, 2) for k, v in gaps.items()},
        "largest_gap_type": max(gaps, key=gaps.get),
        "smallest_gap_type": min(gaps, key=gaps.get),
        "samples": dict(samples),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
