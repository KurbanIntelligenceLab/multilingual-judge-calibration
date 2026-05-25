from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm.provider import LLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "external_validation" / "mrewardbench_panel" / "collection_logs"
DATASET_REPO = "CohereLabsCommunity/multilingual-reward-bench"

SYSTEM_PROMPT = (
    "You are a strict multilingual evaluation judge. "
    "Rate the quality of a single assistant response to a user prompt on a 1-5 integer scale. "
    "Consider helpfulness, correctness, relevance, completeness, clarity, and safety. "
    'Return valid JSON only in the format {"score": <1-5 integer>, "reason": "<brief reason>"}.'
)

USER_PROMPT_TEMPLATE = """Rate this response.

Language code: {language}
Subset: {subset}

User prompt:
{prompt}

Candidate response:
{response}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Evaluator model names passed to LiteLLM. "
            "If omitted, the script uses DEFAULT_LLM from the environment."
        ),
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional list of M-REWARDBENCH language configs, e.g. eng_Latn spa_Latn hin_Deva.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional maximum number of items per language to score.",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DATASET_REPO,
        help="Hugging Face dataset repo containing M-REWARDBENCH.",
    )
    parser.add_argument(
        "--output-stem",
        default="mrewardbench_panel_collection",
        help="Prefix used for the collection files written into the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where collection artifacts are written.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable storing the API key for LiteLLM models.",
    )
    parser.add_argument(
        "--base-url-env",
        default="OPENAI_BASE_URL",
        help="Environment variable storing the LiteLLM-compatible base URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call models; just emit prompt previews and planned collection rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files for this output stem.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output files for this output stem and skip completed rows.",
    )
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in {path}")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def available_languages(repo_id: str) -> list[str]:
    languages = []
    for file_path in list_repo_files(repo_id, repo_type="dataset"):
        if file_path.endswith("/raw.json"):
            language = file_path.split("/", 1)[0]
            if language != "translation":
                languages.append(language)
    return sorted(set(languages))


def build_item_id(row: dict) -> str:
    return f"{row['subset']}::{row['id']}"


def get_language_rows(repo_id: str, language: str, max_items: int | None) -> list[dict]:
    raw_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"{language}/raw.json")
    rows = load_rows(raw_path)
    if max_items is not None:
        rows = rows[:max_items]
    return rows


def build_messages(row: dict, response: str) -> list[dict]:
    prompt = USER_PROMPT_TEMPLATE.format(
        language=row["language"],
        subset=row["subset"],
        prompt=row["prompt"],
        response=response,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def parse_score_payload(text: str) -> tuple[int | None, str]:
    if not text:
        return None, ""
    text = text.strip()
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            score = int(payload.get("score"))
            if 1 <= score <= 5:
                return score, str(payload.get("reason", "")).strip()
        except Exception:
            pass
    result_match = re.search(r"\[RESULT\]\s*([1-5])", text)
    if result_match:
        return int(result_match.group(1)), text
    digit_match = re.search(r"\b([1-5])\b", text)
    if digit_match:
        return int(digit_match.group(1)), text
    return None, text


def build_llm(model: str, api_key_env: str, base_url_env: str) -> LLM:
    return LLM(
        model=model,
        api_key=os.getenv(api_key_env),
        base_url=os.getenv(base_url_env),
        llm_timeout=60,
        llm_temperature=0.0,
        llm_top_p=1.0,
    )


def ensure_outputs(output_dir: Path, output_stem: str, overwrite: bool, resume: bool) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{output_stem}_raw_scores.jsonl"
    margin_path = output_dir / f"{output_stem}_pair_margins.csv"
    meta_path = output_dir / f"{output_stem}_metadata.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and resume:
        raise ValueError("Use either --overwrite or --resume, not both.")
    if not overwrite and not resume:
        for path in (raw_path, margin_path, meta_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    return raw_path, margin_path, meta_path


def load_existing_raw_rows(raw_path: Path) -> dict[tuple[str, str, str, str], dict]:
    if not raw_path.exists():
        return {}
    rows = {}
    for row in load_rows(raw_path):
        key = (
            str(row["item_id"]),
            str(row["language"]),
            str(row["evaluator_model"]),
            str(row["response_role"]),
        )
        rows[key] = row
    return rows


def load_existing_margin_rows(margin_path: Path) -> dict[tuple[str, str, str], dict]:
    if not margin_path.exists():
        return {}
    rows = {}
    with margin_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (
                str(row["item_id"]),
                str(row["language"]),
                str(row["evaluator_model"]),
            )
            rows[key] = row
    return rows


def main() -> None:
    load_dotenv()
    args = parse_args()

    models = args.models or ([os.getenv("DEFAULT_LLM")] if os.getenv("DEFAULT_LLM") else [])
    if not models:
        raise RuntimeError("No evaluator model provided. Use --models or set DEFAULT_LLM.")

    languages = args.languages or available_languages(args.dataset_repo)
    raw_path, margin_path, meta_path = ensure_outputs(
        args.output_dir.resolve(),
        args.output_stem,
        args.overwrite,
        args.resume,
    )

    if args.dry_run:
        llms: dict[str, LLM] = {}
    else:
        llms = {
            model: build_llm(model, args.api_key_env, args.base_url_env)
            for model in models
        }

    pointwise_fieldnames = [
        "item_id",
        "item_numeric_id",
        "subset",
        "language",
        "evaluator_model",
        "response_role",
        "candidate_model",
        "score",
        "reason",
        "llm_response",
        "input_tokens",
        "output_tokens",
        "cost",
        "inference_time",
        "error",
    ]
    margin_fieldnames = [
        "item_id",
        "item_numeric_id",
        "subset",
        "language",
        "evaluator_model",
        "chosen_model",
        "rejected_model",
        "chosen_score",
        "rejected_score",
        "margin",
        "pref_correct",
    ]
    existing_pointwise_rows = load_existing_raw_rows(raw_path) if args.resume else {}
    existing_margin_rows = load_existing_margin_rows(margin_path) if args.resume else {}
    n_pointwise_rows_added = 0
    n_margin_rows_added = 0
    preview_rows: list[dict] = []

    raw_mode = "a" if args.resume else "w"
    margin_mode = "a" if args.resume else "w"
    with raw_path.open(raw_mode, encoding="utf-8") as raw_handle, margin_path.open(
        margin_mode, encoding="utf-8", newline=""
    ) as margin_handle:
        margin_writer = csv.DictWriter(margin_handle, fieldnames=margin_fieldnames)
        if not args.resume or margin_handle.tell() == 0:
            margin_writer.writeheader()

        for language in languages:
            rows = get_language_rows(args.dataset_repo, language, args.max_items)
            for row in rows:
                item_id = build_item_id(row)
                for model in models:
                    margin_key = (item_id, language, model)
                    if margin_key in existing_margin_rows:
                        continue
                    per_role: dict[str, dict] = {}
                    for response_role, response_text, candidate_model in [
                        ("chosen", row["chosen"], row["chosen_model"]),
                        ("rejected", row["rejected"], row["rejected_model"]),
                    ]:
                        pointwise_key = (item_id, language, model, response_role)
                        if pointwise_key in existing_pointwise_rows:
                            per_role[response_role] = existing_pointwise_rows[pointwise_key]
                            continue
                        messages = build_messages(row, response_text)
                        if args.dry_run:
                            result = {
                                "score": None,
                                "reason": "",
                                "llm_response": "",
                                "input_tokens": None,
                                "output_tokens": None,
                                "cost": None,
                                "inference_time": None,
                                "error": "",
                            }
                            preview_rows.append(
                                {
                                    "item_id": item_id,
                                    "language": language,
                                    "evaluator_model": model,
                                    "response_role": response_role,
                                    "messages": messages,
                                }
                            )
                        else:
                            try:
                                llm_result = llms[model]._llm_inference(messages)
                                score, reason = parse_score_payload(llm_result.get("llm_response", ""))
                                result = {
                                    "score": score,
                                    "reason": reason,
                                    "llm_response": llm_result.get("llm_response", ""),
                                    "input_tokens": llm_result.get("input_tokens"),
                                    "output_tokens": llm_result.get("output_tokens"),
                                    "cost": llm_result.get("cost"),
                                    "inference_time": llm_result.get("inference_time"),
                                    "error": "",
                                }
                            except Exception as exc:
                                result = {
                                    "score": None,
                                    "reason": "",
                                    "llm_response": "",
                                    "input_tokens": None,
                                    "output_tokens": None,
                                    "cost": None,
                                    "inference_time": None,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                print(
                                    f"[warn] scoring failed for model={model} language={language} "
                                    f"item_id={item_id} role={response_role}: {type(exc).__name__}: {exc}"
                                )
                                traceback.print_exc()

                        pointwise_row = {
                            "item_id": item_id,
                            "item_numeric_id": row["id"],
                            "subset": row["subset"],
                            "language": language,
                            "evaluator_model": model,
                            "response_role": response_role,
                            "candidate_model": candidate_model,
                            "score": result["score"],
                            "reason": result["reason"],
                            "llm_response": result["llm_response"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "cost": result["cost"],
                            "inference_time": result["inference_time"],
                            "error": result["error"],
                        }
                        raw_handle.write(json.dumps(pointwise_row, ensure_ascii=False) + "\n")
                        raw_handle.flush()
                        existing_pointwise_rows[pointwise_key] = pointwise_row
                        n_pointwise_rows_added += 1
                        per_role[response_role] = pointwise_row

                    if "chosen" not in per_role or "rejected" not in per_role:
                        continue
                    chosen_score = per_role["chosen"]["score"]
                    rejected_score = per_role["rejected"]["score"]
                    margin = (
                        None
                        if chosen_score is None or rejected_score is None
                        else float(chosen_score) - float(rejected_score)
                    )
                    margin_writer.writerow(
                        {
                            "item_id": item_id,
                            "item_numeric_id": row["id"],
                            "subset": row["subset"],
                            "language": language,
                            "evaluator_model": model,
                            "chosen_model": row["chosen_model"],
                            "rejected_model": row["rejected_model"],
                            "chosen_score": chosen_score,
                            "rejected_score": rejected_score,
                            "margin": margin,
                            "pref_correct": None if margin is None else margin > 0,
                        }
                    )
                    margin_handle.flush()
                    existing_margin_rows[margin_key] = {
                        "item_id": item_id,
                        "language": language,
                        "evaluator_model": model,
                    }
                    n_margin_rows_added += 1

    metadata = {
        "dataset_repo": args.dataset_repo,
        "languages": languages,
        "n_languages": len(languages),
        "models": models,
        "n_models": len(models),
        "max_items_per_language": args.max_items,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "n_pointwise_rows": len(existing_pointwise_rows),
        "n_margin_rows": len(existing_margin_rows),
        "n_pointwise_rows_added_this_run": n_pointwise_rows_added,
        "n_margin_rows_added_this_run": n_margin_rows_added,
        "preview_rows": preview_rows[:4],
        "cbc_ready_mapping": {
            "task": "item_id",
            "language": "language",
            "backbone": "evaluator_model",
            "score": "margin",
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Wrote {raw_path}")
    print(f"Wrote {margin_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
