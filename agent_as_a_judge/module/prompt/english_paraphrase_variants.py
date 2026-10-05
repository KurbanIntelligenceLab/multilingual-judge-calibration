from __future__ import annotations

import os


VALID_ENGLISH_PROMPT_VARIANTS = {"EN_P0", "EN_P1", "EN_P2"}


def get_active_english_prompt_variant() -> str:
    variant = os.getenv("AAAJ_ENGLISH_PROMPT_VARIANT", "EN_P0").strip() or "EN_P0"
    if variant not in VALID_ENGLISH_PROMPT_VARIANTS:
        raise ValueError(
            "Unsupported AAAJ_ENGLISH_PROMPT_VARIANT="
            f"{variant!r}. Expected one of {sorted(VALID_ENGLISH_PROMPT_VARIANTS)}."
        )
    return variant


def select_english_variant(
    *,
    default_text: str,
    en_p1_text: str,
    en_p2_text: str,
) -> str:
    variant = get_active_english_prompt_variant()
    if variant == "EN_P1":
        return en_p1_text
    if variant == "EN_P2":
        return en_p2_text
    return default_text
