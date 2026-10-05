ALL_LANGUAGES = (
    "English",
    "Arabic",
    "Turkish",
    "Chinese",
    "Hindi",
    "Japanese",
    "Spanish",
    "Swahili",
)

TRANSLATED_LANGUAGES = tuple(language for language in ALL_LANGUAGES if language != "English")

NEW_LANGUAGES = (
    "Japanese",
    "Spanish",
    "Swahili",
)

FRAMEWORKS = (
    "MetaGPT",
    "GPT-Pilot",
    "OpenHands",
)
