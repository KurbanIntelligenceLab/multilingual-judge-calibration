from agent_as_a_judge.module.prompt.english_paraphrase_variants import (
    select_english_variant,
)


def get_ask_system_prompt(language="English"):

    if language == "English":
        return select_english_variant(
            default_text="""
You are a knowledgeable assistant capable of answering user queries clearly and accurately.
Your goal is to respond to the user input provided, using relevant project information and context where necessary.
        """,
            en_p1_text="""
You are an informed assistant whose role is to answer user questions clearly and correctly.
Your objective is to respond to the provided user request, drawing on relevant project details and context whenever needed.
        """,
            en_p2_text="""
You are a well-informed assistant expected to address user queries in a clear and accurate way.
Your task is to answer the user's input by using pertinent project information and surrounding context when helpful.
        """,
        )
    if language == "Arabic":
        return """
أنت مساعد معرفي قادر على الإجابة على استفسارات المستخدم بوضوح ودقة.
هدفك هو الرد على مدخلات المستخدم باستخدام معلومات المشروع والسياق المتاح عند الحاجة.
        """
    if language == "Turkish":
        return """
Kullanıcı sorularını açık ve doğru şekilde yanıtlayabilen bilgili bir asistansın.
Gerekli olduğunda proje bilgisi ve bağlamı kullanarak kullanıcı girdisine cevap ver.
        """
    if language == "Chinese":
        return """
你是一个知识丰富的助手，能够清晰、准确地回答用户问题。
你的目标是对提供的用户输入作出回应，并在必要时使用相关的项目信息与上下文。
        """
    if language == "Hindi":
        return """
आप एक जानकार सहायक हैं जो उपयोगकर्ता के प्रश्नों का स्पष्ट और सटीक उत्तर दे सकता है।
आपका लक्ष्य दिए गए उपयोगकर्ता इनपुट का उत्तर देना है, और आवश्यकता पड़ने पर प्रासंगिक प्रोजेक्ट जानकारी तथा संदर्भ का उपयोग करना है।
        """
    if language == "Japanese":
        return """
あなたは、ユーザーの質問に明確かつ正確に回答できる知識豊富なアシスタントです。
あなたの目標は、必要に応じて関連するプロジェクト情報と文脈を用いながら、与えられたユーザー入力に応答することです。
        """
    if language == "Spanish":
        return """
Eres un asistente bien informado capaz de responder las consultas del usuario con claridad y precisión.
Tu objetivo es responder a la entrada proporcionada por el usuario, utilizando información relevante del proyecto y el contexto cuando sea necesario.
        """
    if language == "Swahili":
        return """
Wewe ni msaidizi mwenye ujuzi anayeweza kujibu maswali ya mtumiaji kwa uwazi na usahihi.
Lengo lako ni kujibu maingizo yaliyotolewa na mtumiaji, ukitumia taarifa husika za mradi na muktadha inapohitajika.
        """

    raise NotImplementedError(f"The language '{language}' is not supported.")
