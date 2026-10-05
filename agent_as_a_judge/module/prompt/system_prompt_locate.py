from agent_as_a_judge.module.prompt.english_paraphrase_variants import (
    select_english_variant,
)


def get_system_prompt_locate(language="English"):

    if language == "English":
        return select_english_variant(
            default_text="""
You are an advanced AI system specializing in understanding project structures and determining file locations based on provided criteria.
Your task is to locate specific files in the workspace based on the user's criteria and workspace information.
        """,
            en_p1_text="""
You are an advanced AI system specialized in understanding project structures and identifying file locations from the provided criteria.
Your task is to find the relevant files in the workspace using the user's criteria together with the workspace information.
        """,
            en_p2_text="""
You are an advanced AI system focused on interpreting project structure and inferring file locations from given requirements.
Your role is to determine which files in the workspace match the user's criteria based on the available workspace information.
        """,
        )
    if language == "Arabic":
        return """
أنت نظام ذكاء اصطناعي متخصص في فهم بنية المشاريع وتحديد مواقع الملفات بناء على المعايير المقدمة.
مهمتك هي العثور على الملفات المناسبة داخل هيكل المشروع بناء على معيار المستخدم ومعلومات الهيكل.
        """
    if language == "Turkish":
        return """
Proje yapısını anlayıp verilen ölçütlere göre dosya konumlarını belirleyen gelişmiş bir AI sistemisin.
Görevin, kullanıcı ölçütü ve çalışma alanı bilgisini kullanarak ilgili dosyaları bulmaktır.
        """
    if language == "Chinese":
        return """
你是一个擅长理解项目结构并根据条件定位文件的高级 AI 系统。
你的任务是根据用户条件和工作区结构信息，找出最相关的文件路径。
        """
    if language == "Hindi":
        return """
आप एक उन्नत AI सिस्टम हैं जो प्रोजेक्ट संरचना समझकर दिए गए मानदंड के आधार पर फ़ाइलें खोजता है।
आपका कार्य है: उपयोगकर्ता मानदंड और वर्कस्पेस जानकारी के आधार पर संबंधित फ़ाइल पथ पहचानना।
        """
    if language == "Japanese":
        return """
あなたはプロジェクト構造を理解し、与えられた基準に基づいてファイル位置を特定する高度な AI システムです。
あなたの役割は、ユーザーの基準とワークスペース情報に基づいて、最も関連性の高いファイルパスを特定することです。
        """
    if language == "Spanish":
        return """
Eres un sistema de IA avanzado especializado en comprender estructuras de proyectos y determinar ubicaciones de archivos según criterios dados.
Tu tarea es identificar las rutas de archivo más relevantes a partir del criterio del usuario y de la información del workspace.
        """
    if language == "Swahili":
        return """
Wewe ni mfumo wa hali ya juu wa AI unaobobea katika kuelewa miundo ya miradi na kubaini mahali faili zilipo kulingana na vigezo vilivyotolewa.
Kazi yako ni kutambua njia za faili zinazofaa zaidi kwa kutumia kigezo cha mtumiaji na taarifa za workspace.
        """

    raise NotImplementedError(f"The language '{language}' is not supported.")
