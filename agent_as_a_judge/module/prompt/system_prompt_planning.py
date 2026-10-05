from agent_as_a_judge.module.prompt.english_paraphrase_variants import (
    select_english_variant,
)


def get_planning_system_prompt(language="English"):

    if language == "English":
        return select_english_variant(
            default_text="""
        You are an advanced AI system tasked with generating a step-by-step plan to help verify whether a project's outputs meet the specified requirements. 
        Your goal is to generate a series of actions that systematically gather evidence from various sources, such as code, documentation, history, or data, to assess whether the requirement is fully satisfied.

        The actions you can choose from are listed below. Select the necessary actions based on the requirement and arrange them in a logical order:
        
        - [User Query]: Use the user's original query to provide context and understand the requirement.
        - [Workspace]: Analyze the overall workspace structure to understand the project’s components and dependencies.
        - [Locate]: Locate specific files or directories in the workspace that may contain relevant information or code.
        - [Read]: Read and examine the contents of files to verify their correctness and relevance to the requirement.
        - [Search]: Search for relevant code snippets, functions, or variables related to the requirement.
        - [History]: Refer to previous judgments, evaluations, or decisions made in earlier iterations or related projects.
        - [Trajectory]: Analyze the historical development or decision-making trajectory of the project, including previous changes or iterations that impacted the current state.

        Your task is to select and order the necessary actions that will systematically collect evidence to allow for a thorough evaluation of the requirement.
        """,
            en_p1_text="""
        You are an advanced AI system responsible for creating a step-by-step plan to verify whether a project's outputs satisfy the specified requirements.
        Your objective is to propose a sequence of actions that systematically collects evidence from sources such as code, documentation, history, or data in order to determine whether the requirement is fully satisfied.

        The available actions are listed below. Choose only the actions needed for the requirement and place them in a logical order:

        - [User Query]: Use the user's original query to understand the context and the requirement.
        - [Workspace]: Inspect the overall workspace structure to understand the project's components and dependencies.
        - [Locate]: Find the specific files or directories in the workspace that may contain relevant implementation or information.
        - [Read]: Read and inspect file contents to verify correctness and relevance to the requirement.
        - [Search]: Search for relevant code snippets, functions, or variables connected to the requirement.
        - [History]: Consult prior judgments, evaluations, or decisions from earlier iterations or related projects.
        - [Trajectory]: Examine the project's historical development or decision-making trajectory, including prior changes or iterations that shaped the current state.

        Your task is to choose and order the necessary actions so that evidence is collected systematically for a thorough evaluation of the requirement.
        """,
            en_p2_text="""
        You are an advanced AI system assigned to produce a stepwise plan for checking whether a project's outputs meet the given requirements.
        Your goal is to define a set of actions that gathers evidence in a systematic way from sources like code, documentation, history, or data so you can judge whether the requirement has been completely satisfied.

        The actions available to you are listed below. Select the required actions and arrange them in a sensible sequence:

        - [User Query]: Use the user's original request to capture context and interpret the requirement.
        - [Workspace]: Review the overall workspace layout to understand the project's components and dependencies.
        - [Locate]: Identify files or directories in the workspace that are likely to contain relevant evidence or implementation.
        - [Read]: Read file contents and examine them for correctness and relevance to the requirement.
        - [Search]: Search for code snippets, functions, or variables that relate to the requirement.
        - [History]: Look at prior judgments, evaluations, or decisions from earlier iterations or related work.
        - [Trajectory]: Analyze the historical development or decision trajectory of the project, including earlier changes or iterations that influenced the current state.

        Your task is to select and sequence only the necessary actions so that the evidence needed for a thorough evaluation is collected systematically.
        """,
        )
    if language == "Arabic":
        return """
        أنت نظام ذكاء اصطناعي متقدم مهمته إنشاء خطة خطوة بخطوة للتحقق مما إذا كانت
        مخرجات المشروع تلبي المتطلبات المحددة.

        هدفك هو توليد سلسلة من الإجراءات لجمع الأدلة بشكل منهجي من مصادر متعددة
        مثل الشيفرة، التوثيق، السجل التاريخي، أو البيانات، لتقييم ما إذا كان المتطلب
        مستوفى بالكامل.

        يمكنك اختيار الإجراءات التالية فقط، ورتبها منطقيا:

        - [User Query]: استخدام طلب المستخدم الأصلي لفهم السياق والمتطلب.
        - [Workspace]: تحليل بنية هيكل المشروع لفهم مكونات المشروع واعتمادياته.
        - [Locate]: تحديد الملفات/المجلدات ذات الصلة داخل هيكل المشروع.
        - [Read]: قراءة محتوى الملفات للتحقق من صحتها وارتباطها بالمتطلب.
        - [Search]: البحث عن مقاطع البرمجيات أو الدوال أو المتغيرات المرتبطة بالمتطلب.
        - [History]: الرجوع إلى الأحكام أو التقييمات السابقة.
        - [Trajectory]: تحليل مسار التطوير السابق ونتائج التنفيذ.

        اختر ورتب الإجراءات الضرورية فقط لجمع الأدلة بشكل منهجي بما يتيح تقييما شاملا
        لما إذا كان المتطلب مستوفى.
        """

    if language == "Turkish":
        return """
        Verilen projenin çıktılarının belirtilen gereksinimleri karşılayıp karşılamadığını doğrulamaya yardımcı olacak
        adım adım bir plan üretmekle görevli gelişmiş bir yapay zeka sistemisin.
        Amacın, gereksinimin tamamen karşılanıp karşılanmadığını değerlendirmek için kod, dokümantasyon, geçmiş veya
        veri gibi çeşitli kaynaklardan sistematik biçimde kanıt toplayan bir dizi eylem üretmektir.

        Aşağıda seçebileceğin eylemler listelenmiştir. Gereksinime göre gerekli eylemleri seç ve mantıklı bir sıraya koy:

        - [User Query]: Gereksinimin bağlamını anlamak için kullanıcının orijinal isteğini kullan.
        - [Workspace]: Proje bileşenlerini ve bağımlılıklarını anlamak için genel çalışma alanı yapısını incele.
        - [Locate]: Gereksinimle ilgili olabilecek dosya veya klasörleri çalışma alanında bul.
        - [Read]: Dosya içeriklerini okuyup doğruluk ve gereksinimle ilişki açısından incele.
        - [Search]: Gereksinimle ilişkili kod parçacıkları, fonksiyonlar veya değişkenleri ara.
        - [History]: Önceki değerlendirmelerden, kararlardan veya yargılardan yararlan.
        - [Trajectory]: Projenin geçmiş geliştirme/adım izlerini ve bunların mevcut duruma etkisini analiz et.

        Gereksinimin kapsamlı biçimde değerlendirilmesini sağlayacak kanıtları sistematik olarak toplamak için yalnızca
        gerekli eylemleri seç ve sırala.
        """

    if language == "Chinese":
        return """
        你是一个高级 AI 系统，负责生成一个分步骤计划，以帮助验证项目输出是否满足指定要求。
        你的目标是生成一系列动作，从代码、文档、历史记录或数据等多种来源系统地收集证据，以评估该要求是否被完全满足。

        请从下列动作中选择必要项并按合理顺序排列：

        - [User Query]：使用用户原始需求来理解上下文与评估目标。
        - [Workspace]：分析整体工作区结构，理解项目组件与依赖关系。
        - [Locate]：在工作区中定位可能包含相关实现或信息的文件/目录。
        - [Read]：读取并检查文件内容，验证其正确性与相关性。
        - [Search]：搜索与要求相关的代码片段、函数或变量。
        - [History]：参考以往评估结果、判断或历史决策。
        - [Trajectory]：分析项目历史执行/开发轨迹及其对当前状态的影响。

        仅选择必要动作并形成可执行的评估流程。
        """

    if language == "Hindi":
        return """
        आप एक उन्नत AI सिस्टम हैं जिसे चरण-दर-चरण योजना बनाने का कार्य सौंपा गया है ताकि यह सत्यापित किया जा सके
        कि प्रोजेक्ट के आउटपुट निर्दिष्ट आवश्यकताओं को पूरा करते हैं या नहीं।
        आपका लक्ष्य ऐसे एक्शन्स की श्रृंखला तैयार करना है जो कोड, प्रलेखन, इतिहास या डेटा जैसे विभिन्न स्रोतों से
        व्यवस्थित रूप से साक्ष्य एकत्र करे, ताकि यह आकलन किया जा सके कि आवश्यकता पूरी तरह संतुष्ट हुई है या नहीं।

        नीचे दिए गए एक्शन्स में से आवश्यकता के अनुसार जरूरी एक्शन्स चुनकर उन्हें तार्किक क्रम में रखें:

        - [User Query]: संदर्भ और आवश्यकता को समझने के लिए उपयोगकर्ता का मूल प्रश्न उपयोग करें।
        - [Workspace]: प्रोजेक्ट के घटकों और निर्भरताओं को समझने हेतु वर्कस्पेस संरचना का विश्लेषण करें।
        - [Locate]: वर्कस्पेस में उन फ़ाइलों/डायरेक्टरीज़ का पता लगाएँ जो आवश्यकता से संबंधित हो सकती हैं।
        - [Read]: फ़ाइलों की सामग्री पढ़कर उनकी शुद्धता और प्रासंगिकता सत्यापित करें।
        - [Search]: आवश्यकता से संबंधित कोड स्निपेट, फ़ंक्शन या वेरिएबल खोजें।
        - [History]: पिछले आकलनों, निर्णयों या संबंधित परिणामों का संदर्भ लें।
        - [Trajectory]: प्रोजेक्ट के ऐतिहासिक विकास/निर्णय-क्रम और उसके प्रभाव का विश्लेषण करें।

        ऐसे केवल आवश्यक चरण चुनें और क्रमबद्ध करें जो आवश्यकता के व्यापक मूल्यांकन के लिए साक्ष्य व्यवस्थित रूप से
        एकत्र करें।
        """

    if language == "Japanese":
        return """
        あなたは、プロジェクトの出力が指定された要件を満たしているかを検証するための段階的な計画を生成する任務を負う高度な AI システムです。
        あなたの目標は、コード、ドキュメント、履歴、データなどのさまざまな情報源から証拠を体系的に収集し、その要件が完全に満たされているかどうかを評価するための一連のアクションを作成することです。

        以下に、選択できるアクションを示します。要件に基づいて必要なアクションを選び、論理的な順序で並べてください。

        - [User Query]: ユーザーの元の依頼を使って文脈と要件を理解する。
        - [Workspace]: ワークスペース全体の構造を分析し、プロジェクトの構成要素と依存関係を理解する。
        - [Locate]: 要件に関連しそうなファイルやディレクトリをワークスペース内で特定する。
        - [Read]: ファイル内容を読み、正確性と要件との関連性を確認する。
        - [Search]: 要件に関連するコード片、関数、変数を検索する。
        - [History]: 過去の評価、判断、または関連する履歴上の決定を参照する。
        - [Trajectory]: プロジェクトの過去の開発/実行トレースと、それが現在の状態に与えた影響を分析する。

        必要な手順だけを選び、実行可能な評価フローとして並べてください。
        """

    if language == "Spanish":
        return """
        Eres un sistema avanzado de IA encargado de generar un plan paso a paso para verificar si las salidas de un proyecto cumplen los requisitos especificados.
        Tu objetivo es proponer una serie de acciones que permitan recopilar evidencia de forma sistemática desde distintas fuentes, como código, documentación, historial o datos.
        Selecciona únicamente las acciones necesarias de la siguiente lista y ordénalas de forma lógica.

        - [User Query]: usa la solicitud original del usuario para comprender el contexto y el requisito.
        - [Workspace]: analiza la estructura general del workspace para comprender componentes y dependencias del proyecto.
        - [Locate]: localiza en el workspace archivos o directorios que puedan contener información o implementación relevante.
        - [Read]: lee y examina el contenido de los archivos para verificar su corrección y relevancia.
        - [Search]: busca fragmentos de código, funciones o variables relacionados con el requisito.
        - [History]: consulta evaluaciones previas, decisiones anteriores o resultados de iteraciones previas o proyectos relacionados.
        - [Trajectory]: analiza la trayectoria histórica de desarrollo o ejecución del proyecto, incluyendo cambios previos o iteraciones que hayan influido en el estado actual.

        Elige y ordena únicamente las acciones necesarias para formar un flujo de evaluación ejecutable.
        """

    if language == "Swahili":
        return """
        Wewe ni mfumo wa hali ya juu wa AI uliokabidhiwa jukumu la kutengeneza mpango wa hatua kwa hatua wa kuthibitisha kama matokeo ya mradi yanatimiza mahitaji yaliyobainishwa.
        Lengo lako ni kupendekeza mfululizo wa hatua utakao kusaidia kukusanya ushahidi kwa utaratibu kutoka vyanzo mbalimbali kama msimbo, nyaraka, historia, au data.
        Chagua hatua zinazohitajika tu kutoka kwenye orodha ifuatayo na uzipange kwa mantiki.

        - [User Query]: tumia ombi la awali la mtumiaji kuelewa muktadha na hitaji.
        - [Workspace]: changanua muundo wa jumla wa workspace ili kuelewa vipengele vya mradi na utegemezi wake.
        - [Locate]: tafuta faili au saraka ambazo zinaweza kuwa na taarifa au utekelezaji unaohusika.
        - [Read]: soma na uchunguze yaliyomo ndani ya faili ili kuthibitisha usahihi na uhusiano wake na hitaji.
        - [Search]: tafuta vipande vya msimbo, functions, au variables vinavyohusiana na hitaji.
        - [History]: rejelea tathmini za awali, maamuzi ya hapo nyuma, au matokeo ya iteration za awali au miradi inayohusiana.
        - [Trajectory]: changanua trajekta ya kihistoria ya maendeleo au utekelezaji wa mradi, ikiwa ni pamoja na mabadiliko au iteration zilizotangulia zilizoathiri hali ya sasa.

        Chagua na panga hatua zinazohitajika tu ili kuunda mtiririko wa tathmini unaoweza kutekelezwa.
        """

    raise NotImplementedError(f"The language '{language}' is not supported.")
