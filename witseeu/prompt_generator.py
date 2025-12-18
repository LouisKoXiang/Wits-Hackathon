from __future__ import unicode_literals
from textwrap import dedent

class PromptGenerator:
    """
    用於生成各種 Job Description (JD) 和 Resume (履歷) 匹配分析所需的
    System Prompt 和 User Prompt 的類別。

    支援在初始化時設定 appVersion，以控制 JD 提取規則。
    """

    # --- 共通的結構定義 (作為類別屬性) ---
    JD_OUTPUT_STRUCTURE = '[{"jd_name":"技能名稱", "jd_desc":"技能描述, 該技能在 JD 中的描述(如具體要求, 使用範圍)"}, ...]'
    MATCH_SCORE_STRUCTURE = '[{"name":"JD條件", "score":"得分", "desc":"判斷原因"}, ...]'

    def __init__(self, app_version: str = '1.1'):
        self.app_version = app_version

    @staticmethod
    def get_jd_system_prompt() -> str:
        """獲取 JD 提取技能的 System Prompt。"""
        return "你是一個專業的 Job Description (JD) 解析專家, 負責從職缺描述（JD）中提取專業技能。"

    @staticmethod
    def get_match_score_system_prompt() -> str:
        """獲取匹配結果評分的 System Prompt (0-10分細項評分)。"""
        return "你是一位專業的人資顧問和科技顧問, 擅長分析JD(職缺描述)和Resume(求職者的履歷), 並對Resume進行評分"

    @staticmethod
    def get_match_html_system_prompt() -> str:
        """獲取詳細匹配分析報告的 System Prompt (HTML 輸出)。"""
        return "你是一個分析資料結構的專家及人資招募的專家, 擅長分析求職者的履歷 (Resume) 與職缺描述 (JD) 之間的匹配度。"

    @staticmethod
    def get_chat_answer_system_prompt() -> str:
        """獲取 Chat 流程的 System Prompt。"""
        return "你是一位專業的人資顧問，擅長分析求職者的履歷 (Resume) 與職缺描述 (JD) 之間的匹配度。"

    @staticmethod
    def get_resume_rag_system_prompt() -> str:
        """獲取 RAG 流程中 Resume 評分的 System Prompt (強調知識庫)。"""
        return "你是一位專業的人資顧問和科技顧問, 擅長分析JD(職缺描述)和Resume(求職者的履歷), 並對Resume進行評分, 並從**知識庫**中提取評分標準"

    def get_jd_user_prompt(self, jd: str) -> str:
        # 預設 prompt
        extraction_rule = f"""
        ### **提取要求**
        1. 「專業技能」指的是技術能力
        2. 只提取 JD 內 **明確提及** 的技能, 不要添加推測性的內容
        3. 對於 JD 內需求的年資經驗也需提取
        """

        # RAG
        if self.app_version == '2.0':
            extraction_rule = f"""
            ### **提取要求**
            **請嚴格按照知識庫 WITS_HRKL_001 規則**
            """

        return dedent(f"""(
        請仔細閱讀以下 JD，並從中提取與該職位相關的「專業技能」:
        {jd}

        ### **輸出格式**
        請嚴格按照如下格式回覆，僅回覆 JSON，其他文字請勿回覆：
        {self.JD_OUTPUT_STRUCTURE}

        {extraction_rule}
        """)

    def get_match_score_user_prompt(self, jd, resume):
        output_structure = '[{"name":"JD條件", "score":"得分", "desc":"判斷原因"}, ...]'
        prompt = f"""
        ### **JD (職缺描述)**
        {jd}

        ### **Resume (履歷)**
        {resume}
        """

        # RAG
        if self.app_version == '2.0':
            prompt += f"""
            ### **Resume評分標準**
            **請嚴格按照知識庫 WITS_HRKL_002 規則**

            ### **輸出格式**
            請嚴格按照評分標準，並按照如下格式回應，嚴格遵守以 JSON 格式編碼, 其他文字請勿回覆 :
            {output_structure}
            """
        else :
            prompt += f"""
            ### **Resume評分標準**
            1. 依 Resume 裡找出符合 JD.req_name 的項目, 給予評分, 並記下判斷原因
            2. 若 Resume 裡沒有符合 JD.req_name 的項目, 則應該為0分
            3. 注意英文的專有名詞, 前後合併字或前後分開寫的字要視為相同, 例如 SpringBoot, 應視為 Spring Boot, 不必區分文字的大小寫
            4. SQL 部份因為有很多種, 請使用模糊比對, 只要有相關就應該評分
            5. 以下為評分標準:
                - 0分:Resume裡完全沒有提及相關知識或經驗，甚至不知道該技術是什麼。
                - 1分:聽過該技術的名稱，但沒有任何理解或實際應用經驗。
                - 2分:對該技術有基礎認識，但無法實際操作或解決問題。
                - 3分:了解基本概念，能夠在指導下完成簡單的任務，但無法獨立應用。
                - 4分:具備基礎實作能力，能夠完成簡單專案，但遇到問題需要尋求幫助。
                - 5分:具有一定的經驗，能夠獨立處理一般的任務，但仍需參考文件或他人協助。
                - 6分:熟悉該技術，能夠獨立開發應用並解決常見問題，具備一定的優化能力。
                - 7分:具備豐富經驗，能夠高效開發並優化系統，對技術細節有深入理解。
                - 8分:為該技術的高級使用者，能夠設計架構、解決複雜問題，並指導他人。
                - 9分:在該領域具有專家級能力，能夠提出創新方案、最佳實踐，甚至影響行業標準。
                - 10分:世界級專家，參與技術標準制定或核心開發，擁有重大技術貢獻。
            6. 每個配對包含:
                - 'name':JD條件(即 技能名稱, 從 JD.req_name 而來)
                - 'score':所得分數
                - 'desc':判斷原因

            ### **遵守規則**
            請**不要假設 Resume 中包含 JD.req_name**，一定要先從 Resume 中找出實際提到的技能名稱。

            ### **輸出格式**
            請嚴格按照評分標準，並按照如下格式回應，嚴格遵守以 JSON 格式編碼, 其他文字請勿回覆 :
            {output_structure}
            """

        return prompt

    def get_match_html_user_prompt(self, jd: str, resume: str) -> str:
        jd_col_desc = '{"jd_name":"技能名稱", "jd_desc":"技能描述"}'
        resume_col_desc = '{"name":"JD條件", "score":"得分", "desc":"判斷原因"}'
        prompt = f"""
        JD 欄位說明 : {jd_col_desc}
        Resume 欄位說明 : {resume_col_desc}
        JD 結構 : {jd}
        Resume 結構 : {resume}
        """

        # RAG
        if self.app_version == '2.0':
            prompt += f"""
            ### **回覆要求與格式要求**
            **請嚴格按照知識庫 WITS_HRKL_003 規則**
            """
        else :
            prompt += f"""
            嚴格透過 Resume.name 和 JD.jd_name 來比對，並完整回覆以下內容：
            1. **匹配度分析** : (請基於求職者的技能、經歷與 JD 要求進行深入分析，一定要寫出具體細節)。
            2. **詳細評分**：
                - **技能匹配度**（一定要列舉關鍵技能，說明如何符合 JD）。
                - **工作經歷對應**（一定要比較履歷中的經歷與 JD）。
                - **個性與團隊適配性**（一定要分析履歷內容，評估求職者的個性與企業文化的適配度)。
                - **可能的挑戰點**（列出求職者在應徵此職位時可能遇到的挑戰）。
            3. **總體匹配度評分 : X分/(滿分10分)**（請確保 X 不超過 10 分，並且不要將 JD 裡的「加分項目」計入總分。請 AI 確保評分邏輯為：即使求職者符合所有條件，最高分仍然是 10 分。回覆格式必須嚴格遵守：「X分/(滿分10分)」，並附上詳細評分依據）。
            4. **建議面試問題**（根據 JD 和求職者的履歷，提供 2 個有針對性的面試問題，以測試其專業技能、經驗或適應能力）。
            ### **格式要求**
            請使用 HTML 結構化回覆，確保輸出內容能夠直接呈現在網頁上：
            - 使用 `<h2>` 來標示標題
            - 使用 `<p>` 來撰寫段落內容
            - 使用 `<ul>` 和 `<li>` 來列舉清單項目
            - 總分不得超過 10 分, JD 裡若有「加分項目」可列在最終評價的分數上。
            - 內容應該是 AI 根據 Resume 和 JD 生成的分析，而不是單純返回 HTML 標籤
	        - 確保每一個列舉清單項目都有具體的項目和分析內容，不可回空白列。
	        - 嚴格規定在 <HTML_REPORT>...</HTML_REPORT> 標籤內輸出完整的 HTML 分析報告。
            """
        return prompt

    def get_combined_match_user_prompt(self, jd, resume):
        """合併評分與 HTML 報告的 Prompt"""
        rule_score = "按照知識庫 WITS_HRKL_002"
        rule_html = "按照知識庫 WITS_HRKL_003"

        return dedent(f"""
        ### JD 內容
        {jd}
        ### 履歷內容
        {resume}

        ### 任務要求
        1. 評分：請針對 JD 條件進行評分，規則：{rule_score}。
        2. 報告：請撰寫 HTML 格式分析報告，規則：{rule_html}。

        ### 輸出格式限制 (極重要)
        請務必將結果包裝在以下標籤內：
        <SCORE_JSON>
        {self.MATCH_SCORE_STRUCTURE}
        </SCORE_JSON>

        <HTML_REPORT>
        (這裡放入 HTML 格式報告)
        </HTML_REPORT>
        """)