import os
import json
from typing import Optional, Dict, Any, List, Literal

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.loan_schema import LoanInput, LoanPredictResponse
from app.services.loan_service import predict_loan_risk
from app.memory.redis_memory import RedisMemory

router = APIRouter(prefix="/chat", tags=["Chat"])
redis_memory = RedisMemory()

# ================================
# OpenAI Environment
# ================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("未設定 OPENAI_API_KEY")


# ================================
# OpenAI Chat Wrapper
# ================================
def _openai_chat(messages, model=None, temperature=0.2) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30,
        )
    except Exception as e:
        raise HTTPException(500, f"OpenAI 呼叫失敗: {e}")

    if resp.status_code != 200:
        # 官方錯誤碼
        raise HTTPException(
            500, f"OpenAI Error: {resp.status_code} - {resp.text}"
        )

    try:
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        # 回傳格式改版了
        raise HTTPException(500, "OpenAI 回傳格式異常")


# ================================
# Intent Detection
# ================================
IntentType = Literal["predict", "shap", "analyze", "memory", "small_talk", "unknown"]

def _detect_intent(text: str) -> IntentType:
    """
    讓模型自己猜使用者到底想幹嘛。
    """

    sys_prompt = """
你是意圖分類器。你只能回傳以下其一：

predict / shap / analyze / memory / small_talk

判斷規則：
- predict：問風險、會不會過件、機率
- shap：問哪些欄位影響最大、為什麼是這個結果
- analyze：要整體建議、總結、完整分析
- memory：問我剛剛說了什麼、之前的數值
- small_talk：聊天、寒暄
請輸出單一英文單字，不要多餘內容。
""".strip()

    result = _openai_chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": text}],
        model=CHAT_MODEL, temperature=0
    )

    result = result.strip().lower()
    if result not in ["predict", "shap", "analyze", "memory", "small_talk"]:
        # 不在主題 unknown
        return "unknown"
    return result  # type: ignore


# ================================
# Schema
# ================================
class ChatRequest(BaseModel):
    """
    message：使用者說了什麼
    form_data：如果有一起塞貸款資料，就直接啟動預測
    """
    session_id: str = Field(..., example="user-123")
    message: str = Field(..., example="我這樣算高風險嗎？")
    form_data: Optional[LoanInput] = None
    bank_id: str = Field("DEFAULT", example="DEFAULT")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user-001",
                "message": "該客戶的違約風險？",
                "bank_id": "DEFAULT",
                "form_data": {
                    "loan_amnt": 300000,
                    "int_rate": 10.5,
                    "installment": 0,
                    "sub_grade": "B3",
                    "annual_inc": 800000,
                    "dti": 35.2,
                    "open_acc": 5,
                    "pub_rec": 0,
                    "revol_bal": 52000,
                    "revol_util": 68,
                    "total_acc": 15,
                    "mort_acc": 2,
                    "pub_rec_bankruptcies": 0,
                    "earliest_cr_line": "2014",
                    "term": "36 months",
                    "grade": "B",
                    "emp_length": "5 years",
                    "home_ownership": "MORTGAGE",
                    "verification_status": "verified",
                    "purpose": "credit_card",
                    "application_type": "individual",
                }
            }
        }
    }


class ChatResponse(BaseModel):
    """
    回傳給前端的格式：
    - reply：AI 實際回覆
    - prediction：如果此次有跑 risk model，就一起回
    - suggested_questions：順便給前端一些「下一句可以問什麼」
    """
    session_id: str
    reply: str
    model_used: str
    prediction: Optional[LoanPredictResponse] = None
    suggested_questions: Optional[List[str]] = None


# ================================
# Question Suggestion（個人化推薦）
# ================================
def _suggest_questions(intent: str) -> List[str]:
    """
    如果沒有個人化推薦（例如沒有 SHAP、沒有預測資料），
    這裡就是基本款備案問題
    """
    if intent == "predict":
        return [
           "模型輸入欄位是否有缺漏或異常？",
           "模型判定風險的主要驅動因素是否合理？",
           "若調整部分變數，風險是否會顯著改善？",
        ]
    if intent == "shap":
        return [
            "哪些欄位風險影響最大？",
            "模型為什麼這樣判斷？",
        ]
    if intent == "analyze":
        return [
            "是否需要補充其他財務資料以提高分析準確性？",
            "申請人的信用紀錄與模型結果是否一致？",
            "不同貸款方案的風險差異為何？",
        ]
    if intent == "memory":
        return ["最新輸入資料與先前資料是否存在差異?"]
    if intent == "small_talk":
        return ["是否需要進一步分析風險或補充資料？"]

    # unknown
    return ["是否需要進一步分析風險或補充資料？"]


def _personalized_suggestions(
    intent: str,
    prediction: Optional[LoanPredictResponse],
    form_data: Optional[Dict[str, Any]]
) -> List[str]:

    suggestions = []

    # 有預測結果
    if prediction:

        level = prediction.risk_level

        # 依風險等級推薦問題
        if level == "HIGH":
            suggestions += [
                "哪些欄位造成申請人被評為高風險？",
                "申請人若降低負債或提高收入，風險是否能改善？",
                "目前哪些財務項目需要優先查核？",
                "是否存在尚未揭露的負債或異常信用紀錄？",
            ]
        elif level == "MEDIUM":
            suggestions += [
                "申請人目前是接近高風險還是低風險？",
                "哪些欄位若改善可以讓風險下降？",
                "模型認為哪些財務項目需要優先補件或查核？",
            ]
        elif level == "LOW":
            suggestions += [
                "申請人是否具備申請較佳利率或較高額度的資格？",
                "哪些欄位強化了申請人的低風險評等？",
                "若提高貸款金額，風險是否仍在可接受範圍內？",
            ]

        # 如果 SHAP 有出現，拿第一名做個人化問句
        if prediction.shap_top_features:
            top_feat = prediction.shap_top_features[0]
            suggestions.append(f"為什麼「{top_feat.feature_cn}」會成為風險貢獻最高的欄位？",)
            suggestions.append(f"若調整「{top_feat.feature_cn}」，風險是否有改善空間？")

    # 用戶有傳 form_data → 看看要不要補充建議
    if form_data:

        if "dti" in form_data:
            dti = form_data["dti"]
            if dti > 40:
                suggestions.append("DTI 偏高是否已明顯影響申請人的風險評等？")
                suggestions.append("若 DTI 降至 35% 以下，風險是否會改善？")

        if "annual_inc" in form_data:
            suggestions.append("申請人目前的收入水準是否足以支撐申貸金額？")

        if "int_rate" in form_data:
            ir = form_data["int_rate"]
            suggestions.append(f"利率 {ir}% 是否對風險造成明顯影響？")
            suggestions.append("若調整利率，是否能改善整體風險？",)

        if "sub_grade" in form_data:
            sg = form_data["sub_grade"]
            suggestions.append(f"次級信用評級（{sg}）是否與申請人的信用紀錄一致？")

    # 如果還是太少 → 用 intent fallback
    if len(suggestions) < 3:
        suggestions += _suggest_questions(intent)

    # 去重、限制 5 個
    return list(dict.fromkeys(suggestions))[:5]


# ================================
# Main Chat API
# ================================
@router.post("/", response_model=ChatResponse)
def chat_api(payload: ChatRequest):
    """
    ### 1 session_id（必填）
    用於辨識同一位使用者／同一段對話。
    ### 2 message（必填）
    使用者輸入的問題或對話內容，例如：「該客戶的違約風險？」
    bank_id": "DEFAULT"
    ### 3 form_data（分必填）
    貸款欄位資料（LoanInput）。
    首輪對話建議提供；後續對話如無更新可省略。
    ### 4 bank_id（非必填）
    "bank_id": "DEFAULT",
    #### LoanInput 欄位說明：

    | 欄位 | 型別 | 範例 | 說明 |
    |------|------|-------|------|
    | loan_amnt | number | 300000 | 申請貸款金額 |
    | int_rate | number | 10.5 | 貸款利率 (%) |
    | installment | number | 0 | 每期應繳金額 |
    | sub_grade | string | "B3" | 次級信用評級 |
    | annual_inc | number | 800000 | 年收入 |
    | dti | number | 35.2 | 債務收入比 |
    | open_acc | number | 5 | 開放信用額度 |
    | pub_rec | number | 0 | 公共紀錄次數 |
    | revol_bal | number | 52000 | 循環信貸餘額 |
    | revol_util | number | 68 | 信用卡使用率 % |
    | total_acc | number | 15 | 歷史帳戶總數 |
    | mort_acc | number | 2 | 抵押帳戶數 |
    | pub_rec_bankruptcies | number | 0 | 破產紀錄 |
    | earliest_cr_line | string | "2014" | 最早信用記錄年份 |
    | term | string | "36 months" | 貸款期數 |
    | grade | string | "B" | 信用評級 |
    | emp_length | string | "5 years" | 工作年資 |
    | home_ownership | string | "MORTGAGE" | 住房狀況 |
    | verification_status | string | "verified" | 收入驗證狀態 |
    | purpose | string | "credit_card" | 申貸目的 |
    | application_type | string | "individual" | 個人或共同申請 |
    
    Response
    | 欄位名稱           | 型別                    | 說明                 |
    | ------------------------ | --------------------- | ------------------ |
    | **session_id**           | `string`              | 回傳與請求相同的 session。  |
    | **reply**                | `string`              | AI 回覆內容（繁體中文）。     |
    | **model_used**           | `string`              | 實際使用的 OpenAI 模型名稱。 |
    | **prediction**           | `LoanPredictResponse` | 模型的風險預測結果。     |
    | **suggested_questions**  | `string[]`            | 推薦問題。             |

    Interface LoanPredictResponse
    | 欄位名稱                  | 型別                                         | 說明                         |
    | --------------------- | ------------------------------------------ | -------------------------- |
    | **probability**       | `number`                                   | 還款機率（0–1）。                 |
    | **decision_label**    | `"LOW_RISK" / "MEDIUM_RISK" / "HIGH_RISK"` | 模型對該客戶的風險分類。               |
    | **threshold**         | `number`                                   | 分界值（目前固定 0.5）。             |
    | **risk_level**        | `"LOW" / "MEDIUM" / "HIGH"`                | 更易理解的風險等級。                 |
    | **shap_top_features** | `list`                                     | 模型 SHAP 由訓練結果提供 |
    | **policies**          | `object`                                   | 銀行授信政策建議（按風險調整）。           |
    | **meta**              | `object`                                   | 模型資訊（feature 數量、bank_id…）。 |
    ---

    Chat API 主流程：
    1. 讀 session（看之前聊了什麼）
    2. form_data 進來就重新預測
    3. 辨識使用者想要幹嘛（predict, shap, memory…）
    4. 組 prompt（把上下文塞進去）
    5. 跑 OpenAI 聊天
    6. 存新的 session
    7. 回傳結果＋推薦題目
    """

    # ---- Step 1. 先擺好之前的資料     
    session = redis_memory.get_session(payload.session_id)
    history = session["messages"]
    last_prediction_data = session["last_prediction"]
    last_form_data = session["last_form_data"]

    # ---- Step 2. 有傳 form_data → 重新跑預測
    prediction_obj: Optional[LoanPredictResponse] = None
    if payload.form_data is not None:
        prediction_obj = predict_loan_risk(payload.form_data, payload.bank_id)

        # 存本次預測與 form，以便下次聊天可以引用
        last_prediction_data = prediction_obj.model_dump()
        try:
            last_form_data = payload.form_data.model_dump()
        except:
            last_form_data = payload.form_data.dict()

    # ---- Step 3. 如果本次沒算，但之前算過 → 把舊的補回來
    if prediction_obj is None and last_prediction_data:
        try:
            prediction_obj = LoanPredictResponse(**last_prediction_data)
        except:
            prediction_obj = None

    # ---- Step 4. 讓模型猜 user 想問什麼
    intent = _detect_intent(payload.message)

    # ---- Step 5. 組系統 prompt
    sys = [
        "你是一個提供給銀行內部經辦人員使用的風險分析系統。",
        "所有回覆必須以『系統回覆經辦』的口吻撰寫，而不是對客戶說話。",
        "請使用第三人稱，例如：『該客戶』『此申貸人』『申請者』。",
        "禁止使用『您』『你』等第二人稱說法。",
        "語氣必須專業、中性、客觀，符合金融業授信風險報告格式。",
        "若提供建議，請採用：『建議』『可考慮』『風險評估如下』『模型顯示』等用語。",
        "內容需聚焦於：風險等級、影響因素、授信建議、整體分析，不提供情緒性描述。",
        "你的任務包括：風險預測解釋、特徵影響度分析、整體授信建議、回想使用者填寫的欄位資料。"
    ]

    if last_form_data:
        sys.append("以下是客戶最近提供的貸款欄位 (JSON)：")
        sys.append(json.dumps(last_form_data, ensure_ascii=False))

    if prediction_obj:
        sys.append("以下是最新的風險模型預測結果 (JSON)：")
        sys.append(json.dumps(prediction_obj.model_dump(), ensure_ascii=False))

    sys.append(f"判斷出使用者意圖：{intent}")

    # Intent 特別規則補強
    if intent == "predict":
         sys.append(
            "模型輸出的 probability 是『還款機率（repay probability）』，不是違約機率。"
            "請務必使用以下邏輯回答："
            "還款機率 = probability；還款違約機率 = 1 - probability。"
            "請根據 risk_level 解釋此客戶是高、中或低風險，並避免混淆。"
            "回覆請描述該客戶的風險狀態，而不是對客戶說明。"
        )

    elif intent == "shap":
        shap_list = None
        if prediction_obj and prediction_obj.shap_top_features:
            shap_list = [s.model_dump() for s in prediction_obj.shap_top_features]

        if shap_list:
            sys.append("以下是模型整體 SHAP 重要特徵摘要 (JSON)：")
            sys.append(json.dumps(shap_list, ensure_ascii=False))

        sys.append(
            "請說明在『模型整體層級』，哪些特徵對風險判斷影響最大，"
            "並描述這些特徵變動時，通常會如何拉高或降低風險。"
        )

    elif intent == "analyze":
        sys.append("請以『授信審查摘要』的格式提供回覆，包含：風險定位、主要驅動因子、建議授信策略（2~3 點）。" )
        

    elif intent == "memory":
        sys.append(
            "使用者正在詢問之前提供的資料。請根據 JSON 回憶該客戶的欄位與模型結果，"
            "並以『系統提供資料』的角度回答，整理出先前填寫的關鍵數值與當時的風險判斷。"
        )

    elif intent == "small_talk":
        sys.append(
            "使用者正在進行非技術性對話。請保持簡短、正式、中性，避免使用第二人稱，"
            "並適度引導其回到本系統的用途，例如風險分析、授信建議、特徵解釋等問題。"
        )

    else:
        sys.append(
            "意圖無法辨識，請提示經辦可詢問：風險、特徵影響、授信建議、"
            "或請求整理先前輸入的欄位資料等主題。"
        )

    system_prompt = "\n\n".join(sys)

    # ---- Step 6. 完整訊息（含歷史）
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for msg in history[-8:]:
        messages.append(msg)

    messages.append({"role": "user", "content": payload.message})

    # ---- Step 7. 呼叫 OpenAI 拿回答
    reply = _openai_chat(messages, CHAT_MODEL, 0.2)

    # ---- Step 8. 存回 session（聊天 log + 最新預測）
    history.append({"role": "user", "content": payload.message})
    history.append({"role": "assistant", "content": reply})

    session["messages"] = history
    session["last_prediction"] = last_prediction_data
    session["last_form_data"] = last_form_data

    redis_memory.save_session(payload.session_id, session)

    # ---- Step 9. 補幾個建議問題
    suggested = _personalized_suggestions(intent, prediction_obj, last_form_data)

    return ChatResponse(
        session_id=payload.session_id,
        reply=reply,
        model_used=CHAT_MODEL,
        prediction=prediction_obj,
        suggested_questions=suggested
    )
