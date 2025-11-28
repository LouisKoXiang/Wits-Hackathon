# app/services/loan_service.py
from typing import List, Dict, Any, Tuple
import numpy as np

from app.ml_module.xgb_loader import load_xgb_artifacts
from app.schemas.loan_schema import (
    LoanInput,
    LoanPredictResponse,
    ShapTopFeature,
    RiskLabel,
    RiskLevel,
)

from app.ml_module.policy_loader import load_policy

# ============================================================
# Encoding Helper - 各欄位編碼小工具
# ============================================================

def _encode_sub_grade(sub: str) -> int:
    """
    次級信用評級（如 B3, C4）→ 數值
    A~G 共有 7 個等級，每個等級有 5 個子級 → 25 個整數空間
    """
    try:
        base = ord(sub[0].upper()) - ord("A")   # A=0, B=1, ...
        num = int(sub[1])                       # 1~5
        return base * 5 + (num - 1)             # 每個主級 5 個子級
    except:
        return 0


def _encode_emp_length(v: str) -> Dict[str, float]:
    """
    將“工齡”轉成 one-hot 類型。
    原始 ML 訓練資料對工齡處理採用多個工齡欄位，因此需要逐一建立。
    """
    v = v.lower().strip()
    return {
        "emp_length_10_plus_years": 1.0 if "10" in v else 0.0,
        "emp_length_1_year": 1.0 if v.startswith("1") else 0.0,
        "emp_length_2_years": 1.0 if v.startswith("2") else 0.0,
        "emp_length_3_years": 1.0 if v.startswith("3") else 0.0,
        "emp_length_4_years": 1.0 if v.startswith("4") else 0.0,
        "emp_length_5_years": 1.0 if v.startswith("5") else 0.0,
        "emp_length_6_years": 1.0 if v.startswith("6") else 0.0,
        "emp_length_7_years": 1.0 if v.startswith("7") else 0.0,
        "emp_length_8_years": 1.0 if v.startswith("8") else 0.0,
        "emp_length_9_years": 1.0 if v.startswith("9") else 0.0,
    }


def _encode_home_ownership(v: str) -> Dict[str, float]:
    """房產擁有狀況 → one-hot"""
    v = v.upper()
    return {
        "home_ownership_MORTGAGE": 1.0 if v == "MORTGAGE" else 0.0,
        "home_ownership_NONE": 1.0 if v == "NONE" else 0.0,
        "home_ownership_OTHER": 1.0 if v == "OTHER" else 0.0,
        "home_ownership_OWN": 1.0 if v == "OWN" else 0.0,
        "home_ownership_RENT": 1.0 if v == "RENT" else 0.0,
    }


def _encode_purpose(v: str, feature_cols: List[str]) -> Dict[str, float]:
    """
    loan.purpose → one-hot
    特色：
    ✔ 動態從模型 feature_cols 判斷有哪些 purpose_* 欄位
    ✔ 使用者輸入不在訓練資料 → 自動 fallback 到 purpose_other
    """
    v = v.lower().strip()

    keys = [col for col in feature_cols if col.startswith("purpose_")]
    result = {key: 0.0 for key in keys}

    key = f"purpose_{v}"
    if key in result:
        result[key] = 1.0
    else:
        # fallback
        if "purpose_other" in result:
            result["purpose_other"] = 1.0

    return result

# ============================================================
# Main Encoding Function（LoanInput → XGBoost features）
# ============================================================

def _encode_loan_input(loan: LoanInput, feature_cols: List[str]) -> np.ndarray:
    """
    將 LoanInput（前端傳入的 JSON）轉成 **模型需要的特徵向量**
    並依照 feature_cols 的順序輸出 numpy array
    """
    row: List[float] = []

    # earliest credit year (信用紀錄最早年份)
    earliest_year = 2010
    if loan.earliest_cr_line:
        try:
            earliest_year = int(str(loan.earliest_cr_line)[:4])
        except:
            pass

    # 數值欄位（直接轉換成 float）
    numeric_map = {
        "loan_amnt": float(loan.loan_amnt),
        "int_rate": float(loan.int_rate),
        "installment": float(loan.installment),
        "sub_grade": _encode_sub_grade(loan.sub_grade),
        "annual_inc": float(loan.annual_inc),
        "dti": float(loan.dti),
        "open_acc": float(loan.open_acc),
        "pub_rec": float(loan.pub_rec),
        "revol_bal": float(loan.revol_bal),
        "revol_util": float(loan.revol_util),
        "total_acc": float(loan.total_acc),
        "mort_acc": float(loan.mort_acc),
        "pub_rec_bankruptcies": float(loan.pub_rec_bankruptcies),
        "earliest_cr_line_year": float(earliest_year),
    }

    # 其他 one-hot 類欄位
    term_map = {"term_60_months": 1.0 if loan.term.startswith("60") else 0.0}

    grade_map = {
        f"grade_{g}": 1.0 if loan.grade.upper() == g else 0.0
        for g in ["B", "C", "D", "E", "F", "G"]
    }

    emp_length_map = _encode_emp_length(loan.emp_length)
    home_ownership_map = _encode_home_ownership(loan.home_ownership)

    # 驗證狀態
    ver = loan.verification_status.replace(" ", "_")
    verification_map = {
        "verification_status_Source_Verified": 1.0 if ver == "Source_Verified" else 0.0,
        "verification_status_Verified": 1.0 if ver == "Verified" else 0.0,
    }

    # loan.purpose（動態 one-hot）
    purpose_map = _encode_purpose(loan.purpose, feature_cols)
    # print("purpose_map keys:", purpose_map.keys())

    application_map = {
        "application_type_INDIVIDUAL": 1.0 if loan.application_type.upper() == "INDIVIDUAL" else 0.0,
        "application_type_JOINT": 1.0 if loan.application_type.upper() == "JOINT" else 0.0,
    }

    # 合併所有欄位成一個 dict
    encoding_full = {
        **numeric_map,
        **term_map,
        **grade_map,
        **emp_length_map,
        **home_ownership_map,
        **verification_map,
        **purpose_map,
        **application_map,
    }

    # 按照 feature_cols 的固定順序，把數值塞入 row
    for col in feature_cols:
        row.append(float(encoding_full.get(col, 0.0)))

    return np.array(row, dtype=float)

# ============================================================
# Risk logic（還款機率越高 → 風險越低）
# ============================================================

def _compute_risk_label(prob: float, threshold: float) -> Tuple[RiskLabel, RiskLevel, float]:
    """
    根據預測機率計算風險分類
    threshold：訓練時找到的最佳門檻（預設 0.5）
    """
    low_cut = threshold * 0.5  # EX: threshold=0.6 → low_cut=0.3

    if prob >= threshold:
        return "LOW_RISK", "LOW", threshold
    elif prob >= low_cut:
        return "MEDIUM_RISK", "MEDIUM", threshold
    else:
        return "HIGH_RISK", "HIGH", threshold


def _build_policies(risk_level: str, bank_id: str) -> Dict[str, Any]:
    """
    整合 policy_loader
    """
    policy_full = load_policy(bank_id)
    strategy = policy_full["risk_strategies"].get(risk_level, {})
    return {
        "bank_id": bank_id,
        "bank_name": policy_full.get("bank_name"),
        "risk_level": risk_level,
        "summary": strategy.get("summary"),
        "suggestions": strategy.get("suggestions", []),
    }


# ============================================================
# Main Prediction Function（API 入口函式）
# ============================================================

def predict_loan_risk(loan: LoanInput, bank_id: str = "DEFAULT") -> LoanPredictResponse:
    """
    完整預測流程：
    1. 讀取模型 artifacts（模型、Scaler、特徵欄位、SHAP 統計）
    2. 編碼 LoanInput → XGBoost 特徵向量
    3. 套用 Scaler（與訓練資料保持一致）
    4. 使用模型推論還款機率
    5. 依門檻計算風險等級
    6. 回傳 SHAP 前 5 名重要特徵
    """

    # ------------------------------------------------------------
    #  1. 讀取模型 artifacts
    # ------------------------------------------------------------
    # load_xgb_artifacts() 會一次回傳：
    #   artifacts.model            → 訓練後的 XGBoost 模型
    #   artifacts.scaler           → MinMaxScaler / StandardScaler（訓練時的 scaling）
    #   artifacts.feature_cols     → 特徵欄位順序
    #   artifacts.optimal_threshold→ 0.6
    #   artifacts.shap_summary     → SHAP 全量重要性摘要
    #
    artifacts = load_xgb_artifacts()

    # ------------------------------------------------------------
    # 2. 編碼 LoanInput → XGBoost 特徵向量
    # ------------------------------------------------------------
    X = _encode_loan_input(loan, artifacts.feature_cols)

    # ------------------------------------------------------------
    # 3. Scaler（保持與訓練一致）
    # ------------------------------------------------------------
    import pandas as pd
    X_df = pd.DataFrame([X], columns=artifacts.feature_cols)
    X_scaled = artifacts.scaler.transform(X_df)

    # ------------------------------------------------------------
    # 4. 使用模型推論
    # ------------------------------------------------------------
    prob = float(artifacts.model.predict_proba(X_scaled)[0, 1])

    # ------------------------------------------------------------
    # 5. 使用訓練時最佳門檻分類
    # ------------------------------------------------------------
    threshold = float(getattr(artifacts, "optimal_threshold", 0.5))

    decision_label, risk_level, threshold = _compute_risk_label(prob, threshold)

    # ------------------------------------------------------------
    # 6. SHAP 前五名特徵（若 artifacts 有提供）
    # ------------------------------------------------------------
    shap_top = []
    if artifacts.shap_summary:
        sorted_feats = sorted(
            artifacts.shap_summary,
            key=lambda x: float(x.get("Mean_Absolute_SHAP_Value", 0)),
            reverse=True,
        )[:5]

        shap_top = [
            ShapTopFeature(
                feature_en=item.get("Feature_EN", ""),
                feature_cn=item.get("Feature_CN", ""),
                mean_abs_shap=float(item.get("Mean_Absolute_SHAP_Value", 0.0)),
            )
            for item in sorted_feats
        ]

    # ------------------------------------------------------------
    # 7. 建立銀行策略建議
    # ------------------------------------------------------------
    policy = _build_policies(risk_level, bank_id)

    # ------------------------------------------------------------
    # 8. 回傳 metadata
    # ------------------------------------------------------------
    meta = {
        "model": "xgboost_classifier",
        "feature_count": len(artifacts.feature_cols),
        "bank_id": bank_id,
        "optimal_threshold_used": threshold,
    }

    # ------------------------------------------------------------
    # 9. 組合並輸出統一格式的 API Response
    # ------------------------------------------------------------
    return LoanPredictResponse(
        probability=prob,
        decision_label=decision_label,
        threshold=threshold,
        risk_level=risk_level,
        shap_top_features=shap_top,
        policies=policy,
        meta=meta,
    )
