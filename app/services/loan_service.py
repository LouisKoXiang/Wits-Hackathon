# app/services/loan_service.py
from typing import List, Dict, Any, Tuple
import numpy as np

import pandas as pd
from app.schemas.loan_schema import (
    LoanInput,
    LoanPredictResponse,
    ShapTopFeature,
    RiskLabel,
    RiskLevel,
)

from app.ml_module.model_pipeline.artifacts_loader import (
    load_pipeline_artifacts,
)

from app.ml_module.policy_loader import load_policy

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
    # ------------------------------------------------------------
    # 使用新版 pipeline + model：
    # 流程：
    # 1. LoanInput (Pydantic) → dict → DataFrame
    # 2. pipeline.transform() 進行特徵工程（FeatureEngineer）
    # 3. model.predict_proba() 得到還款機率
    # 4. 使用 optimal_threshold 進行風險判斷
    #  5. SHAP → 完整回吐 lending_shap_summary.json 的內容
    # 6. 套用銀行 policy
    # 7. 組成 LoanPredictResponse
    # ------------------------------------------------------------
    # 1. LoanInput → DataFrame
    # ------------------------------------------------------------
    df = pd.DataFrame([loan.model_dump()])

    # ------------------------------------------------------------
    # 2. 取得 artifacts（pipeline / model / threshold / shap_summary）
    # ------------------------------------------------------------
    artifacts = load_pipeline_artifacts()

    # ------------------------------------------------------------
    # 3. pipeline 特徵工程
    # ------------------------------------------------------------
    try:
        X = artifacts.pipeline.transform(df)
    except Exception as e:
        raise RuntimeError(f"pipeline.transform err：{e}")

    # ------------------------------------------------------------
    # 4. 使用模型推論 預測還款機率（proba[:, 1]）
    # ------------------------------------------------------------
    try:
        prob = float(artifacts.model.predict_proba(X)[0, 1])
    except Exception as e:
        raise RuntimeError(f"pipeline.transform err：{e}")

    # ------------------------------------------------------------
    # 5. 使用訓練時最佳門檻分類
    # ------------------------------------------------------------
    threshold = float(artifacts.threshold)

    decision_label, risk_level, threshold = _compute_risk_label(prob, threshold)

    # ------------------------------------------------------------
    # 6. SHAP 前五名特徵（若 artifacts 有提供）
    # ------------------------------------------------------------
    shap_top = [
        ShapTopFeature(
            feature_en=item.get("Feature_EN", "") or "",
            feature_cn=item.get("Feature_CN", "") or "",
            mean_abs_shap=float(item.get("Mean_Absolute_SHAP_Value") or 0.0),
        )
        for item in (artifacts.shap_summary or [])
    ]

    # ------------------------------------------------------------
    # 7. 建立銀行策略建議
    # ------------------------------------------------------------
    policy = _build_policies(risk_level, bank_id)

    # ------------------------------------------------------------
    # 8. 回傳 metadata
    # ------------------------------------------------------------
    meta = {
        "model": "xgboost_pipeline",
        "feature_count": X.shape[1],
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
