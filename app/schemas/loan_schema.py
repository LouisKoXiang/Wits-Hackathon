# schemas/loan_schema.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any


class LoanInput(BaseModel):
    loan_amnt: float = Field(..., description="貸款金額", example=300000)
    int_rate: float = Field(..., description="年利率（%）", example=10.5)
    installment: float = Field(..., description="每月付款金額", example=9500)
    sub_grade: str = Field(..., description="次級信用評級", example="B3")
    annual_inc: float = Field(..., description="年收入", example=800000)
    dti: float = Field(..., description="債務收入比（%）", example=35.2)
    open_acc: int = Field(..., description="開放信用帳戶數", example=5)
    pub_rec: int = Field(..., description="公共紀錄數", example=0)
    revol_bal: float = Field(..., description="循環信貸餘額", example=52000)
    revol_util: float = Field(..., description="循環信貸使用率（%）", example=68)
    total_acc: int = Field(..., description="總信用帳戶數", example=15)
    mort_acc: int = Field(..., description="房貸帳戶數", example=2)
    pub_rec_bankruptcies: int = Field(..., description="公共破產紀錄數", example=0)
    earliest_cr_line: str = Field(..., description="最早信用紀錄", example="2014")
    term: str = Field(..., description="貸款期限", example="36 months")
    grade: str = Field(..., description="信用等級 A~G", example="B")
    emp_length: str = Field(..., description="工作年限", example="5 years")
    home_ownership: str = Field(..., description="居住狀態", example="MORTGAGE")
    verification_status: str = Field(..., description="驗證狀態", example="verified")
    purpose: str = Field(..., description="貸款用途", example="credit_card")
    application_type: str = Field(..., description="申請類型", example="individual")



class ShapTopFeature(BaseModel):
    feature_en: str
    feature_cn: str
    mean_abs_shap: float


RiskLabel = Literal["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class LoanPredictResponse(BaseModel):
    probability: float
    decision_label: RiskLabel
    threshold: float
    risk_level: RiskLevel

    shap_top_features: List[ShapTopFeature]

    policies: Dict[str, Any]

    meta: Dict[str, Any]
