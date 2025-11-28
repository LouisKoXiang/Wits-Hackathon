from fastapi import APIRouter, Query
from app.schemas.loan_schema import LoanInput, LoanPredictResponse
from app.services.loan_service import predict_loan_risk

router = APIRouter(prefix="/predict", tags=["ML 預測"])


@router.post("/", response_model=LoanPredictResponse)
def loan_predict(
    payload: LoanInput,
    bank_id: str = Query("DEFAULT", description="銀行代碼，用於載入不同 policy"),
) -> LoanPredictResponse:
    """
    貸款還款機率預測 + 風險等級 + SHAP Top 特徵 + 銀行政策建議
    """
    return predict_loan_risk(payload, bank_id=bank_id)
