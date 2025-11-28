from typing import List
from fastapi import APIRouter, Query

from app.ml_module.xgb_loader import load_xgb_artifacts
from app.schemas.loan_schema import ShapTopFeature

router = APIRouter(prefix="/shap", tags=["SHAP 分析"])

def shap_top_features_local(shap_summary_raw, top_k: int = 5):
    if not shap_summary_raw:
        return []

    def get_value(item, key, default=None):
        return item.get(key) or item.get(key.lower()) or default

    sorted_items = sorted(
        shap_summary_raw,
        key=lambda x: float(get_value(x, "Mean_Absolute_SHAP_Value", 0.0)),
        reverse=True,
    )

    top_feats = []
    for item in sorted_items[:top_k]:
        top_feats.append(
            ShapTopFeature(
                feature_en=str(get_value(item, "Feature_EN", "")),
                feature_cn=str(get_value(item, "Feature_CN", "")),
                mean_abs_shap=float(get_value(item, "Mean_Absolute_SHAP_Value", 0.0)),
            )
        )

    return top_feats


@router.get("/summary", response_model=List[ShapTopFeature])
def shap_summary(top_k: int = Query(10, ge=1, le=50)):
    """
    回傳整體模型的 SHAP 重要特徵 Top K。
    """
    artifacts = load_xgb_artifacts()
    return shap_top_features_local(artifacts.shap_summary, top_k=top_k)
