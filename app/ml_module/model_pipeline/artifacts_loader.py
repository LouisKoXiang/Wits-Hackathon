from pathlib import Path
from typing import Any, List
import json
import joblib
from pydantic import BaseModel

from loan_predictor_package import FeatureEngineer, LoanPredictor

BASE_DIR = Path(__file__).resolve().parent

PIPELINE_PATH = BASE_DIR / "lending_pipeline.pkl"
MODEL_PATH = BASE_DIR / "lending_model.pkl" # 目前沒引用 
THRESHOLD_PATH = BASE_DIR / "lending_optimal_threshold.json"
SHAP_SUMMARY_PATH = BASE_DIR / "lending_shap_summary.json"


class PipelineArtifacts(BaseModel):
    pipeline: Any
    model: Any
    threshold: float
    shap_summary: List[dict]


def _load_pipeline_predictor() -> LoanPredictor:
    # laod pipeline.pkl
    predictor = joblib.load(PIPELINE_PATH)

    if not isinstance(predictor, LoanPredictor):
        raise TypeError(
            f"lending_pipeline.pkl 應為 LoanPredictor，實際為 {type(predictor)}"
        )
    return predictor


def load_shap_summary() -> List[dict]:
    # 讀 shap summary，文件缺失時回傳空 list
    if not SHAP_SUMMARY_PATH.exists():
        return []

    with open(SHAP_SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, list) else []


def _load_optimal_threshold(fallback: float) -> float:
    # 讀 threshold，無檔案時使用預設值
    if not THRESHOLD_PATH.exists():
        return fallback

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return float(data.get("optimal_threshold", fallback))


def load_pipeline_artifacts() -> PipelineArtifacts:
    # pipeline、model、threshold、shap
    predictor = _load_pipeline_predictor()

    threshold = _load_optimal_threshold(
        fallback=getattr(predictor, "optimal_threshold", 0.5)
    )

    shap_summary = load_shap_summary()

    return PipelineArtifacts(
        pipeline=predictor.pipeline,
        model=predictor.model,
        threshold=threshold,
        shap_summary=shap_summary,
    )
