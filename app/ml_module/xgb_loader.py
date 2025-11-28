from pathlib import Path
import json
import joblib
import xgboost as xgb
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 模型相關檔案路徑
# ---------------------------------------------------------
# BASE_DIR → 模組目錄 (app/ml_module)
BASE_DIR = Path(__file__).resolve().parent

# 所有 XGBoost 模型與前處理資料放在此資料夾
ARTIFACT_DIR = BASE_DIR / "wesley_xgb_files" / "bin_model"

FEATURE_COLS_PATH = ARTIFACT_DIR / "model_artifacts_wesley_lending_feature_cols.json"
SCALER_PATH = ARTIFACT_DIR / "model_artifacts_wesley_lending_scaler_xgb.pkl"
MODEL_PKL_PATH = ARTIFACT_DIR / "model_artifacts_wesley_lending_model_xgb.pkl"
SHAP_SUMMARY_PATH = ARTIFACT_DIR / "model_artifacts_wesley_lending_shap_summary.json"
OPTIMAL_THRESHOLD_PATH = ARTIFACT_DIR / "model_artifacts_wesley_lending_optimal_threshold.json"


# ---------------------------------------------------------
# Artifacts 容器
# - 模型、Scaler、特徵欄位與其他附屬資訊會被封裝在此物件
# ---------------------------------------------------------
class XGBArtifacts:
    def __init__(
        self,
        model: xgb.XGBClassifier,
        scaler: MinMaxScaler,
        feature_cols: List[str],
        shap_summary: Optional[list],
        optimal_threshold: float = 0.5,
    ):
        self.model = model               # 主模型
        self.scaler = scaler             # scaler
        self.feature_cols = feature_cols # 順序
        self.shap_summary = shap_summary or []  # SHAP
        self.optimal_threshold = optimal_threshold # New 0.6


# 模型 artifacts 緩存
_artifacts: Optional[XGBArtifacts] = None


# ---------------------------------------------------------
# 載入整套 XGB 模型與相關 artifacts
#
# 會載入：
# - 模型本體
# - scaler
# - 特徵欄位
# - SHAP summary（若存在）
# - optimal threshold（若存在）
#
# 讀取後會快取在 _artifacts，避免重複 I/O
# ---------------------------------------------------------
def load_xgb_artifacts() -> XGBArtifacts:
    global _artifacts
    # 若已載入過，直接回傳 cache
    if _artifacts is not None:
        return _artifacts

    # --- 1. 讀特徵欄位
    with FEATURE_COLS_PATH.open("r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # --- 2. 讀取scaler
    scaler = joblib.load(SCALER_PATH)

    # --- 3. 讀取模型
    model = joblib.load(MODEL_PKL_PATH)

    # --- 4. 讀SHAP summary
    shap_summary = None
    if SHAP_SUMMARY_PATH.exists():
        with SHAP_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            shap_summary = json.load(f)

    # --- 5. 讀threshold
    optimal_threshold = 0.5
    if OPTIMAL_THRESHOLD_PATH.exists():
        try:
            with OPTIMAL_THRESHOLD_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                optimal_threshold = float(data.get("optimal_threshold", 0.5))
        except Exception:
            optimal_threshold = 0.5

    # --- 打包 artifacts
    _artifacts = XGBArtifacts(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        shap_summary=shap_summary,
        optimal_threshold=optimal_threshold,
    )

    return _artifacts
