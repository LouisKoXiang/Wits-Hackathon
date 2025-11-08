# =========================================
# Lending Club Risk API (Cross-Env Safe)
# =========================================
# Author: Louis Ko
# Description:
#   - FastAPI 服務，用於載入訓練好的模型與前處理器
#   - 支援 VSCode、本機、Colab、雲端部署環境
# =========================================

from fastapi import FastAPI, HTTPException
import tensorflow as tf
import joblib, json, numpy as np, os
from pathlib import Path

# -----------------------------------------
# 自動偵測 model 資料夾路徑
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = None

# 嘗試依序找 /model 與 /ml_module/model
for candidate in [BASE_DIR / "model", BASE_DIR / "ml_module" / "model"]:
    if (candidate / "LendingClub.keras").exists():
        MODEL_DIR = candidate
        break

if MODEL_DIR is None:
    raise FileNotFoundError("找不到任何模型資料夾（請確認 model/ 或 ml_module/model/ 是否存在）")

MODEL_PATH = MODEL_DIR / "LendingClub.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
COLUMNS_PATH = MODEL_DIR / "columns.json"

# -----------------------------------------
# 初始化 FastAPI
# -----------------------------------------
app = FastAPI(title="Lending Club Risk Predictor API")

# -----------------------------------------
# 檢查模型檔案是否存在
# -----------------------------------------
missing = [f.name for f in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH] if not f.exists()]
if missing:
    raise FileNotFoundError(f"缺少必要檔案於 {MODEL_DIR}: {missing}")

# -----------------------------------------
# 載入模型與前處理器
# -----------------------------------------
print(f"使用模型目錄：{MODEL_DIR}")
print("載入模型與前處理器中...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(COLUMNS_PATH) as f:
    columns = json.load(f)

print("模型與前處理器載入完成。")

# -----------------------------------------
# /predict API
# -----------------------------------------
@app.post("/predict")
def predict(data: dict):
    """預測貸款違約風險"""
    try:
        X = np.array([[data.get(col, 0) for col in columns]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

    # 設定風險等級（門檻可依實際模型調整）
    if pred > 0.8:
        risk_level = "Low"
    elif pred > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_score": float(pred),
        "risk_level": risk_level,
        "model_columns": columns
    }
