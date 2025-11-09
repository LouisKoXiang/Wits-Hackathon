# Lending Club Risk Predictor

FastAPI + TensorFlow + LangChain + ngrok  
黑客松原型專案：以 AI 協助經辦人快速預測貸款風險等級

---

## 專案結構

```
lending-club-risk/
┣ 📂 api/
│ ┗ main.py # FastAPI 主程式，提供 /predict API
┣ 📂 ml_module/
│ ┣ train_model.ipynb # 模型訓練 Notebook（可改為 .py 版本）
│ ┣ 📂 model/ # 訓練後輸出模型檔案
│ │ ┣ LendingClub.keras
│ │ ┣ scaler.pkl
│ │ ┗ columns.json
┣ lending_club_risk_api.ipynb # 主 Notebook：啟動 API、測試、LangChain 工具
┣ lending_club_risk_api_demo.ipynb # 整體架構 可直接打開 colab 跑整體流程
┣ .env # 儲存 ngrok token（範例：NGROK_AUTHTOKEN=xxxx）
┣ .gitignore # 忽略模型、虛擬環境與暫存檔
┗ README.md # 專案說明文件（本檔案）
```


---

## 專案簡介

本專案展示如何透過 AI、API 與前端整合，  
建立一個能預測貸款違約風險的完整原型系統。

### 系統流程概述

1. **ML 模組 (`ml_module/train_model.ipynb`)**  
   使用模擬資料訓練神經網路模型，輸出以下檔案：
   - `LendingClub.keras`：模型主體  
   - `scaler.pkl`：特徵縮放器  
   - `columns.json`：模型欄位定義  

2. **API 模組 (`api/main.py`)**  
   使用 FastAPI 提供 `/predict` 端點，自動偵測 `model/` 或 `ml_module/model/` 資料夾。  
   回傳風險分數與風險等級（Low / Medium / High）。

3. **執行端 (`lending_club_risk_api.ipynb`)**  
   - 載入 `.env` 的 ngrok token  
   - 啟動 FastAPI 本機服務  
   - 建立 ngrok 公開通道  
   - 呼叫 `/predict` 進行測試  
   - 提供 LangChain Tool：`risk_predict_tool`

---

## 環境需求

| 套件 | 用途 |
|------|------|
| fastapi, uvicorn | 後端 API |
| tensorflow, scikit-learn | 模型訓練與推論 |
| joblib, pandas, numpy | 資料處理 |
| langchain, openai | AI 對話整合（選配） |
| python-dotenv | 載入環境變數 |
| pyngrok | 建立公開 API 通道 |

安裝方式：

```bash
pip install -r requirements.txt
```
#
# 前端初版架構
## 開發環境
- Node.js 18.14.2 或以上
- npm 9 或 yarn 1.x
- Vite 5 +

🚀 專案啟動

1️⃣ 安裝依賴
```
cd f2e
npm install
```

2️⃣ 啟動開發伺服器
```
npm run dev
```

3️⃣ 開啟瀏覽器
預設運行於：
```
http://localhost:5173/
```

## API 串接設定

```
# f2e/.env
VITE_API_BASE_URL="https://xxxxxx.ngrok-free.app"
```

## 專案結構
```
f2e/
 ┣ 📂 src/
 ┃ ┣ 📂 components/
 ┃ ┃ ┗ RiskForm.vue        # 主表單元件
 ┃ ┣ 📂 views/
 ┃ ┃ ┣ HomeView.vue        # 表單頁面
 ┃ ┃ ┗ ResultView.vue      # 顯示預測結果
 ┃ ┣ 📂 router/
 ┃ ┃ ┗ index.ts            # Vue Router 設定
 ┃ ┣ App.vue               # 主應用入口
 ┃ ┗ main.ts               # Vue 啟動點
 ┣ .env                    # API 連線設定
 ┣ .env.example            # 範例自己cp
 ┣ package.json
 ┣ tsconfig.json
 ┣ vite.config.ts
 ┗ README.md
```
