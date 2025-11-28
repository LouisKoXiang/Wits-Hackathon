# 後端

## 專案結構

```
app/
├── main.py                    # FastAPI 入口
│
├── routes/                    # 所有 API 路由
│   ├── predict.py             # /predict   → 貸款風險預測API
│   ├── shap.py                # /shap      → SHAP
│   ├── chat.py                # /chat      → Chat
│   └── debug_redis.py         # /debug/redis → Redis Key Debug
│
├── ml_module/
│   ├── xgb_loader.py          # 載 XGBoost
│   └── policy_loader.py       # 銀行授信策略
│
├── services/
│   └── loan_service.py        # 編碼 → Scaler → 模型推論 → SHAP → 授信策略
│
├── schemas/
│   └── loan_schema.py         # schema
│
├── memory/
│   └── redis_memory.py        # redis
└── DockerFile                 # docker

```

## 環境變數（Environment Variables）

請於 `.env` 或 Cloud Run 中設定：

| 變數 | 說明 | 預設 |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI 金鑰（必要） | 無 |
| `OPENAI_CHAT_MODEL` | Chat 模型 | gpt-4o-mini |
| `OPENAI_API_BASE` | OpenAI API 端點 | https://api.openai.com/v1 |
| `USE_REDIS` | 0=本地記憶體 / 1=Redis | 0 |
| `REDIS_HOST` | Redis 主機 | localhost |
| `REDIS_PORT` | Redis Port | 6379 |
| `REDIS_PASSWORD` | Redis 密碼 | 無 |
| `ALLOWED_ORIGINS` | CORS | * |

---

# Docker 建置

## 建置 Image

```
docker build -t wits-api .
```

---

## 本機啟動（不使用 Redis）

```
docker run -p 8000:8080 \
  -e USE_REDIS=0 \
  -e OPENAI_API_KEY="your-key-here" \
  wits-api
```


# 推送至 Artifact Registry

```
docker tag wits-api asia-east1-docker.pkg.dev/xxx/xxx/xxxx
docker push asia-east1-docker.pkg.dev/xxx/xxx/xxxx
```

---

# 部署至 Cloud Run

```
gcloud run deploy wits-api \
  --image asia-east1-docker.pkg.dev/xxxxx/xxxxxxx-repo/wits-api \
  --region asia-east1 \
  --allow-unauthenticated
```
---

# 測試 API

Swagger：

```
http://localhost:8000/docs
```

## API Summary

| Endpoint | 方法 | 說明 |
|---------|------|------|
| `POST /predict` | ML inference | 機率 + 風險分類 + SHAP + 授信策略 |
| `POST /chat` | Chat | 多輪對話、意圖分類、引用之前資料 |
| `GET /shap/summary` | SHAP Summary | 模型整體特徵重要性 |
| `GET /debug/redis/keys` | Debug | 列出 Redis Keys |
| `GET /debug/redis/key/{key}` | Debug | 查看特定 Key |


## 系統架構圖

```
┌───────────────────────────────────────────────────────┐
│                      FastAPI (Cloud Run)              │
│                                                       │
│  ┌──────────────────────┐   ┌──────────────────────┐  │
│  │     /chat/chat.py    │   │   /shap/shap.py      │  │
│  └───────────┬──────────┘   └───────────┬──────────┘  │
│              │                          │             │
│  ┌───────────▼───────────┐    ┌─────────▼──────────┐  │
│  │ /predict/predict.py   │    │ /debug/redis.py    │  │
│  └───────────────────────┘    └────────────────────┘  │
└───────────────────────────────────────────────────────┘

                   │
                   │ 若 form_data 存在 → 觸發預測
                   ▼

┌───────────────────────────────────────────────────┐
│                loan_service.py                    │
│                主推論流程：Encoding → ML            │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │Encoding                                     │  │
│  ├─────────────────────────────────────────────┤  │
│  │ Scaler Transform                            │  │
│  ├─────────────────────────────────────────────┤  │
│  │ XGBoost predict_proba（還款機率）             │  │
│  ├─────────────────────────────────────────────┤  │
│  │ SHAP Top K（前五大重要特徵）                   │  │
│  ├─────────────────────────────────────────────┤  │
│  │ 授信策略建議(call policy_loader)              │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘

             ▲                                ▲
             │                                │
             │                                │

┌────────────────────────────┐   ┌──────────────────────────┐
│    ML Module (Artifacts)   │   │       policy_loader.py   │
│    xgb_loader.py           │   │   依 bank_id 載入授信策略  │
│    - model                 │   └──────────────────────────┘
│    - scaler                │
│    - features              │
│    - SHAP summary          │
│    - threshold             │
└────────────────────────────┘

─────────────────────────────────────────────────────────────

                （Chat / predict Memory）
┌─────────────────────────────────────────────────────────┐
│                  Session Memory                         │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ redis_memory.py / Session Manager                 │  │
│  │ - Save messages                                   │  │
│  │ - Save last_prediction                            │  │
│  │ - Save last_form_data                             │  │
│  └───────────────┬───────────────────────────────────┘  │
│                  │                                      │
│          ┌───────▼──────────┐                           │
│          │ Redis / In-Memory│  ← USE_REDIS=1 / 0        │
│          └──────────────────┘                           │
└─────────────────────────────────────────────────────────┘

```