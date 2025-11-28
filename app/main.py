from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.routes import shap, predict, chat, debug_redis

# Application
app = FastAPI(
    title="Lending Club Risk API (Cloud Run)",
    version=os.getenv("API_VERSION", "1.0.0"),
)

# CORS
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

allow_credentials = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(shap.router)
app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(debug_redis.router)

@app.get("/")
def root():
    return {"status": "ok"}

