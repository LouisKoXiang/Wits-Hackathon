FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_PROGRESS_BAR=off
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV http_proxy=
ENV https_proxy=
ENV ALL_PROXY=
ENV all_proxy=
ENV NO_PROXY=
ENV no_proxy=

WORKDIR /app

# ---- FIX apt errors ----
RUN rm -rf /var/lib/apt/lists/*

# ---- Install system deps ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---- Install Python deps ----
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy source ----
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
