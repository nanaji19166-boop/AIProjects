FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-core fonts-noto-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY telugusetu/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY telugusetu/ .
RUN mkdir -p data

EXPOSE 8000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
