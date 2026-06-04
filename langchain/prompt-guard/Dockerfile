FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/guard/.cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10002 guard

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip wheel \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY --chown=guard:guard server.py .

USER guard
EXPOSE 8088

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8088"]
