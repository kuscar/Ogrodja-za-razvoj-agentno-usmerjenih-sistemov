FROM python:3.12-slim AS builder

WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BACKEND_PORT=8080 \
    BACKEND_HOST=0.0.0.0 \
    GOOGLE_GENAI_USE_VERTEXAI=1

COPY --from=builder /install /usr/local

WORKDIR /app

COPY rag_*.md ./
COPY imgs/ ./imgs/
COPY backend/ ./backend/

RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

WORKDIR /app/backend

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
