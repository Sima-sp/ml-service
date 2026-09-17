# Serviço de IA do SIMA-SP — imagem de produção (só o necessário para servir).
# Build:  docker build -t sima-ml-service .
# Rodar:  docker run --rm -p 5000:5000 sima-ml-service
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: paralelismo do LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-servico.txt .
RUN pip install -r requirements-servico.txt

# src/ é necessário: features.py é o MESMO código do treino e o joblib referencia classes dele
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

RUN useradd --create-home --uid 10001 sima && chown -R sima /app
USER sima

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
