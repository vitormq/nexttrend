# Dockerfile - AIOps Locaweb
# Build:  docker build -t aiops-locaweb:latest .
# Run:    docker run -p 8000:8000 aiops-locaweb:latest

FROM python:3.11-slim

# variaveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# diretorio de trabalho
WORKDIR /app

# dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copia e instala dependencias Python antes do codigo
# (aproveita o cache do Docker). Usa o conjunto COMPLETO (pipeline + API + dashboard).
# O requirements.txt enxuto da raiz e usado pelo deploy do painel (Streamlit Cloud).
COPY requirements-full.txt .
RUN pip install --upgrade pip && pip install -r requirements-full.txt

# copia o restante do codigo
COPY . .

# cria pastas de dados e mlruns (montadas como volumes em producao)
RUN mkdir -p data/raw data/processed data/splits models/saved reports mlruns

# expoe a porta da API
EXPOSE 8000

# comando padrao: API FastAPI via uvicorn (arquivo real na raiz do projeto)
CMD ["uvicorn", "etapa9_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]