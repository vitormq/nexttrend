import os
import sys

# =============================================================================
# AIOPS LOCAWEB - ETAPA 1: ARQUIVO COMPLETO DE CONFIGURAÇÃO E SETUP
# =============================================================================
# Copie este arquivo inteiro para o VS Code.
# Execute-o para verificar a instalação e criar a estrutura de pastas.
# O conteúdo de cada arquivo de configuração está em blocos de comentário.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. src/config.py - CONTEÚDO COMPLETO
# -----------------------------------------------------------------------------
# Salve o conteúdo abaixo em:  <projeto>/src/config.py
# -----------------------------------------------------------------------------

CONFIG_PY_CONTENT = '''
# src/config.py
# Gerado automaticamente pelo setup da Etapa 1 - AIOps Locaweb

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# COLUNAS ESPERADAS NO DATASET
# ---------------------------------------------------------------------------
COLUNAS = [
    "numero",
    "data_abertura",
    "data_fechamento",
    "data_resolucao",
    "prioridade",
    "categoria",
    "subcategoria",
    "grupo_atendimento",
    "atribuido_a",
    "estado",
    "descricao_curta",
    "descricao",
    "comentarios",
    "tempo_resolucao_horas",
    "dentro_ola",
    "mes_abertura",
    "ano_abertura",
    "dia_semana_abertura",
    "hora_abertura",
    "volume_diario",
    "volume_semanal",
    "volume_mensal",
    "reaberto",
    "numero_reaberturas",
    "tempo_primeira_resposta_horas",
    "satisfacao_cliente",
    "canal_abertura",
    "tipo_incidente",
    "impacto",
    "urgencia",
]

# ---------------------------------------------------------------------------
# PRIORIDADES USADAS NO KPI E NO OLA
# ---------------------------------------------------------------------------
PRIORIDADES_KPI = ["1 - Critica", "2 - Alta", "3 - Media"]
PRIORIDADES_OLA = ["2 - Alta", "3 - Media"]

# ---------------------------------------------------------------------------
# OLA - LIMITE DE HORAS POR PRIORIDADE
# ---------------------------------------------------------------------------
OLA_LIMITE_HORAS = {
    "P1": 4,
    "P2": 4,
    "P3": 12,
    "P4": 24,
    "P5": 96,
}

# ---------------------------------------------------------------------------
# OLA - META MENSAL POR PRIORIDADE
# ---------------------------------------------------------------------------
OLA_META_MENSAL_P2 = 3          # maximo de violacoes P2 por mes
OLA_META_MENSAL_P3 = None       # sem limite definido para P3

# ---------------------------------------------------------------------------
# KPI OLA - FAIXAS DE CUMPRIMENTO (%) POR PRIORIDADE
# ---------------------------------------------------------------------------
KPI_OLA_FAIXAS = {
    "P2": {
        "critico":  (0.00, 0.85),   # abaixo de 85 %
        "atencao":  (0.85, 0.93),   # entre 85 % e 93 %
        "normal":   (0.93, 1.00),   # acima de 93 %
    },
    "P3": {
        "critico":  (0.00, 0.90),   # abaixo de 90 %
        "atencao":  (0.90, 0.95),   # entre 90 % e 95 %
        "normal":   (0.95, 1.00),   # acima de 95 %
    },
}

# ---------------------------------------------------------------------------
# KPI VOLUME - FAIXAS DE DESVIO (%) EM RELACAO A MEDIA MOVEL
# ---------------------------------------------------------------------------
KPI_VOLUME_FAIXAS = {
    "P2": {
        "critico":  0.40,   # desvio > 40 % da media movel
        "atencao":  0.20,   # desvio entre 20 % e 40 %
        "normal":   0.00,   # desvio ate 20 %
    },
    "P3": {
        "critico":  0.50,
        "atencao":  0.25,
        "normal":   0.00,
    },
}

# ---------------------------------------------------------------------------
# SPLIT DE DADOS - PROPORÇÕES
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, (
    "Os ratios de split devem somar 1.0"
)

# ---------------------------------------------------------------------------
# CAMINHOS BASE DO PROJETO
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"
DATA_SPLIT_DIR = BASE_DIR / "data" / "split"
MLRUNS_DIR     = BASE_DIR / "mlruns"

# ---------------------------------------------------------------------------
# MLFLOW
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}",
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "aiops-locaweb")
'''

print("=" * 70)
print("AIOPS LOCAWEB - ETAPA 1")
print("=" * 70)
print("\n[1/6] Conteúdo de src/config.py carregado em memória.\n")


# -----------------------------------------------------------------------------
# 2. requirements.txt
# -----------------------------------------------------------------------------
# Salve o conteúdo abaixo em:  <projeto>/requirements.txt
# -----------------------------------------------------------------------------

REQUIREMENTS_TXT = """
# requirements.txt - AIOps Locaweb
# Gerado automaticamente pela Etapa 1

# dados e numerica
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0

# machine learning
scikit-learn==1.5.0
xgboost==2.0.3
lightgbm==4.3.0
shap==0.45.0

# mlflow
mlflow==2.13.0
mlflow-skinny==2.13.0

# api
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1

# dashboard
streamlit==1.35.0
plotly==5.22.0
altair==5.3.0

# banco / cache
sqlalchemy==2.0.30
redis==5.0.4

# utilitarios
python-dotenv==1.0.1
loguru==0.7.2
tqdm==4.66.4
joblib==1.4.2
httpx==0.27.0
tenacity==8.3.0

# qualidade de codigo
pytest==8.2.0
pytest-cov==5.0.0
ruff==0.4.4
mypy==1.10.0
"""

print("[2/6] requirements.txt:\n")
print(REQUIREMENTS_TXT)


# -----------------------------------------------------------------------------
# 3. Dockerfile
# -----------------------------------------------------------------------------
# Salve o conteúdo abaixo em:  <projeto>/Dockerfile
# -----------------------------------------------------------------------------

DOCKERFILE = """
# Dockerfile - AIOps Locaweb
# Build:  docker build -t aiops-locaweb:latest .
# Run:    docker run -p 8000:8000 aiops-locaweb:latest

FROM python:3.11-slim

# variaveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# diretorio de trabalho
WORKDIR /app

# dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential \\
        libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# copia e instala dependencias Python antes do codigo
# (aproveita o cache do Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copia o restante do codigo
COPY . .

# cria pastas de dados e mlruns (montadas como volumes em producao)
RUN mkdir -p data/raw data/processed data/split mlruns

# expoe a porta da API
EXPOSE 8000

# comando padrao: API FastAPI via uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""

print("[3/6] Dockerfile:\n")
print(DOCKERFILE)


# -----------------------------------------------------------------------------
# 4. docker-compose.yml
# -----------------------------------------------------------------------------
# Salve o conteúdo abaixo em:  <projeto>/docker-compose.yml
# -----------------------------------------------------------------------------

DOCKER_COMPOSE_YML = """
# docker-compose.yml - AIOps Locaweb
# Uso: docker compose up --build

version: "3.9"

services:

  # API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aiops_api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_EXPERIMENT_NAME=aiops-locaweb
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./mlruns:/app/mlruns
    depends_on:
      - mlflow
      - redis
    networks:
      - aiops_net

  # DASHBOARD
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: aiops_dashboard
    restart: unless-stopped
    command: >
      streamlit run src/dashboard/app.py
        --server.port=8501
        --server.address=0.0.0.0
        --server.headless=true
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://api:8000
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/app/data
    depends_on:
      - api
    networks:
      - aiops_net

  # MLFLOW
  mlflow:
    image: python:3.11-slim
    container_name: aiops_mlflow
    restart: unless-stopped
    command: >
      bash -c "pip install mlflow==2.13.0 --quiet &&
               mlflow server
                 --backend-store-uri sqlite:////mlruns/mlflow.db
                 --default-artifact-root /mlruns/artifacts
                 --host 0.0.0.0
                 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    networks:
      - aiops_net

  # REDIS
  redis:
    image: redis:7-alpine
    container_name: aiops_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    networks:
      - aiops_net

networks:
  aiops_net:
    driver: bridge
"""

print("[4/6] docker-compose.yml:\n")
print(DOCKER_COMPOSE_YML)


# -----------------------------------------------------------------------------
# 5. CRIAÇÃO DA ESTRUTURA DE PASTAS
# -----------------------------------------------------------------------------

ESTRUTURA = [
    "src",
    "src/api",
    "src/dashboard",
    "src/models",
    "src/features",
    "src/utils",
    "data/raw",
    "data/processed",
    "data/split",
    "mlruns/artifacts",
    "notebooks",
    "tests",
    "scripts",
]

print("[5/6] Criando estrutura de pastas do projeto...\n")

BASE_PROJETO = os.path.join(os.path.expanduser("~"), "aiops_locaweb")

criadas = []
ja_existiam = []

for pasta in ESTRUTURA:
    caminho = os.path.join(BASE_PROJETO, pasta)
    if not os.path.exists(caminho):
        os.makedirs(caminho, exist_ok=True)
        criadas.append(caminho)
    else:
        ja_existiam.append(caminho)

    # cria __init__.py em pastas src/*
    if pasta.startswith("src"):
        init_file = os.path.join(caminho, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# AIOps Locaweb\n")

# grava src/config.py com o conteúdo real
config_path = os.path.join(BASE_PROJETO, "src", "config.py")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(CONFIG_PY_CONTENT.strip())

# grava requirements.txt
req_path = os.path.join(BASE_PROJETO, "requirements.txt")
with open(req_path, "w", encoding="utf-8") as f:
    f.write(REQUIREMENTS_TXT.strip())

# grava Dockerfile
docker_path = os.path.join(BASE_PROJETO, "Dockerfile")
with open(docker_path, "w", encoding="utf-8") as f:
    f.write(DOCKERFILE.strip())

# grava docker-compose.yml
compose_path = os.path.join(BASE_PROJETO, "docker-compose.yml")
with open(compose_path, "w", encoding="utf-8") as f:
    f.write(DOCKER_COMPOSE_YML.strip())

# grava .env de exemplo
env_path = os.path.join(BASE_PROJETO, ".env.example")
with open(env_path, "w", encoding="utf-8") as f:
    f.write(
        "MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db\n"
        "MLFLOW_EXPERIMENT_NAME=aiops-locaweb\n"
        "REDIS_URL=redis://localhost:6379/0\n"
        "API_BASE_URL=http://localhost:8000\n"
    )

print(f"  Pasta raiz do projeto : {BASE_PROJETO}")
print(f"  Pastas criadas        : {len(criadas)}")
print(f"  Pastas já existentes  : {len(ja_existiam)}")
print(f"  Arquivos gravados     : src/config.py, requirements.txt,")
print(f"                          Dockerfile, docker-compose.yml, .env.example")

# exibe árvore de diretórios criada
print("\n  Estrutura gerada:\n")
for pasta in ESTRUTURA:
    nivel = pasta.count("/")
    nome  = pasta.split("/")[-1]
    indent = "    " + "  " * nivel
    print(f"{indent}{'--- ' if nivel else ''}{nome}/")


# -----------------------------------------------------------------------------
# 6. VERIFICAÇÃO DE INSTALAÇÃO DAS LIBS
# -----------------------------------------------------------------------------

print("\n[6/6] Verificando instalação das bibliotecas principais...\n")

LIBS_PARA_VERIFICAR = [
    ("pandas",       "pd"),
    ("numpy",        "np"),
    ("scipy",        None),
    ("sklearn",      None),
    ("xgboost",      None),
    ("lightgbm",     None),
    ("shap",         None),
    ("mlflow",       None),
    ("fastapi",      None),
    ("uvicorn",      None),
    ("pydantic",     None),
    ("streamlit",    None),
    ("plotly",       None),
    ("altair",       None),
    ("sqlalchemy",   None),
    ("redis",        None),
    ("dotenv",       None),
    ("loguru",       None),
    ("tqdm",         None),
    ("joblib",       None),
    ("httpx",        None),
    ("tenacity",     None),
]

instaladas   = []
faltando     = []

col_w = 16

print(f"  {'BIBLIOTECA':<{col_w}} {'STATUS':<12} VERSÃO")
print(f"  {'-'*col_w} {'-'*12} {'-'*10}")

for lib, alias in LIBS_PARA_VERIFICAR:
    try:
        mod = __import__(lib)
        versao = getattr(mod, "__version__", "n/d")
        print(f"  {lib:<{col_w}} {'OK':<12} {versao}")
        instaladas.append(lib)
    except ImportError:
        print(f"  {lib:<{col_w}} {'FALTANDO':<12} -")
        faltando.append(lib)

print(f"\n  Instaladas : {len(instaladas)}")
print(f"  Faltando   : {len(faltando)}")

if faltando:
    print("\n  Para instalar as libs faltando, execute:")
    print(f"    pip install {' '.join(faltando)}\n")
    print(f"  Ou instale tudo de uma vez (recomendado):")
    print(f"    pip install -r {req_path}\n")
else:
    print("\n  Todas as bibliotecas estão instaladas. \n")


# -----------------------------------------------------------------------------
# RESUMO FINAL
# -----------------------------------------------------------------------------

print("=" * 70)
print("ETAPA 1 - RESUMO FINAL")
print("=" * 70)
print(f"""
  Projeto criado em  : {BASE_PROJETO}

  Arquivos prontos:
    - src/config.py          - constantes de negócio e caminhos
    - requirements.txt       - dependências Python com versões fixas
    - Dockerfile             - imagem da API / dashboard
    - docker-compose.yml     - API + Dashboard + MLflow + Redis
    - .env.example           - variáveis de ambiente de exemplo

  Próximos passos:
    1. cd {BASE_PROJETO}
    2. python -m venv .venv && source .venv/bin/activate  (Linux/Mac)
       ou .venv\\Scripts\\activate  (Windows)
    3. pip install -r requirements.txt
    4. cp .env.example .env  (e ajuste conforme necessário)
    5. docker compose up --build   (opcional, sobe todos os serviços)

  Etapa 2:  ingestão e validação dos dados (data/raw -> data/processed)
""")
print("=" * 70)
