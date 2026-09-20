# src/config.py
# Gerado automaticamente pelo setup da Etapa 1 — AIOps Locaweb

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
# OLA — LIMITE DE HORAS POR PRIORIDADE
# ---------------------------------------------------------------------------
OLA_LIMITE_HORAS = {
    "P1": 4,
    "P2": 4,
    "P3": 12,
    "P4": 24,
    "P5": 96,
}

# ---------------------------------------------------------------------------
# OLA — META MENSAL POR PRIORIDADE
# ---------------------------------------------------------------------------
OLA_META_MENSAL_P2 = 3          # maximo de violacoes P2 por mes
OLA_META_MENSAL_P3 = None       # sem limite definido para P3

# ---------------------------------------------------------------------------
# KPI OLA — FAIXAS DE CUMPRIMENTO (%) POR PRIORIDADE
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
# KPI VOLUME — FAIXAS DE DESVIO (%) EM RELACAO A MEDIA MOVEL
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
# SPLIT DE DADOS — PROPORÇÕES
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