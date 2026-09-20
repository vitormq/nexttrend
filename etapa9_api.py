"""
etapa9_api.py - API REST (FastAPI) que serve as previsões do projeto:
risco de violação de OLA e previsão de volume de incidentes.

Roda com: uvicorn etapa9_api:app --reload --port 8000
Documentação interativa em /docs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Caminhos
BASE_DIR = Path(__file__).resolve().parent
MDL_DIR  = BASE_DIR / "models" / "saved"

# Limites OLA por prioridade (em horas)
OLA_LIMITES = {
    "1 - Crítica":    4.0,
    "2 - Alta":       4.0,
    "3 - Média":     12.0,
    "4 - Baixa":     24.0,
    "5 - Muito Baixa": 96.0,
}

# Metas de volume mensal (anual / 12)
KPI_VOLUME_MENSAL = {
    "2 - Alta":  {"meta_100": 6168 / 12, "limite_0": 6336 / 12},
    "3 - Média": {"meta_100": 22524 / 12, "limite_0": 24276 / 12},
}


# Schemas pydantic

class IncidenteRequest(BaseModel):
    numero:            str            = Field(...,  example="INC8654075")
    prioridade:        str            = Field(...,  example="2 - Alta")
    grupo_designado:   str            = Field(...,  example="Team09")
    item_configuracao: Optional[str]  = Field(None, example="IC00055")
    aberto_por:        str            = Field(...,  example="Manual")
    solucao:           Optional[str]  = Field(None, example="Contorno")
    hora_abertura:     int            = Field(...,  ge=0,  le=23, example=14)
    dia_semana:        int            = Field(...,  ge=0,  le=6,  example=1)
    duracao_segundos:  Optional[float]= Field(None, example=0.0)
    tem_incidente_pai: bool           = Field(False, example=False)
    categoria:         Optional[str]  = Field(None, example="cat85")
    produto:           Optional[str]  = Field(None, example="lrel")

    class Config:
        json_schema_extra = {"example": {
            "numero": "INC9999999", "prioridade": "2 - Alta",
            "grupo_designado": "Team09", "item_configuracao": "IC00055",
            "aberto_por": "Manual", "solucao": None,
            "hora_abertura": 14, "dia_semana": 1,
            "duracao_segundos": 0.0, "tem_incidente_pai": False,
        }}


class FatorRisco(BaseModel):
    feature:    str
    shap_valor: float
    descricao:  str


class OLARiskResponse(BaseModel):
    numero:           str
    prioridade:       str
    prob_violacao:    float
    nivel_risco:      str
    ola_limite_horas: float
    fatores_risco:    List[FatorRisco]
    recomendacao:     str
    timestamp:        datetime


class PrevisaoVolumeRequest(BaseModel):
    prioridade: str = Field(..., example="2 - Alta")
    horizonte:  int = Field(7,   ge=1, le=30)

    class Config:
        json_schema_extra = {"example": {"prioridade": "2 - Alta", "horizonte": 7}}


class PrevisaoDia(BaseModel):
    data:            str
    volume_previsto: float
    limite_inferior: float
    limite_superior: float


class VolumeResponse(BaseModel):
    prioridade:     str
    horizonte_dias: int
    previsoes:      List[PrevisaoDia]
    total_previsto: float
    alerta_kpi:     Optional[str]
    timestamp:      datetime


class HealthResponse(BaseModel):
    status:             str
    modelos_carregados: List[str]
    versao:             str
    timestamp:          datetime


# CARREGAMENTO DE MODELOS (uma vez na inicialização)

def _carregar_modelo(nome: str):
    caminho = MDL_DIR / nome
    if caminho.exists():
        try:
            return joblib.load(caminho)
        except Exception as e:
            print(f" Erro ao carregar {nome}: {e}")
    return None

_pipeline     = _carregar_modelo("ola_predictor_pipeline.pkl")
_feature_cols = _carregar_modelo("ola_feature_cols.pkl")
_explainer    = _carregar_modelo("shap_explainer.pkl")
_feat_names   = _carregar_modelo("shap_feature_names.pkl")

_MODELOS_OLA_OK = all([_pipeline is not None, _feature_cols is not None])


def _carregar_base_features():
    """Base P2/P3 KPI com as 42 features (Etapa 4) - usada como template point-in-time."""
    p = BASE_DIR / "data" / "processed" / "incidents_features.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return df[df["entrou_para_kpi"] & df["prioridade_cod"].isin(["P2", "P3"])].copy()
    return None

_base_feats = _carregar_base_features()


# Helpers

def _nivel_risco(prob: float) -> str:
    if prob >= 0.80: return "CRÍTICO"
    if prob >= 0.60: return "ALTO"
    if prob >= 0.35: return "MÉDIO"
    return "BAIXO"

def _recomendacao(nivel: str, prioridade: str) -> str:
    msgs = {
        "CRÍTICO": f"Acionar escalação imediata. {prioridade} com >80% de chance de violar OLA.",
        "ALTO":    "Monitorar ativamente. Verificar disponibilidade do grupo responsável.",
        "MÉDIO":   "Acompanhar evolução. Garantir que o incidente está em tratamento.",
        "BAIXO":   "Risco controlado. Manter atendimento padrão conforme SLA.",
    }
    return msgs.get(nivel, "Sem recomendação disponível.")

def _montar_features(req: IncidenteRequest) -> pd.DataFrame:
    """
    Monta o vetor de 42 features point-in-time do modelo (Etapa 4), SEM leakage.
    As features históricas (IC/grupo, carga) vêm do incidente mais recente do IC
    na base; os campos conhecidos na abertura são sobrescritos pela requisição.
    Campos leaky da requisição (solucao, duracao_segundos) são IGNORADOS.
    """
    try:
        prio_num = int(str(req.prioridade).split(" - ")[0])
    except Exception:
        prio_num = 3

    row = {}
    if _base_feats is not None and _feature_cols is not None:
        base = _base_feats
        if req.item_configuracao:
            sub = base[base["Item de configuração"] == req.item_configuracao]
            if not sub.empty:
                base = sub
        tmpl = base.sort_values("dt_aberto").iloc[-1]
        row = {c: tmpl[c] for c in _feature_cols if c in tmpl.index}

    row.update({
        "Prioridade": req.prioridade, "Grupo designado": req.grupo_designado,
        "Item de configuração": req.item_configuracao or "DESCONHECIDO",
        "Aberto por": req.aberto_por, "hora_abertura": req.hora_abertura,
        "dia_semana": req.dia_semana, "prioridade_num": prio_num,
        "is_weekend": int(req.dia_semana >= 5),
        "is_horario_comercial": int(8 <= req.hora_abertura <= 18),
        "is_madrugada": int(0 <= req.hora_abertura <= 5),
        "hora_x_prioridade": req.hora_abertura * prio_num,
        "ola_limite_horas": {1: 4, 2: 4, 3: 12}.get(prio_num, 24),
    })
    df = pd.DataFrame([row])
    for col in (_feature_cols or []):
        if col not in df.columns:
            df[col] = 0
    return df[_feature_cols] if _feature_cols else df

def _carregar_prophet(prioridade: str):
    """Carrega modelo Prophet para a prioridade informada."""
    fname = (f"prophet_{prioridade.split(' - ')[0].strip()}_"
             f"{prioridade.split(' - ')[-1].strip()}.pkl")
    modelo = _carregar_modelo(fname)
    if modelo is None:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo Prophet para '{prioridade}' não encontrado. Execute a Etapa 5."
        )
    return modelo


# Aplicação fastapi

app = FastAPI(
    title="Locaweb AIOps API",
    description=(
        "API de previsão de incidentes operacionais e risco de violação de OLA. "
        "Solução agnóstica de cloud - roda em Docker em qualquer ambiente."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "projeto":   "Locaweb AIOps - Challenge FIAP",
        "versao":    "1.0.0",
        "docs":      "/docs",
        "endpoints": ["/health", "/ola/prever", "/ola/prever/lote",
                      "/volume/prever", "/volume/tendencia"],
    })


# Health Check

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Verifica status da API e quais modelos estão carregados."""
    modelos_esperados = [
        "ola_predictor_pipeline.pkl",
        "prophet_2_Alta.pkl",
        "prophet_3_Média.pkl",
    ]
    carregados = [m for m in modelos_esperados if (MDL_DIR / m).exists()]
    return HealthResponse(
        status="ok" if len(carregados) == len(modelos_esperados) else "degraded",
        modelos_carregados=carregados,
        versao="1.0.0",
        timestamp=datetime.now(timezone.utc),
    )


# Risco de OLA

@app.post(
    "/ola/prever",
    response_model=OLARiskResponse,
    tags=["Risco de OLA"],
    summary="Prevê risco de violação de OLA para um incidente",
)
def prever_risco_ola(req: IncidenteRequest):
    """
    Recebe dados de um incidente e retorna:
    - Probabilidade de violação de OLA (0 a 1)
    - Nível de risco: BAIXO / MÉDIO / ALTO / CRÍTICO
    - Top 5 fatores que mais influenciam o risco (SHAP)
    - Recomendação operacional textual
    """
    if not _MODELOS_OLA_OK:
        raise HTTPException(503, "Modelos OLA não disponíveis. Execute a Etapa 5.")

    df = _montar_features(req)

    try:
        prob = float(_pipeline.predict_proba(df[_feature_cols])[0][1])
    except Exception as e:
        raise HTTPException(422, f"Erro na predição: {e}")

    nivel = _nivel_risco(prob)

    # SHAP local (opcional - não bloqueia se falhar)
    fatores = []
    if _explainer is not None and _feat_names is not None:
        try:
            preprocessor = _pipeline[:-1]
            X_t          = preprocessor.transform(df[_feature_cols])
            sv           = _explainer.shap_values(X_t)
            sv           = sv[1][0] if isinstance(sv, list) else sv[0]
            top_idx      = np.argsort(np.abs(sv))[::-1][:5]
            desc_map = {
                "hora_abertura":      "Hora de abertura do incidente",
                "duracao_horas":      "Tempo decorrido desde abertura",
                "dia_semana":         "Dia da semana (0=Seg, 6=Dom)",
                "e_contorno":         "Solução temporária aplicada",
                "tem_incidente_pai":  "É sub-incidente de outro",
                "qtd_filhos":         "Quantidade de sub-incidentes",
            }
            for i in top_idx:
                if i < len(_feat_names):
                    fn = _feat_names[i]
                    fatores.append(FatorRisco(
                        feature=fn,
                        shap_valor=round(float(sv[i]), 4),
                        descricao=desc_map.get(fn, fn),
                    ))
        except Exception:
            pass

    return OLARiskResponse(
        numero=req.numero,
        prioridade=req.prioridade,
        prob_violacao=round(prob, 4),
        nivel_risco=nivel,
        ola_limite_horas=OLA_LIMITES.get(req.prioridade, 4.0),
        fatores_risco=fatores,
        recomendacao=_recomendacao(nivel, req.prioridade),
        timestamp=datetime.now(timezone.utc),
    )


@app.post(
    "/ola/prever/lote",
    tags=["Risco de OLA"],
    summary="Predição em lote (até 500 incidentes)",
)
def prever_lote(incidentes: List[IncidenteRequest]):
    """Aceita lista de incidentes e retorna risco de OLA para cada um."""
    if not _MODELOS_OLA_OK:
        raise HTTPException(503, "Modelos OLA não disponíveis.")
    if len(incidentes) > 500:
        raise HTTPException(400, "Máximo de 500 incidentes por lote.")

    resultados = []
    for inc in incidentes:
        try:
            resultados.append(prever_risco_ola(inc))
        except Exception as e:
            resultados.append({"numero": inc.numero, "erro": str(e)})
    return resultados


# Previsão de Volume

@app.post(
    "/volume/prever",
    response_model=VolumeResponse,
    tags=["Previsão de Volume"],
    summary="Prevê volume de incidentes D+1 até D+N",
)
def prever_volume(req: PrevisaoVolumeRequest):
    """
    Usa Prophet para prever o volume diário de incidentes por prioridade.
    Retorna previsão central + intervalo de confiança + alerta de KPI.
    """
    modelo = _carregar_prophet(req.prioridade)

    futuro   = modelo.make_future_dataframe(
        periods=req.horizonte, freq="D", include_history=False
    )
    # Preenche regressores extras do Prophet (ex.: ic_barulhento_medio_dia) para o horizonte
    for reg in list(getattr(modelo, "extra_regressors", {}).keys()):
        futuro[reg] = (futuro["ds"].dt.dayofweek.isin([5, 6]).astype(int)
                       if reg == "is_weekend" else 0.0)
    forecast = modelo.predict(futuro)

    previsoes = [
        PrevisaoDia(
            data=str(row["ds"].date()),
            volume_previsto=max(0.0, round(float(row["yhat"]),       1)),
            limite_inferior=max(0.0, round(float(row["yhat_lower"]), 1)),
            limite_superior=max(0.0, round(float(row["yhat_upper"]), 1)),
        )
        for _, row in forecast.iterrows()
    ]

    total  = sum(p.volume_previsto for p in previsoes)
    alerta = None

    if req.prioridade in KPI_VOLUME_MENSAL:
        kpi   = KPI_VOLUME_MENSAL[req.prioridade]
        vol_m = total * (30 / req.horizonte)
        if vol_m > kpi["limite_0"]:
            alerta = (
                f"ALERTA: volume projetado ({vol_m:.0f}/mês) "
                f"ultrapassa o limite 0% de KPI para {req.prioridade}."
            )
        elif vol_m > kpi["meta_100"]:
            alerta = (
                f"ATENÇÃO: volume projetado ({vol_m:.0f}/mês) "
                f"acima da meta 100% para {req.prioridade}."
            )

    return VolumeResponse(
        prioridade=req.prioridade,
        horizonte_dias=req.horizonte,
        previsoes=previsoes,
        total_previsto=round(total, 1),
        alerta_kpi=alerta,
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/volume/tendencia",
    tags=["Previsão de Volume"],
    summary="Tendência histórica de volume por prioridade",
)
def tendencia_historica(prioridade: str = "2 - Alta", ultimos_dias: int = 90):
    """Retorna série histórica de volume diário de incidentes KPI."""
    caminho = BASE_DIR / "data" / "processed" / "incidents_clean.parquet"
    if not caminho.exists():
        raise HTTPException(404, "Dataset processado não encontrado. Execute a Etapa 2.")

    df = pd.read_parquet(caminho)
    df["dt_aberto"] = pd.to_datetime(df["Aberto"], errors="coerce")
    df = df[
        (df["Prioridade"] == prioridade) &
        (df["entrou_para_kpi"] == True)
    ].copy()

    corte = pd.Timestamp.today() - pd.Timedelta(days=ultimos_dias)
    df    = df[df["dt_aberto"] >= corte]

    serie = (
        df.groupby(df["dt_aberto"].dt.date)
        .size()
        .reset_index()
        .rename(columns={"dt_aberto": "data", 0: "volume"})
    )
    serie["data"] = serie["data"].astype(str)
    return serie.to_dict("records")
