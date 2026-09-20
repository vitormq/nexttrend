# AIOps Locaweb - Previsão de Incidentes e Atingimento de KPIs

Challenge FIAP x Locaweb - Equipe **Python Rangers** (solução *NextTrend*).
Ciência de dados sobre incidentes de ITSM para **antecipar volume**, **prever risco de violação de OLA** e **projetar o atingimento dos KPIs** - com recomendações operacionais.

## Modelos

| Modelo | Técnica | Entrega |
|---|---|---|
| **Volume** | Prophet | Previsão diária D+1 / D+7 por prioridade (P2, P3) |
| **Risco de OLA** | XGBoost (42 features *point-in-time*, sem leakage) | Probabilidade de violação por chamado + ranking |
| **Atingimento de KPI** | Monte Carlo sobre as faixas do dicionário | P(KPI saudável >= 100%) por KPI/prioridade, com incerteza |
| **Explicabilidade** | SHAP | Fatores de risco + mapa de riscos por grupo |
| **Clusterização** | KMeans / TF-IDF / DBSCAN | Perfis, causas recorrentes e ICs barulhentos |

**Definições dos KPIs** (Dicionário de Dados, confirmadas nos dados de 2025):
- **KPI-OLA** = nº de incidentes com OLA violado no ano (`kpi_violado_bin`), por prioridade.
- **KPI-Volume** = nº de incidentes tratados no ano (`entrou_para_kpi`), por prioridade.
- **Saudável** = atingimento >= 100% (contagem menor -> atingimento maior). Limites de OLA: P1/P2 = 4h, P3 = 12h, P4 = 24h, P5 = 96h.

## Pipeline (etapas 1-10)

| # | Arquivo | Papel |
|---|---|---|
| 1 | `etapa1_setup.py` | Setup do ambiente |
| 2 | `etapa2_preprocessamento.py` | Ingestão do `.xlsx`, limpeza, flags de KPI/OLA, splits temporais |
| 3 | `etapa3_eda.py` | Análise exploratória (distribuições, sazonalidade, grupos de risco) |
| 4 | `etapa4_features.py` | 37 features **point-in-time** (histórico de IC/grupo só com encerrados-antes) |
| 5 | `etapa5_modelos.py` | Prophet (volume) + XGBoost (risco de OLA) |
| 6 | `etapa6_clusterizacao.py` | KMeans + TF-IDF + DBSCAN |
| 7 | `etapa7_explicabilidade.py` | SHAP + mapa de riscos por grupo |
| 8 | `etapa8_kpi_atingimento.py` | Projeção Monte Carlo do atingimento dos KPIs |
| 9 | `etapa9_api.py` | API REST (FastAPI) |
| 10 | `etapa10_dashboard.py` | Dashboard (Streamlit) - 9 módulos, inclui Fila de Risco e Atingimento de KPI |

**Ordem de execução** (via `scripts/run_pipeline.py`): 2 -> 4 -> 5 -> 7 -> 8 -> 6 (clusterização, opcional). A EDA (etapa 3) roda à parte. "Opcional" significa que, se a clusterização falhar, o pipeline principal (modelos e KPI) não é interrompido.

## Como rodar

### Docker (recomendado - reprodutível)

```bash
# 1. coloque o dataset em data/raw/LW-DATASET.xlsx
# 2. gere dados, modelos e relatórios (one-shot)
docker compose run --rm pipeline
# 3. suba API + dashboard
docker compose up api dashboard
```

- Dashboard: http://localhost:8501
- API (Swagger): http://localhost:8000/docs
- MLflow (opcional): `docker compose up mlflow` -> http://localhost:5000

O MLflow usa o *file store* local (`./mlruns`), montado como volume - sem servidor obrigatório e sem condição de corrida.

### Local (venv)

```bash
python -m venv venv && source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py
streamlit run etapa10_dashboard.py
```

> No Windows, rode com `PYTHONIOENCODING=utf-8` para evitar erro de encoding no console.

## Notas técnicas

- **Sem data leakage**: o modelo de OLA usa apenas informação conhecida **na abertura** do incidente. `Solução`, `Duração` e datas de resolução ficam de fora; features históricas de violação usam somente incidentes já **encerrados antes** da abertura.
- **Alvo autoritativo**: `KPI Violado?` vem do motor de SLA do ITSM e **não** é reconstruível de `Duração > limite` (proxy fraco). Usa-se o campo autoritativo.
- **Avaliação honesta**: as métricas usam validação cruzada (*out-of-fold*), não in-sample. O AUC do risco de OLA é **0,79** (o 0,93 in-sample infla por avaliar o modelo nos próprios dados de treino).
- **Pré-processamento validado**: a ablação em `scripts/ablacao_preprocessamento.py` (resultados em `reports/ablacao_preprocessamento.json`) mostra que normalização, capping de outliers e log-transform **não melhoram o AUC** (todos ~0,795) - coerente com um modelo baseado em árvore, invariante à escala. Não há nulos numéricos; nulos de texto viram categoria própria.
- **Artefatos**: modelos em `models/saved/`, relatórios (SHAP, mapa de riscos, clusters) em `reports/`, experimentos em `mlruns/`.
