"""
etapa5_modelos.py - Modelagem preditiva do projeto.
Prophet para prever o volume de incidentes e XGBoost para o risco de violar OLA.
Lê os splits de data/splits/ e salva os modelos em models/saved/.

Roda com: python etapa5_modelos.py
"""

# Imports
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from prophet import Prophet

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats as _sps

import mlflow
import mlflow.sklearn

# Constantes
MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/splits")

# Caminho absoluto convertido para file URI (file:///C:/.../mlruns no Windows).
# "file://./mlruns" era interpretado como host de rede "." -> UNC \\.\mlruns -> WinError 2.
# Em Docker, MLFLOW_TRACKING_URI (ex.: http://mlflow:5000) tem prioridade.
MLRUNS_DIR = Path("mlruns").resolve()
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI") or MLRUNS_DIR.as_uri()
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "locaweb-aiops")

# Features completas produzidas pela Etapa 4
FEATURE_COLS = [
    # Temporais
    "hora_abertura",
    "dia_semana",
    "dia_mes",
    "mes",
    "trimestre",
    "is_weekend",
    "is_horario_comercial",
    "is_madrugada",
    "is_fim_de_mes",
    # Prioridade / SLA
    "prioridade_num",
    "ola_limite_horas",
    # Fila / roteamento
    "n_reatribuicoes",
    "trocou_grupo",
    "Grupo designado",
    "Item de configuração",
    "Aberto por",
    "Prioridade",
    # Histórico do item de configuração (IC)
    "ic_vol_30d",
    "ic_vol_7d",
    "ic_taxa_violacao_30d",
    "ic_taxa_violacao_7d",
    "ic_media_duracao_30d",
    "ic_barulhento_medio_dia",
    # Histórico do grupo
    "grupo_vol_30d",
    "grupo_vol_7d",
    "grupo_taxa_violacao_30d",
    "grupo_taxa_violacao_7d",
    "grupo_media_duracao_30d",
    # Carga operacional
    "carga_grupo_abertos",
    "carga_grupo_atrasados",
    "carga_ic_abertos",
    # Descrição / texto (TF-IDF reduzido)
    "desc_n_palavras",
    "desc_n_chars",
    "desc_tem_erro",
    "desc_tem_lento",
    "desc_tem_fora",
    "desc_tem_critico",
    "desc_tem_cliente",
    # Interações
    "alta_x_barulhento",
    "alta_x_carga_grupo",
    "hora_x_prioridade",
    # Acumuladas do dia
    "vol_dia_ate_agora",
    "violacoes_dia_ate_agora",
    "taxa_violacao_dia_ate_agora",
    # Flags
    "entrou_para_kpi",
]

# Features categóricas (OrdinalEncoder). "Solução" foi REMOVIDA - é leakage
# (só se conhece após a resolução; não existe no momento do score em tempo real).
CATEGORICAS_FEATURES = [
    "Prioridade",
    "Grupo designado",
    "Item de configuração",
    "Aberto por",
]


# Modelo 1 - prophet (volume)

def preparar_serie_temporal(df: pd.DataFrame, prioridade: str) -> pd.DataFrame:
    """
    Filtra o DataFrame pela prioridade e retorna uma série diária
    no formato Prophet: colunas 'ds' (datetime) e 'y' (volume).
    Preenche datas faltantes com 0.
    """
    mask = (df["Prioridade"] == prioridade) & (df["entrou_para_kpi"] == True)
    df_filtrado = df.loc[mask].copy()

    if df_filtrado.empty:
        print(f"  [AVISO] Nenhum dado para prioridade '{prioridade}' após filtro.")
        return pd.DataFrame(columns=["ds", "y"])

    # Garantir que dt_aberto seja datetime
    if "dt_aberto" in df_filtrado.columns:
        df_filtrado["dt_aberto"] = pd.to_datetime(df_filtrado["dt_aberto"], errors="coerce")
        df_filtrado["data"] = df_filtrado["dt_aberto"].dt.date
    elif "Aberto" in df_filtrado.columns:
        df_filtrado["Aberto"] = pd.to_datetime(df_filtrado["Aberto"], errors="coerce")
        df_filtrado["data"] = df_filtrado["Aberto"].dt.date
    else:
        # Tentar coluna genérica de data
        col_data = [c for c in df_filtrado.columns if "aberto" in c.lower() or "data" in c.lower()]
        if col_data:
            df_filtrado["data"] = pd.to_datetime(df_filtrado[col_data[0]], errors="coerce").dt.date
        else:
            raise ValueError("Coluna de data de abertura não encontrada no DataFrame.")

    # Agrupar por data -> contagem diária
    serie = (
        df_filtrado.groupby("data")
        .size()
        .reset_index(name="y")
    )
    serie["ds"] = pd.to_datetime(serie["data"])
    serie = serie[["ds", "y"]].sort_values("ds").reset_index(drop=True)

    # Preencher datas faltantes com 0
    if len(serie) > 1:
        datas_completas = pd.date_range(serie["ds"].min(), serie["ds"].max(), freq="D")
        serie = serie.set_index("ds").reindex(datas_completas, fill_value=0).reset_index()
        serie.columns = ["ds", "y"]

    return serie


def treinar_prophet(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    prioridade: str,
    mlflow_run,
) -> tuple:
    """
    Treina um modelo Prophet para previsão de volume diário de incidentes
    de uma determinada prioridade. Avalia no conjunto de validação.

    Retorna: (modelo_prophet, dict_metricas)
    """
    print(f"\n  -- Prophet | Prioridade: {prioridade} --")

    # Preparar séries
    serie_train = preparar_serie_temporal(df_train, prioridade)
    serie_val   = preparar_serie_temporal(df_val,   prioridade)

    if serie_train.empty or len(serie_train) < 10:
        print(f"  [AVISO] Série de treino insuficiente para '{prioridade}'. Pulando.")
        return None, {}

    # Adicionar regressores extras se disponíveis
    usar_ic_barulhento = "ic_barulhento_medio_dia" in df_train.columns
    usar_is_weekend    = "is_weekend" in df_train.columns

    if usar_ic_barulhento:
        # Calcular média diária do regressor
        col_data_ic = "dt_aberto" if "dt_aberto" in df_train.columns else "Aberto"
        reg_train = (
            df_train.assign(_data=pd.to_datetime(df_train[col_data_ic], errors="coerce").dt.date)
            .groupby("_data")["ic_barulhento_medio_dia"]
            .mean()
            .reset_index()
        )
        reg_train.columns = ["ds", "ic_barulhento_medio_dia"]
        reg_train["ds"] = pd.to_datetime(reg_train["ds"])
        serie_train = serie_train.merge(reg_train, on="ds", how="left")
        serie_train["ic_barulhento_medio_dia"] = serie_train["ic_barulhento_medio_dia"].fillna(0)

    if usar_is_weekend:
        serie_train["is_weekend"] = serie_train["ds"].dt.dayofweek.isin([5, 6]).astype(int)

    # Instanciar Prophet
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    # Sazonalidade mensal customizada
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    # Regressores opcionais
    if usar_ic_barulhento and "ic_barulhento_medio_dia" in serie_train.columns:
        model.add_regressor("ic_barulhento_medio_dia")
    if usar_is_weekend and "is_weekend" in serie_train.columns:
        model.add_regressor("is_weekend")

    # Treinar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(serie_train)

    # Previsão no período de validação
    if not serie_val.empty and len(serie_val) > 0:
        future_val = model.make_future_dataframe(
            periods=len(serie_val), freq="D", include_history=False
        )

        if usar_ic_barulhento and "ic_barulhento_medio_dia" in serie_train.columns:
            future_val["ic_barulhento_medio_dia"] = 0.0
        if usar_is_weekend and "is_weekend" in serie_train.columns:
            future_val["is_weekend"] = future_val["ds"].dt.dayofweek.isin([5, 6]).astype(int)

        forecast_val = model.predict(future_val)
        forecast_val["yhat"] = forecast_val["yhat"].clip(lower=0)

        # Alinhar pelo índice de datas
        y_real = serie_val["y"].values[: len(forecast_val)]
        y_pred = forecast_val["yhat"].values[: len(y_real)]

        # MAPE (evitar divisão por zero)
        mask_nonzero = y_real != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_real[mask_nonzero] - y_pred[mask_nonzero]) / y_real[mask_nonzero])) * 100
        else:
            mape = np.nan

        mae = np.mean(np.abs(y_real - y_pred))

        print(f"  Validação -> MAPE: {mape:.2f}%  |  MAE: {mae:.2f}")
    else:
        mape, mae = np.nan, np.nan
        print("  [AVISO] Conjunto de validação vazio. Métricas não calculadas.")

    # Log MLflow
    prioridade_slug = prioridade.replace(" - ", "_").replace(" ", "_")
    prefix = f"prophet_{prioridade_slug}"

    if not np.isnan(mape):
        mlflow.log_metric(f"{prefix}_mape", round(mape, 4))
    if not np.isnan(mae):
        mlflow.log_metric(f"{prefix}_mae", round(mae, 4))

    # Salvar modelo
    nome_arquivo = prioridade.split(" - ")[0].strip() + "_" + prioridade.split(" - ")[-1].strip()
    caminho_modelo = MODEL_DIR / f"prophet_{nome_arquivo}.pkl"
    joblib.dump(model, caminho_modelo)
    print(f"  Modelo salvo em: {caminho_modelo}")

    metricas = {"mape": mape, "mae": mae, "prioridade": prioridade}
    return model, metricas


def prever_volume(modelo: Prophet, horizonte_dias: int = 7) -> pd.DataFrame:
    """
    Gera previsão de volume para os próximos `horizonte_dias` dias.
    Retorna DataFrame com colunas: ds, yhat, yhat_lower, yhat_upper.
    """
    future = modelo.make_future_dataframe(
        periods=horizonte_dias, freq="D", include_history=False
    )

    # Preencher regressores adicionais se existirem.
    # No Prophet, o nome do regressor é a CHAVE do dict extra_regressors
    # (o dict-valor guarda prior_scale/mu/std/mode, não um campo "name").
    regressores = list(modelo.extra_regressors.keys()) \
        if hasattr(modelo, "extra_regressors") else []

    for reg in regressores:
        if reg == "is_weekend":
            future[reg] = future["ds"].dt.dayofweek.isin([5, 6]).astype(int)
        else:
            future[reg] = 0.0

    forecast = modelo.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["yhat"]       = result["yhat"].clip(lower=0)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0)

    return result


# Modelo 2 - xgboost (risco de violação de ola)

def preparar_dados_ola(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple:
    """
    Filtra e prepara os dados para o modelo de risco de violação de OLA.

    IMPORTANTE: NÃO inclui 'Duração' como feature - data leak!
    A duração só é conhecida APÓS a resolução do incidente.
    Usamos apenas features disponíveis NO MOMENTO da abertura.

    Retorna: X_train, y_train, X_val, y_val, X_test, y_test, feature_cols_usadas
    """
    prioridades_alvo = ["2 - Alta", "3 - Média"]
    target = "kpi_violado_bin"

    def filtrar(df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (df["entrou_para_kpi"] == True) &
            (df["Prioridade"].isin(prioridades_alvo))
        )
        return df.loc[mask].copy()

    df_tr = filtrar(df_train)
    df_vl = filtrar(df_val)
    df_te = filtrar(df_test)

    print(f"\n  Dados OLA filtrados -> Train: {len(df_tr)} | Val: {len(df_vl)} | Test: {len(df_te)}")

    # Guarda anti-leakage: nada conhecido só APÓS a resolução pode virar feature.
    colunas_proibidas = [
        # duração / tempos de resolução
        "Duração", "duracao_horas", "duracao_minutos", "duracao_real",
        "tempo_resolucao", "Resolvido", "Encerrado", "dt_encerrado",
        "dt_resolucao", "Aberto", "Fechado", "Atualizado", "dt_aberto",
        # desfecho pós-resolução
        "Solução", "Código de fechamento", "Status", "violou_ola_calculado",
        "mes_encerramento",
        # alvo / rótulos
        "kpi_violado_bin", "kpi_violado", "KPI Violado?", "Entrou para KPI?",
        "entrou_para_kpi",
        # identificadores
        "Número", "número",
    ]

    feature_cols_disponiveis = [
        c for c in FEATURE_COLS
        if c in df_tr.columns and c not in colunas_proibidas
    ]

    # Garantir que target existe
    for df_ in [df_tr, df_vl, df_te]:
        if target not in df_.columns:
            raise ValueError(f"Coluna target '{target}' não encontrada. Verifique a Etapa 4.")

    X_train = df_tr[feature_cols_disponiveis]
    y_train = df_tr[target].astype(int)

    X_val   = df_vl[feature_cols_disponiveis]
    y_val   = df_vl[target].astype(int)

    X_test  = df_te[feature_cols_disponiveis]
    y_test  = df_te[target].astype(int)

    print(f"  Features usadas: {len(feature_cols_disponiveis)}")
    print(f"  Taxa de violação -> Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols_disponiveis


def construir_pipeline_xgboost(feature_cols: list) -> Pipeline:
    """
    Constrói Pipeline scikit-learn com:
      - ColumnTransformer (numérico + categórico)
      - XGBClassifier
    """
    # Separar features por tipo
    cats_presentes = [c for c in CATEGORICAS_FEATURES if c in feature_cols]
    nums_presentes  = [c for c in feature_cols if c not in cats_presentes]

    print(f"\n  Numéricas: {len(nums_presentes)} | Categóricas: {len(cats_presentes)}")

    # Preprocessador numérico
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Preprocessador categórico
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="DESCONHECIDO")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  nums_presentes),
            ("cat", categorical_transformer, cats_presentes),
        ],
        remainder="drop",
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    pipeline = Pipeline([
        ("preprocessador", preprocessor),
        ("classificador",  xgb),
    ])

    return pipeline


def encontrar_threshold_otimo(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Varre thresholds de 0.1 a 0.9 e retorna aquele com melhor F1-score.
    """
    melhor_f1        = -1.0
    melhor_threshold = 0.5

    for thresh in np.arange(0.1, 0.91, 0.01):
        y_pred_t = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        if f1 > melhor_f1:
            melhor_f1        = f1
            melhor_threshold = thresh

    return round(float(melhor_threshold), 2)


def _bootstrap_ic(y_true, y_prob, metric="auc", B=2000, seed=42):
    """
    Intervalo de confiança 95% (bootstrap por reamostragem) de AUC ou AP.
    Reamostra as linhas com reposição B vezes; ignora reamostras degeneradas
    (uma só classe). Com poucos positivos, o IC sai largo - e isso é honesto.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    n = len(y_true)
    fn = roc_auc_score if metric == "auc" else average_precision_score
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():            # precisa das duas classes
            continue
        vals.append(fn(yt, y_prob[idx]))
    if not vals:
        return (np.nan, np.nan, np.nan, 0)
    vals = np.array(vals)
    return (float(vals.mean()), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)), len(vals))


def _cv_estratificada(pipeline, X_all, y_all, n_estimators, n_splits=5, seed=42):
    """
    Validação cruzada estratificada (k folds) sobre TODOS os rótulos.
    Ataca o problema de o teste temporal ter só ~6 positivos: cada fold de teste
    fica com ~50 positivos, dando uma estimativa estável de AUC/AP + IC.
    As features são point-in-time (calculadas na abertura), então o split
    aleatório NÃO reintroduz leakage - estima discriminação, complementando o
    holdout temporal (que simula produção).
    """
    y_arr = np.asarray(y_all)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, aps = [], []
    for tr, te in skf.split(X_all, y_arr):
        ytr = y_arr[tr]
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        pipe = clone(pipeline)
        pipe.named_steps["classificador"].set_params(
            n_estimators=n_estimators, scale_pos_weight=spw)
        pipe.fit(X_all.iloc[tr], ytr)
        p = pipe.predict_proba(X_all.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y_arr[te], p))
        aps.append(average_precision_score(y_arr[te], p))
    aucs, aps = np.array(aucs), np.array(aps)

    def _ic95(v):
        if len(v) < 2:
            return (float(v.mean()), float(v.mean()))
        sem = _sps.sem(v)
        lo, hi = _sps.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=sem)
        return (float(lo), float(hi))

    return {
        "auc_folds": aucs, "ap_folds": aps,
        "auc_mean": float(aucs.mean()), "auc_ic": _ic95(aucs),
        "ap_mean": float(aps.mean()),  "ap_ic": _ic95(aps),
    }


def treinar_xgboost(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    pipeline: Pipeline,
    mlflow_run,
) -> tuple:
    """
    Treina o pipeline XGBoost para classificação de risco de violação de OLA.

    Estratégias:
      - scale_pos_weight para desbalanceamento
      - Early stopping monitorando AUC na validação
      - Threshold ótimo por F1

    Retorna: (pipeline_treinado, dict_metricas)
    """
    print("\n  -- XGBoost | Risco de Violação de OLA --")

    # Calcular scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / max(n_pos, 1)
    print(f"  Classe 0: {n_neg} | Classe 1: {n_pos} | scale_pos_weight: {spw:.2f}")

    pipeline.named_steps["classificador"].set_params(scale_pos_weight=spw)

    # Pré-processar validação para early stopping
    preprocessor = pipeline.named_steps["preprocessador"]
    preprocessor.fit(X_train)

    X_val_prep = preprocessor.transform(X_val)

    # Treinar com early stopping
    pipeline.named_steps["classificador"].fit(
        preprocessor.transform(X_train),
        y_train,
        eval_set=[(X_val_prep, y_val)],
        early_stopping_rounds=30,
        verbose=False,
    )

    # Re-fit completo do pipeline (para que predict/predict_proba funcionem end-to-end)
    # Usamos o melhor n_estimators identificado pelo early stopping.
    # XGBoost 2.0 removeu best_ntree_limit -> usar best_iteration (0-indexado -> +1).
    _clf = pipeline.named_steps["classificador"]
    _best_iter = getattr(_clf, "best_iteration", None)
    best_n = (_best_iter + 1) if _best_iter is not None else 300
    print(f"  Melhor iteração (early stopping): {best_n}")

    pipeline.named_steps["classificador"].set_params(n_estimators=best_n)
    pipeline.fit(X_train, y_train)

    # Métricas na Validação
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_auc    = roc_auc_score(y_val, y_val_prob)
    val_ap     = average_precision_score(y_val, y_val_prob)
    threshold  = encontrar_threshold_otimo(y_val.values, y_val_prob)
    y_val_pred = (y_val_prob >= threshold).astype(int)
    val_f1     = f1_score(y_val, y_val_pred, zero_division=0)
    val_prec   = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec    = recall_score(y_val, y_val_pred, zero_division=0)

    print(f"\n  [Validação]")
    print(f"  AUC-ROC:           {val_auc:.4f}")
    print(f"  Average Precision: {val_ap:.4f}")
    print(f"  Threshold ótimo:   {threshold}")
    print(f"  F1:  {val_f1:.4f}  |  Precisão: {val_prec:.4f}  |  Recall: {val_rec:.4f}")

    # Métricas no Teste
    y_test_prob  = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred  = (y_test_prob >= threshold).astype(int)
    test_auc     = roc_auc_score(y_test, y_test_prob)
    test_ap      = average_precision_score(y_test, y_test_prob)
    test_f1      = f1_score(y_test, y_test_pred, zero_division=0)
    test_prec    = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec     = recall_score(y_test, y_test_pred, zero_division=0)

    print(f"\n  [Teste]")
    print(f"  AUC-ROC:           {test_auc:.4f}")
    print(f"  Average Precision: {test_ap:.4f}")
    print(f"  F1:  {test_f1:.4f}  |  Precisão: {test_prec:.4f}  |  Recall: {test_rec:.4f}")

    print(f"\n  Classification Report (Teste | threshold={threshold}):")
    print(classification_report(y_test, y_test_pred,
                                 target_names=["Não Violado", "Violado"],
                                 zero_division=0))

    print("  Confusion Matrix (Teste):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  {cm}")

    # Avaliação robusta (alvo raro: teste tem poucos positivos)
    print(f"\n  [Avaliação robusta]")
    # 1) IC bootstrap 95% no teste - mostra o quão (in)certo é o AUC do teste
    n_pos_test = int((y_test == 1).sum())
    tb_auc = _bootstrap_ic(y_test, y_test_prob, "auc")
    tb_ap  = _bootstrap_ic(y_test, y_test_prob, "ap")
    print(f"  Teste tem apenas {n_pos_test} positivos -> IC bootstrap (95%):")
    print(f"    AUC teste: {test_auc:.3f}  IC[{tb_auc[1]:.3f}, {tb_auc[2]:.3f}]")
    print(f"    AP  teste: {test_ap:.3f}  IC[{tb_ap[1]:.3f}, {tb_ap[2]:.3f}]")

    # 2) CV estratificada 5-fold sobre TODOS os rótulos - estimativa estável
    X_all = pd.concat([X_train, X_val, X_test], axis=0)
    y_all = pd.concat([y_train, y_val, y_test], axis=0)
    cv = _cv_estratificada(pipeline, X_all, y_all, n_estimators=best_n, n_splits=5)
    print(f"  CV estratificada 5-fold ({int((y_all==1).sum())} positivos no total):")
    print(f"    AUC: {cv['auc_mean']:.3f}  IC95[{cv['auc_ic'][0]:.3f}, {cv['auc_ic'][1]:.3f}]"
          f"  folds={np.round(cv['auc_folds'], 3).tolist()}")
    print(f"    AP:  {cv['ap_mean']:.3f}  IC95[{cv['ap_ic'][0]:.3f}, {cv['ap_ic'][1]:.3f}]")

    # Log MLflow
    # Parâmetros
    xgb_params = pipeline.named_steps["classificador"].get_params()
    params_log = {
        "xgb_n_estimators":    xgb_params.get("n_estimators"),
        "xgb_max_depth":       xgb_params.get("max_depth"),
        "xgb_learning_rate":   xgb_params.get("learning_rate"),
        "xgb_subsample":       xgb_params.get("subsample"),
        "xgb_colsample_bytree":xgb_params.get("colsample_bytree"),
        "xgb_scale_pos_weight":round(spw, 4),
        "threshold_otimo":     threshold,
    }
    mlflow.log_params(params_log)

    # Métricas
    metricas_log = {
        "val_auc_roc":          round(val_auc,  4),
        "val_avg_precision":    round(val_ap,   4),
        "val_f1":               round(val_f1,   4),
        "val_precisao":         round(val_prec, 4),
        "val_recall":           round(val_rec,  4),
        "test_auc_roc":         round(test_auc,  4),
        "test_avg_precision":   round(test_ap,   4),
        "test_f1":              round(test_f1,   4),
        "test_precisao":        round(test_prec, 4),
        "test_recall":          round(test_rec,  4),
        # avaliação robusta
        "test_auc_ic_low":      round(tb_auc[1], 4),
        "test_auc_ic_high":     round(tb_auc[2], 4),
        "cv_auc_mean":          round(cv["auc_mean"], 4),
        "cv_auc_ic_low":        round(cv["auc_ic"][0], 4),
        "cv_auc_ic_high":       round(cv["auc_ic"][1], 4),
        "cv_ap_mean":           round(cv["ap_mean"], 4),
        "cv_ap_ic_low":         round(cv["ap_ic"][0], 4),
        "cv_ap_ic_high":        round(cv["ap_ic"][1], 4),
    }
    mlflow.log_metrics(metricas_log)

    # Salvar artefatos
    pipeline_path     = MODEL_DIR / "ola_predictor_pipeline.pkl"
    feature_cols_path = MODEL_DIR / "ola_feature_cols.pkl"

    joblib.dump(pipeline,     pipeline_path)
    joblib.dump(X_train.columns.tolist(), feature_cols_path)

    print(f"\n  Pipeline salvo em:      {pipeline_path}")
    print(f"  Feature cols salvas em: {feature_cols_path}")

    metricas = {
        "val_auc": val_auc, "val_ap": val_ap,
        "test_auc": test_auc, "test_ap": test_ap,
        "test_auc_ic": (tb_auc[1], tb_auc[2]),
        "cv_auc_mean": cv["auc_mean"], "cv_auc_ic": cv["auc_ic"],
        "cv_ap_mean": cv["ap_mean"], "cv_ap_ic": cv["ap_ic"],
        "threshold": threshold, "melhor_n_estimators": best_n,
    }
    return pipeline, metricas


# Pipeline principal com mlflow

def _gerar_dados_sinteticos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Gera DataFrames sintéticos para demonstração quando os arquivos
    parquet da Etapa 4 não estiverem disponíveis.
    """
    print("\n  [INFO] Arquivos parquet não encontrados. Gerando dados sintéticos para demo...")

    rng   = np.random.default_rng(42)
    n_tr, n_vl, n_te = 3000, 600, 600
    total = n_tr + n_vl + n_te

    prioridades = rng.choice(
        ["1 - Crítica", "2 - Alta", "3 - Média", "4 - Baixa"],
        total, p=[0.05, 0.25, 0.45, 0.25]
    )

    datas = pd.date_range("2022-01-01", periods=total, freq="3h")

    df_base = pd.DataFrame({
        "dt_aberto":               datas,
        "Prioridade":              prioridades,
        "Grupo designado":         rng.choice(["GRP_A", "GRP_B", "GRP_C", "GRP_D"], total),
        "Item de configuração":    rng.choice(["IC_001", "IC_002", "IC_003", "IC_004"], total),
        "Solução":                 rng.choice(["SOL_X", "SOL_Y", "SOL_Z"], total),
        "Aberto por":              rng.choice(["usr1", "usr2", "usr3", "usr4"], total),
        "entrou_para_kpi":         rng.choice([True, False], total, p=[0.85, 0.15]),
        "kpi_violado_bin":         rng.integers(0, 2, total),
        # Numéricas
        "hora_abertura":           rng.integers(0, 24, total).astype(float),
        "dia_semana":              rng.integers(0, 7, total).astype(float),
        "dia_mes":                 rng.integers(1, 29, total).astype(float),
        "mes":                     rng.integers(1, 13, total).astype(float),
        "trimestre":               rng.integers(1, 5, total).astype(float),
        "is_weekend":              rng.integers(0, 2, total).astype(float),
        "is_horario_comercial":    rng.integers(0, 2, total).astype(float),
        "is_madrugada":            rng.integers(0, 2, total).astype(float),
        "is_fim_de_mes":           rng.integers(0, 2, total).astype(float),
        "prioridade_num":          rng.integers(1, 6, total).astype(float),
        "ola_limite_horas":        rng.choice([4, 8, 24, 72], total).astype(float),
        "n_reatribuicoes":         rng.integers(0, 6, total).astype(float),
        "trocou_grupo":            rng.integers(0, 2, total).astype(float),
        "ic_vol_30d":              rng.integers(0, 50, total).astype(float),
        "ic_vol_7d":               rng.integers(0, 15, total).astype(float),
        "ic_taxa_violacao_30d":    rng.uniform(0, 1, total),
        "ic_taxa_violacao_7d":     rng.uniform(0, 1, total),
        "ic_media_duracao_30d":    rng.uniform(1, 50, total),
        "ic_barulhento_medio_dia": rng.uniform(0, 5, total),
        "grupo_vol_30d":           rng.integers(10, 200, total).astype(float),
        "grupo_vol_7d":            rng.integers(2, 50, total).astype(float),
        "grupo_taxa_violacao_30d": rng.uniform(0, 1, total),
        "grupo_taxa_violacao_7d":  rng.uniform(0, 1, total),
        "grupo_media_duracao_30d": rng.uniform(1, 80, total),
        "carga_grupo_abertos":     rng.integers(0, 30, total).astype(float),
        "carga_grupo_atrasados":   rng.integers(0, 10, total).astype(float),
        "carga_ic_abertos":        rng.integers(0, 10, total).astype(float),
        "desc_n_palavras":         rng.integers(5, 100, total).astype(float),
        "desc_n_chars":            rng.integers(20, 500, total).astype(float),
        "desc_tem_erro":           rng.integers(0, 2, total).astype(float),
        "desc_tem_lento":          rng.integers(0, 2, total).astype(float),
        "desc_tem_fora":           rng.integers(0, 2, total).astype(float),
        "desc_tem_critico":        rng.integers(0, 2, total).astype(float),
        "desc_tem_cliente":        rng.integers(0, 2, total).astype(float),
        "alta_x_barulhento":       rng.uniform(0, 5, total),
        "alta_x_carga_grupo":      rng.uniform(0, 30, total),
        "hora_x_prioridade":       rng.uniform(0, 100, total),
        "vol_dia_ate_agora":       rng.integers(0, 20, total).astype(float),
        "violacoes_dia_ate_agora": rng.integers(0, 10, total).astype(float),
        "taxa_violacao_dia_ate_agora": rng.uniform(0, 1, total),
    })

    df_train = df_base.iloc[:n_tr].copy()
    df_val   = df_base.iloc[n_tr:n_tr + n_vl].copy()
    df_test  = df_base.iloc[n_tr + n_vl:].copy()

    return df_train, df_val, df_test


def executar_treinamento() -> None:
    """
    Pipeline principal de treinamento:
      1. Carrega splits (ou gera dados sintéticos)
      2. Treina Prophet para Alta e Média
      3. Treina XGBoost para risco de OLA
      4. Registra tudo no MLflow
      5. Imprime resumo final
    """
    print("=" * 60)
    print("  AIOps Locaweb - Etapa 5: Modelagem Preditiva")
    print("=" * 60)

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Carregar dados
    train_path = DATA_DIR / "train.parquet"
    val_path   = DATA_DIR / "val.parquet"
    test_path  = DATA_DIR / "test.parquet"

    if train_path.exists() and val_path.exists() and test_path.exists():
        print("\n  Carregando splits da Etapa 4...")
        df_train = pd.read_parquet(train_path)
        df_val   = pd.read_parquet(val_path)
        df_test  = pd.read_parquet(test_path)
        print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    else:
        df_train, df_val, df_test = _gerar_dados_sinteticos()

    # Iniciar run MLflow
    with mlflow.start_run(run_name="treinamento_completo") as run:
        print(f"\n  MLflow Run ID: {run.info.run_id}")

        metricas_resumo = {}

        # MODELO 1A - Prophet | 2 - Alta
        print("\n" + "-" * 50)
        print("  MODELO 1 - Prophet (Volume de Incidentes)")
        print("-" * 50)

        model_alta, met_alta = treinar_prophet(
            df_train, df_val,
            prioridade="2 - Alta",
            mlflow_run=run,
        )
        if model_alta:
            metricas_resumo["prophet_alta_mape"] = met_alta.get("mape")
            metricas_resumo["prophet_alta_mae"]  = met_alta.get("mae")

            # Demo de previsão D+1 e D+7
            prev_d1 = prever_volume(model_alta, horizonte_dias=1)
            prev_d7 = prever_volume(model_alta, horizonte_dias=7)
            print(f"\n  Previsão D+1 (Alta): {prev_d1['yhat'].iloc[0]:.1f} incidentes")
            print(f"  Previsão D+7 (Alta):\n{prev_d7[['ds','yhat','yhat_lower','yhat_upper']].to_string(index=False)}")

        # MODELO 1B - Prophet | 3 - Média
        model_media, met_media = treinar_prophet(
            df_train, df_val,
            prioridade="3 - Média",
            mlflow_run=run,
        )
        if model_media:
            metricas_resumo["prophet_media_mape"] = met_media.get("mape")
            metricas_resumo["prophet_media_mae"]  = met_media.get("mae")

            prev_d1_m = prever_volume(model_media, horizonte_dias=1)
            print(f"\n  Previsão D+1 (Média): {prev_d1_m['yhat'].iloc[0]:.1f} incidentes")

        # MODELO 2 - XGBoost (Risco de OLA)
        print("\n" + "-" * 50)
        print("  MODELO 2 - XGBoost (Risco de Violação de OLA)")
        print("-" * 50)

        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols_usadas = \
            preparar_dados_ola(df_train, df_val, df_test)

        pipeline_xgb = construir_pipeline_xgboost(feature_cols_usadas)

        pipeline_xgb, met_xgb = treinar_xgboost(
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            pipeline_xgb,
            mlflow_run=run,
        )

        metricas_resumo["xgb_val_auc"]  = met_xgb.get("val_auc")
        metricas_resumo["xgb_test_auc"] = met_xgb.get("test_auc")
        metricas_resumo["xgb_val_ap"]   = met_xgb.get("val_ap")
        metricas_resumo["xgb_test_ap"]  = met_xgb.get("test_ap")

        # Log artefatos MLflow (caminhos)
        mlflow.log_artifact(str(MODEL_DIR / "ola_predictor_pipeline.pkl"))
        mlflow.log_artifact(str(MODEL_DIR / "ola_feature_cols.pkl"))
        if (MODEL_DIR / "prophet_2_Alta.pkl").exists():
            mlflow.log_artifact(str(MODEL_DIR / "prophet_2_Alta.pkl"))
        if (MODEL_DIR / "prophet_3_Média.pkl").exists():
            mlflow.log_artifact(str(MODEL_DIR / "prophet_3_Média.pkl"))

        # Resumo final
        print("\n" + "=" * 60)
        print("  RESUMO FINAL - Etapa 5 Concluída")
        print("=" * 60)

        for chave, valor in metricas_resumo.items():
            if valor is not None and not (isinstance(valor, float) and np.isnan(valor)):
                print(f"  {chave:<30} {valor:.4f}")

        print("\n  Artefatos gerados:")
        for artefato in MODEL_DIR.glob("*.pkl"):
            tamanho_kb = artefato.stat().st_size / 1024
            print(f"    {artefato.name:<45} ({tamanho_kb:.1f} KB)")

        print(f"\n  MLflow tracking: {MLFLOW_URI}")
        print(f"  Experimento:     {EXPERIMENT}")
        print(f"  Run ID:          {run.info.run_id}")
        print("\n  Para visualizar os experimentos, execute:")
        print("    mlflow ui --backend-store-uri ./mlruns")
        print("=" * 60)


# Entry point
if __name__ == "__main__":
    executar_treinamento()
