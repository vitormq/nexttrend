"""
etapa7_explicabilidade.py - Explicabilidade do modelo de risco de OLA.
Usa SHAP para mostrar o que pesa na previsão e gera o mapa de riscos por grupo.

Roda com: python etapa7_explicabilidade.py
"""

import pandas as pd
import numpy as np
import joblib
import shap
import mlflow
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Caminhos
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data" / "processed"
SPLIT_DIR  = BASE_DIR / "data" / "splits"
MODEL_DIR  = BASE_DIR / "models" / "saved"
OUT_DIR    = BASE_DIR / "reports" / "explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# file URI absoluto (evita o bug de UNC \\.\mlruns no Windows - ver etapa4).
# Em Docker, MLFLOW_TRACKING_URI (ex.: http://mlflow:5000) tem prioridade.
import os
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI") or (BASE_DIR / "mlruns").resolve().as_uri()
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "locaweb-aiops")


# 1. carregar modelo e dados de teste

def carregar_modelo_e_dados() -> tuple:
    """
    Carrega o pipeline XGBoost treinado na Etapa 5 e o conjunto de teste.
    Filtra apenas incidentes P2/P3 que entraram para o KPI.
    """
    print("Carregando modelo e dados...")

    pipeline     = joblib.load(MODEL_DIR / "ola_predictor_pipeline.pkl")
    feature_cols = joblib.load(MODEL_DIR / "ola_feature_cols.pkl")

    # Tentar carregar split de teste; fallback para dataset completo
    caminho_test = SPLIT_DIR / "test.parquet"
    if caminho_test.exists():
        df_test = pd.read_parquet(caminho_test)
    else:
        print("  Split de teste não encontrado, usando dataset completo (últimos 10%)")
        df = pd.read_parquet(DATA_DIR / "incidents_clean.parquet")
        n  = len(df)
        df_test = df.iloc[int(n * 0.90):].copy()

    # Filtrar apenas P2/P3 que entraram para KPI
    df_test = df_test[
        (df_test["entrou_para_kpi"] == True) &
        (df_test["Prioridade"].isin(["2 - Alta", "3 - Média"]))
    ].copy()

    print(f"  Incidentes no teste (P2/P3 KPI): {len(df_test):,}")

    # Garantir que todas as feature_cols existam
    for col in feature_cols:
        if col not in df_test.columns:
            df_test[col] = 0

    X_test = df_test[feature_cols]
    y_test = df_test["kpi_violado_bin"].astype(int)

    return pipeline, X_test, y_test, df_test, feature_cols


# 2. calcular shap global

def calcular_shap_global(pipeline, X_test: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Calcula valores SHAP para todo o conjunto de teste usando TreeExplainer.
    Retorna explainer, matriz de valores SHAP, nomes de features e importância.
    """
    print("\n[1/4] Calculando SHAP global...")

    modelo_xgb   = pipeline.named_steps["classificador"]
    preprocessor = pipeline[:-1]
    X_transformed = preprocessor.transform(X_test)

    # Recuperar nomes das features após transformação
    try:
        nomes_features = list(pipeline.named_steps["preprocessador"].get_feature_names_out())
    except Exception:
        nomes_features = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # TreeExplainer - otimizado para XGBoost
    explainer   = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(X_transformed)

    # Para classificação binária pegar classe positiva (violação)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    print(f"  Shape dos valores SHAP: {sv.shape}")

    # Importância global = média do |SHAP| por feature
    importancia = pd.DataFrame({
        "feature":    nomes_features,
        "shap_medio": np.abs(sv).mean(axis=0),
    }).sort_values("shap_medio", ascending=False).reset_index(drop=True)

    print("\n  Top 15 features por importância SHAP:")
    print(importancia.head(15).to_string(index=False))

    # Salvar explainer e nomes para uso posterior na API e no dashboard
    joblib.dump(explainer,      MODEL_DIR / "shap_explainer.pkl")
    joblib.dump(nomes_features, MODEL_DIR / "shap_feature_names.pkl")

    return explainer, sv, X_transformed, nomes_features, importancia


# 3. gráficos shap globais

def gerar_graficos_shap_global(
    sv: np.ndarray,
    X_transformed: np.ndarray,
    nomes_features: list,
    importancia: pd.DataFrame,
) -> None:
    """
    Gera e salva 3 tipos de gráfico SHAP global:
    1. Beeswarm - distribuição de valores SHAP por feature
    2. Bar plot  - importância média
    3. Dependence plots - top 3 features
    """
    print("\n[2/4] Gerando gráficos SHAP globais...")

    # data= permite ao beeswarm colorir por valor da feature (necessário no SHAP 0.45)
    exp = shap.Explanation(values=sv, data=np.asarray(X_transformed),
                           feature_names=nomes_features)

    # Gráficos são acessórios: uma falha de plot não pode abortar o pipeline
    # (o mapa de riscos e os CSVs vêm depois e o dashboard depende deles).
    # Beeswarm
    try:
        plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(exp, max_display=15, show=False)
        plt.title("SHAP Beeswarm - Impacto das Features na Violação de OLA",
                  fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  shap_beeswarm.png")
    except Exception as e:
        plt.close("all")
        print(f"   beeswarm falhou (ignorado): {e}")

    # Bar plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        top15 = importancia.head(15).sort_values("shap_medio")
        ax.barh(top15["feature"], top15["shap_medio"], color="#E74C3C")
        ax.set_xlabel("Importância SHAP (média |SHAP|)", fontsize=11)
        ax.set_title("Top 15 Features - Importância Global para Violação de OLA",
                     fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_barplot.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  shap_barplot.png")
    except Exception as e:
        plt.close("all")
        print(f"   barplot falhou (ignorado): {e}")

    # Dependence plots (top 3)
    top3 = importancia.head(3)["feature"].tolist()
    for feat in top3:
        if feat in nomes_features:
            try:
                idx = list(nomes_features).index(feat)
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.dependence_plot(
                    idx, sv, np.asarray(X_transformed),
                    feature_names=nomes_features,
                    ax=ax, show=False, interaction_index=None,
                )
                ax.set_title(f"SHAP Dependence: {feat}", fontsize=12)
                plt.tight_layout()
                nome_arq = f"shap_dependence_{feat.replace(' ', '_')[:30]}.png"
                plt.savefig(OUT_DIR / nome_arq, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  {nome_arq}")
            except Exception as e:
                plt.close("all")
                print(f"   dependence '{feat}' falhou (ignorado): {e}")


# 4. EXPLICAÇÃO LOCAL (por incidente)

def explicar_incidente(
    pipeline,
    explainer,
    df_test: pd.DataFrame,
    feature_cols: list,
    nomes_features: list,
    numero_incidente: str = None,
    indice: int = None,
) -> dict:
    """
    Explica a previsão de um incidente específico.
    Gera waterfall plot e retorna dict com fatores de risco/proteção.

    Uso:
        resultado = explicar_incidente(pipeline, explainer, df_test,
                                       feature_cols, nomes_features,
                                       numero_incidente="INC8654075")
    """
    # Selecionar linha
    if numero_incidente:
        row = df_test[df_test["Número"] == numero_incidente]
        if row.empty:
            raise ValueError(f"Incidente {numero_incidente} não encontrado.")
        idx = row.index[0]
    elif indice is not None:
        idx = df_test.index[indice]
    else:
        raise ValueError("Informe numero_incidente ou indice.")

    inc   = df_test.loc[idx]
    X_inc = df_test.loc[[idx], feature_cols]

    # Pré-processar e prever
    preprocessor = pipeline[:-1]
    X_inc_t      = preprocessor.transform(X_inc)
    prob_violacao = float(pipeline.predict_proba(X_inc)[0][1])

    # Nível de risco
    nivel = (
        "CRÍTICO" if prob_violacao >= 0.80 else
        "ALTO"    if prob_violacao >= 0.60 else
        "MÉDIO"   if prob_violacao >= 0.35 else
        "BAIXO"
    )

    # SHAP local
    sv_local = explainer.shap_values(X_inc_t)
    sv_local = sv_local[1][0] if isinstance(sv_local, list) else sv_local[0]

    fatores_df = pd.DataFrame({
        "feature":    nomes_features,
        "shap_valor": sv_local,
    }).sort_values("shap_valor", ascending=False)

    fatores_risco    = fatores_df[fatores_df["shap_valor"] > 0].head(5)
    fatores_protecao = fatores_df[fatores_df["shap_valor"] < 0].tail(5)

    resultado = {
        "numero":           str(inc.get("Número", "N/A")),
        "prioridade":       str(inc.get("Prioridade", "N/A")),
        "grupo":            str(inc.get("Grupo designado", "N/A")),
        "ic":               str(inc.get("Item de configuração", "N/A")),
        "prob_violacao":    round(prob_violacao, 4),
        "nivel_risco":      nivel,
        "fatores_risco":    fatores_risco.to_dict("records"),
        "fatores_protecao": fatores_protecao.to_dict("records"),
    }

    print(f"\n  Incidente: {resultado['numero']} | {resultado['prioridade']}")
    print(f"  Risco: {nivel} ({prob_violacao:.1%})")
    print("  Fatores de risco (top 5):")
    for f in resultado["fatores_risco"]:
        print(f"    + {f['feature']}: {f['shap_valor']:+.4f}")
    print("  Fatores de proteção (top 5):")
    for f in resultado["fatores_protecao"]:
        print(f"    - {f['feature']}: {f['shap_valor']:+.4f}")

    # Waterfall plot
    expected_val = (
        float(explainer.expected_value[1])
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )
    try:
        exp_local = shap.Explanation(
            values=sv_local,
            base_values=expected_val,
            feature_names=nomes_features,
        )
        plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(exp_local, max_display=12, show=False)
        plt.title(
            f"SHAP Waterfall - {resultado['numero']} | Risco: {nivel}",
            fontsize=12
        )
        plt.tight_layout()
        fname = f"shap_waterfall_{resultado['numero']}.png"
        plt.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {fname}")
    except Exception as e:
        plt.close("all")
        print(f"   waterfall falhou (ignorado): {e}")

    return resultado


# 5. mapa de riscos operacionais

def gerar_mapa_riscos(
    pipeline,
    df_test: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Gera mapa de riscos por Grupo x Prioridade com probabilidade
    média de violação de OLA. Responde: "onde agir preventivamente?"
    """
    print("\n[3/4] Gerando mapa de riscos operacionais...")

    # Mapa de riscos usa a base COMPLETA (com features) para ficar robusto por grupo;
    # o split de teste sozinho tem poucas centenas de linhas. Fallback: df_test.
    caminho_feat = DATA_DIR / "incidents_features.parquet"
    if caminho_feat.exists():
        df = pd.read_parquet(caminho_feat)
        df = df[(df["entrou_para_kpi"] == True) &
                (df["Prioridade"].isin(["2 - Alta", "3 - Média"]))].copy()
        print(f"  Base do mapa (features completas, P2/P3 KPI): {len(df):,}")
    else:
        df = df_test.copy()

    # Garantir colunas de tempo
    df["dt_encerrado"] = pd.to_datetime(df.get("Encerrado", pd.NaT), errors="coerce")
    df["mes_enc"]      = df["dt_encerrado"].dt.to_period("M").astype(str)

    # Probabilidades
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df["prob_violacao"] = pipeline.predict_proba(df[feature_cols])[:, 1]

    # Mapa por Grupo + Prioridade
    mapa_grupo = (
        df.groupby(["Grupo designado", "Prioridade"])
        .agg(
            prob_media_violacao = ("prob_violacao",   "mean"),
            total_incidentes    = ("Número",           "count"),
            violacoes_reais     = ("kpi_violado_bin",  "sum"),
        )
        .reset_index()
        .sort_values("prob_media_violacao", ascending=False)
    )
    mapa_grupo["perc_risco"] = (mapa_grupo["prob_media_violacao"] * 100).round(1)
    mapa_grupo["nivel"] = mapa_grupo["prob_media_violacao"].apply(
        lambda p: "CRÍTICO" if p >= 0.80 else
                  "ALTO"    if p >= 0.60 else
                  "MÉDIO"   if p >= 0.35 else "BAIXO"
    )

    # Mapa temporal (tendência por mês)
    mapa_mensal = (
        df.groupby(["mes_enc", "Prioridade"])
        .agg(prob_media_violacao=("prob_violacao", "mean"))
        .reset_index()
    )

    print(f"\n  Top 10 grupos com maior risco:")
    print(
        mapa_grupo.head(10)[
            ["Grupo designado", "Prioridade", "perc_risco", "nivel", "total_incidentes"]
        ].to_string(index=False)
    )

    # Salvar
    mapa_grupo.to_parquet(OUT_DIR / "mapa_riscos_grupo.parquet", index=False)
    mapa_grupo.to_csv(OUT_DIR / "mapa_riscos_grupo.csv", index=False)
    print("  mapa_riscos_grupo.parquet + .csv")

    return mapa_grupo, mapa_mensal


# 6. recomendações automáticas

def gerar_recomendacoes(
    importancia: pd.DataFrame,
    mapa_riscos: pd.DataFrame,
) -> list:
    """
    Gera recomendações práticas baseadas nos resultados SHAP e mapa de riscos.
    Retorna lista de dicts com tipo, mensagem e prioridade.
    """
    recomendacoes = []

    # Recomendação 1 - features críticas
    top5 = importancia.head(5)["feature"].tolist()
    recomendacoes.append({
        "tipo":       "FEATURE_CRITICA",
        "prioridade": "ALTA",
        "mensagem": (
            f"As 5 features mais críticas para violação de OLA são: {', '.join(top5)}. "
            "Monitorar esses indicadores em tempo real permite antecipar crises operacionais."
        ),
    })

    # Recomendação 2 - grupos em risco
    grupos_criticos = mapa_riscos[mapa_riscos["prob_media_violacao"] >= 0.60]
    for _, row in grupos_criticos.head(3).iterrows():
        recomendacoes.append({
            "tipo":       "GRUPO_EM_RISCO",
            "prioridade": "ALTA",
            "mensagem": (
                f"Grupo '{row['Grupo designado']}' - {row['Prioridade']}: "
                f"{row['perc_risco']}% de risco médio de violação de OLA. "
                "Revisar capacidade e escalação preventiva."
            ),
        })

    # Recomendação 3 - KPI mensal P2
    recomendacoes.append({
        "tipo":       "KPI_MENSAL_P2",
        "prioridade": "CRÍTICA",
        "mensagem": (
            "P2: máximo de 3 violações de OLA por mês. "
            "Configurar alerta automático ao atingir 2 violações no mês "
            "para acionar plano preventivo antes de ferir o KPI."
        ),
    })

    # Recomendação 4 - contorno e reincidência
    recomendacoes.append({
        "tipo":       "CONTORNO_RISCO",
        "prioridade": "MÉDIA",
        "mensagem": (
            "Incidentes resolvidos por CONTORNO têm maior probabilidade de "
            "reincidência em 72h. Priorizar investigação de causa raiz para "
            "ICs com múltiplos contornos consecutivos."
        ),
    })

    return recomendacoes


# Pipeline principal com mlflow

def executar_explicabilidade() -> dict:
    """
    Pipeline completo de explicabilidade com rastreamento MLflow.
    Roda todas as etapas em sequência e salva todos os artefatos.
    """
    print("=" * 60)
    print("ETAPA 7 - EXPLICABILIDADE (SHAP)")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="explicabilidade_shap"):

        # 1. Carregar dados e modelo
        pipeline, X_test, y_test, df_test, feature_cols = carregar_modelo_e_dados()

        # 2. SHAP global
        explainer, sv, X_transformed, nomes_features, importancia = calcular_shap_global(
            pipeline, X_test, feature_cols
        )

        # 3. Gráficos globais
        gerar_graficos_shap_global(sv, X_transformed, nomes_features, importancia)

        # 4. Explicar incidente de maior risco (exemplo automático)
        print("\n[3/4] Explicando incidente de maior risco...")
        df_temp = df_test.copy()
        for col in feature_cols:
            if col not in df_temp.columns:
                df_temp[col] = 0
        df_temp["prob_temp"] = pipeline.predict_proba(df_temp[feature_cols])[:, 1]
        inc_alto = df_temp.sort_values("prob_temp", ascending=False).iloc[0]["Número"]
        explicar_incidente(pipeline, explainer, df_test,
                           feature_cols, nomes_features,
                           numero_incidente=inc_alto)

        # 5. Mapa de riscos
        mapa_riscos, mapa_mensal = gerar_mapa_riscos(pipeline, df_test, feature_cols)

        # 6. Recomendações
        print("\n[4/4] Gerando recomendações...")
        recomendacoes = gerar_recomendacoes(importancia, mapa_riscos)
        for r in recomendacoes:
            print(f"  [{r['prioridade']}] {r['tipo']}: {r['mensagem'][:80]}...")

        # 7. Salvar CSV de importância
        importancia.to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)

        # 8. Log MLflow
        mlflow.log_metric("shap_top1_score",   float(importancia.iloc[0]["shap_medio"]))
        mlflow.log_metric("grupos_alto_risco",  int((mapa_riscos["prob_media_violacao"] >= 0.60).sum()))
        mlflow.log_metric("total_recomendacoes", len(recomendacoes))
        mlflow.log_artifacts(str(OUT_DIR),    artifact_path="shap_reports")
        mlflow.log_artifacts(str(MODEL_DIR),  artifact_path="shap_models")

    print("\n" + "=" * 60)
    print("ETAPA 7 CONCLUÍDA")
    print(f"  Relatórios em: {OUT_DIR}")
    print(f"  Modelos em:    {MODEL_DIR}")
    print("=" * 60)

    return {
        "importancia":    importancia,
        "mapa_riscos":    mapa_riscos,
        "mapa_mensal":    mapa_mensal,
        "recomendacoes":  recomendacoes,
        "explainer":      explainer,
        "shap_values":    sv,
        "nomes_features": nomes_features,
    }


# Execução direta

if __name__ == "__main__":
    resultados = executar_explicabilidade()

    print("\n\nRECOMENDAÇÕES FINAIS:")
    print("-" * 60)
    for i, r in enumerate(resultados["recomendacoes"], 1):
        print(f"\n{i}. [{r['prioridade']}] {r['tipo']}")
        print(f"   {r['mensagem']}")
