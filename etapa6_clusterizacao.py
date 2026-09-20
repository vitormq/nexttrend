"""
etapa6_clusterizacao.py - Agrupamento dos incidentes.
Perfis de comportamento (KMeans) e análise de recorrência/causas
(TF-IDF + KMeans nas descrições). Salva os modelos e os relatórios.

Roda com: python etapa6_clusterizacao.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import warnings
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Caminhos
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data" / "processed"
MODEL_DIR   = BASE_DIR / "models" / "saved"
REPORT_DIR  = BASE_DIR / "reports" / "clustering"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# file URI absoluto (evita UNC \\.\mlruns no Windows); env tem prioridade em Docker.
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI") or (BASE_DIR / "mlruns").resolve().as_uri()
EXPERIMENT  = os.getenv("MLFLOW_EXPERIMENT_NAME", "locaweb-aiops")


# 1. CLUSTER DE PERFIL NUMÉRICO (KMeans)

def preparar_features_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara features numéricas para clusterização. Usa apenas incidentes KPI."""
    df_kpi = df[df["entrou_para_kpi"] == True].copy()

    mapa_prioridade = {
        "1 - Crítica": 1, "2 - Alta": 2, "3 - Média": 3,
        "4 - Baixa": 4,   "5 - Muito Baixa": 5,
    }
    df_kpi["prioridade_num"]      = df_kpi["Prioridade"].map(mapa_prioridade)
    df_kpi["hora_abertura"]       = pd.to_datetime(df_kpi["Aberto"], errors="coerce").dt.hour
    df_kpi["dia_semana_abertura"] = pd.to_datetime(df_kpi["Aberto"], errors="coerce").dt.dayofweek
    df_kpi["duracao_horas"]       = df_kpi["Duração"] / 3600
    df_kpi["e_contorno"]          = (df_kpi["Solução"] == "Contorno").astype(int)
    df_kpi["e_definitiva"]        = (df_kpi["Solução"] == "Definitiva").astype(int)
    df_kpi["ola_violado_num"]     = (df_kpi["KPI Violado?"] == "SIM").astype(int)

    return df_kpi


FEATURES_PERFIL = [
    "prioridade_num", "hora_abertura", "dia_semana_abertura",
    "duracao_horas",  "e_contorno",    "e_definitiva", "ola_violado_num",
]


def encontrar_k_otimo(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
    """Encontra k ótimo por Silhouette Score."""
    melhor_k, melhor_score = k_min, -1
    print("  Silhouette Score por k:")
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        print(f"    k={k} -> {score:.4f}")
        if score > melhor_score:
            melhor_score, melhor_k = score, k
    print(f"  -> Melhor k = {melhor_k} (score={melhor_score:.4f})")
    return melhor_k


def treinar_kmeans_perfil(df: pd.DataFrame) -> dict:
    """KMeans sobre features numéricas de incidentes KPI."""
    print("\n[1/4] KMeans - Perfil Numérico de Incidentes")

    df_feat = preparar_features_numericas(df)
    X       = df_feat[FEATURES_PERFIL].dropna().values
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    k = encontrar_k_otimo(X_sc)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_feat = df_feat.dropna(subset=FEATURES_PERFIL).copy()
    df_feat["cluster_perfil"] = km.fit_predict(X_sc)

    # PCA 2D para visualização
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    df_feat["pca_x"] = coords[:, 0]
    df_feat["pca_y"] = coords[:, 1]

    # Perfil médio por cluster
    perfil = df_feat.groupby("cluster_perfil")[FEATURES_PERFIL].mean().round(2)
    print("\n  Perfil médio por cluster:")
    print(perfil.to_string())

    joblib.dump(km,     MODEL_DIR / "kmeans_perfil.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler_perfil.pkl")
    joblib.dump(pca,    MODEL_DIR / "pca_perfil.pkl")

    sil = silhouette_score(X_sc, df_feat["cluster_perfil"].values)
    print(f"  Silhouette final = {sil:.4f}")

    return {"modelo": km, "scaler": scaler, "pca": pca,
            "df_result": df_feat, "k": k, "perfil": perfil, "silhouette": sil}


# 2. CLUSTER DE CAUSAS (TF-IDF + KMeans)

def limpar_descricao(texto: str) -> str:
    """Remove prefixos automáticos da descrição."""
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    for p in ["problem:", "alarm application monitoring", "check application monitoring"]:
        texto = texto.replace(p, "")
    return texto.strip()


def treinar_cluster_causas(df: pd.DataFrame, n_clusters: int = 8) -> dict:
    """TF-IDF nas descrições + KMeans para agrupar causas técnicas recorrentes."""
    print(f"\n[2/4] TF-IDF + KMeans - Causas Recorrentes (k={n_clusters})")

    df_desc = df[df["Descrição resumida"].notna()].copy()
    df_desc["desc_limpa"] = df_desc["Descrição resumida"].apply(limpar_descricao)
    df_desc = df_desc[df_desc["desc_limpa"].str.len() > 5].copy()
    print(f"  Incidentes com descrição válida: {len(df_desc):,}")

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=5)
    X_tfidf = tfidf.fit_transform(df_desc["desc_limpa"])

    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_desc["cluster_causa"] = km.fit_predict(X_svd)

    # Top termos por cluster
    termos = tfidf.get_feature_names_out()
    centroides_orig = svd.inverse_transform(km.cluster_centers_)
    top_termos = {}
    print("\n  Top termos por cluster:")
    for i, centro in enumerate(centroides_orig):
        top_idx = centro.argsort()[-8:][::-1]
        palavras = [termos[j] for j in top_idx]
        top_termos[i] = palavras
        qtd = (df_desc["cluster_causa"] == i).sum()
        print(f"    Cluster {i} ({qtd:,}): {', '.join(palavras)}")

    joblib.dump(km,    MODEL_DIR / "kmeans_causas.pkl")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_causas.pkl")
    joblib.dump(svd,   MODEL_DIR / "svd_causas.pkl")

    return {"modelo": km, "tfidf": tfidf, "svd": svd,
            "df_result": df_desc, "top_termos": top_termos}


# 3. ICs BARULHENTOS (DBSCAN)

def analisar_ics_barulhentos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica ICs com alta frequência de incidentes automáticos.
    Usa DBSCAN nos top ICs para encontrar clusters de instabilidade crônica.
    """
    print("\n[3/4] DBSCAN - ICs Barulhentos")

    df_auto = df[
        (df["Status"] == "Sem Intervenção") &
        (df["Aberto por"] == "Monitoramento") &
        (df["Item de configuração"].notna())
    ].copy()
    df_auto["data_abertura"] = pd.to_datetime(df_auto["Aberto"], errors="coerce").dt.date

    freq = (
        df_auto
        .groupby(["Item de configuração", "data_abertura"])
        .size()
        .reset_index(name="qtd_diaria")
    )

    ic_stats = freq.groupby("Item de configuração").agg(
        total_incidentes   = ("qtd_diaria", "sum"),
        dias_com_incidente = ("qtd_diaria", "count"),
        media_diaria       = ("qtd_diaria", "mean"),
        max_diario         = ("qtd_diaria", "max"),
        std_diario         = ("qtd_diaria", "std"),
    ).reset_index().fillna(0)

    ic_stats["indice_ruido"] = (
        ic_stats["total_incidentes"] * ic_stats["max_diario"]
    ) / (ic_stats["dias_com_incidente"] + 1)

    ic_stats = ic_stats.sort_values("indice_ruido", ascending=False)

    # DBSCAN nos top 200 ICs
    top = ic_stats.head(200).copy()
    feats_db = ["total_incidentes", "media_diaria", "max_diario", "std_diario"]
    X_db = StandardScaler().fit_transform(top[feats_db].values)
    top["cluster_dbscan"] = DBSCAN(eps=0.8, min_samples=3).fit_predict(X_db)

    n_cl  = len(set(top["cluster_dbscan"])) - (1 if -1 in top["cluster_dbscan"].values else 0)
    n_rui = (top["cluster_dbscan"] == -1).sum()
    print(f"  DBSCAN: {n_cl} clusters | {n_rui} anomalias únicas")

    # Cruzar com incidentes manuais P2/P3
    df_man = df[
        (df["Aberto por"] == "Manual") &
        (df["Prioridade"].isin(["2 - Alta", "3 - Média"])) &
        (df["Item de configuração"].notna())
    ].groupby("Item de configuração").size().reset_index(name="total_manuais_p2p3")

    ic_stats = ic_stats.merge(df_man, on="Item de configuração", how="left").fillna(0)
    ic_stats["correlacao_risco"] = (
        ic_stats["total_manuais_p2p3"] / (ic_stats["total_incidentes"] + 1)
    )

    print(f"\n  Top 15 ICs mais barulhentos:")
    print(
        ic_stats.head(15)[
            ["Item de configuração", "total_incidentes", "max_diario", "indice_ruido", "correlacao_risco"]
        ].to_string(index=False)
    )

    joblib.dump(ic_stats, MODEL_DIR / "ic_barulhentos.pkl")
    ic_stats.to_csv(REPORT_DIR / "ic_barulhentos.csv", index=False)

    return ic_stats


# 4. ANÁLISE DE REINCIDÊNCIA (Contorno -> Reabertura)

def analisar_reincidencia(df: pd.DataFrame, janela_horas: int = 72) -> pd.DataFrame:
    """
    Detecta incidentes que reabriram no mesmo IC dentro de `janela_horas`
    após um encerramento com solução por CONTORNO.
    """
    print(f"\n[4/4] Reincidência após Contorno (janela={janela_horas}h)")

    df["dt_aberto"]    = pd.to_datetime(df["Aberto"],    errors="coerce")
    df["dt_encerrado"] = pd.to_datetime(df["Encerrado"], errors="coerce")

    contornos = df[
        (df["Solução"] == "Contorno") &
        (df["Item de configuração"].notna()) &
        (df["dt_encerrado"].notna())
    ][["Número", "Item de configuração", "Prioridade", "dt_encerrado"]].copy()
    contornos.columns = ["num_c", "ic", "prio_c", "enc_c"]

    todos = df[df["Item de configuração"].notna()][
        ["Número", "Item de configuração", "dt_aberto"]
    ].copy()
    todos.columns = ["num_r", "ic", "aberto_r"]

    merged = contornos.merge(todos, on="ic")
    merged = merged[merged["num_r"] != merged["num_c"]]
    merged["delta_h"] = (merged["aberto_r"] - merged["enc_c"]).dt.total_seconds() / 3600
    reinci = merged[(merged["delta_h"] > 0) & (merged["delta_h"] <= janela_horas)].copy()

    taxa = len(reinci) / (len(contornos) + 1) * 100
    print(f"  Contornos analisados:      {len(contornos):,}")
    print(f"  Reincidências ({janela_horas}h):  {len(reinci):,}")
    print(f"  Taxa de reincidência:      {taxa:.1f}%")

    top_ics = (
        reinci.groupby("ic").size()
        .sort_values(ascending=False).head(10)
        .reset_index().rename(columns={"ic": "IC", 0: "reincidencias"})
    )
    print(f"\n  Top ICs com mais reincidências:")
    print(top_ics.to_string(index=False))

    reinci.to_csv(REPORT_DIR / "reincidencias.csv", index=False)
    return reinci


# Pipeline principal

def executar_clusterizacao(caminho_parquet: str = None) -> dict:
    """Pipeline completo de clusterização com rastreamento MLflow."""
    if caminho_parquet is None:
        caminho_parquet = str(DATA_DIR / "incidents_clean.parquet")

    print("=" * 60)
    print("ETAPA 6 - CLUSTERIZAÇÃO")
    print("=" * 60)
    print(f"Carregando: {caminho_parquet}")

    df = pd.read_parquet(caminho_parquet)
    print(f"Shape: {df.shape}\n")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    resultados = {}

    with mlflow.start_run(run_name="clusterizacao_completa"):

        # 1. Perfil numérico
        res_perfil = treinar_kmeans_perfil(df)
        mlflow.log_param("kmeans_perfil_k", res_perfil["k"])
        mlflow.log_metric("kmeans_perfil_silhouette", res_perfil["silhouette"])
        resultados["perfil"] = res_perfil

        # 2. Causas por descrição
        res_causas = treinar_cluster_causas(df, n_clusters=8)
        mlflow.log_param("kmeans_causas_k", 8)
        resultados["causas"] = res_causas

        # 3. ICs barulhentos
        ic_stats = analisar_ics_barulhentos(df)
        mlflow.log_metric("ics_analisados", len(ic_stats))
        mlflow.log_metric("ics_alto_risco",
                          (ic_stats["correlacao_risco"] > 0.1).sum())
        resultados["ics"] = ic_stats

        # 4. Reincidência
        reinci = analisar_reincidencia(df, janela_horas=72)
        mlflow.log_metric("total_reincidencias_72h", len(reinci))
        resultados["reincidencias"] = reinci

        # Artefatos
        mlflow.log_artifacts(str(MODEL_DIR),  artifact_path="clustering_models")
        mlflow.log_artifacts(str(REPORT_DIR), artifact_path="clustering_reports")

    print("\n" + "=" * 60)
    print("ETAPA 6 CONCLUÍDA")
    print(f"  Modelos salvos em: {MODEL_DIR}")
    print(f"  Relatórios em:     {REPORT_DIR}")
    print("=" * 60)

    return resultados


# Execução direta

if __name__ == "__main__":
    resultados = executar_clusterizacao()
    print("\nResultados disponíveis:", list(resultados.keys()))
