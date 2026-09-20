import pandas as pd
import numpy as np
from pathlib import Path
import os
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTES (standalone - duplicadas do config.py para arquivo ser autossuficiente)
# ==============================================================================

# Limites oficiais do Dicionário de Dados (campo Duração / relógio de SLA).
# Fonte: "Dicionário de Dados - v2" -> 1-Crítica 4h, 2-Alta 4h, 3-Média 12h,
# 4-Baixa 24h, 5-Muito Baixa 96h. (Consistente com src/config.py.)
OLA_LIMITE_HORAS = {
    "P1": 4.0,
    "P2": 4.0,
    "P3": 12.0,
    "P4": 24.0,
    "P5": 96.0,
}

PRIORIDADES_KPI = ["P1", "P2", "P3"]

# Caminhos
BASE_DIR       = Path(__file__).parent
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"
DATA_SPLIT_DIR = BASE_DIR / "data" / "splits"
ARQUIVO_FONTE  = DATA_RAW_DIR / "LW-DATASET.xlsx"

# Garante que os diretórios existam
for _dir in [DATA_RAW_DIR, DATA_PROC_DIR, DATA_SPLIT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. INGESTÃO
# ==============================================================================

def carregar_dataset(caminho: Path = ARQUIVO_FONTE) -> pd.DataFrame:
    """
    Lê LW-DATASET.xlsx com openpyxl e padroniza os nomes de colunas.
    Retorna DataFrame raw.
    """
    print(f"[LOADER] Lendo arquivo: {caminho}")

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Coloque o arquivo LW-DATASET.xlsx em data/raw/ antes de rodar."
        )

    df = pd.read_excel(caminho, engine="openpyxl")
    print(f"[LOADER] Shape bruto: {df.shape}")

    # Mapeia nomes originais -> nomes limpos para lookup (não renomeia ainda)
    col_map = {col: col.strip() for col in df.columns}
    df.rename(columns=col_map, inplace=True)

    print(f"[LOADER] Colunas encontradas: {list(df.columns)}")
    return df


# ==============================================================================
# 2. LIMPEZA
# ==============================================================================

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza e tipagem de todas as colunas do dataset.
    Retorna DataFrame limpo.
    """
    print("[PREPROCESSOR] Iniciando limpeza de dados...")
    df = df.copy()

    # Datas
    for col in ["Aberto", "Resolvido", "Encerrado"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            n_nulos = df[col].isna().sum()
            print(f"  [parse] '{col}': {n_nulos} nulos após parse")

    # Duração (segundos)
    if "Duração" in df.columns:
        df["Duração"] = pd.to_numeric(df["Duração"], errors="coerce").fillna(0)

    # Campos de texto com strip simples
    campos_texto = ["Prioridade", "Status", "Aberto por", "Solução"]
    for campo in campos_texto:
        if campo in df.columns:
            df[campo] = df[campo].astype(str).str.strip()

    # Entrou para KPI?
    if "Entrou para KPI?" in df.columns:
        df["Entrou para KPI?"] = (
            df["Entrou para KPI?"].astype(str).str.strip().str.upper()
        )

    # KPI Violado?
    if "KPI Violado?" in df.columns:
        df["KPI Violado?"] = (
            df["KPI Violado?"].astype(str).str.strip().str.upper()
        )
        # "N/A", "NAN", "" -> NaN
        df["KPI Violado?"] = df["KPI Violado?"].replace(
            {"N/A": np.nan, "NAN": np.nan, "": np.nan}
        )

    # Incidente Pai
    if "Incidente Pai" in df.columns:
        df["Incidente Pai"] = df["Incidente Pai"].astype(str).str.strip()
        # astype(str) transforma NaN em "nan" (minúsculo); comparar em lower
        # garante que os sentinelas virem NaN de fato (senão "nan" fica literal).
        sentinelas = {"nan", "none", "nat", "", "0"}
        df["Incidente Pai"] = df["Incidente Pai"].where(
            ~df["Incidente Pai"].str.lower().isin(sentinelas), np.nan
        )

    # Preencher nulos de texto com ""
    cols_objeto = df.select_dtypes(include="object").columns
    df[cols_objeto] = df[cols_objeto].fillna("")

    print(f"[PREPROCESSOR] Limpeza concluída. Shape: {df.shape}")
    return df


# ==============================================================================
# 3. COLUNAS DE DATAS DERIVADAS
# ==============================================================================

def calcular_data_resolucao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria colunas padronizadas de datas:
      - dt_aberto      : cópia de 'Aberto'
      - dt_encerrado   : cópia de 'Encerrado'
      - dt_resolucao   : COALESCE(Resolvido, Encerrado)
                         Se ambas preenchidas -> usa Resolvido
                         Se só Encerrado      -> usa Encerrado
    """
    print("[PREPROCESSOR] Calculando colunas de data...")
    df = df.copy()

    # Garante que as colunas existem como datetime
    for col in ["Aberto", "Resolvido", "Encerrado"]:
        if col not in df.columns:
            df[col] = pd.NaT

    df["dt_aberto"]    = df["Aberto"]
    df["dt_encerrado"] = df["Encerrado"]

    # COALESCE: Resolvido tem prioridade; cai para Encerrado se Resolvido for NaT
    df["dt_resolucao"] = df["Resolvido"].combine_first(df["Encerrado"])

    n_resolucao = df["dt_resolucao"].notna().sum()
    print(f"  [datas] dt_resolucao preenchida: {n_resolucao} / {len(df)}")

    return df


# ==============================================================================
# 4. FLAGS E FEATURES DE KPI
# ==============================================================================

def calcular_kpi_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula flags de KPI e cria features derivadas:
      - entrou_para_kpi     : bool - regras de negócio
      - kpi_violado_bin     : bool - entrou_kpi E KPI Violado? == SIM
      - duracao_horas       : float - Duração / 3600
      - ola_limite_horas    : float - OLA por prioridade
      - violou_ola_calculado: bool  - duracao_horas > ola_limite_horas
      - mes_encerramento    : str   - período "YYYY-MM" (mês de encerramento)
    """
    print("[PREPROCESSOR] Calculando flags de KPI e OLA...")
    df = df.copy()

    # ------------------------------------------------------------------
    # prioridade_cod  - normaliza "2 - Alta" -> "P2", "3 - Média" -> "P3", etc.
    # As constantes PRIORIDADES_KPI / OLA_LIMITE_HORAS usam códigos P1..P5;
    # o dataset traz rótulos por extenso ("2 - Alta"). Extrair o dígito
    # inicial torna o match imune a acento/encoding.
    # ------------------------------------------------------------------
    df["prioridade_cod"] = "P" + df["Prioridade"].str.extract(r"^\s*(\d)")[0]

    # ------------------------------------------------------------------
    # entrou_para_kpi
    # ------------------------------------------------------------------
    cond_prioridade    = df["prioridade_cod"].isin(PRIORIDADES_KPI)
    cond_status        = df["Status"].str.upper() != "SEM INTERVENÇÃO"
    cond_sem_pai       = df["Incidente Pai"].isna() | (df["Incidente Pai"] == "")

    df["entrou_para_kpi"] = cond_prioridade & cond_status & cond_sem_pai

    n_kpi = df["entrou_para_kpi"].sum()
    print(f"  [kpi] Registros que entram para KPI: {n_kpi}")

    # ------------------------------------------------------------------
    # kpi_violado_bin
    # ------------------------------------------------------------------
    cond_violado = df["KPI Violado?"].str.upper() == "SIM"
    df["kpi_violado_bin"] = df["entrou_para_kpi"] & cond_violado

    n_violado = df["kpi_violado_bin"].sum()
    print(f"  [kpi] KPI violados (binário): {n_violado}")

    # ------------------------------------------------------------------
    # duracao_horas
    # ------------------------------------------------------------------
    df["duracao_horas"] = df["Duração"] / 3600.0

    # ------------------------------------------------------------------
    # ola_limite_horas  (por prioridade)
    # ------------------------------------------------------------------
    df["ola_limite_horas"] = df["prioridade_cod"].map(OLA_LIMITE_HORAS)

    # ------------------------------------------------------------------
    # violou_ola_calculado  (validação independente do campo original)
    # ------------------------------------------------------------------
    df["violou_ola_calculado"] = df["duracao_horas"] > df["ola_limite_horas"]

    # ------------------------------------------------------------------
    # mes_encerramento  (OLA contabilizado no mês de encerramento)
    # ------------------------------------------------------------------
    df["mes_encerramento"] = (
        df["dt_encerrado"]
        .dt.to_period("M")
        .astype(str)
    )
    # Registros sem dt_encerrado ficam como "NaT"
    df["mes_encerramento"] = df["mes_encerramento"].replace("NaT", np.nan)

    return df


# ==============================================================================
# 5. SPLITS TEMPORAIS
# ==============================================================================

def gerar_splits(df: pd.DataFrame) -> dict:
    """
    Gera splits temporais (80/10/10) ordenados por dt_aberto.
    Salva cada split como Parquet e imprime estatísticas.
    Retorna dict {"train": df_train, "val": df_val, "test": df_test}.
    """
    print("\n[SPLITS] Gerando splits temporais (80 / 10 / 10)...")

    # Ordena cronologicamente (sem embaralhar - split temporal)
    df_sorted = df.sort_values("dt_aberto").reset_index(drop=True)

    n = len(df_sorted)
    i_val  = int(n * 0.80)
    i_test = int(n * 0.90)

    splits = {
        "train": df_sorted.iloc[:i_val].copy(),
        "val":   df_sorted.iloc[i_val:i_test].copy(),
        "test":  df_sorted.iloc[i_test:].copy(),
    }

    print(f"\n{'Split':<8} {'Registros':>10} {'% dataset':>10} {'OLA Violado':>13} {'% Violação':>12}")
    print("-" * 58)

    for nome, split_df in splits.items():
        total    = len(split_df)
        pct_ds   = total / n * 100
        violados = split_df["kpi_violado_bin"].sum()
        pct_viol = violados / total * 100 if total > 0 else 0.0

        print(f"{nome:<8} {total:>10,} {pct_ds:>9.1f}% {violados:>13,} {pct_viol:>11.1f}%")

        # Salva parquet
        destino = DATA_SPLIT_DIR / f"{nome}.parquet"
        split_df.to_parquet(destino, index=False)
        print(f"  [salvo] {destino}")

    return splits


# ==============================================================================
# 6. PIPELINE COMPLETO
# ==============================================================================

def pipeline_completo():
    """
    Executa todas as etapas de ingestão e pré-processamento em sequência.
    """
    print("=" * 65)
    print("  AIOps Locaweb - Etapa 2: Ingestão e Pré-processamento")
    print("=" * 65)

    # 1. Ingestão
    df = carregar_dataset()

    # 2. Limpeza
    df = limpar_dados(df)

    # 3. Datas derivadas
    df = calcular_data_resolucao(df)

    # 4. Flags e features de KPI
    df = calcular_kpi_flags(df)

    # 5. Salvar dataset limpo
    destino_clean = DATA_PROC_DIR / "incidents_clean.parquet"
    df.to_parquet(destino_clean, index=False)
    print(f"\n[SAVE] Dataset limpo salvo em: {destino_clean}")

    # 6. Splits temporais
    splits = gerar_splits(df)

    # 7. Estatísticas finais
    print("\n" + "=" * 65)
    print("  ESTATÍSTICAS FINAIS DO DATASET")
    print("=" * 65)

    total = len(df)
    n_kpi     = df["entrou_para_kpi"].sum()
    n_violado = df["kpi_violado_bin"].sum()
    pct_total = n_violado / n_kpi * 100 if n_kpi > 0 else 0.0

    print(f"  Total de registros       : {total:,}")
    print(f"  Entram para KPI          : {n_kpi:,}")
    print(f"  OLA violados             : {n_violado:,}  ({pct_total:.1f}% dos que entram no KPI)")

    # % violação por prioridade P2 e P3
    for pri in ["P2", "P3"]:
        mask_pri  = df["prioridade_cod"] == pri
        mask_kpi  = df["entrou_para_kpi"] & mask_pri
        mask_viol = df["kpi_violado_bin"] & mask_pri
        n_pri     = mask_kpi.sum()
        n_viol    = mask_viol.sum()
        pct       = n_viol / n_pri * 100 if n_pri > 0 else 0.0
        print(f"  Violação {pri}             : {n_viol:,} / {n_pri:,}  ({pct:.1f}%)")

    # Período do dataset
    dt_min = df["dt_aberto"].min()
    dt_max = df["dt_aberto"].max()
    print(f"  Período do dataset       : {dt_min:%Y-%m-%d}  ->  {dt_max:%Y-%m-%d}")

    print("\n[DONE] Etapa 2 concluída com sucesso.")
    print("=" * 65)

    return df, splits


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pipeline_completo()
