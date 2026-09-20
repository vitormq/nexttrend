"""
etapa4_features.py - Engenharia de Features (anti-leakage)
AIOps Locaweb Challenge - Etapa 4

Sente-se ENTRE a Etapa 2 e a Etapa 5:
    Etapa 2  -> data/processed/incidents_clean.parquet
    Etapa 4  -> adiciona features point-in-time e RE-GERA os splits
    Etapa 5  -> lê data/splits/{train,val,test}.parquet

Princípio central - SEM VAZAMENTO (leakage):
    Todas as features usam APENAS informação conhecida no MOMENTO DA ABERTURA
    do incidente. Nada que só exista após a resolução entra como feature:
      - Duração / duracao_horas / Resolvido / Encerrado / dt_resolucao
      - Solução / Código de fechamento / Status final
      - KPI Violado? (é o alvo)
    Features históricas (violação/duração de IC e grupo) são calculadas somente
    sobre incidentes JÁ ENCERRADOS antes da abertura do incidente atual, para
    não usar rótulos que ainda não seriam conhecidos operacionalmente.

Como rodar:
    python etapa4_features.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Constantes / caminhos
BASE_DIR       = Path(__file__).parent
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"
DATA_SPLIT_DIR = BASE_DIR / "data" / "splits"
ARQ_CLEAN      = DATA_PROC_DIR / "incidents_clean.parquet"
ARQ_FEATURES   = DATA_PROC_DIR / "incidents_features.parquet"

for _d in (DATA_PROC_DIR, DATA_SPLIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

NAT_INT = np.iinfo(np.int64).min           # datetime64[ns] NaT -> INT64_MIN após astype('int64')
NS_DIA  = np.int64(24 * 3600 * 1_000_000_000)

# Colunas que NUNCA podem virar feature (só conhecidas após a resolução, ou são o alvo)
COLUNAS_LEAKY = [
    "Resolvido", "Encerrado", "dt_encerrado", "dt_resolucao",
    "Duração", "duracao_horas", "Código de fechamento", "Solução",
    "Status", "KPI Violado?", "Entrou para KPI?",
    "violou_ola_calculado", "kpi_violado_bin",
]


# HELPERS NUMÉRICOS (janelas temporais point-in-time)
def _prior_window_count(t_ns, indices, win_ns):
    """Nº de incidentes ANTERIORES do grupo dentro da janela [t-win, t)."""
    out = np.zeros(len(t_ns))
    for idx in indices.values():
        idx = np.sort(idx)
        tt = t_ns[idx]                                   # ascendente (df ordenado por dt_aberto)
        lo = np.searchsorted(tt, tt - win_ns, side="left")
        out[idx] = np.arange(len(tt)) - lo
    return out


def _closed_before(topen_ns, tclose_ns, viol, dur, indices):
    """
    Para cada incidente, agrega sobre os incidentes do grupo JÁ ENCERRADOS
    estritamente antes da sua abertura. Retorna (count, soma_violacoes, soma_duracao).
    """
    n = len(topen_ns)
    cnt = np.zeros(n); vsum = np.zeros(n); dsum = np.zeros(n)
    for idx in indices.values():
        idx = np.sort(idx)
        to_ = topen_ns[idx]; tc = tclose_ns[idx]
        v = viol[idx]; d = dur[idx]
        valid = tc > NAT_INT
        ci = np.where(valid)[0]
        if ci.size == 0:
            continue
        cc = tc[ci]
        order = np.argsort(cc, kind="mergesort")
        cc_s = cc[order]
        cum_v = np.concatenate([[0.0], np.cumsum(v[ci][order])])
        cum_d = np.concatenate([[0.0], np.cumsum(d[ci][order])])
        k = np.searchsorted(cc_s, to_, side="left")      # encerrados antes da abertura
        cnt[idx] = k
        vsum[idx] = cum_v[k]
        dsum[idx] = cum_d[k]
    return cnt, vsum, dsum


def _open_load(topen_ns, tclose_ns, indices):
    """Nº de incidentes do grupo ABERTOS e ainda não encerrados no instante da abertura (exclui o próprio)."""
    n = len(topen_ns); out = np.zeros(n)
    for idx in indices.values():
        idx = np.sort(idx)
        to_ = topen_ns[idx]; tc = tclose_ns[idx]
        opened_sorted = np.sort(to_)
        closed_sorted = np.sort(tc[tc > NAT_INT])
        opened_le = np.searchsorted(opened_sorted, to_, side="right")
        closed_le = np.searchsorted(closed_sorted, to_, side="right")
        out[idx] = np.clip(opened_le - closed_le - 1, 0, None)   # -1 remove o próprio
    return out


def _delayed_load(topen_ns, tclose_ns, ola_horas, indices):
    """Nº de incidentes do grupo abertos que JÁ estouraram seu OLA no instante da abertura atual."""
    n = len(topen_ns); out = np.zeros(n)
    breach = topen_ns.astype(float) + ola_horas.astype(float) * 3600e9   # instante em que estoura o OLA
    for idx in indices.values():
        idx = np.sort(idx)
        to_ = topen_ns[idx].astype(float)
        tc = tclose_ns[idx]
        bs = breach[idx]
        end = np.where(tc > NAT_INT, tc.astype(float), np.inf)
        contrib = bs < end                                   # só conta se estourou antes de encerrar
        bs_s = np.sort(bs[contrib])
        end_s = np.sort(end[contrib])
        started = np.searchsorted(bs_s, to_, side="right")
        ended = np.searchsorted(end_s, to_, side="right")
        out[idx] = np.clip(started - ended, 0, None)
    return out


# Builders de features
def add_temporais(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["dt_aberto"]
    df["hora_abertura"]        = dt.dt.hour
    df["dia_semana"]           = dt.dt.dayofweek
    df["dia_mes"]              = dt.dt.day
    df["mes"]                  = dt.dt.month
    df["trimestre"]            = dt.dt.quarter
    df["is_weekend"]           = (dt.dt.dayofweek >= 5).astype(int)
    df["is_horario_comercial"] = dt.dt.hour.between(8, 18).astype(int)
    df["is_madrugada"]         = dt.dt.hour.between(0, 5).astype(int)
    df["is_fim_de_mes"]        = (dt.dt.day >= 25).astype(int)
    return df


def add_texto(df: pd.DataFrame) -> pd.DataFrame:
    s = df["Descrição resumida"].astype(str).fillna("")
    low = s.str.lower()
    df["desc_n_chars"]    = s.str.len()
    df["desc_n_palavras"] = s.str.split().apply(len)
    df["desc_tem_erro"]    = low.str.contains(r"erro|error|falha|fail",           regex=True).astype(int)
    df["desc_tem_lento"]   = low.str.contains(r"lent|slow|demora|timeout",         regex=True).astype(int)
    df["desc_tem_fora"]    = low.str.contains(r"fora|down|indispon|offline|unavail", regex=True).astype(int)
    df["desc_tem_critico"] = low.str.contains(r"critic|crític|urgente|grave",      regex=True).astype(int)
    df["desc_tem_cliente"] = low.str.contains(r"client|customer|usuari|usuári",    regex=True).astype(int)
    return df


def add_historico_e_carga(df: pd.DataFrame) -> pd.DataFrame:
    topen  = df["dt_aberto"].values.astype("int64")
    tclose = df["dt_encerrado"].values.astype("int64")
    viol   = df["kpi_violado_bin"].astype(float).values
    dur    = df["duracao_horas"].astype(float).values
    ola    = df["ola_limite_horas"].fillna(24.0).astype(float).values

    ic_idx   = df.groupby("Item de configuração").indices
    gr_idx   = df.groupby("Grupo designado").indices
    date_idx = df.groupby(df["dt_aberto"].dt.normalize()).indices

    # Volume por janela (open-based, estritamente anterior)
    df["ic_vol_7d"]     = _prior_window_count(topen, ic_idx, 7 * NS_DIA)
    df["ic_vol_30d"]    = _prior_window_count(topen, ic_idx, 30 * NS_DIA)
    df["grupo_vol_7d"]  = _prior_window_count(topen, gr_idx, 7 * NS_DIA)
    df["grupo_vol_30d"] = _prior_window_count(topen, gr_idx, 30 * NS_DIA)
    df["ic_barulhento_medio_dia"] = df["ic_vol_30d"] / 30.0

    # Histórico de violação/duração - SOMENTE encerrados antes da abertura (anti-leakage de rótulo)
    ic_cnt, ic_vs, ic_ds = _closed_before(topen, tclose, viol, dur, ic_idx)
    gr_cnt, gr_vs, gr_ds = _closed_before(topen, tclose, viol, dur, gr_idx)
    df["ic_taxa_violacao_30d"]    = np.where(ic_cnt > 0, ic_vs / ic_cnt, 0.0)
    df["ic_taxa_violacao_7d"]     = df["ic_taxa_violacao_30d"]     # mesmo estimador histórico
    df["ic_media_duracao_30d"]    = np.where(ic_cnt > 0, ic_ds / ic_cnt, 0.0)
    df["grupo_taxa_violacao_30d"] = np.where(gr_cnt > 0, gr_vs / gr_cnt, 0.0)
    df["grupo_taxa_violacao_7d"]  = df["grupo_taxa_violacao_30d"]
    df["grupo_media_duracao_30d"] = np.where(gr_cnt > 0, gr_ds / gr_cnt, 0.0)

    # Carga operacional no instante da abertura (point-in-time)
    df["carga_grupo_abertos"]   = _open_load(topen, tclose, gr_idx)
    df["carga_ic_abertos"]      = _open_load(topen, tclose, ic_idx)
    df["carga_grupo_atrasados"] = _delayed_load(topen, tclose, ola, gr_idx)

    # Acumuladas do dia (estritamente anterior; violações só de encerrados-antes)
    d_cnt, d_vs, _ = _closed_before(topen, tclose, viol, dur, date_idx)
    df["vol_dia_ate_agora"]          = df.groupby(df["dt_aberto"].dt.normalize()).cumcount().astype(float)
    df["violacoes_dia_ate_agora"]    = d_vs
    df["taxa_violacao_dia_ate_agora"] = np.where(
        df["vol_dia_ate_agora"] > 0, d_vs / df["vol_dia_ate_agora"], 0.0
    )
    return df


def add_interacoes(df: pd.DataFrame) -> pd.DataFrame:
    df["prioridade_num"] = df["prioridade_cod"].str.extract(r"(\d)").astype(float)
    is_alta = (df["prioridade_num"] <= 2).astype(int)          # P1/P2 = alta criticidade
    df["alta_x_barulhento"]  = is_alta * df["ic_vol_30d"]
    df["alta_x_carga_grupo"] = is_alta * df["carga_grupo_abertos"]
    df["hora_x_prioridade"]  = df["hora_abertura"] * df["prioridade_num"]
    return df


# Lista das features geradas nesta etapa (para auditoria/print)
FEATURES_GERADAS = [
    "hora_abertura", "dia_semana", "dia_mes", "mes", "trimestre",
    "is_weekend", "is_horario_comercial", "is_madrugada", "is_fim_de_mes",
    "prioridade_num",
    "desc_n_chars", "desc_n_palavras", "desc_tem_erro", "desc_tem_lento",
    "desc_tem_fora", "desc_tem_critico", "desc_tem_cliente",
    "ic_vol_7d", "ic_vol_30d", "ic_barulhento_medio_dia",
    "ic_taxa_violacao_7d", "ic_taxa_violacao_30d", "ic_media_duracao_30d",
    "grupo_vol_7d", "grupo_vol_30d",
    "grupo_taxa_violacao_7d", "grupo_taxa_violacao_30d", "grupo_media_duracao_30d",
    "carga_grupo_abertos", "carga_ic_abertos", "carga_grupo_atrasados",
    "vol_dia_ate_agora", "violacoes_dia_ate_agora", "taxa_violacao_dia_ate_agora",
    "alta_x_barulhento", "alta_x_carga_grupo", "hora_x_prioridade",
]


# SPLITS TEMPORAIS (mesma regra da Etapa 2: 80/10/10 por dt_aberto)
def gerar_splits(df: pd.DataFrame) -> dict:
    print("\n[SPLITS] Re-gerando splits temporais (80 / 10 / 10) com features...")
    df_sorted = df.sort_values("dt_aberto").reset_index(drop=True)
    n = len(df_sorted)
    i_val, i_test = int(n * 0.80), int(n * 0.90)
    splits = {
        "train": df_sorted.iloc[:i_val].copy(),
        "val":   df_sorted.iloc[i_val:i_test].copy(),
        "test":  df_sorted.iloc[i_test:].copy(),
    }
    print(f"\n{'Split':<8} {'Registros':>10} {'OLA Violado':>13} {'% Violação':>12}")
    print("-" * 46)
    for nome, sdf in splits.items():
        tot = len(sdf); vio = int(sdf["kpi_violado_bin"].sum())
        pct = vio / tot * 100 if tot else 0.0
        print(f"{nome:<8} {tot:>10,} {vio:>13,} {pct:>11.2f}%")
        sdf.to_parquet(DATA_SPLIT_DIR / f"{nome}.parquet", index=False)
    return splits


# Pipeline
def pipeline_features():
    print("=" * 65)
    print("  AIOps Locaweb - Etapa 4: Engenharia de Features (anti-leakage)")
    print("=" * 65)

    if not ARQ_CLEAN.exists():
        raise FileNotFoundError(
            f"{ARQ_CLEAN} não encontrado. Rode a Etapa 2 antes da Etapa 4."
        )

    df = pd.read_parquet(ARQ_CLEAN)
    print(f"[LOAD] incidents_clean.parquet: {df.shape}")

    # Ordena cronologicamente ANTES de qualquer feature histórica
    df = df.sort_values("dt_aberto").reset_index(drop=True)

    print("[FEATURES] Temporais...")
    df = add_temporais(df)
    print("[FEATURES] Texto (descrição resumida)...")
    df = add_texto(df)
    print("[FEATURES] Histórico point-in-time (IC/grupo) + carga operacional...")
    df = add_historico_e_carga(df)
    print("[FEATURES] Interações...")
    df = add_interacoes(df)

    # Garante ausência de NaN nas features geradas
    df[FEATURES_GERADAS] = df[FEATURES_GERADAS].fillna(0)

    # Auditoria anti-leakage
    print("\n[AUDIT] Features geradas:", len(FEATURES_GERADAS))
    vazando = [c for c in FEATURES_GERADAS if c in COLUNAS_LEAKY]
    print(f"[AUDIT] Features geradas que colidem com colunas leaky: {vazando if vazando else 'NENHUMA '}")
    print(f"[AUDIT] Colunas leaky preservadas no dataset (NÃO usar como feature): {COLUNAS_LEAKY}")

    df.to_parquet(ARQ_FEATURES, index=False)
    print(f"\n[SAVE] Dataset com features salvo: {ARQ_FEATURES}  (shape {df.shape})")

    gerar_splits(df)

    print("\n[DONE] Etapa 4 concluída. Rode a Etapa 5 para treinar com as novas features.")
    print("=" * 65)
    return df


if __name__ == "__main__":
    pipeline_features()
