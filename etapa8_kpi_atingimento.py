"""
etapa8_kpi_atingimento.py - Projeção de Atingimento dos KPIs
AIOps Locaweb Challenge - Etapa 8

Entrega o item-título do desafio: "Projeção de atingimento dos KPIs
(probabilidade de atingimento)".

Dois KPIs anuais (medidos mensalmente), por prioridade (2-Alta, 3-Média):
  KPI-OLA    = nº de incidentes com OLA quebrado no ano  (= kpi_violado_bin)
  KPI-VOLUME = nº de incidentes tratados no ano           (= entrou_para_kpi)
Cada contagem anual é mapeada em faixas de % de atingimento (dicionário de dados).
"Saudável" = atingimento >= 100%. Contagem MENOR -> atingimento MAIOR.

Método: a partir de um mês-corte, soma o realizado no ano até o mês (YTD) com uma
projeção Monte Carlo dos meses restantes (Poisson para quebras raras; Normal com
overdispersion empírica para volume), gerando a DISTRIBUIÇÃO do total anual -> daí
P(cada faixa), P(atingimento saudável) e a contagem esperada com intervalo.

Como rodar:
    python etapa8_kpi_atingimento.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

BASE_DIR      = Path(__file__).parent
ARQ_CLEAN     = BASE_DIR / "data" / "processed" / "incidents_clean.parquet"
MLRUNS_DIR    = BASE_DIR / "mlruns"
EXPERIMENT    = "locaweb-aiops"

# FAIXAS DE ATINGIMENTO (dicionário de dados)
# Cada faixa = (limite_superior_INCLUSIVO, pct). Ordenadas asc; última = (inf, 0).
FAIXAS = {
    ("OLA", "P2"): [(30, 150), (35, 125), (39, 100), (45, 75), (53, 50), (np.inf, 0)],
    ("OLA", "P3"): [(200, 150), (230, 125), (263, 100), (290, 75), (320, 50), (np.inf, 0)],
    ("VOL", "P2"): [(4584, 150), (5388, 125), (6168, 100), (6252, 75), (6336, 50), (np.inf, 0)],
    ("VOL", "P3"): [(19488, 150), (22116, 125), (22524, 100), (23892, 75), (24276, 50), (np.inf, 0)],
}
PRIORIDADES = {"P2": "2 - Alta", "P3": "3 - Média"}
PCT_VALIDOS = [150, 125, 100, 75, 50, 0]
SAUDAVEL_MIN = 100          # atingimento >= 100% é "saudável"


def atingimento_vec(counts, faixas):
    """Mapeia contagens (array) -> % de atingimento (array), vetorizado."""
    lims = np.array([f[0] for f in faixas], dtype=float)
    pcts = np.array([f[1] for f in faixas], dtype=float)
    idx = np.searchsorted(lims, np.asarray(counts, dtype=float), side="left")
    idx = np.clip(idx, 0, len(pcts) - 1)
    return pcts[idx]


# Séries mensais
def carregar_series(ano: int):
    """
    Retorna dict {(metric, pc): np.array(12,)} com a contagem mensal (meses 1..12)
    do ano informado, para metric in {'OLA','VOL'} e pc in {'P2','P3'}.
    """
    df = pd.read_parquet(ARQ_CLEAN, columns=[
        "dt_aberto", "prioridade_cod", "entrou_para_kpi", "kpi_violado_bin"
    ])
    df = df[df["entrou_para_kpi"] & (df["dt_aberto"].dt.year == ano)].copy()
    df["mes"] = df["dt_aberto"].dt.month
    out = {}
    for pc in PRIORIDADES:
        sub = df[df["prioridade_cod"] == pc]
        vol = sub.groupby("mes").size().reindex(range(1, 13), fill_value=0).values.astype(float)
        bre = sub.groupby("mes")["kpi_violado_bin"].sum().reindex(range(1, 13), fill_value=0).values.astype(float)
        out[("VOL", pc)] = vol
        out[("OLA", pc)] = bre
    return out


# Projeção monte carlo
def projetar(serie12, metric, pc, mes_corte, n_sim=30000, seed=42):
    """
    serie12: array (12,) com a contagem mensal realizada do ano.
    mes_corte: usa meses 1..mes_corte como realizado (YTD) e projeta o resto.
               mes_corte=0 -> projeta o ano inteiro (uso forward, sem YTD).
    Retorna dict com estatísticas de atingimento.
    """
    faixas = FAIXAS[(metric, pc)]
    rng = np.random.default_rng(seed)

    obs = serie12[:mes_corte] if mes_corte > 0 else np.array([])
    ytd = float(obs.sum())
    rem = 12 - mes_corte

    if rem > 0:
        base = obs if obs.size > 0 else serie12          # sem YTD -> usa o próprio ano como base de taxa
        rate = float(base.mean())
        if metric == "OLA":                              # quebras: contagem rara -> Poisson
            fut = rng.poisson(max(rate, 1e-9), size=(n_sim, rem)).sum(axis=1)
        else:                                            # volume: alto e com overdispersion -> Normal empírica
            sd = float(base.std(ddof=1)) if base.size > 1 else np.sqrt(max(rate, 1.0))
            fut = rng.normal(rate, max(sd, 1e-6), size=(n_sim, rem)).clip(min=0).sum(axis=1)
    else:
        fut = np.zeros(n_sim)

    total = ytd + fut
    pct = atingimento_vec(total, faixas)

    p_faixa = {int(p): float((pct == p).mean()) for p in PCT_VALIDOS}
    return {
        "metric": metric, "pc": pc, "mes_corte": mes_corte,
        "ytd": ytd,
        "total_esperado": float(total.mean()),
        "total_p05": float(np.percentile(total, 5)),
        "total_p95": float(np.percentile(total, 95)),
        "atingimento_esperado": float(pct.mean()),
        "faixa_provavel": int(PCT_VALIDOS[int(np.argmax([p_faixa[p] for p in PCT_VALIDOS]))]),
        "p_saudavel": float((pct >= SAUDAVEL_MIN).mean()),
        "p_faixa": p_faixa,
    }


# Relatório
def _linha(r, atingimento_real=None):
    real = f"{atingimento_real:>4.0f}%" if atingimento_real is not None else "  - "
    return (f"    corte M{r['mes_corte']:>2}  | YTD {r['ytd']:>7.0f} | "
            f"projeção {r['total_esperado']:>8.1f} "
            f"[{r['total_p05']:>7.1f}, {r['total_p95']:>8.1f}] | "
            f"faixa {r['faixa_provavel']:>3}% | P(saudável) {r['p_saudavel']:>6.1%} | real {real}")


def demonstrar(ano=2025, cortes=(3, 6, 9, 12)):
    print("=" * 78)
    print(f"  AIOps Locaweb - Etapa 8: Projeção de Atingimento dos KPIs - ANO {ano}")
    print("=" * 78)

    series = carregar_series(ano)
    resumo_final = {}

    if _HAS_MLFLOW:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or MLRUNS_DIR.resolve().as_uri())
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", EXPERIMENT))
        ctx = mlflow.start_run(run_name=f"kpi_atingimento_{ano}")
        ctx.__enter__()

    for metric, titulo in [("OLA", "KPI-OLA (quebras de OLA no ano)"),
                           ("VOL", "KPI-VOLUME (incidentes tratados no ano)")]:
        print(f"\n-- {titulo} --")
        for pc, nome in PRIORIDADES.items():
            serie = series[(metric, pc)]
            real_total = float(serie.sum())
            real_pct = float(atingimento_vec([real_total], FAIXAS[(metric, pc)])[0])
            print(f"\n  {nome}  |  realizado no ano: {real_total:.0f}  ->  atingimento {real_pct:.0f}%"
                  f"  ({'SAUDÁVEL' if real_pct >= SAUDAVEL_MIN else 'ABAIXO DA META'})")
            for mc in cortes:
                r = projetar(serie, metric, pc, mc)
                print(_linha(r, atingimento_real=real_pct if mc == 12 else None))
            # registro do ponto de decisão típico (meio do ano, M6)
            r6 = projetar(serie, metric, pc, 6)
            resumo_final[f"{metric}_{pc}_real_pct"] = real_pct
            resumo_final[f"{metric}_{pc}_M6_p_saudavel"] = r6["p_saudavel"]
            resumo_final[f"{metric}_{pc}_M6_total_esperado"] = r6["total_esperado"]

    # Projeção forward (ano seguinte, sem ground truth)
    print(f"\n-- Projeção FORWARD {ano + 1} (taxa base = {ano}, sem realizado) --")
    for metric in ("OLA", "VOL"):
        for pc, nome in PRIORIDADES.items():
            r = projetar(series[(metric, pc)], metric, pc, mes_corte=0)
            print(f"  {metric:3} {nome:10} | esperado {r['total_esperado']:>8.1f} "
                  f"[{r['total_p05']:>7.1f}, {r['total_p95']:>8.1f}] | "
                  f"faixa provável {r['faixa_provavel']:>3}% | P(saudável) {r['p_saudavel']:>6.1%}")

    if _HAS_MLFLOW:
        mlflow.log_params({"ano": ano, "n_sim": 30000, "saudavel_min_pct": SAUDAVEL_MIN})
        mlflow.log_metrics({k: v for k, v in resumo_final.items()})
        ctx.__exit__(None, None, None)

    print("\n" + "=" * 78)
    print("  Leitura: P(saudável) = probabilidade de o total anual cair em faixa >= 100%.")
    print("  KPI-OLA e KPI-VOLUME projetados por prioridade, com incerteza (IC 5-95%).")
    print("=" * 78)
    return resumo_final


if __name__ == "__main__":
    demonstrar(ano=2025)
