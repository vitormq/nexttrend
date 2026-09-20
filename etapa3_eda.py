"""
etapa3_eda.py - Análise exploratória dos incidentes.
Distribuições, sazonalidade, duração vs. limite de OLA, ICs recorrentes,
grupos de risco e correlações. Gera gráficos HTML e um resumo em reports/eda/.

Roda com: python etapa3_eda.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Caminhos
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR   = BASE_DIR / "reports" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLA_LIMITES = {
    "1 - Crítica": 4, "2 - Alta": 4,
    "3 - Média": 12, "4 - Baixa": 24, "5 - Muito Baixa": 96,
}


def salvar(fig, nome: str) -> None:
    """Salva figura Plotly como HTML interativo."""
    caminho = OUT_DIR / nome
    fig.write_html(str(caminho))
    print(f"  {nome}")


# Carregar e preparar dados

def carregar_dados() -> pd.DataFrame:
    caminho = DATA_DIR / "incidents_clean.parquet"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Execute a Etapa 2 (etapa2_preprocessamento.py) primeiro."
        )
    df = pd.read_parquet(caminho)
    print(f"Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    # Parsear datas
    df["dt_aberto"]    = pd.to_datetime(df["Aberto"],    errors="coerce")
    df["dt_encerrado"] = pd.to_datetime(df["Encerrado"], errors="coerce")

    # Features de tempo
    df["hora"]        = df["dt_aberto"].dt.hour
    df["dia_semana"]  = df["dt_aberto"].dt.dayofweek
    df["mes"]         = df["dt_aberto"].dt.month
    df["nome_dia"]    = df["dt_aberto"].dt.day_name()

    # Duração em horas
    df["duracao_horas"] = df["Duração"] / 3600

    # OLA violado flag
    df["ola_violado"] = (df["KPI Violado?"] == "SIM")

    # Mês de encerramento
    df["mes_enc"] = df["dt_encerrado"].dt.to_period("M").astype(str)

    # Flag KPI (fallback se não existir)
    if "entrou_para_kpi" not in df.columns:
        df["entrou_para_kpi"] = (
            df["Prioridade"].isin(["1 - Crítica", "2 - Alta", "3 - Média"]) &
            (df["Status"] != "Sem Intervenção") &
            (~df["Incidente Pai"].notna())
        )

    return df


# Análises

def eda_01_distribuicao(df: pd.DataFrame) -> None:
    """Distribuição por prioridade e status."""
    print("\n[01] Distribuição por Prioridade e Status")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Distribuição por Prioridade",
                                        "Distribuição por Status"))

    prio = df["Prioridade"].value_counts().reset_index()
    prio.columns = ["Prioridade", "Quantidade"]
    fig.add_trace(
        go.Bar(x=prio["Prioridade"], y=prio["Quantidade"],
               marker_color=["#C0392B","#E74C3C","#E67E22","#3498DB","#95A5A6"],
               name="Prioridade"),
        row=1, col=1,
    )

    status = df["Status"].value_counts().reset_index()
    status.columns = ["Status", "Quantidade"]
    fig.add_trace(
        go.Bar(x=status["Status"], y=status["Quantidade"],
               marker_color=["#2ECC71","#3498DB","#9B59B6","#E74C3C"],
               name="Status"),
        row=1, col=2,
    )
    fig.update_layout(title_text="Distribuição Geral dos Incidentes",
                      showlegend=False, template="plotly_white", height=420)
    salvar(fig, "01_distribuicao.html")


def eda_02_sazonalidade(df: pd.DataFrame) -> None:
    """Sazonalidade: hora, dia da semana, mês e heatmap."""
    print("[02] Sazonalidade")

    dias_pt = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
    meses_pt = ["Jan","Fev","Mar","Abr","Mai","Jun",
                "Jul","Ago","Set","Out","Nov","Dez"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Volume por Hora do Dia",
                        "Volume por Dia da Semana",
                        "Volume por Mês",
                        "Heatmap: Hora x Dia da Semana"),
    )

    hora_v = df.groupby("hora").size().reset_index(name="v")
    fig.add_trace(go.Bar(x=hora_v["hora"], y=hora_v["v"],
                         marker_color="#3498DB", name="Hora"),
                  row=1, col=1)

    dia_v = df.groupby("dia_semana").size().reset_index(name="v").sort_values("dia_semana")
    dia_v["nome"] = [dias_pt[d] for d in dia_v["dia_semana"]]
    fig.add_trace(go.Bar(x=dia_v["nome"], y=dia_v["v"],
                         marker_color="#E74C3C", name="Dia"),
                  row=1, col=2)

    mes_v = df.groupby("mes").size().reset_index(name="v")
    mes_v["nome"] = [meses_pt[m-1] for m in mes_v["mes"]]
    fig.add_trace(go.Bar(x=mes_v["nome"], y=mes_v["v"],
                         marker_color="#2ECC71", name="Mês"),
                  row=2, col=1)

    heat = df.groupby(["dia_semana","hora"]).size().reset_index(name="v")
    piv  = heat.pivot(index="dia_semana", columns="hora", values="v").fillna(0)
    piv.index = dias_pt
    fig.add_trace(go.Heatmap(z=piv.values,
                             x=[f"{h}h" for h in piv.columns],
                             y=piv.index,
                             colorscale="Reds", showscale=True),
                  row=2, col=2)

    fig.update_layout(title_text="Sazonalidade dos Incidentes",
                      showlegend=False, template="plotly_white", height=700)
    salvar(fig, "02_sazonalidade.html")


def eda_03_ola_por_prioridade(df: pd.DataFrame) -> None:
    """Distribuição de duração vs. limite OLA por prioridade."""
    print("[03] OLA por Prioridade")

    df_kpi = df[df["entrou_para_kpi"] == True].copy()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["P1 - Crítica (4h)", "P2 - Alta (4h)", "P3 - Média (12h)"],
    )

    for i, (prio, limite) in enumerate([
        ("1 - Crítica", 4), ("2 - Alta", 4), ("3 - Média", 12)
    ], 1):
        sub    = df_kpi[df_kpi["Prioridade"] == prio]["duracao_horas"].clip(upper=limite*3)
        dentro = sub[sub <= limite]
        fora   = sub[sub > limite]
        n, nv  = len(sub), len(fora)
        pct    = nv / n * 100 if n > 0 else 0

        fig.add_trace(go.Histogram(x=dentro, name="Dentro do OLA",
                                   marker_color="#2ECC71", opacity=0.7, nbinsx=25),
                      row=1, col=i)
        fig.add_trace(go.Histogram(x=fora, name="Violou OLA",
                                   marker_color="#E74C3C", opacity=0.7, nbinsx=25),
                      row=1, col=i)
        fig.add_vline(x=limite, line_dash="dash", line_color="black",
                      annotation_text=f"OLA {limite}h", row=1, col=i)

        print(f"  {prio}: {n:,} incidentes | {nv:,} violações ({pct:.1f}%)")

    fig.update_layout(title_text="Duração vs. Limite OLA por Prioridade",
                      template="plotly_white", showlegend=False, height=400)
    salvar(fig, "03_ola_por_prioridade.html")


def eda_04_ola_por_mes(df: pd.DataFrame) -> None:
    """Violações de OLA por mês - P2 e P3."""
    print("[04] OLA por Mês")

    df_kpi = df[
        df["Prioridade"].isin(["2 - Alta", "3 - Média"]) &
        (df["entrou_para_kpi"] == True)
    ].copy()

    resumo = (
        df_kpi.groupby(["mes_enc", "Prioridade"])
        .agg(total=("Número","count"),
             ola_violado=("ola_violado","sum"))
        .reset_index()
    )
    resumo["taxa"] = (resumo["ola_violado"] / resumo["total"] * 100).round(1)

    fig = px.bar(
        resumo, x="mes_enc", y="ola_violado",
        color="Prioridade", barmode="group",
        title="OLA Violado por Mês - P2 (limite: 3/mês) e P3",
        color_discrete_map={"2 - Alta": "#E74C3C", "3 - Média": "#F39C12"},
        template="plotly_white",
        labels={"mes_enc": "Mês", "ola_violado": "Violações de OLA"},
    )
    fig.add_hline(y=3, line_dash="dash", line_color="red",
                  annotation_text="Meta máx P2: 3/mês")
    fig.update_layout(xaxis_tickangle=-45, height=450)
    salvar(fig, "04_ola_por_mes.html")


def eda_05_ics_barulhentos(df: pd.DataFrame) -> None:
    """Top 20 ICs com mais incidentes automáticos."""
    print("[05] ICs Barulhentos")

    df_auto = df[
        (df["Status"] == "Sem Intervenção") &
        (df["Aberto por"] == "Monitoramento") &
        (df["Item de configuração"].notna())
    ].copy()

    top = (
        df_auto.groupby("Item de configuração")
        .agg(total=("Número","count"),
             dur_media_h=("duracao_horas","mean"),
             prioridades=("Prioridade","nunique"))
        .reset_index()
        .sort_values("total", ascending=False)
        .head(20)
    )
    top["dur_media_h"] = top["dur_media_h"].round(2)

    fig = px.bar(
        top, x="Item de configuração", y="total",
        color="dur_media_h",
        color_continuous_scale="Reds",
        title="Top 20 ICs - Mais Incidentes Automáticos (cor = duração média em horas)",
        labels={"total": "Qtd. Incidentes", "dur_media_h": "Duração Média (h)"},
        template="plotly_white",
    )
    fig.update_layout(xaxis_tickangle=-45, height=480)
    salvar(fig, "05_ics_barulhentos.html")

    print("  Top 10 ICs:")
    print(top.head(10)[["Item de configuração","total","dur_media_h"]].to_string(index=False))


def eda_06_tipo_solucao(df: pd.DataFrame) -> None:
    """Distribuição de tipos de solução."""
    print("[06] Tipo de Solução")

    df_sol = df["Solução"].fillna("Não informado").value_counts().reset_index()
    df_sol.columns = ["Solução", "Quantidade"]

    fig = px.pie(
        df_sol, values="Quantidade", names="Solução",
        title="Tipo de Solução Aplicada",
        color_discrete_sequence=["#E74C3C","#2ECC71","#95A5A6","#3498DB"],
        template="plotly_white",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    salvar(fig, "06_tipo_solucao.html")

    df_kpi_sol = df[df["entrou_para_kpi"] == True]["Solução"].fillna("Não informado").value_counts(normalize=True) * 100
    print("  Proporção nos incidentes KPI:")
    print(df_kpi_sol.round(1).to_string())


def eda_07_grupos_risco(df: pd.DataFrame) -> None:
    """Scatter: volume x taxa de violação por grupo."""
    print("[07] Grupos de Risco")

    df_kpi = df[df["entrou_para_kpi"] == True].copy()
    df_kpi["ola_violado_bin"] = df_kpi["ola_violado"].astype(int)

    stats = (
        df_kpi.groupby("Grupo designado")
        .agg(total=("Número","count"),
             violacoes=("ola_violado_bin","sum"),
             dur_media=("duracao_horas","mean"))
        .reset_index()
    )
    stats["taxa_pct"] = (stats["violacoes"] / stats["total"] * 100).round(1)
    stats = stats.sort_values("total", ascending=False).head(15)

    fig = px.scatter(
        stats,
        x="total", y="taxa_pct",
        size="dur_media", color="taxa_pct",
        text="Grupo designado",
        color_continuous_scale="RdYlGn_r",
        title="Grupos: Volume x Taxa de Violação de OLA (tamanho = duração média)",
        labels={"total":"Total de Incidentes KPI",
                "taxa_pct":"Taxa de Violação OLA (%)"},
        template="plotly_white",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=520)
    salvar(fig, "07_grupos_risco.html")


def eda_08_correlacao(df: pd.DataFrame) -> None:
    """Correlação entre features numéricas e violação de OLA."""
    print("[08] Correlação de Features")

    df_kpi = df[df["entrou_para_kpi"] == True].copy()
    df_kpi["ola_violado_bin"] = df_kpi["ola_violado"].astype(int)

    feats = ["duracao_horas", "hora", "dia_semana", "mes", "ola_violado_bin"]
    corr  = df_kpi[feats].dropna().corr()

    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto", zmin=-1, zmax=1,
        title="Correlação entre Features Numéricas e Violação de OLA",
    )
    salvar(fig, "08_correlacao.html")

    print("  Correlação com OLA Violado:")
    print(corr["ola_violado_bin"].drop("ola_violado_bin").sort_values(ascending=False).round(4).to_string())


# Relatório de insights

def gerar_relatorio_insights(df: pd.DataFrame) -> None:
    """Gera arquivo de texto com os principais insights do EDA."""
    print("\n[09] Gerando relatório de insights...")

    df_kpi = df[df["entrou_para_kpi"] == True].copy()

    total        = len(df)
    kpi_sim      = len(df_kpi)
    violados     = int((df["KPI Violado?"] == "SIM").sum())
    taxa_viol    = violados / (kpi_sim + 1) * 100
    pct_auto     = (df["Aberto por"] == "Monitoramento").mean() * 100
    pct_sem_int  = (df["Status"] == "Sem Intervenção").mean() * 100
    pct_contorno = (df["Solução"] == "Contorno").mean() * 100

    hora_pico    = df.groupby("hora").size().idxmax()
    dia_pico     = df.groupby("dia_semana").size().idxmax()
    dias_pt      = ["Segunda","Terça","Quarta","Quinta","Sexta","Sábado","Domingo"]

    top_ic = (
        df[df["Status"] == "Sem Intervenção"]
        .groupby("Item de configuração").size()
        .nlargest(3).index.tolist()
    ) if "Item de configuração" in df.columns else []

    linhas = [
        "=" * 65,
        "  RELATÓRIO DE INSIGHTS - EDA AIOps Locaweb",
        "=" * 65,
        "",
        "VISÃO GERAL",
        f"  Total de incidentes:        {total:,}",
        f"  Entram para KPI:            {kpi_sim:,} ({kpi_sim/total*100:.1f}%)",
        f"  OLA Violado:                {violados:,} ({taxa_viol:.1f}% dos KPI)",
        f"  Abertos por Monitoramento:  {pct_auto:.1f}%",
        f"  Status Sem Intervenção:     {pct_sem_int:.1f}%",
        f"  Solução por Contorno:       {pct_contorno:.1f}%",
        "",
        "SAZONALIDADE",
        f"  Hora de pico: {hora_pico}h",
        f"  Dia de pico:  {dias_pt[dia_pico]}",
        "  Recomendação: criar features hora_sin/cos e dia_sem_sin/cos",
        "",
        "ICs BARULHENTOS",
        f"  Top 3 ICs automáticos: {', '.join(str(x) for x in top_ic)}",
        "  Recomendação: criar feature 'alertas_ic_24h_antes' como preditor de P2 iminente",
        "",
        "OLA E KPI",
        "  OLA medido pela data de ENCERRAMENTO (não abertura)",
        "  P2: máx 3 violações/mês | P3: meta em definição",
        "  Incidentes Sem Intervenção e com Incidente Pai NÃO entram para KPI",
        "",
        "RECOMENDAÇÕES DE FEATURE ENGINEERING",
        "  1. Encoding cíclico (sin/cos) para hora, dia, mês",
        "  2. Frequência de alertas por IC (7d e 30d)",
        "  3. Janela de 24h antes: alertas automáticos no mesmo IC",
        "  4. Reincidência: reabertura em 72h após contorno",
        "  5. Hierarquia Pai-Filho: qtd_filhos e proporcao_filhos_violaram",
        "  6. Carga de equipe: volume do grupo nos últimos 7 dias",
        "",
        "=" * 65,
    ]

    relatorio = "\n".join(linhas)
    print(relatorio)

    with open(OUT_DIR / "insights.txt", "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"\n  insights.txt salvo em {OUT_DIR}")


# Pipeline principal

def executar_eda() -> None:
    print("=" * 60)
    print("ETAPA 3 - EDA (ANÁLISE EXPLORATÓRIA)")
    print("=" * 60)

    df = carregar_dados()

    print(f"\nPeríodo: {df['dt_aberto'].min().date()} -> {df['dt_aberto'].max().date()}")
    print(f"Colunas: {df.columns.tolist()}\n")
    print("Gerando gráficos...\n")

    eda_01_distribuicao(df)
    eda_02_sazonalidade(df)
    eda_03_ola_por_prioridade(df)
    eda_04_ola_por_mes(df)
    eda_05_ics_barulhentos(df)
    eda_06_tipo_solucao(df)
    eda_07_grupos_risco(df)
    eda_08_correlacao(df)
    gerar_relatorio_insights(df)

    print("\n" + "=" * 60)
    print("ETAPA 3 CONCLUÍDA")
    print(f"  Gráficos HTML em: {OUT_DIR}")
    print(f"  Abra os .html no navegador para interagir com os gráficos")
    print("=" * 60)


# Execução direta

if __name__ == "__main__":
    executar_eda()
