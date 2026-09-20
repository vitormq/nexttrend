"""
NextTrend - painel operacional (Streamlit).
Reúne a previsão de volume, o risco de violação de OLA, a projeção de
atingimento dos indicadores e a fila de risco num só lugar para a operação.

Roda com: streamlit run etapa10_dashboard.py --server.port 8501
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

import etapa8_kpi_atingimento as kpi

st.set_page_config(page_title="NextTrend",
                   layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MDL_DIR  = BASE_DIR / "models" / "saved"
REP_DIR  = BASE_DIR / "reports" / "explainability"

TIER = {150: "#0e9f6e", 125: "#4faa77", 100: "#3a80c6", 75: "#d99a1e", 50: "#e0763a", 0: "#d6474b"}
PLOT = dict(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
DIAS = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
        "Sexta-feira", "Sábado", "Domingo"]
MESES = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
         "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
PRIOS = ["2 - Alta", "3 - Média"]

# Estilo (cards premium)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0e14;--s1:#141c28;--s2:#0f1620;--bd:#212c3d;--bdh:#35496a;
  --tx:#e8edf4;--mut:#8595ab;--acc:#ff6a3d;--acc2:#ff8a63;}
html,body,[class*="css"],.stApp,button,input,textarea,select{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif !important;}
.stApp{background:var(--bg);}
h1{font-weight:800 !important;letter-spacing:-.03em;font-size:2.1rem !important;color:var(--tx);}
h2{font-weight:700 !important;letter-spacing:-.02em;color:var(--tx);}
h3{font-weight:700 !important;letter-spacing:-.015em;color:var(--tx);}
[data-testid="stMainBlockContainer"]{padding-top:2.4rem;max-width:1440px;}
hr{border-color:var(--bd) !important;opacity:.7;margin:1rem 0;}
/* menus nativos escondidos */
.modebar,.js-plotly-plot .modebar{display:none !important;}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important;}
#MainMenu{visibility:hidden !important;}
header[data-testid="stHeader"]{height:0 !important;background:transparent !important;}
/* métricas -> cards premium */
div[data-testid="stMetric"]{background:linear-gradient(180deg,var(--s1),var(--s2));
  border:1px solid var(--bd);border-radius:16px;padding:16px 20px;
  box-shadow:0 1px 0 rgba(255,255,255,.02) inset,0 10px 30px rgba(0,0,0,.35);transition:border-color .2s;}
div[data-testid="stMetric"]:hover{border-color:var(--bdh);}
div[data-testid="stMetricValue"]{font-size:2.05rem;font-weight:800;letter-spacing:-.03em;color:var(--tx);}
div[data-testid="stMetricLabel"] p{font-size:.78rem;color:var(--mut);font-weight:600;letter-spacing:.02em;}
/* cards custom */
.cards{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:4px;}
.kpi{flex:1;min-width:210px;background:linear-gradient(180deg,var(--s1),var(--s2));
  border:1px solid var(--bd);border-radius:18px;padding:18px 20px;
  box-shadow:0 10px 30px rgba(0,0,0,.35);transition:border-color .2s,transform .2s;}
.kpi:hover{border-color:var(--bdh);transform:translateY(-2px);}
.kpi .top{display:flex;align-items:center;gap:9px;color:var(--mut);font-size:.8rem;font-weight:600;letter-spacing:.02em;}
.kpi .dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none;box-shadow:0 0 0 3px rgba(255,255,255,.03);}
.kpi .val{font-size:2.8rem;font-weight:800;line-height:1.02;margin:10px 0 2px;letter-spacing:-.035em;}
.kpi .bar{height:5px;border-radius:4px;background:#1c2636;overflow:hidden;margin:8px 0 6px;}
.kpi .bar i{display:block;height:100%;border-radius:4px;}
.kpi .meta{font-size:.66rem;color:#6f8098;text-transform:uppercase;letter-spacing:.09em;}
.kpi .status{font-size:.82rem;font-weight:700;margin-top:8px;}
.hero{background:linear-gradient(135deg,#10231a,#0c1913);border:1px solid #1c4d38;border-radius:20px;
  padding:22px 26px;box-shadow:0 12px 32px rgba(0,0,0,.4);}
.hero .lab{color:var(--mut);font-size:.76rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;}
.hero .big{font-size:3.1rem;font-weight:800;color:#2ee08f;line-height:1.02;margin-top:6px;letter-spacing:-.035em;}
.hero .sub{color:var(--mut);font-size:.86rem;margin-top:8px;}
.foco{background:linear-gradient(180deg,var(--s1),var(--s2));border:1px solid var(--bd);border-radius:18px;padding:18px 20px;}
.foco h4{margin:0 0 14px;font-size:.76rem;color:var(--mut);font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.foco .row{margin:11px 0;}
.foco .rl{display:flex;justify-content:space-between;font-size:.86rem;color:#cdd8e6;margin-bottom:5px;}
.foco .rl b{color:var(--tx);font-weight:600;}
.foco .rb{height:7px;border-radius:5px;background:#1c2636;overflow:hidden;}
.foco .rb i{display:block;height:100%;border-radius:5px;}
.acao{display:flex;gap:16px;align-items:flex-start;background:linear-gradient(180deg,var(--s1),var(--s2));
  border:1px solid var(--bd);border-radius:16px;padding:16px 18px;margin-bottom:11px;transition:border-color .2s;}
.acao:hover{border-color:var(--bdh);}
.acao .n{flex:none;width:38px;height:38px;border-radius:11px;background:rgba(255,106,61,.12);
  border:1px solid rgba(255,106,61,.4);color:var(--acc2);font-weight:800;
  display:flex;align-items:center;justify-content:center;font-size:1.15rem;}
.acao .t{font-weight:700;color:var(--tx);font-size:1.02rem;letter-spacing:-.01em;}
.acao .d{color:var(--mut);font-size:.87rem;margin-top:3px;line-height:1.4;}
.alerta{background:linear-gradient(135deg,#2a1618,#1d1214);border:1px solid #6b2b30;border-radius:16px;
  padding:18px 22px;color:#f0d6d8;font-size:.95rem;line-height:1.45;}
/* navegação: pílulas elegantes (fixa via JS) */
div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]){
  position:sticky;top:0;z-index:1000;background:var(--bg);
  padding:10px 0 10px;border-bottom:1px solid var(--bd);margin-bottom:8px;}
div[data-testid="stRadio"] [role="radiogroup"]{gap:6px;flex-wrap:wrap;}
div[data-testid="stRadio"] [role="radiogroup"] label{background:rgba(255,255,255,.02);border:1px solid var(--bd);
  border-radius:11px;padding:8px 16px;margin:0;cursor:pointer;transition:all .18s;
  font-size:.86rem;font-weight:500;color:var(--mut);}
div[data-testid="stRadio"] [role="radiogroup"] label:hover{border-color:var(--bdh);color:var(--tx);background:rgba(255,255,255,.04);}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked){background:var(--acc);border-color:var(--acc);
  box-shadow:0 6px 18px rgba(255,106,61,.35);}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) *{color:#fff !important;font-weight:600;}
div[data-testid="stRadio"] [role="radiogroup"] label>div:first-child{display:none !important;}
/* botões, expander, dataframe, popover */
button[kind="primary"]{background:var(--acc) !important;border:none !important;border-radius:11px !important;
  font-weight:600 !important;box-shadow:0 6px 18px rgba(255,106,61,.3) !important;}
[data-testid="stExpander"]{border:1px solid var(--bd) !important;border-radius:14px !important;background:var(--s2);}
[data-testid="stExpander"] summary{font-weight:600;}
[data-testid="stDataFrame"]{border:1px solid var(--bd);border-radius:12px;overflow:hidden;}
[data-testid="stPopover"] button{border:1px solid var(--bd) !important;border-radius:12px !important;
  background:linear-gradient(180deg,var(--s1),var(--s2)) !important;color:var(--tx) !important;font-weight:600 !important;}
/* fila de risco operacional */
.fila{display:flex;flex-direction:column;gap:8px;}
.frow{display:grid;grid-template-columns:132px 1fr 98px 196px;align-items:center;gap:14px;
  background:linear-gradient(180deg,var(--s1),var(--s2));border:1px solid var(--bd);
  border-radius:13px;padding:12px 16px;transition:border-color .18s,transform .18s;}
.frow:hover{border-color:var(--bdh);transform:translateX(2px);}
.fid{font-weight:700;color:var(--tx);font-size:.92rem;letter-spacing:-.01em;}
.fgr{font-size:.9rem;color:var(--tx);line-height:1.25;}
.fgr b{font-weight:600;}
.fgr span{display:block;color:var(--mut);font-size:.74rem;margin-top:1px;}
.fbadge{font-size:.68rem;font-weight:700;letter-spacing:.06em;text-align:center;
  padding:5px 0;border-radius:8px;border:1px solid;}
.fscore{display:flex;align-items:center;gap:10px;}
.fsbar{flex:1;height:8px;border-radius:5px;background:#1c2636;overflow:hidden;}
.fsbar i{display:block;height:100%;border-radius:5px;}
.fpct{font-weight:800;font-size:1.05rem;color:var(--tx);min-width:32px;text-align:right;letter-spacing:-.02em;}
</style>
""", unsafe_allow_html=True)


def _kpi_card(dot, titulo, pct, valcor, status, scor):
    w = min(pct, 150) / 150 * 100
    return (f'<div class="kpi"><div class="top"><span class="dot" style="background:{dot}"></span>{titulo}</div>'
            f'<div class="val" style="color:{valcor}">{pct:.0f}%</div>'
            f'<div class="bar"><i style="width:{w:.0f}%;background:{valcor}"></i></div>'
            f'<div class="meta">meta: 100%</div>'
            f'<div class="status" style="color:{scor}">{status}</div></div>')

def _br(n):
    return f"{n:,.0f}".replace(",", ".")


# Carregadores
@st.cache_data(ttl=600)
def carregar_limpo():
    p = DATA_DIR / "incidents_clean.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=600)
def carregar_features():
    p = DATA_DIR / "incidents_features.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_resource
def carregar_modelo():
    try:
        return (joblib.load(MDL_DIR / "ola_predictor_pipeline.pkl"),
                joblib.load(MDL_DIR / "ola_feature_cols.pkl"))
    except Exception:
        return None, None

@st.cache_resource
def carregar_previsor(prioridade):
    nome = f"prophet_{prioridade.split(' - ')[0].strip()}_{prioridade.split(' - ')[-1].strip()}.pkl"
    cam = MDL_DIR / nome
    return joblib.load(cam) if cam.exists() else None

@st.cache_data(ttl=600)
def series_indicadores(ano):
    return kpi.carregar_series(ano)

@st.cache_data(ttl=600)
def mapa_riscos():
    p = REP_DIR / "mapa_riscos_grupo.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=600)
def calcular_perfis(k=4, amostra=4000, seed=42):
    """Agrupa incidentes por comportamento, nomeia cada grupo pela sua característica
    mais marcante e projeta em 2 dimensões para o mapa de perfis."""
    d = carregar_features()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = d[d["entrou_para_kpi"] & d["prioridade_cod"].isin(["P2", "P3"])].copy()
    cols = [c for c in ["ic_vol_30d", "grupo_vol_30d", "grupo_taxa_violacao_30d",
                        "ic_taxa_violacao_30d", "carga_grupo_abertos", "hora_abertura"]
            if c in d.columns]
    Xs = StandardScaler().fit_transform(d[cols].fillna(0).values)
    d["_perfil"] = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xs)
    proj = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    d["_x"], d["_y"] = proj[:, 0], proj[:, 1]

    # Nomeia cada grupo pela característica que mais o destaca (derivado dos dados).
    zc = pd.DataFrame(Xs, columns=cols); zc["_perfil"] = d["_perfil"].values
    zmed = zc.groupby("_perfil").mean()
    risco_cols = [c for c in ["ic_taxa_violacao_30d", "grupo_taxa_violacao_30d"] if c in cols]
    zmed["_risco"] = zmed[risco_cols].mean(axis=1) if risco_cols else 0
    nomes, livres = {}, set(zmed.index)

    def _atribui(coluna, rotulo, limiar=0.35):
        if coluna not in zmed.columns or not livres:
            return
        c = zmed.loc[list(livres), coluna].idxmax()
        if zmed.loc[c, coluna] > limiar:
            nomes[c] = rotulo; livres.discard(c)

    _atribui("_risco",              "Alto risco de violação")
    _atribui("grupo_vol_30d",       "Grupos de altíssimo volume")
    _atribui("ic_vol_30d",          "Itens recorrentes (barulhentos)")
    _atribui("carga_grupo_abertos", "Grupos sobrecarregados")
    for c in livres:
        nomes[c] = "Operação rotineira"
    d["Perfil"] = d["_perfil"].map(nomes)

    resumo = (d.groupby("Perfil")
                .agg(**{"Incidentes": ("Número", "count"),
                        "Taxa de violação (%)": ("kpi_violado_bin", lambda s: round(s.mean()*100, 1)),
                        "Volume médio do grupo": ("grupo_vol_30d", lambda s: round(s.mean())),
                        "Volume médio do item": ("ic_vol_30d", lambda s: round(s.mean())),
                        "Hora média de abertura": ("hora_abertura", lambda s: round(s.mean()))})
                .reset_index().sort_values("Taxa de violação (%)", ascending=False))
    ds = d.sample(min(amostra, len(d)), random_state=seed).copy()
    return ds, resumo

@st.cache_data(ttl=1800)
def treinar_triagem(amostra=40000, seed=42):
    """Classificação multi-classe: dado um incidente, prevê o GRUPO responsável (triagem/roteamento)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    d = carregar_features()
    if d.empty:
        return None
    alvo = "Grupo designado"
    cat = [c for c in ["Prioridade", "Categoria", "Produto", "Aberto por"] if c in d.columns]
    num = [c for c in ["hora_abertura", "dia_semana", "is_weekend", "is_horario_comercial",
                       "desc_n_palavras", "desc_tem_erro", "desc_tem_lento", "desc_tem_fora",
                       "desc_tem_critico", "desc_tem_cliente"] if c in d.columns]
    d = d.dropna(subset=[alvo]).copy()
    vc = d[alvo].value_counts()
    d = d[d[alvo].isin(vc[vc >= 100].index)]
    if len(d) > amostra:
        d = d.sample(amostra, random_state=seed)
    X = d[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype(str)
    if cat:
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1).fit_transform(X[cat])
    labels = sorted(d[alvo].unique())
    y = d[alvo].map({g: i for i, g in enumerate(labels)})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=140, max_depth=20, n_jobs=-1,
                                 random_state=seed, class_weight="balanced_subsample")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    cm = confusion_matrix(yte, pred, labels=range(len(labels))).astype(float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cmn, labels, acc


def nivel_risco(p):
    return ("Crítico" if p >= 0.80 else "Alto" if p >= 0.60 else
            "Médio" if p >= 0.35 else "Baixo")

def prever_volume(prioridade, dias):
    m = carregar_previsor(prioridade)
    if m is None:
        return None
    fut = m.make_future_dataframe(periods=dias, freq="D", include_history=False)
    for reg in list(getattr(m, "extra_regressors", {}).keys()):
        fut[reg] = fut["ds"].dt.dayofweek.isin([5, 6]).astype(int) if reg == "is_weekend" else 0.0
    fc = m.predict(fut)
    fc["yhat"] = fc["yhat"].clip(lower=0)
    return fc


# Fila de risco e causa-raiz por incidente (SHAP)
def nivel_fila(p):
    return ("Crítico" if p >= 0.60 else "Alto" if p >= 0.50 else
            "Médio" if p >= 0.40 else "Baixo")

COR_NIVEL = {"Crítico": "#d6474b", "Alto": "#e0763a", "Médio": "#d99a1e", "Baixo": "#3a80c6"}

LEGIVEL = {
    "grupo_taxa_violacao_30d": "Histórico de violação do grupo (30 dias)",
    "grupo_taxa_violacao_7d": "Histórico de violação do grupo (7 dias)",
    "ic_taxa_violacao_30d": "Histórico de violação do item (30 dias)",
    "ic_taxa_violacao_7d": "Histórico de violação do item (7 dias)",
    "Grupo designado": "Grupo responsável",
    "Item de configuração": "Item de configuração",
    "Prioridade": "Prioridade do chamado",
    "prioridade_num": "Nível de prioridade",
    "ola_limite_horas": "Prazo de OLA (horas)",
    "hora_abertura": "Hora de abertura",
    "dia_semana": "Dia da semana",
    "is_horario_comercial": "Aberto em horário comercial",
    "is_weekend": "Aberto no fim de semana",
    "is_madrugada": "Aberto de madrugada",
    "is_fim_de_mes": "Aberto no fim do mês",
    "carga_grupo_abertos": "Chamados abertos no grupo",
    "carga_grupo_atrasados": "Chamados atrasados no grupo",
    "carga_ic_abertos": "Chamados abertos no item",
    "ic_vol_30d": "Volume do item (30 dias)",
    "ic_vol_7d": "Volume do item (7 dias)",
    "grupo_vol_30d": "Volume do grupo (30 dias)",
    "grupo_vol_7d": "Volume do grupo (7 dias)",
    "ic_media_duracao_30d": "Duração média do item (30 dias)",
    "grupo_media_duracao_30d": "Duração média do grupo (30 dias)",
    "alta_x_barulhento": "Alta prioridade em item recorrente",
    "alta_x_carga_grupo": "Alta prioridade com grupo sobrecarregado",
    "hora_x_prioridade": "Hora x prioridade",
    "vol_dia_ate_agora": "Volume do dia até a abertura",
    "violacoes_dia_ate_agora": "Violações do dia até a abertura",
    "taxa_violacao_dia_ate_agora": "Taxa de violação do dia até a abertura",
    "desc_n_palavras": "Tamanho da descrição (palavras)",
    "desc_tem_critico": "Descrição menciona 'crítico'",
    "desc_tem_erro": "Descrição menciona 'erro'",
    "desc_tem_cliente": "Descrição menciona 'cliente'",
    "Aberto por": "Origem da abertura",
}

def _rotulo_feature(f):
    return LEGIVEL.get(f, f.replace("_", " ").capitalize())

@st.cache_data(ttl=1800)
def base_com_oof():
    """Base P2/P3 pontuada por validação cruzada (out-of-fold): cada incidente
    recebe a nota de risco de um modelo que NÃO o usou no treino. É a medida
    HONESTA de desempenho - evita o número inflado de avaliar o modelo nos
    próprios dados de treino (que superestima o AUC e o lift)."""
    pipe, cols = carregar_modelo()
    feats = carregar_features()
    if pipe is None or feats.empty:
        return None
    base = feats[feats["entrou_para_kpi"] & feats["prioridade_cod"].isin(["P2", "P3"])].copy()
    oof_path = DATA_DIR / "oof_scores.parquet"
    if oof_path.exists():
        # scores honestos já calculados (scripts/gerar_oof.py) - carrega pronto,
        # rápido e leve no deploy (não recalcula a validação cruzada ao vivo)
        oof = pd.read_parquet(oof_path).set_index("Número")["risco"]
        base["risco"] = base["Número"].map(oof)
        if base["risco"].isna().any():
            base["risco"] = base["risco"].fillna(base["risco"].median())
    else:
        # fallback: calcula a validação cruzada ao vivo (cada incidente pontuado
        # por um modelo que não o viu no treino)
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        y = base["kpi_violado_bin"].astype(int).values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base["risco"] = cross_val_predict(clone(pipe), base[cols], y, cv=skf,
                                          method="predict_proba", n_jobs=-1)[:, 1]
    return base, cols

@st.cache_data(ttl=600)
def fila_risco():
    """Ranqueia os chamados que entram no indicador pelo risco de violar OLA
    (usando a nota honesta out-of-fold, a mesma da avaliação do modelo)."""
    res = base_com_oof()
    if res is None:
        return None
    base, cols = res
    base = base.copy()
    stats = dict(
        n=int(len(base)),
        crit=int((base["risco"] >= 0.60).sum()),
        alto=int(((base["risco"] >= 0.50) & (base["risco"] < 0.60)).sum()),
        med=int(((base["risco"] >= 0.40) & (base["risco"] < 0.50)).sum()),
        media=float(base["risco"].mean()),
    )
    base = base.sort_values(["risco", "grupo_taxa_violacao_30d", "Número"],
                            ascending=[False, False, True])
    return base.head(80).reset_index(drop=True), cols, stats

@st.cache_resource
def carregar_shap_expl():
    try:
        import shap
        pipe, _ = carregar_modelo()
        if pipe is None:
            return None
        return shap.TreeExplainer(pipe.named_steps["classificador"])
    except Exception:
        return None

def causa_raiz_incidente(row_df, cols):
    """Contribuição de cada fator (SHAP) ao risco de UM chamado, com nome legível."""
    expl = carregar_shap_expl()
    pipe, _ = carregar_modelo()
    if expl is None or pipe is None:
        return None
    try:
        pre = pipe[:-1]
        Xt = pre.transform(row_df[cols])
        try:
            nomes = list(pipe.named_steps["preprocessador"].get_feature_names_out())
        except Exception:
            nomes = [f"f{i}" for i in range(Xt.shape[1])]
        sv = expl.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        vals = np.asarray(sv)[0]
        limpos = [n.split("__", 1)[1] if "__" in n else n for n in nomes]
        out = pd.DataFrame({"feature": limpos, "shap": vals})
        out["abs"] = out["shap"].abs()
        return out.sort_values("abs", ascending=False).reset_index(drop=True)
    except Exception:
        return None


# Oráculo (respostas baseadas em dados)
def oraculo_responder(pergunta: str) -> str:
    q = (pergunta or "").lower()
    df = carregar_limpo()
    if df.empty:
        return "Ainda não há dados carregados. Gere os dados operacionais primeiro."

    def bloco_volume():
        linhas = ["**Previsão de volume de incidentes:**"]
        for pr in PRIOS:
            fc = prever_volume(pr, 7)
            if fc is None:
                continue
            d1 = fc["yhat"].iloc[0]
            d7 = fc["yhat"].iloc[:7].sum()
            linhas.append(f"- {pr}: amanhã ~ **{d1:.0f}** incidentes; próximos 7 dias ~ **{d7:.0f}**.")
        return "\n".join(linhas)

    def bloco_kpi():
        s = series_indicadores(2025)
        linhas = ["**Tendência dos indicadores (projeção para o próximo ano):**"]
        rotulo = {"OLA": "Quebras de OLA", "VOL": "Volume tratado"}
        for metrica in ("OLA", "VOL"):
            for pc in ("P2", "P3"):
                r = kpi.projetar(s[(metrica, pc)], metrica, pc, mes_corte=0)
                chance = r["p_saudavel"] * 100
                sit = "tende a fechar saudável" if chance >= 50 else "corre risco de ficar abaixo da meta"
                linhas.append(f"- {rotulo[metrica]} - {kpi.PRIORIDADES[pc]}: {sit} "
                              f"(chance de ficar saudável ~ **{chance:.0f}%**).")
        return "\n".join(linhas)

    def bloco_grupos():
        m = mapa_riscos()
        if m.empty:
            return "O mapa de riscos por grupo ainda não foi gerado."
        top = m.sort_values("prob_media_violacao", ascending=False).head(5)
        linhas = ["**Grupos com maior risco médio de violação de OLA:**"]
        for _, r in top.iterrows():
            linhas.append(f"- {r['Grupo designado']} - {r['Prioridade']}: "
                          f"risco médio **{r['prob_media_violacao']*100:.0f}%** "
                          f"({int(r['total_incidentes'])} incidentes).")
        return "\n".join(linhas)

    def bloco_horarios():
        d = df[df["entrou_para_kpi"]].copy()
        d["hora"] = d["dt_aberto"].dt.hour
        vol_h = d.groupby("hora").size()
        pico = vol_h.idxmax()
        d["dia"] = d["dt_aberto"].dt.dayofweek
        pior_dia = DIAS[int(d.groupby("dia")["kpi_violado_bin"].mean().idxmax())]
        return ("**Padrões de horário e dia:**\n"
                f"- Pico de abertura de incidentes por volta das **{pico}h**.\n"
                f"- Maior taxa de violação de OLA acontece na **{pior_dia}**.")

    if any(k in q for k in ["volume", "quantos", "previs", "amanhã", "amanha", "semana", "demanda"]):
        return bloco_volume()
    if any(k in q for k in ["grupo", "time", "equipe", "onde", "quem"]):
        return bloco_grupos()
    if any(k in q for k in ["hora", "horário", "horario", "pico", "dia da semana", "quando"]):
        return bloco_horarios()
    if any(k in q for k in ["indicador", "kpi", "atingimento", "saudável", "saudavel", "meta",
                            "risco", "viola", "ola", "tendência", "tendencia", "futuro", "próximo ano", "2026"]):
        return bloco_kpi()
    # resposta geral
    return ("Posso responder sobre:\n"
            "- **Volume futuro** - ex.: \"qual a previsão de volume para a próxima semana?\"\n"
            "- **Tendência dos indicadores** - ex.: \"qual o risco do indicador de OLA da prioridade Alta?\"\n"
            "- **Grupos em risco** - ex.: \"quais times têm mais risco de violar OLA?\"\n"
            "- **Padrões de horário/dia** - ex.: \"qual o horário de pico de incidentes?\"\n\n"
            "_Resumo atual:_\n" + bloco_kpi())


# Cabeçalho (navegação no topo, para dar mais largura às visualizações)
hc1, hc2 = st.columns([5, 2])
with hc1:
    st.markdown("## NextTrend")
    st.caption("Previsão de incidentes e tendências operacionais - Locaweb")
with hc2:
    st.write("")
    with st.popover("Oráculo - perguntas e respostas", use_container_width=True):
        st.markdown("**Pergunte sobre os incidentes e as tendências futuras.**")
        if "oraculo_hist" not in st.session_state:
            st.session_state.oraculo_hist = []
        exemplos = ["Previsão de volume para a próxima semana",
                    "Qual o risco dos indicadores de OLA?",
                    "Quais times têm mais risco de violar OLA?",
                    "Qual o horário de pico de incidentes?"]
        for i, ex in enumerate(exemplos):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.oraculo_pergunta = ex
        pergunta = st.text_input("Sua pergunta:", key="oraculo_pergunta",
                                 placeholder="Digite sua pergunta...")
        if st.button("Perguntar ao Oráculo", type="primary", use_container_width=True) and pergunta:
            st.session_state.oraculo_hist.insert(0, (pergunta, oraculo_responder(pergunta)))
        for perg, resp in st.session_state.oraculo_hist[:4]:
            st.markdown(f"**Você:** {perg}")
            st.markdown(resp)
            st.markdown("---")

aba = st.radio("Navegação", [
    "Central de Decisão", "Visão Geral", "Previsão de Volume",
    "Atingimento de Indicadores", "Risco de Violação de OLA", "Fila de Risco",
    "Perfis de Incidentes", "Mapa de Riscos", "Triagem (bônus)",
], horizontal=True, label_visibility="collapsed")
st.markdown("---")

# Fixa a barra de navegação no topo ao rolar (CSS sticky não vinga sozinho no Streamlit;
# este script aplica a fixação no contêiner certo e destrava ancestrais que cortavam).
components.html("""
<script>
(function(){
  const pdoc = window.parent.document;
  function pin(){
    const radio = pdoc.querySelector('div[data-testid="stRadio"]');
    if(!radio){ return setTimeout(pin, 250); }
    const wrap = radio.closest('[data-testid="stElementContainer"]') || radio.parentElement;
    if(!wrap){ return setTimeout(pin, 250); }
    wrap.style.position = 'sticky';
    wrap.style.top = '0px';
    wrap.style.zIndex = '1000';
    const bg = getComputedStyle(pdoc.body).backgroundColor;
    wrap.style.background = (bg && bg !== 'rgba(0, 0, 0, 0)') ? bg : '#0e1117';
    wrap.style.paddingTop = '8px';
    wrap.style.paddingBottom = '8px';
    let el = wrap.parentElement;
    while(el && el !== pdoc.body){
      const ov = getComputedStyle(el).overflow;
      if(ov === 'hidden' || ov === 'clip'){ el.style.overflow = 'visible'; }
      el = el.parentElement;
    }
  }
  pin();
})();
</script>
""", height=0)

df = carregar_limpo()


# CENTRAL DE DECISÃO (cockpit executivo)
if aba == "Central de Decisão":
    st.title("Central de Decisão")
    st.caption("Onde agir agora: situação dos indicadores, alerta principal, focos críticos e ações recomendadas.")
    if df.empty:
        st.warning("Gere os dados operacionais primeiro."); st.stop()
    series = series_indicadores(2025)

    st.subheader("Situação dos indicadores - 2025")
    ordem = [("OLA", "P2"), ("OLA", "P3"), ("VOL", "P2"), ("VOL", "P3")]
    cards, pior = [], None
    for m, pc in ordem:
        total = float(series[(m, pc)].sum())
        pct = float(kpi.atingimento_vec([total], kpi.FAIXAS[(m, pc)])[0])
        valcor = TIER.get(int(pct), "#0e9f6e" if pct >= 100 else "#d99a1e")
        if pct >= 100:   dot, status, scor = "#0e9f6e", "saudável", "#2ecb86"
        elif pct >= 75:  dot, status, scor = "#d99a1e", "abaixo da meta", "#e6b23a"
        else:            dot, status, scor = "#d6474b", "crítico", "#e5686b"
        titulo = ("Quebras de OLA" if m == "OLA" else "Volume tratado") + " - " + kpi.PRIORIDADES[pc]
        cards.append(_kpi_card(dot, titulo, pct, valcor, status, scor))
        if pct < 100 and (pior is None or pct < pior[2]):
            pior = (m, pc, pct)
    st.markdown('<div class="cards">' + "".join(cards) + '</div>', unsafe_allow_html=True)

    st.write("")
    if pior:
        m, pc, pct = pior
        r3 = kpi.projetar(series[(m, pc)], m, pc, 3)
        st.markdown(
            f'<div class="alerta"><b style="color:#fff">Alerta principal - '
            f'{"Quebras de OLA" if m == "OLA" else "Volume"} - {kpi.PRIORIDADES[pc]}:</b> '
            f'fechou o ano em <b style="color:#ff8a6e">{pct:.0f}%</b> (a meta é 100%). Já em <b>março</b> '
            f'o sistema sinalizava apenas <b style="color:#ff8a6e">{r3["p_saudavel"]*100:.0f}%</b> de chance '
            f'de fechar saudável - meses de antecedência para agir, em vez de descobrir só no fim do ano.</div>',
            unsafe_allow_html=True)
    else:
        st.success("Todos os indicadores fecharam saudáveis (100% ou mais).")

    st.markdown("---")
    cA, cB = st.columns([1.05, 1])
    with cA:
        st.subheader("Impacto estimado")
        viol = int(df["kpi_violado_bin"].sum())
        i1, i2 = st.columns(2)
        custo = i1.number_input(
            "Custo por violação de OLA (R$)", min_value=0, value=5000, step=500,
            help="Quanto cada violação de OLA custa à operação (multa de contrato, retrabalho, "
                 "insatisfação do cliente). Use o valor real do seu SLA.")
        evit = i2.slider(
            "Quanto dá para evitar agindo cedo (%)", 0, 100, 30,
            help="Nem toda violação é evitável. Este é o percentual que a operação conseguiria evitar "
                 "ao agir no aviso antecipado do sistema. Ex.: 30% = de cada 10 violações, 3 seriam "
                 "evitadas. Ajuste conforme a sua realidade.")
        total_custo = viol * custo
        economia = total_custo * evit / 100
        st.markdown(
            f'<div class="hero"><div class="lab">Economia potencial com antecipação</div>'
            f'<div class="big">R$ {_br(economia)}</div>'
            f'<div class="sub">de R$ {_br(total_custo)} em multas no ano - evitando {evit}% - '
            f'{viol} violações registradas</div></div>', unsafe_allow_html=True)
        st.caption("Premissas ajustáveis acima - troque pelos valores reais do contrato de SLA.")
    with cB:
        st.subheader("Focos críticos")
        mapa = mapa_riscos()
        rows = ""
        if not mapa.empty:
            mx = max(0.01, float(mapa["prob_media_violacao"].max()))
            for _, rr in mapa.sort_values("prob_media_violacao", ascending=False).head(3).iterrows():
                p = rr["prob_media_violacao"] * 100
                cor = "#d6474b" if p >= 60 else ("#e0763a" if p >= 40 else "#d99a1e")
                rows += (f'<div class="row"><div class="rl"><span><b>{rr["Grupo designado"]}</b> - '
                         f'{rr["Prioridade"]}</span><span>{p:.0f}%</span></div><div class="rb">'
                         f'<i style="width:{rr["prob_media_violacao"]/mx*100:.0f}%;background:{cor}"></i></div></div>')
        st.markdown(f'<div class="foco"><h4>Grupos com maior risco de violação</h4>{rows}</div>', unsafe_allow_html=True)
        ic = df.groupby("Item de configuração").size().sort_values(ascending=False).head(3)
        mxi = int(ic.max()); irows = ""
        for nome, q in ic.items():
            irows += (f'<div class="row"><div class="rl"><span><b>{nome}</b></span><span>{q} incidentes</span></div>'
                      f'<div class="rb"><i style="width:{q/mxi*100:.0f}%;background:#3a80c6"></i></div></div>')
        st.markdown(f'<div class="foco" style="margin-top:12px"><h4>Itens que mais geram incidentes</h4>{irows}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Ações recomendadas para a semana")
    mapa = mapa_riscos()
    g_top = (mapa.sort_values("prob_media_violacao", ascending=False).iloc[0]["Grupo designado"]
             if not mapa.empty else "o grupo de maior risco")
    ic_top = df.groupby("Item de configuração").size().sort_values(ascending=False).head(3).index.tolist()
    acoes = [
        (f"Reforçar {g_top}", "Grupo com maior risco de violação de OLA. Revisar capacidade e priorizar "
                              "escalonamento preventivo."),
        ("Atacar a causa-raiz dos itens recorrentes", f"Poucos itens concentram muitos incidentes "
                              f"({', '.join(map(str, ic_top))}). Resolver a causa reduz o volume na fonte."),
        ("Vigiar o OLA da prioridade Alta", "Meta de no máximo 3 violações por mês. Configurar alerta ao "
                              "atingir 2 no mês para agir antes de furar o indicador."),
    ]
    st.markdown("".join(
        f'<div class="acao"><div class="n">{i}</div><div><div class="t">{t}</div>'
        f'<div class="d">{d}</div></div></div>' for i, (t, d) in enumerate(acoes, 1)),
        unsafe_allow_html=True)


# Visão geral
elif aba == "Visão Geral":
    st.title("Visão Geral Operacional")
    if df.empty:
        st.warning("Dados não encontrados. Gere os dados operacionais primeiro."); st.stop()

    entram = int(df["entrou_para_kpi"].sum())
    violados = int(df["kpi_violado_bin"].sum())
    taxa = violados / max(entram, 1) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de incidentes", f"{len(df):,}".replace(",", "."))
    c2.metric("Considerados no indicador", f"{entram:,}".replace(",", "."))
    c3.metric("Violações de OLA", f"{violados:,}".replace(",", "."))
    c4.metric("Taxa de violação", f"{taxa:.2f}%".replace(".", ","))
    st.markdown("---")

    ca, cb = st.columns(2)
    with ca:
        dist = df["Prioridade"].value_counts().reset_index()
        dist.columns = ["Prioridade", "Quantidade"]
        fig = px.bar(dist, x="Prioridade", y="Quantidade", title="Incidentes por prioridade",
                     color="Prioridade",
                     color_discrete_sequence=["#C0392B", "#E74C3C", "#E67E22", "#3498DB", "#95A5A6"])
        fig.update_layout(**PLOT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        dk = df[df["entrou_para_kpi"]].copy()
        serie = dk.groupby(dk["dt_aberto"].dt.date).size().reset_index()
        serie.columns = ["Data", "Volume"]
        fig2 = px.area(serie.tail(90), x="Data", y="Volume",
                       title="Volume diário - últimos 90 dias", color_discrete_sequence=["#3498DB"])
        fig2.update_layout(**PLOT)
        fig2.update_xaxes(tickformat="%d/%m")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Violações de OLA por mês - prioridades Alta e Média")
    dp = df[df["prioridade_cod"].isin(["P2", "P3"]) & df["entrou_para_kpi"]].copy()
    dp["Mês"] = dp["dt_aberto"].dt.month.map(lambda m: f"{m:02d} - {MESES[m-1]}")
    resumo = (dp.groupby(["Mês", "Prioridade"])
                .agg(**{"Total": ("Número", "count"), "Violações de OLA": ("kpi_violado_bin", "sum")})
                .reset_index())
    resumo["Taxa de violação (%)"] = (resumo["Violações de OLA"] / resumo["Total"] * 100).round(2)
    st.dataframe(resumo.sort_values("Mês", ascending=False), use_container_width=True, hide_index=True)


# Previsão de volume
elif aba == "Previsão de Volume":
    st.title("Previsão de Volume de Incidentes")
    c1, c2 = st.columns([1, 2])
    with c1:
        prio = st.selectbox("Prioridade", PRIOS)
        dias = st.slider("Quantos dias prever", 1, 30, 7)
        gerar = st.button("Gerar previsão", type="primary")
    with c2:
        st.info("A faixa clara ao redor da linha representa a incerteza da previsão. "
                "A previsão para **amanhã** ajuda o dia a dia; a de **uma semana** ajuda o planejamento de escala.")
    if gerar:
        fc = prever_volume(prio, dias)
        if fc is None:
            st.error("Previsão indisponível. Gere os modelos primeiro."); st.stop()
        sub = fc.iloc[:dias]
        d1 = float(sub["yhat"].iloc[0]); tot = float(sub["yhat"].sum())
        media = float(sub["yhat"].mean()); pico = float(sub["yhat"].max())
        pico_data = sub.loc[sub["yhat"].idxmax(), "ds"].strftime("%d/%m")
        hist = df[(df["Prioridade"] == prio) & df["entrou_para_kpi"]]
        hist_media = (hist.groupby(hist["dt_aberto"].dt.date).size().tail(90).mean()
                      if not hist.empty else d1)
        delta = d1 - hist_media

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Previsão para amanhã", f"{d1:.0f}",
                  f"{delta:+.0f} vs média recente", delta_color="inverse")
        k2.metric(f"Total em {dias} dias", f"{tot:.0f}")
        k3.metric("Média por dia", f"{media:.0f}")
        k4.metric(f"Pico previsto - {pico_data}", f"{pico:.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"].clip(lower=0), mode="lines",
                                 line_color="rgba(255,255,255,.15)", name="Limite superior"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"].clip(lower=0), mode="lines",
                                 fill="tonexty", line_color="rgba(255,255,255,.15)",
                                 fillcolor="rgba(255,92,57,.10)", name="Faixa de incerteza"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines+markers",
                                 line=dict(color="#ff6a3d", width=3),
                                 marker=dict(size=7), name="Previsão"))
        fig.add_hline(y=hist_media, line_dash="dot", line_color="#6b7c96",
                      annotation_text="média recente", annotation_position="top left")
        fig.update_layout(title=f"Previsão de volume - {prio}", xaxis_title="Data",
                          yaxis_title="Incidentes por dia", **PLOT)
        fig.update_xaxes(tickformat="%d/%m")
        st.plotly_chart(fig, use_container_width=True)
        tend = "acima" if delta > 0 else "abaixo"
        st.caption(f"A previsão para amanhã está **{abs(delta):.0f} incidentes {tend}** da média diária dos "
                   f"últimos 90 dias (**{hist_media:.0f}/dia**). A faixa sombreada é a incerteza da previsão "
                   f"(intervalo de 95%); a linha pontilhada é a média recente para comparação.")

        st.markdown("---")
        st.markdown("**Qualidade do modelo - real x previsto**")
        m = carregar_previsor(prio)
        if m is not None and getattr(m, "history", None) is not None:
            pr = m.predict(m.history.copy())
            real = m.history["y"].to_numpy(dtype=float)
            prev = pr["yhat"].clip(lower=0).to_numpy(dtype=float)
            mae = float(np.mean(np.abs(real - prev)))
            lim = float(max(real.max(), prev.max())) * 1.05
            figq = go.Figure()
            figq.add_trace(go.Scatter(x=real, y=prev, mode="markers",
                                      marker=dict(color="#ff6a3d", size=6, opacity=0.4), name="Dias"))
            figq.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                      line=dict(color="#8091a8", dash="dash"), name="Acerto perfeito"))
            figq.update_layout(**PLOT, height=440,
                               title=f"Real x Previsto - erro médio de {mae:.1f} incidentes/dia",
                               xaxis_title="Volume real por dia", yaxis_title="Volume previsto por dia")
            st.plotly_chart(figq, use_container_width=True)
            st.caption("Cada ponto é um dia do histórico. **Quanto mais perto da linha tracejada, mais a "
                       f"previsão acertou** o volume real daquele dia. Em média, o modelo erra cerca de "
                       f"**{mae:.0f} incidentes por dia** para essa prioridade.")


# Atingimento de indicadores
elif aba == "Atingimento de Indicadores":
    st.title("Projeção de Atingimento dos Indicadores")
    st.caption("Chance de cada indicador anual fechar saudável (atingimento igual ou acima de 100%). "
               "Quanto menor a contagem, maior o atingimento.")
    ano = 2025
    try:
        series = series_indicadores(ano)
    except Exception as e:
        st.error(f"Não foi possível carregar os indicadores: {e}"); st.stop()

    cols = st.columns(4)
    ordem = [("OLA", "P2"), ("OLA", "P3"), ("VOL", "P2"), ("VOL", "P3")]
    for (metrica, pc), col in zip(ordem, cols):
        total = float(series[(metrica, pc)].sum())
        pctv = float(kpi.atingimento_vec([total], kpi.FAIXAS[(metrica, pc)])[0])
        titulo = "Quebras de OLA" if metrica == "OLA" else "Volume tratado"
        marca = "saudável" if pctv >= 100 else "abaixo da meta"
        col.metric(f"{titulo} - {kpi.PRIORIDADES[pc]}", f"{pctv:.0f}%",
                   f"{pctv-100:+.0f} pontos vs meta - {marca}",
                   delta_color="normal")
    st.markdown("---")

    c1, c2, c3 = st.columns([1, 1, 2])
    _MET = {"Quebras de OLA": "OLA", "Volume tratado": "VOL"}
    _PRI = {kpi.PRIORIDADES["P2"]: "P2", kpi.PRIORIDADES["P3"]: "P3"}
    metrica = _MET[c1.selectbox("Indicador", list(_MET.keys()))]
    pc = _PRI[c2.selectbox("Prioridade", list(_PRI.keys()))]
    ini, fim = c3.select_slider("Período analisado (meses)", options=list(range(1, 13)),
                                value=(1, 12), format_func=lambda m: MESES[m-1])

    serie = series[(metrica, pc)]
    r = kpi.projetar(serie, metrica, pc, fim)
    realizado = float(serie.sum())
    realizado_pct = float(kpi.atingimento_vec([realizado], kpi.FAIXAS[(metrica, pc)])[0])
    realizado_periodo = float(serie[ini-1:fim].sum())

    unidade = "Quebras de OLA" if metrica == "OLA" else "Incidentes"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Atingimento no ano", f"{realizado_pct:.0f}%",
              "saudável" if realizado_pct >= 100 else "abaixo da meta", delta_color="off",
              help="Quanto MENOR a contagem, MAIOR o atingimento (faixas do dicionário de dados). "
                   "100% ou mais = saudável.")
    m2.metric(f"{unidade} no ano", f"{realizado:.0f}")
    if fim >= 12:
        m3.metric("Total no ano (fechado)", f"{realizado:.0f}")
    else:
        m3.metric(f"Projeção até dezembro (dados até {MESES[fim-1]})", f"{r['total_esperado']:.0f}")
    m4.metric("Chance de fechar saudável", f"{r['p_saudavel']*100:.0f}%")
    if fim < 12:
        st.caption(f"A **projeção** estima como o ano fecharia usando só os dados até {MESES[fim-1]}: "
                   f"cerca de **{r['total_esperado']:.0f}** {unidade.lower()}, com faixa provável entre "
                   f"**{r['total_p05']:.0f}** e **{r['total_p95']:.0f}**. Em dezembro, o ano já está fechado.")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Contagem realizada por mês (no período selecionado)**")
        meses_sel = list(range(ini, fim + 1))
        dfm = pd.DataFrame({"Mês": [MESES[m-1] for m in meses_sel],
                            "Contagem": [serie[m-1] for m in meses_sel]})
        figm = px.bar(dfm, x="Mês", y="Contagem", color_discrete_sequence=["#3a80c6"])
        figm.update_layout(**PLOT, showlegend=False, height=340)
        st.plotly_chart(figm, use_container_width=True)
    with g2:
        st.markdown("**Chance de fechar saudável ao longo do período**")
        xs = list(range(ini, fim + 1))
        ys = [kpi.projetar(serie, metrica, pc, m)["p_saudavel"] * 100 for m in xs]
        figt = go.Figure()
        figt.add_trace(go.Scatter(x=[MESES[m-1] for m in xs], y=ys, mode="lines+markers",
                                  line=dict(color="#ff6a3d", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(255,92,57,.10)"))
        figt.update_layout(**PLOT, height=340, yaxis_title="Chance de fechar saudável (%)",
                           yaxis_range=[-2, 104], showlegend=False)
        st.plotly_chart(figt, use_container_width=True)

    fwd = kpi.projetar(serie, metrica, pc, mes_corte=0)
    situacao = "tende a fechar saudável" if fwd["p_saudavel"] >= 0.5 else "corre risco de ficar abaixo da meta"
    st.info(f"**Projeção para o próximo ano:** esperado **{fwd['total_esperado']:.0f}** "
            f"({fwd['total_p05']:.0f}-{fwd['total_p95']:.0f}) - {situacao} - "
            f"chance de ficar saudável **{fwd['p_saudavel']*100:.0f}%**.")


# Risco de violação de ola
elif aba == "Risco de Violação de OLA":
    st.title("Risco de Violação de OLA")
    st.caption("Analise o risco de violação por grupo, prioridade, horário e origem. "
               "Deixe um filtro como está para incluir tudo e ter a visão completa. "
               "As notas de risco são de validação cruzada (o modelo pontuando dados que "
               "não viu no treino), por isso os números aqui são honestos, não inflados.")
    res = base_com_oof()
    if res is None:
        st.warning("Modelo ou base indisponível. Gere os dados e os modelos primeiro."); st.stop()

    base, feature_cols = res
    grupos_all = sorted(base["Grupo designado"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    prios = c1.multiselect("Prioridade", PRIOS, default=PRIOS,
                           placeholder="Todas as prioridades")
    grupos = c2.multiselect("Grupo (vazio = todos)", grupos_all, default=[],
                            placeholder="Todos os grupos")
    origem = c3.multiselect("Aberto por", ["Manual", "Monitoramento"], default=["Manual", "Monitoramento"],
                            placeholder="Todas as origens")
    c4, c5 = st.columns([3, 2])
    dias_sel = c4.multiselect("Dia da semana", DIAS, default=DIAS,
                              placeholder="Todos os dias")
    h_ini, h_fim = c5.slider("Faixa de hora de abertura", 0, 23, (0, 23))
    gerar = st.button("Gerar análise", type="primary")

    if gerar:
        d = base.copy()
        if prios:    d = d[d["Prioridade"].isin(prios)]
        if grupos:   d = d[d["Grupo designado"].isin(grupos)]
        if origem:   d = d[d["Aberto por"].isin(origem)]
        if dias_sel: d = d[d["dia_semana"].isin([DIAS.index(x) for x in dias_sel])]
        d = d[(d["hora_abertura"] >= h_ini) & (d["hora_abertura"] <= h_fim)]
        if d.empty:
            st.warning("Nenhum incidente encontrado com esses filtros."); st.stop()

        d["Risco previsto"] = d["risco"]   # nota honesta out-of-fold (validação cruzada)
        viol = int(d["kpi_violado_bin"].sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Incidentes no filtro", f"{len(d):,}".replace(",", "."))
        k2.metric("Violações de OLA reais", f"{viol:,}".replace(",", "."))
        k3.metric("Taxa de violação real", f"{viol/len(d)*100:.2f}%".replace(".", ","),
                  help="O que de fato aconteceu no histórico: quantos dos incidentes filtrados violaram o OLA.")
        k4.metric("Índice médio de risco", f"{d['Risco previsto'].mean()*100:.1f}".replace(".", ","),
                  help="Pontuação de risco do modelo (0 a 100), usada para PRIORIZAR chamados - quanto maior, "
                       "mais provável violar EM RELAÇÃO aos demais. Não é a chance literal de violar: o modelo "
                       "é calibrado para ordenar bem, não para acertar o percentual exato.")
        st.caption("A **taxa de violação real** é o que aconteceu de fato (~1%). O **índice de risco** é a "
                   "pontuação que o modelo usa para **priorizar** - serve para ordenar quem é mais arriscado, "
                   "não para prever o percentual. Por isso os dois números são bem diferentes.")

        # Poder de priorização (lift) - a prova positiva, imune à raridade do evento
        d_ord = d.sort_values("Risco previsto", ascending=False)
        n_top = max(1, int(len(d) * 0.10))
        viol_total = float(d["kpi_violado_bin"].sum())
        cap = float(d_ord.head(n_top)["kpi_violado_bin"].sum()) / viol_total * 100 if viol_total > 0 else 0
        taxa_geral = d["kpi_violado_bin"].mean() * 100
        taxa_top = d_ord.head(n_top)["kpi_violado_bin"].mean() * 100
        lift = taxa_top / taxa_geral if taxa_geral > 0 else 0
        st.success(f"**Poder de priorização:** os **10% de maior risco** apontados pelo modelo concentram "
                   f"**{cap:.0f}% de todas as violações** do período - e a taxa de violação nesse grupo é "
                   f"**{lift:.1f}x a média**. Ou seja: atacando essa fila primeiro, a equipe pega a maior "
                   f"parte das violações olhando uma fração pequena dos chamados. É para isso que o modelo serve.")
        st.markdown("---")

        g1, g2 = st.columns(2)
        with g1:
            top = (d.groupby("Grupo designado")["Risco previsto"].mean()
                     .sort_values(ascending=False).head(12).reset_index())
            top["Risco previsto"] = (top["Risco previsto"] * 100).round(1)
            fig = px.bar(top, x="Risco previsto", y="Grupo designado", orientation="h",
                         title="Risco médio previsto por grupo (%)",
                         color="Risco previsto", color_continuous_scale="OrRd")
            fig.update_layout(**PLOT, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            hh = d.groupby("hora_abertura")["kpi_violado_bin"].mean().reindex(range(24), fill_value=0) * 100
            fig = px.line(x=hh.index, y=hh.values, markers=True,
                          title="Taxa de violação real por hora de abertura (%)")
            fig.update_traces(line_color="#ff6a3d")
            fig.update_layout(**PLOT, xaxis_title="Hora do dia", yaxis_title="Taxa de violação (%)", height=380)
            st.plotly_chart(fig, use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            dd = d.groupby("dia_semana")["kpi_violado_bin"].mean().reindex(range(7), fill_value=0) * 100
            fig = px.bar(x=[DIAS[i] for i in dd.index], y=dd.values,
                         title="Taxa de violação real por dia da semana (%)",
                         color=dd.values, color_continuous_scale="OrRd")
            fig.update_layout(**PLOT, xaxis_title="", yaxis_title="Taxa de violação (%)",
                              coloraxis_showscale=False, height=360)
            st.plotly_chart(fig, use_container_width=True)
        with g4:
            fig = px.histogram(d, x="Risco previsto", nbins=30,
                               title="Distribuição do risco previsto", color_discrete_sequence=["#3a80c6"])
            fig.update_layout(**PLOT, xaxis_title="Risco previsto de violar o OLA",
                              yaxis_title="Nº de incidentes", height=360)
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Desempenho do modelo no conjunto filtrado**")
        y_true = d["kpi_violado_bin"].astype(int).values
        risco = d["Risco previsto"].values
        y_pred = (risco >= 0.54).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        g5, g6 = st.columns(2)
        with g5:
            fig = px.imshow(cmn, text_auto=".0%", color_continuous_scale="Blues",
                            x=["Previsto: não viola", "Previsto: viola"],
                            y=["Real: não viola", "Real: viola"],
                            title="Matriz de confusão (proporção por linha)")
            fig.update_layout(**PLOT, height=360, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with g6:
            if 0 < y_true.sum() < len(y_true):
                fpr, tpr, _ = roc_curve(y_true, risco)
                area = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         line=dict(color="#ff6a3d", width=2.5),
                                         name=f"Modelo (nota {area:.2f} de 1,00)"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(color="gray", dash="dash"),
                                         name="Referência: modelo sem informação"))
                fig.update_layout(**PLOT, height=360, title="Poder de separação do modelo",
                                  xaxis_title="Falsos alarmes",
                                  yaxis_title="Violações identificadas")
                fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Gráfico indisponível: o filtro não tem os dois desfechos (violou / não violou).")
        pego, limpo = cmn[1, 1] * 100, cmn[0, 0] * 100
        tp, fp = float(cm[1, 1]), float(cm[0, 1])
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        st.info(f"**Como ler:** das que **realmente violaram**, o modelo pegou **{pego:.0f}%**; das que "
                f"**não violaram**, acertou **{limpo:.0f}%** (os outros **{100-limpo:.0f}% são falsos "
                f"alarmes**). No **gráfico de separação**, o eixo horizontal são os falsos alarmes e o "
                f"vertical, as violações pegas.")
        st.caption(f"*Nota técnica:* por ser evento raro (~1%), se o modelo fosse usado como **alarme "
                   f"liga/desliga** num limiar fixo, a maioria dos disparos seria falsa (só ~{prec:.0f} de "
                   f"cada 100 seriam violações reais). É exatamente por isso que ele é usado como "
                   f"**priorização** - o ranking mostrado acima - e não como gatilho automático. A curva de "
                   f"separação confirma que esse ranking é bom.")

        st.subheader("Incidentes de maior risco")
        tab = d.sort_values("Risco previsto", ascending=False).head(15).copy()
        tab["Risco previsto"]     = (tab["Risco previsto"] * 100).round(1).astype(str) + "%"
        tab["Histórico do grupo"] = (tab["grupo_taxa_violacao_30d"] * 100).round(1).astype(str) + "%"
        tab["Histórico do item"]  = (tab["ic_taxa_violacao_30d"] * 100).round(1).astype(str) + "%"
        tab["Dia da semana"] = tab["dia_semana"].map(lambda i: DIAS[int(i)])
        tab = tab.rename(columns={"Número": "Incidente", "Grupo designado": "Grupo", "hora_abertura": "Hora"})
        st.dataframe(tab[["Incidente", "Grupo", "Prioridade", "Aberto por", "Dia da semana", "Hora",
                          "Histórico do grupo", "Histórico do item", "Risco previsto"]],
                     use_container_width=True, hide_index=True)
        st.caption("O **risco previsto** é definido principalmente pelo **histórico de violação do grupo e do "
                   "item** e pela prioridade - por isso incidentes do mesmo grupo têm risco parecido (o modelo "
                   "aprendeu que aquele grupo viola OLA com mais frequência). O horário e o dia aparecem como "
                   "contexto, mas pesam pouco neste modelo.")


# FILA DE RISCO (priorização operacional + causa-raiz por chamado)
elif aba == "Fila de Risco":
    st.title("Fila de Risco Operacional")
    st.caption("Os chamados que entram no indicador, ordenados pela prioridade de risco do modelo. "
               "Atenda de cima para baixo: é onde se concentram as violações de OLA. "
               "Selecione um chamado para ver a causa-raiz - o que eleva o risco dele.")
    res = fila_risco()
    if res is None:
        st.warning("Modelo ou base indisponível. Gere os dados e os modelos primeiro."); st.stop()
    fila, cols, stats = res

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chamados na fila", _br(stats["n"]))
    c2.metric("Nível crítico", _br(stats["crit"]),
              help="Chamados no topo do ranking de risco - prioridade máxima de atendimento preventivo.")
    c3.metric("Nível alto", _br(stats["alto"]))
    c4.metric("Índice médio de risco", f"{stats['media']*100:.0f}".replace(".", ","))
    st.markdown("---")

    st.subheader("Prioridade de atendimento - os 12 chamados mais críticos")
    linhas = ""
    for _, r in fila.head(12).iterrows():
        p = float(r["risco"]); nv = nivel_fila(p); cor = COR_NIVEL[nv]
        gtx = float(r.get("grupo_taxa_violacao_30d", 0)) * 100
        linhas += (
            f'<div class="frow">'
            f'<div class="fid">{r["Número"]}</div>'
            f'<div class="fgr"><b>{r["Grupo designado"]}</b>'
            f'<span>{r["Prioridade"]} - histórico do grupo {gtx:.0f}%</span></div>'
            f'<div class="fbadge" style="color:{cor};border-color:{cor}66;background:{cor}1f">{nv.upper()}</div>'
            f'<div class="fscore"><div class="fsbar"><i style="width:{p*100:.0f}%;background:{cor}"></i></div>'
            f'<span class="fpct">{p*100:.0f}</span></div>'
            f'</div>')
    st.markdown(f'<div class="fila">{linhas}</div>', unsafe_allow_html=True)
    st.caption("O **índice de risco (0-100)** é a pontuação de priorização do modelo: quanto maior, mais "
               "arriscado **em relação aos demais**. É um ranking para ordenar o atendimento, não a chance "
               "literal de violar - mesmo critério da aba **Risco de Violação de OLA**.")

    st.markdown("---")
    st.subheader("Causa-raiz do chamado - por que o risco está alto")
    opcoes = {f'{r["Número"]} - {r["Grupo designado"]} (índice {r["risco"]*100:.0f})': i
              for i, r in fila.iterrows()}
    sel = st.selectbox("Escolha um chamado da fila", list(opcoes.keys()))
    row = fila.iloc[[opcoes[sel]]]
    r0 = row.iloc[0]
    cr = causa_raiz_incidente(row, cols)

    ca, cb = st.columns([1.5, 1])
    with ca:
        if cr is None:
            st.info("Explicabilidade indisponível neste ambiente.")
        else:
            top_c = cr.head(8).iloc[::-1]
            labels = [_rotulo_feature(f) for f in top_c["feature"]]
            cores = ["#d6474b" if v > 0 else "#3a80c6" for v in top_c["shap"]]
            figc = go.Figure(go.Bar(x=top_c["shap"], y=labels, orientation="h", marker_color=cores))
            figc.update_layout(**PLOT, height=420, showlegend=False,
                               title=f"O que mais pesa no risco do chamado {r0['Número']}",
                               xaxis_title="Contribuição ao risco  (<- reduz  -  aumenta ->)",
                               yaxis_title="")
            st.plotly_chart(figc, use_container_width=True)
            st.caption("Cada barra é um fator. **Vermelho empurra o risco deste chamado para cima**; "
                       "**azul puxa para baixo.** O tamanho é o peso do fator na decisão do modelo.")
    with cb:
        nvsel = nivel_fila(float(r0["risco"]))
        st.markdown("**Ficha do chamado**")
        st.markdown(
            f'<div class="foco">'
            f'<div class="row"><div class="rl"><span>Índice de risco</span>'
            f'<span style="color:{COR_NIVEL[nvsel]};font-weight:700">{r0["risco"]*100:.0f} - {nvsel}</span></div></div>'
            f'<div class="row"><div class="rl"><span>Grupo</span><b>{r0["Grupo designado"]}</b></div></div>'
            f'<div class="row"><div class="rl"><span>Prioridade</span><b>{r0["Prioridade"]}</b></div></div>'
            f'<div class="row"><div class="rl"><span>Item</span><b>{r0["Item de configuração"]}</b></div></div>'
            f'<div class="row"><div class="rl"><span>Histórico do grupo (30d)</span>'
            f'<b>{float(r0.get("grupo_taxa_violacao_30d",0))*100:.0f}%</b></div></div>'
            f'<div class="row"><div class="rl"><span>Histórico do item (30d)</span>'
            f'<b>{float(r0.get("ic_taxa_violacao_30d",0))*100:.0f}%</b></div></div>'
            f'<div class="row"><div class="rl"><span>Hora de abertura</span>'
            f'<b>{int(r0["hora_abertura"])}h</b></div></div>'
            f'</div>', unsafe_allow_html=True)
        if cr is not None:
            motivos = [_rotulo_feature(f) for f in cr[cr["shap"] > 0]["feature"].head(3)]
            if motivos:
                st.markdown("**Principais motivos do risco elevado:** " + "; ".join(motivos) + ".")

    st.markdown("---")
    with st.expander("Ver a fila completa (80 chamados de maior risco)"):
        full = fila.copy()
        full["Índice de risco"] = (full["risco"] * 100).round(0).astype(int)
        full["Nível"] = full["risco"].map(nivel_fila)
        full["Histórico do grupo"] = ((full["grupo_taxa_violacao_30d"] * 100)
                                      .round(0).astype(int).astype(str) + "%")
        full["Hora"] = full["hora_abertura"].astype(int)
        full = full.rename(columns={"Número": "Incidente", "Grupo designado": "Grupo"})
        st.dataframe(
            full[["Incidente", "Grupo", "Prioridade", "Hora", "Histórico do grupo",
                  "Nível", "Índice de risco"]],
            use_container_width=True, hide_index=True,
            column_config={"Índice de risco": st.column_config.ProgressColumn(
                "Índice de risco", min_value=0, max_value=100, format="%d")})


# Perfis de incidentes (agrupamento)
elif aba == "Perfis de Incidentes":
    st.title("Perfis de Incidentes")
    st.caption("Os incidentes são agrupados automaticamente por comportamento (o quanto o item de "
               "configuração e o grupo costumam gerar e violar, a carga do grupo e o horário). "
               "Cada perfil recebe um nome pela característica que mais o destaca.")
    ds, resumo = calcular_perfis()
    if ds.empty:
        st.warning("Base indisponível para o agrupamento."); st.stop()

    nomes = list(resumo["Perfil"])
    paleta = px.colors.qualitative.Bold
    cmap = {n: paleta[i % len(paleta)] for i, n in enumerate(nomes)}

    figb = px.scatter(resumo, x="Taxa de violação (%)", y="Volume médio do grupo",
                      size="Incidentes", color="Perfil", text="Perfil", color_discrete_map=cmap,
                      size_max=95, hover_data={"Incidentes": True, "Perfil": False})
    figb.update_traces(textposition="top center", textfont=dict(size=12, color="#e7edf5"))
    figb.update_layout(**PLOT, height=470, showlegend=False,
                       title="Perfis por risco x volume - cada bolha é um perfil (tamanho = nº de incidentes)",
                       xaxis_title="Taxa de violação de OLA (%)",
                       yaxis_title="Volume médio do grupo (incidentes / 30 dias)")
    st.plotly_chart(figb, use_container_width=True)
    st.caption("Cada bolha é um perfil. **Mais à direita = mais viola OLA; mais acima = grupos mais "
               "movimentados; maior a bolha = mais incidentes.** O perfil de **alto risco** é pequeno em "
               "volume, mas fica isolado à direita - é onde mais vale a pena agir.")

    with st.expander("Ver o mapa detalhado (todos os incidentes)"):
        fig = px.scatter(ds, x="_x", y="_y", color="Perfil", color_discrete_map=cmap,
                         hover_data={"Grupo designado": True, "Prioridade": True,
                                     "_x": False, "_y": False, "Perfil": False},
                         title="Mapa de similaridade de comportamento")
        fig.update_traces(marker=dict(size=7, opacity=0.75))
        fig.update_layout(**PLOT, height=520, legend_title_text="Perfil",
                          legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                                      bgcolor="rgba(20,28,42,0.72)", bordercolor="#2a3648", borderwidth=1),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada ponto é um incidente; pontos próximos têm comportamento parecido "
                   "(é uma projeção, então os eixos não têm unidade).")

    st.subheader("O que caracteriza cada perfil")
    DESCR = {
        "Alto risco de violação": "Incidentes em itens de configuração com histórico alto de violar OLA "
                                  "(itens problemáticos/reincidentes). São poucos, mas violam muito mais que a média.",
        "Grupos de altíssimo volume": "Incidentes de times que tratam um volume enorme de chamados.",
        "Itens recorrentes (barulhentos)": "Incidentes concentrados nos mesmos itens, que geram muitos "
                                           "chamados seguidos - candidatos a causa-raiz.",
        "Grupos sobrecarregados": "Incidentes de grupos com muita carga aberta ao mesmo tempo.",
        "Operação rotineira": "O grosso da operação - comportamento normal e risco baixo.",
    }
    resumo_show = resumo.copy()
    resumo_show.insert(1, "O que define", resumo_show["Perfil"].map(DESCR).fillna("-"))
    st.dataframe(resumo_show, use_container_width=True, hide_index=True)


# TRIAGEM AUTOMÁTICA (classificação multi-classe)
elif aba == "Triagem (bônus)":
    st.title("Triagem Automática de Incidentes")
    st.info("**Demonstração complementar (bônus).** A classificação central do projeto - identificar os "
            "incidentes críticos que vão violar OLA - está na aba **Risco de Violação de OLA**. Aqui "
            "exploramos, como capacidade extra, um roteamento automático: dado um novo incidente, sugerir "
            "para qual time encaminhar. É um caminho para evolução futura, não o foco do desafio.")
    with st.spinner("Treinando o classificador de triagem..."):
        res = treinar_triagem()
    if res is None:
        st.warning("Base indisponível."); st.stop()
    cmn, labels, acc = res
    macro = float(np.mean(np.diag(cmn)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Acerto geral", f"{acc*100:.0f}%")
    m2.metric("Acerto médio por time", f"{macro*100:.0f}%",
              help="Média do acerto de cada time (macro) - é a métrica mais justa: o 'acerto geral' é "
                   "puxado pelos times de maior volume, então sozinho ele engana.")
    m3.metric("Times possíveis", f"{len(labels)}")
    m4.metric("Avaliado em", "25% (teste)")
    st.markdown("")

    fig = px.imshow(cmn, x=labels, y=labels, color_continuous_scale="Blues", zmin=0, zmax=1,
                    aspect="auto", text_auto=".0%",
                    labels=dict(x="Grupo previsto", y="Grupo real", color="Proporção"),
                    title="Matriz de confusão da triagem - grupo real x grupo previsto (proporção por linha)")
    fig.update_traces(textfont_size=8)
    fig.update_layout(**PLOT, height=680)
    fig.update_xaxes(tickangle=45, side="bottom")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Como ler:** cada **linha** é o grupo que de fato atendeu; cada **coluna**, o grupo que o "
               "modelo sugeriu. A **diagonal** são os acertos. O modelo acerta bem os **times de maior "
               "volume**, mas ainda **confunde alguns times menores** - tende a sugerir o time dominante "
               "(por isso o *acerto médio por time* é bem menor que o geral). É um bom ponto de partida "
               "para roteamento automático, com espaço claro para evoluir.")


# Mapa de riscos
elif aba == "Mapa de Riscos":
    st.title("Mapa de Riscos Operacionais")
    st.caption("Risco médio de violação de OLA por grupo e prioridade - onde agir preventivamente.")
    mapa = mapa_riscos()
    if mapa.empty:
        st.warning("Mapa de riscos ainda não gerado."); st.stop()
    piv = mapa.pivot_table(index="Grupo designado", columns="Prioridade",
                           values="prob_media_violacao", aggfunc="mean").fillna(0)
    fig = px.imshow(piv, text_auto=".0%", color_continuous_scale="RdYlGn_r",
                    aspect="auto", zmin=0, zmax=max(0.2, float(piv.values.max())),
                    title="Risco médio de violação - grupo x prioridade")
    fig.update_layout(**PLOT, height=560)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Detalhamento por grupo")
    det = mapa.sort_values("prob_media_violacao", ascending=False).rename(columns={
        "Grupo designado": "Grupo", "prob_media_violacao": "Risco médio",
        "total_incidentes": "Incidentes", "violacoes_reais": "Violações reais"})
    det["Risco médio"] = (det["Risco médio"] * 100).round(1).astype(str) + "%"
    st.dataframe(det[["Grupo", "Prioridade", "Risco médio", "Incidentes", "Violações reais"]],
                 use_container_width=True, hide_index=True)
