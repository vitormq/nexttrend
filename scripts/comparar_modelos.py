# -*- coding: utf-8 -*-
# Compara famílias de modelos no MESMO problema (risco de violação de OLA),
# usando AUC de validação cruzada (fora da amostra). Serve para JUSTIFICAR a
# escolha do XGBoost e responder objetivamente "redes neurais dariam mais
# assertividade?". Salva reports/comparacao_modelos.json e .png (para o deck).
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "reports"; OUT.mkdir(exist_ok=True)

# modelo/colunas treinados + base de features
pipe = joblib.load(BASE / "models" / "saved" / "ola_predictor_pipeline.pkl")
cols = joblib.load(BASE / "models" / "saved" / "ola_feature_cols.pkl")
d = pd.read_parquet(BASE / "data" / "processed" / "incidents_features.parquet")
base = d[d["entrou_para_kpi"] & d["prioridade_cod"].isin(["P2", "P3"])].copy()
X = base[cols]
y = base["kpi_violado_bin"].astype(int).values

pre = pipe[:-1]  # reaproveita o MESMO pré-processamento (ColumnTransformer)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def avaliar(nome, modelo):
    s = cross_val_score(modelo, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"  {nome:26s} AUC {s.mean():.3f} +/- {s.std():.3f}")
    return {"modelo": nome, "auc": float(s.mean()), "desvio": float(s.std())}

# rede neural e logística precisam de escala; XGBoost não
def com_escala(clf):
    return Pipeline([("pre", clone(pre)),
                     ("sc", StandardScaler(with_mean=False)),
                     ("clf", clf)])

print("Comparando famílias de modelos (AUC de validação cruzada)...")
resultados = [
    avaliar("XGBoost (atual)", clone(pipe)),
    avaliar("Regressão Logística", com_escala(LogisticRegression(max_iter=1000, class_weight="balanced"))),
    avaliar("Rede Neural (64,32)", com_escala(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42))),
    avaliar("Rede Neural (128,64,32)", com_escala(MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, early_stopping=True, random_state=42))),
]
json.dump(resultados, open(OUT / "comparacao_modelos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# gráfico de barras (tema escuro, para combinar com o deck/painel)
BG, CORAL, INK, MUT, GRID = "#0F1826", "#FF6A3D", "#E7EDF5", "#94A2BA", "#2A3648"
nomes = [r["modelo"] for r in resultados][::-1]
aucs = [r["auc"] for r in resultados][::-1]
erros = [r["desvio"] for r in resultados][::-1]
cores = [CORAL if "XGBoost" in n else "#3A80C6" for n in nomes]
fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.barh(nomes, aucs, xerr=erros, color=cores, height=0.62,
        error_kw=dict(ecolor=MUT, capsize=4, lw=1.2))
for i, a in enumerate(aucs):
    ax.text(a + 0.006, i, f"{a:.3f}".replace(".", ","), va="center",
            color=INK, fontsize=11, fontweight="bold")
ax.set_xlim(0.70, 0.83)
ax.axvline(0.5, color=GRID, lw=0.8)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUT); ax.set_xlabel("AUC (validação cruzada)", color=MUT)
ax.set_title("Risco de OLA - XGBoost supera redes neurais neste dado tabular",
             color=INK, fontsize=11.5, pad=12)
plt.tight_layout()
fig.savefig(OUT / "comparacao_modelos.png", facecolor=BG, bbox_inches="tight")
plt.close()
print("OK: reports/comparacao_modelos.json e .png")
