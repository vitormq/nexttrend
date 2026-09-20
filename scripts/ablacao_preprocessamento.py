# -*- coding: utf-8 -*-
# Ablacao de pre-processamento do modelo de risco de OLA.
# Mantem o mesmo modelo (XGBoost) e a mesma validacao cruzada, variando so o
# tratamento dos dados, para medir se normalizacao, capping de outliers ou
# log-transform trazem beneficio real no AUC out-of-fold.
# Salva o resultado em reports/ablacao_preprocessamento.json.
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE = Path(__file__).resolve().parent.parent
pipe = joblib.load(BASE / "models" / "saved" / "ola_predictor_pipeline.pkl")
cols = joblib.load(BASE / "models" / "saved" / "ola_feature_cols.pkl")
d = pd.read_parquet(BASE / "data" / "processed" / "incidents_features.parquet")
base = d[d["entrou_para_kpi"] & d["prioridade_cod"].isin(["P2", "P3"])].copy()
X = base[cols].copy()
y = base["kpi_violado_bin"].astype(int).values

cat_cols = [c for c in ["Grupo designado", "Item de configuração", "Aberto por", "Prioridade"] if c in cols]
num_cols = [c for c in cols if c not in cat_cols]
n_nan = int(X[num_cols].isna().sum().sum())
print(f"n = {len(X)} | positivos = {int(y.sum())} ({y.mean()*100:.1f}%) | NaN numericos = {n_nan}")


class NumOp(BaseEstimator, TransformerMixin):
    """Operacao numerica fold-aware (aprende no treino, aplica no teste)."""
    def __init__(self, num_cols, mode):
        self.num_cols = num_cols
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode == "cap":
            self.lo_ = X[self.num_cols].quantile(0.01)
            self.hi_ = X[self.num_cols].quantile(0.99)
        elif self.mode == "median":
            self.med_ = X[self.num_cols].median()
        return self

    def transform(self, X):
        X = X.copy()
        if self.mode == "cap":
            X[self.num_cols] = X[self.num_cols].clip(lower=self.lo_, upper=self.hi_, axis=1)
        elif self.mode == "median":
            X[self.num_cols] = X[self.num_cols].fillna(self.med_)
        elif self.mode == "log":
            X[self.num_cols] = np.sign(X[self.num_cols]) * np.log1p(np.abs(X[self.num_cols]))
        return X


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def auc(model):
    s = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return float(s.mean()), float(s.std())

variantes = {
    "Baseline (atual)":                clone(pipe),
    "Normalizacao (StandardScaler)":   Pipeline([("pre", clone(pipe[:-1])), ("scale", StandardScaler(with_mean=False)), ("clf", clone(pipe[-1]))]),
    "Capping de outliers [1%,99%]":    Pipeline([("cap", NumOp(num_cols, "cap")), ("base", clone(pipe))]),
    "Log nas numericas":               Pipeline([("log", NumOp(num_cols, "log")), ("base", clone(pipe))]),
    "Capping + Normalizacao":          Pipeline([("cap", NumOp(num_cols, "cap")), ("pre", clone(pipe[:-1])), ("scale", StandardScaler(with_mean=False)), ("clf", clone(pipe[-1]))]),
}

print(f"\n{'variante':32s} {'AUC (CV)':>10} {'desvio':>8}")
res = {}
for nome, modelo in variantes.items():
    m, s = auc(modelo)
    res[nome] = {"auc": round(m, 4), "std": round(s, 4)}
    print(f"{nome:32s} {m:>10.4f} {s:>8.4f}")

saida = BASE / "reports" / "ablacao_preprocessamento.json"
json.dump({"n_nan_numericos": n_nan, "resultados": res}, open(saida, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nOK: {saida}")
