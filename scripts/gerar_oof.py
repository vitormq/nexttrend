# -*- coding: utf-8 -*-
# Gera os scores de risco "out-of-fold" (validação cruzada) usados pelo painel.
# Cada incidente recebe a nota de um modelo que NÃO o usou no treino - é a
# medida HONESTA de desempenho, que evita o AUC/lift inflado de avaliar o
# modelo nos próprios dados de treino.
# O resultado é salvo em data/processed/oof_scores.parquet para o dashboard
# carregar pronto (rápido e leve no deploy, sem recalcular a validação cruzada).
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent

# modelo treinado + colunas de entrada + base de features
pipe = joblib.load(BASE / "models" / "saved" / "ola_predictor_pipeline.pkl")
cols = joblib.load(BASE / "models" / "saved" / "ola_feature_cols.pkl")
d = pd.read_parquet(BASE / "data" / "processed" / "incidents_features.parquet")

# só os incidentes que entram no indicador, nas prioridades Alta (P2) e Média (P3)
base = d[d["entrou_para_kpi"] & d["prioridade_cod"].isin(["P2", "P3"])].copy()
y = base["kpi_violado_bin"].astype(int).values

# 5 dobras estratificadas: cada previsão vem da dobra em que o incidente ficou fora do treino
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base["risco"] = cross_val_predict(
    clone(pipe), base[cols], y, cv=skf, method="predict_proba", n_jobs=-1
)[:, 1]

# confere o AUC honesto e a unicidade da chave antes de salvar
auc = roc_auc_score(y, base["risco"].values)
assert base["Número"].is_unique, "Número não é único - ajustar a chave do merge"

out = base[["Número", "risco"]].copy()
out.to_parquet(BASE / "data" / "processed" / "oof_scores.parquet", index=False)
print(f"OK oof_scores.parquet: {len(out)} linhas | AUC out-of-fold = {auc:.3f}")
