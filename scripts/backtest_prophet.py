# -*- coding: utf-8 -*-
# Backtest FORA DA AMOSTRA da previsão de volume (Prophet), para tornar a
# "capacidade de antecipação (D+1 e D+7)" defensável - em vez de reportar só o
# erro medido no próprio histórico (in-sample), que é otimista.
# Usa validação cruzada temporal: treina até um ponto, prevê os 7 dias seguintes
# (que o modelo NÃO viu) e compara com o real. Salva reports/backtest_prophet.json.
import warnings
warnings.filterwarnings("ignore")
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
from prophet.diagnostics import cross_validation

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "reports"; OUT.mkdir(exist_ok=True)

MODELOS = {"2 - Alta": "prophet_2_Alta.pkl", "3 - Média": "prophet_3_Média.pkl"}
resultados = {}

for prio, arq in MODELOS.items():
    cam = BASE / "models" / "saved" / arq
    if not cam.exists():
        print(f"[SKIP] {arq} não encontrado"); continue
    m = joblib.load(cam)

    # erro IN-SAMPLE (no próprio histórico) - só para comparar
    hist = m.history.copy()
    pin = m.predict(hist)["yhat"].clip(lower=0).to_numpy()
    real = hist["y"].to_numpy(dtype=float)
    mae_in = float(np.mean(np.abs(real - pin)))

    # backtest FORA DA AMOSTRA: treina até o corte, prevê os 7 dias seguintes
    cv = cross_validation(m, initial="270 days", period="20 days",
                          horizon="7 days", parallel=None)
    cv["yhat"] = cv["yhat"].clip(lower=0)
    cv["erro_abs"] = (cv["y"] - cv["yhat"]).abs()
    cv["h"] = (cv["ds"] - cv["cutoff"]).dt.days  # 1..7 dias à frente

    mae_out = float(cv["erro_abs"].mean())
    d1 = float(cv.loc[cv["h"] == 1, "erro_abs"].mean())     # D+1
    d7 = float(cv.loc[cv["h"] == 7, "erro_abs"].mean())     # D+7
    # MAPE só onde houve volume real (evita divisão por zero em dias vazios)
    mask = cv["y"] > 0
    mape = float((cv.loc[mask, "erro_abs"] / cv.loc[mask, "y"]).mean() * 100)
    media_real = float(cv["y"].mean())

    resultados[prio] = {
        "mae_in_sample": round(mae_in, 2),
        "mae_out_of_sample": round(mae_out, 2),
        "mae_D1": round(d1, 2),
        "mae_D7": round(d7, 2),
        "mape_out_of_sample_pct": round(mape, 1),
        "volume_medio_dia": round(media_real, 1),
        "n_janelas": int(cv["cutoff"].nunique()),
    }
    print(f"{prio}: MAE in={mae_in:.2f} | out={mae_out:.2f} | D+1={d1:.2f} | "
          f"D+7={d7:.2f} | MAPE={mape:.1f}% | vol.médio={media_real:.1f}/dia | "
          f"{cv['cutoff'].nunique()} janelas")

json.dump(resultados, open(OUT / "backtest_prophet.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("OK: reports/backtest_prophet.json")
