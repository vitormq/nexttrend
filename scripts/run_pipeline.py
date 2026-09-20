"""
run_pipeline.py - Executa o pipeline AIOps Locaweb de ponta a ponta.

Ordem: Etapa 2 (ingestão/limpeza) -> 4 (features) -> 5 (Prophet+XGBoost)
       -> 7 (SHAP/mapa de riscos) -> 8 (projeção de atingimento)
       -> 6 (clusterização, opcional). A EDA (Etapa 3) roda à parte.

Uso:
    python scripts/run_pipeline.py
    (no Docker: `docker compose run --rm pipeline`)

Requer data/raw/LW-DATASET.xlsx presente.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # raiz do projeto
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# (arquivo, obrigatório?)
ETAPAS = [
    ("etapa2_preprocessamento.py", True),
    ("etapa4_features.py",         True),
    ("etapa5_modelos.py",          True),
    ("etapa7_explicabilidade.py",  True),
    ("etapa8_kpi_atingimento.py",  True),
    ("etapa6_clusterizacao.py",    False),   # opcional: não aborta o pipeline se falhar
]


def rodar(script: str, obrigatorio: bool) -> bool:
    caminho = BASE_DIR / script
    if not caminho.exists():
        print(f"\n[SKIP] {script} não encontrado.")
        return not obrigatorio
    print("\n" + "=" * 70)
    print(f"  RODANDO {script}")
    print("=" * 70)
    res = subprocess.run([sys.executable, str(caminho)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        msg = "OBRIGATÓRIA" if obrigatorio else "opcional"
        print(f"\n[FALHA] {script} retornou {res.returncode} (etapa {msg}).")
        return not obrigatorio
    print(f"[OK] {script} concluída.")
    return True


def main() -> int:
    print("#" * 70)
    print("#  AIOps Locaweb - Pipeline completo")
    print(f"#  Raiz: {BASE_DIR}")
    print("#" * 70)
    for script, obrig in ETAPAS:
        if not rodar(script, obrig):
            print(f"\n[ABORTADO] Pipeline interrompido em {script}.")
            return 1
    print("\n" + "#" * 70)
    print("#  PIPELINE CONCLUÍDO - modelos em models/saved, relatórios em reports/")
    print("#  Suba os serviços:  docker compose up api dashboard")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
