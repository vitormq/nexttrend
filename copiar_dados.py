# Documentação: Script para copiar o dataset original para a pasta do projeto
import shutil
import os

caminho_origem = r"C:\Users\Vítor\Desktop\Enterprise - python\LW-DATASET.xlsx"
caminho_destino = r"C:\Users\Vítor\aiops_locaweb\data\raw\LW-DATASET.xlsx"

print("Iniciando a cópia do arquivo...")

if os.path.exists(caminho_origem):
    shutil.copy2(caminho_origem, caminho_destino)
    print(f"Sucesso! O arquivo foi copiado para:\n{caminho_destino}")
else:
    print("Erro: Arquivo original não encontrado.")
