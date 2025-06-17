import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import time
import re
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import os

def carregar_dataframes_com_runs(pasta, NE, TCC, TCT, O, NC=500, num_runs=100):
    """
    Lê arquivos .dat_run_X_presas_por_passo.csv com base nos parâmetros fornecidos.

    Retorna:
    - dicionário de dicionários com os DataFrames
      Ex: dataframes["SR_TCC_10_SR_TCT_5"][3] -> DataFrame da run 3
    """
    dataframes = {}

    for sr_tcc in range(1, 51):
        for sr_tct in range(1, 51):
            chave_config = f"SR_TCC_{sr_tcc}_SR_TCT_{sr_tct}"
            dataframes[chave_config] = {}

            for run in range(1, num_runs + 1):
                nome_arquivo = (
                    f"NC_{NC}_NE_{NE}_O_{O}_TCC_{TCC}_SR_{sr_tcc}"
                    f"_TCT_{TCT}_SR_{sr_tct}.dat_run_{run}_presas_por_passo.csv")
                caminho_arquivo = os.path.join(pasta, nome_arquivo)

                if os.path.exists(caminho_arquivo):
                    try:
                        df = pd.read_csv(caminho_arquivo)
                        dataframes[chave_config][run] = df
                    except Exception as e:
                        print(f" Erro ao ler {nome_arquivo}: {e}")
                # else:
                #     print(f"Arquivo não encontrado: {nome_arquivo}")

    return dataframes

def plotar_curvas_media_e_desvio(dataframes, chaves_escolhidas, titulo="Média das Presas Vivas por Configuração",
                                  salvar_pdf=False, nome_pdf="grafico.pdf", mostrar=True, valor_inicial=None):
    """
    Plota a média e desvio padrão das curvas de presas vivas ao longo do tempo para cada configuração.

    Parâmetros:
    - dataframes: dicionário do tipo dataframes["SR_TCC_x_SR_TCT_y"][run] = DataFrame
    - chaves_escolhidas: lista de chaves de configuração
    - titulo: título do gráfico
    - salvar_pdf: se True, salva como PDF
    - nome_pdf: nome do arquivo PDF
    - mostrar: se True, exibe na tela
    - valor_inicial: valor que será colocado no passo 0, e os demais valores serão deslocados em +10 passos
    """
    plt.figure(figsize=(12, 7))

    for chave in chaves_escolhidas:
        if chave in dataframes:
            runs_dict = dataframes[chave]
            if not runs_dict:
                print(f"[Aviso] Nenhuma run encontrada para: {chave}")
                continue

            # Obter o número máximo de passos e preparar arrays
            max_passo = max(df['passo'].iloc[-1] for df in runs_dict.values() if not df.empty)
            intervalo = runs_dict[next(iter(runs_dict))]['passo'].iloc[1] - runs_dict[next(iter(runs_dict))]['passo'].iloc[0]

            # Lista de curvas de presas por passo
            curvas = []
            for run_df in runs_dict.values():
                if not run_df.empty:
                    curva = run_df.set_index('passo').reindex(range(0, max_passo + 1, intervalo))['presas_vivas'].fillna(method='ffill').fillna(method='bfill')
                    curva_np = curva.to_numpy()

                    # Inserir valor inicial, se fornecido
                    if valor_inicial is not None:
                        curva_np = np.insert(curva_np, 0, valor_inicial)

                    curvas.append(curva_np)

            if curvas:
                matriz = np.array(curvas)
                media = matriz.mean(axis=0)
                desvio = matriz.std(axis=0)

                # Criar vetor de passos com deslocamento
                passos = np.arange(len(media)) * intervalo
                if valor_inicial is not None:
                    passos = passos - intervalo  # desloca todos os dados +intervalo pra frente
                    passos[0] = 0  # passo 0 reservado para valor_inicial

                plt.plot(passos, media, label=chave)
                plt.fill_between(passos, media - desvio, media + desvio, alpha=0.3)
        else:
            print(f"[Aviso] Chave não encontrada: {chave}")

    plt.xlabel('Steps')
    plt.ylabel('Escapers_Alive')
    plt.title(titulo)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    if salvar_pdf:
        plt.savefig(nome_pdf, format='pdf')

    if mostrar:
        plt.show()


base_path = Path.home()/"Dados_Doc"/"TCC_095"/"TCT_095"/"Distribuição_quadrada"/"NC_500_NE_100_TCC_095_TCT_095"
NE = 100
TCC = 0.95
TCT = 0.95
O = 0
chaves_escolhidas=["SR_TCC_5_SR_TCT_5", "SR_TCC_10_SR_TCT_5", "SR_TCC_15_SR_TCT_5","SR_TCC_20_SR_TCT_5","SR_TCC_30_SR_TCT_5", 
                    "SR_TCC_35_SR_TCT_5","SR_TCC_40_SR_TCT_5", "SR_TCC_45_SR_TCT_5", "SR_TCC_50_SR_TCT_5"]
df_NC_500_NE_500_TCC_095_TCT_095_O_00 = carregar_dataframes_com_runs(base_path, NE, TCC, TCT, O)
plotar_curvas_media_e_desvio(df_NC_500_NE_500_TCC_095_TCT_095_O_00, chaves_escolhidas, valor_inicial=100)