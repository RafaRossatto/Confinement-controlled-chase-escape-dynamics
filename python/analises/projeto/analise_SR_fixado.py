import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import time
import re
from pathlib import Path








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gerar_heatmap_matriz_media(
    dfs,
    limite=680,
    titulo="Média de Escapers (valores > {limite} marcados como 'infinito')",
    salvar_pdf=False,
    nome_pdf="heatmap_escapers.pdf"
):
    """
    Gera um mapa de calor a partir de um dicionário de DataFrames com colunas 'steps'.

    Parâmetros:
    - dfs: dicionário no formato dfs["SR_TCC_x_SR_TCT_y"] = DataFrame
    - limite: valor máximo que será exibido no heatmap (valores maiores serão limitados)
    - titulo: título do gráfico (você pode usar {limite} para inserir o valor automaticamente)
    - salvar_pdf: se True, salva o gráfico como PDF
    - nome_pdf: nome do arquivo PDF a ser salvo
    """
    matriz_medias = np.full((50, 50), np.nan)

    for chave, df in dfs.items():
        try:
            sr_tcc = int(chave.split("_")[2])
            sr_tct = int(chave.split("_")[5])

            df.columns = df.columns.str.strip()

            if "steps" in df.columns:
                media = pd.to_numeric(df["steps"], errors='coerce').mean()
                matriz_medias[sr_tcc - 1, sr_tct - 1] = media
            else:
                print(f"⚠️ Coluna 'steps' não encontrada em {chave}")
        except Exception as e:
            print(f"❌ Erro ao processar {chave}: {e}")

    df_matriz = pd.DataFrame(
        matriz_medias,
        index=[f"SR_TCC_{i+1}" for i in range(50)],
        columns=[f"SR_TCT_{j+1}" for j in range(50)]
    )

    df_limitada = df_matriz.clip(upper=limite)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        df_limitada,
        cmap='viridis',
        fmt='',
        linewidths=0.5,
        cbar_kws={'label': 'Média de escapers'}
    )

    plt.title(titulo.format(limite=limite))
    plt.xlabel("SR_TCT")
    plt.ylabel("SR_TCC")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if salvar_pdf:
        plt.savefig(nome_pdf, format='pdf')
        print(f"📄 Gráfico salvo em: {nome_pdf}")













base_path = Path.home() / "Dados_Doc" / "TCC_095" / "TCT_095" / "Distribuição_quadrada" / "NC_500_NE_100_TCC_095_TCT_095"

NE = 100
TCC = 0.95
TCT = 0.95
O = 0

chaves_escolhidas = ["SR_TCC_5_SR_TCT_5", "SR_TCC_10_SR_TCT_5", "SR_TCC_15_SR_TCT_5",
                    "SR_TCC_20_SR_TCT_5", "SR_TCC_30_SR_TCT_5", "SR_TCC_35_SR_TCT_5",
                    "SR_TCC_40_SR_TCT_5", "SR_TCC_45_SR_TCT_5", "SR_TCC_50_SR_TCT_5"]

df_dados = carregar_dataframes_com_runs(base_path, NE, TCC, TCT,O, usar_cache=True, forcar_recarregar=False  # Coloque True se quiser ignorar o cache
)
#plotar_curvas_media_e_desvio(df_dados, chaves_escolhidas, valor_inicial=100)

dfs = carregar_dataframes(
    pasta=base_path,
    NE=100,
    TCC=0.95,
    TCT=0.95,
    O=0,
    usar_cache=True,
    forcar_recarregar=False)

""""
gerar_heatmap_matriz_media(
    dfs,
    limite=680,
    titulo="Média de Escapers (valores > {limite} marcados como 'infinito')",
    salvar_pdf=True,
    nome_pdf="mapa_escapers.pdf"
)
"""
df_matriz, max_abaixo = gerar_matriz_medias_steps(dfs, limite=1000)
