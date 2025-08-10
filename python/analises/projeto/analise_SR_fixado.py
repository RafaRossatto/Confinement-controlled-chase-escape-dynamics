import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.plotagem import plotar_media_steps_por_NC # type: ignore


from utils.plotagem import gerar_heatmap_matriz_media # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore

import matplotlib.pyplot as plt

base_path = Path.home() / "Dados_Doc" / "p_0.3" 
valores_ne_nc = [25, 50, 100, 200, 400, 800]
medias_tau = {}

for ne_nc in valores_ne_nc:
    NE  = ne_nc
    NC = int(ne_nc * 0.8)
    O = 4915
    #print(f"\n=== Analisando NE = NC = {NE} ===")
    
    # Carregar os dados
    df_dict = carregar_dataframes_com_runs(
        pasta=base_path,
        NE=NE, TCC=1.00, TCT=1.00, O=O,
        NC=NC, num_runs=100,
        usar_cache=True, forcar_recarregar=False
    )
    
    limite = (1 / np.e) * NE
    passos_tau = []

    for chave, runs in df_dict.items():
        for run, df in runs.items():
            if "presas_vivas" in df.columns and "passo" in df.columns:
                abaixo_limite = df[df["presas_vivas"] < limite]
                if not abaixo_limite.empty:
                    passo_tau = abaixo_limite.iloc[0]["passo"]
                    passos_tau.append(passo_tau)
                else:
                    print(f"⚠️ Run {run} em {chave} NUNCA caiu abaixo de {limite:.2f} presas vivas.")
            else:
                print(f"[❌] Run {run} em {chave} NÃO tem as colunas esperadas. Colunas: {df.columns.tolist()}")

    if passos_tau:
        media_tau = np.mean(passos_tau)
        medias_tau[NE] = media_tau
        #print(f"📊 Média de τ: {media_tau:.2f}")
    else:
        medias_tau[NE] = None
        print("❌ Nenhuma run atingiu o limite (1/e) * NE.")

# Mostrar todas as médias
print("\n=== Médias de τ por NE ===")
for NE, media in medias_tau.items():
    if media is not None:
        print(f"NE = {NE}: τ médio ≈ {media:.2f}")
    else:
        print(f"NE = {NE}: nenhum τ encontrado")


"""

# Gera o gráfico do numero de escapers por tempo
df_NE_50_NC_5= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 5, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_10= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 10, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_25= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 25, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_50= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 50, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_100= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 100, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_250= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 250, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_500= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 500, usar_cache=True, forcar_recarregar=False)
df_NE_50_NC_1000= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 1000, usar_cache=True, forcar_recarregar=False)
#df_NE_50_NC_5000= carregar_dataframes(base_path, NE = 50, TCC= 0.99, TCT=0.99,O=0,NC = 5000, usar_cache=True, forcar_recarregar=False)

dfs_por_nc = {
    5: df_NE_50_NC_5,
    10: df_NE_50_NC_10,
    25: df_NE_50_NC_25,
    50: df_NE_50_NC_50,
    100: df_NE_50_NC_100,
    250: df_NE_50_NC_250,
    500: df_NE_50_NC_500,
    1000: df_NE_50_NC_1000,
    #5000: df_NE_50_NC_5000,
}

#plotar_media_steps_por_NC(dfs_por_nc, titulo="number of initial escapees $N^{O}_{E}=25$", salvar=True, nome_arquivo="N_E_25.pdf")
ncs, medias,desvio = plotar_media_steps_por_NC(dfs_por_nc, titulo="number of initial escapees $N^{O}_{E}=25$", salvar=True, nome_arquivo="N_E_25.pdf")


print (medias)
print (desvio)
"""