import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.plotagem import plotar_media_steps_por_NC # type: ignore


from utils.plotagem import gerar_heatmap_matriz_media # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore

import matplotlib.pyplot as plt

base_path = Path.home() / "Dados_Doc" / "CLI"/"Teste"


# Gera o gráfico do numero de escapers por tempo
df_NE_10_NC_5= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 5, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_10= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 10, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_25= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 25, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_50= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 50, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_100= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 100, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_250= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 250, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_500= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 500, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_1000= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 1000, usar_cache=True, forcar_recarregar=False)
df_NE_10_NC_5000= carregar_dataframes(base_path, NE = 25, TCC= 1.00, TCT=1.00,O=0,NC = 5000, usar_cache=True, forcar_recarregar=False)

dfs_por_nc = {
    5: df_NE_10_NC_5,
    10: df_NE_10_NC_10,
    25: df_NE_10_NC_25,
    50: df_NE_10_NC_50,
    100: df_NE_10_NC_100,
    250: df_NE_10_NC_250,
    500: df_NE_10_NC_500,
    1000: df_NE_10_NC_1000,
    5000: df_NE_10_NC_5000,
}


# Dados extraídos da linha vermelha do artigo
nc_artigo = [
    5.35, 7.23, 10.09, 13.65, 18.01, 23.37, 30.51, 39.51,
    51.20, 66.15, 85.51, 110.79, 143.14, 184.63, 237.88,
    306.14, 394.53, 508.56, 655.51, 844.26, 1086.73, 1397.31,
    1793.53, 2300.80, 2954.88, 3794.79, 4875.91, 6263.71,
    8040.97, 10307.67
]

t_artigo = [
    103775.29, 81279.62, 66273.38, 54310.73, 44560.39, 36402.96, 29567.57,
    24014.33, 19457.61, 15711.49, 12676.75, 10246.97, 8286.83, 6702.06,
    5425.58, 4373.86, 3524.47, 2822.72, 2260.66, 1810.81, 1457.78, 1176.82,
    952.81, 774.36, 630.95, 513.69, 419.52, 344.23, 281.96, 237.89
]


#plotar_media_steps_por_NC(dfs_por_nc, titulo="number of initial escapees $N^{O}_{E}=25$", salvar=True, nome_arquivo="N_E_25.pdf")
ncs, medias = plotar_media_steps_por_NC(dfs_por_nc)



plt.figure(figsize=(8, 6))

print (medias)
# Seus dados aqui
plt.plot(ncs, medias, 'o-', label='Meus dados', color='blue')

# Dados do artigo
plt.plot(nc_artigo, t_artigo, 'o--', label='Artigo (linha vermelha)', color='red')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("the number of chasers $N_C$")
plt.ylabel("trapping time $T$")
plt.title("Comparação dos dados")
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()