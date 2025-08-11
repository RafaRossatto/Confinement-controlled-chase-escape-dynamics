import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.plotagem import plotar_media_steps_por_NC # type: ignore
from utils.plotagem import gerar_heatmap_matriz_media # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore


# Lista das pastas obs_XXXX
obs_list = [
    "obs_00", "obs_1638", "obs_3276", "obs_4914", "obs_6553",
    "obs_8192", "obs_9830", "obs_11468", "obs_13107"
]

# Lista dos nC_XX a percorrer
nc_list = ["nC_05", "nC_10", "nC_50", "nC_100", "nC_500", "nC_1000", "nC_1500"]

# Área da rede
L = 128
area = L**2

# Caminho base
base_path = Path.home() / "Dados_Doc"

# Configuração do colormap
cmap = plt.cm.viridis

plt.figure(figsize=(10, 6))

for idx, nc_folder in enumerate(nc_list):
    obs_normalizados = []
    medias = []
    stds = []
    presas_totais_list = []  # <-- novo

    for obs_folder in obs_list:
        pasta_nc = base_path / obs_folder / nc_folder
        todos_os_valores = []

        if pasta_nc.is_dir():
            for i in range(100):
                pasta_run = pasta_nc / f"run_{i:02d}"
                arquivo = pasta_run / "inaccessible_preys.txt"
                if arquivo.exists():
                    with open(arquivo, "r") as f:
                        conteudo = f.read().strip()
                    try:
                        valores = [float(x) for x in conteudo.split()]
                        todos_os_valores.extend(valores)
                    except ValueError:
                        print(f"⚠ Erro ao converter valores em {arquivo}")

        if todos_os_valores:
            media = np.mean(todos_os_valores)
            std = np.std(todos_os_valores)

            num_obs = int(obs_folder.split("_")[1])
            n = num_obs / area
            presas_totais = (area - num_obs) / 2  # <-- depende de CADA obs_folder

            obs_normalizados.append(n)
            medias.append(media)
            stds.append(std)
            presas_totais_list.append(presas_totais)  # <-- guarda alinhado

    if medias:
        # normaliza ponto a ponto pelo total de presas daquele n (obs)
        medias_norm = [m / p for m, p in zip(medias, presas_totais_list)]
        stds_norm   = [s / p for s, p in zip(stds, presas_totais_list)]

        # ordena por densidade n
        obs_normalizados, medias_norm, stds_norm = zip(
            *sorted(zip(obs_normalizados, medias_norm, stds_norm))
        )

        plt.errorbar(obs_normalizados, medias_norm, yerr=stds_norm,
                     fmt='o-', capsize=5, markersize=5,
                     color=cmap(idx / (len(nc_list) - 1)),
                     label=f"$N_{{O}}^{{C}} = {nc_folder.split('_')[1]}$")

# Linha vertical em x = 0.59
plt.axvline(x=0.59, color='black', linestyle='--', linewidth=1.5, label='$n$ = 0.59')

plt.xlabel("$n$")
plt.ylabel("Inaccessible_preys")
plt.title("Curvas normalizadas para diferentes nC")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('test.pdf')
plt.show()



