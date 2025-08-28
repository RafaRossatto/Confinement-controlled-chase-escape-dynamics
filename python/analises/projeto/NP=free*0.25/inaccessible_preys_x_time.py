from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- parâmetros principais ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
            "obs_8192", "obs_9830","obs_11468", "obs_13107", "obs_14745"]

bases = [
    (r"$N_C=N_P$",      "Nc=Np"),
    (r"$N_C=0.5\,N_P$", "Nc=Np*0.5"),
    (r"$N_C=0.8\,N_P$", "Nc=Np*0.8"),
]

L = 128
area = L**2
NUM_RUNS = 100
# -------------------------------------------------------------------

def total_presas_por_obs(num_obs: int) -> float:
    """N_P = (área - #obs)/2, isto é, metade dos sítios livres após obstáculos."""
    livres = area - num_obs
    return livres / 4.0

cmap = plt.cm.viridis
plt.figure(figsize=(10, 6))

for base_idx, (label_tex, base_dirname) in enumerate(bases):
    base_path = Path.home() / f"Dados_Doc/Np=free*0.25/{base_dirname}"

    xs_n = []      # densidade de obstáculos n = #obs / área
    ys_mean = []   # média normalizada por N_P
    ys_std  = []   # desvio normalizado por N_P

    for obs_folder in obs_list:
        pasta_obs = base_path / obs_folder
        if not pasta_obs.exists():
            print(f" Pasta não encontrada: {pasta_obs}")
            continue

        subpastas = sorted([p for p in pasta_obs.iterdir() if p.is_dir()])
        if not subpastas:
            print(f" Nenhuma subpasta encontrada em {pasta_obs}")
            continue
        pasta_dentro = subpastas[0]

        valores_finais_runs = []
        for i in range(NUM_RUNS):
            pasta_run = pasta_dentro / f"run_{i:02d}"
            arquivo = pasta_run / "inaccessible_preys.txt"
            if arquivo.exists():
                conteudo = arquivo.read_text().split()
                if conteudo:
                    try:
                        ultimo = float(conteudo[-1])  # usa o valor final do run
                        valores_finais_runs.append(ultimo)
                    except ValueError:
                        print(f" Erro ao converter último valor em {arquivo}")

        if not valores_finais_runs:
            print(f" Sem valores em {pasta_dentro}")
            continue

        media_bruta = float(np.mean(valores_finais_runs))
        std_bruto   = float(np.std(valores_finais_runs))

        # extrai #obs do nome "obs_XXXX"
        try:
            num_obs = int(obs_folder.split("_")[1])
        except Exception:
            print(f" Não consegui extrair número de obstáculos de '{obs_folder}'")
            continue

        # N_P = (área - #obs) / 2  (mesmo para ambas as bases)
        N_P = total_presas_por_obs(num_obs)
        if N_P <= 0:
            print(f" N_P inválido para {obs_folder}")
            continue

        media_norm = media_bruta / N_P
        std_norm   = std_bruto   / N_P

        n = num_obs / area  # densidade de obstáculos no eixo x

        xs_n.append(n)
        ys_mean.append(media_norm)
        ys_std.append(std_norm)

    if ys_mean:
        xs_n, ys_mean, ys_std = zip(*sorted(zip(xs_n, ys_mean, ys_std)))

        plt.errorbar(xs_n, ys_mean, yerr=ys_std,
                     fmt='o-', capsize=5, markersize=5,
                     color=cmap(base_idx / max(1, len(bases) - 1)),
                     label=label_tex)

# linha de referência ~ limiar de percolação de sítio em rede quadrada
plt.axvline(x=0.59, color='black', linestyle='--', linewidth=1.5, label='$n \\approx 0.59$')

plt.xlabel("$\phi$")
plt.ylabel("$N_{\\text{inacc}}/N_P$")
plt.title("$N_C = N_P$  vs  $N_C = 0.5\\,N_P$  ")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('inaccessible_preys_normalized_by_half_free.pdf')
plt.show()
