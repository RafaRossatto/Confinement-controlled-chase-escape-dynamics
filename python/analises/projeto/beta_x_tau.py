import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

# Caminho base
base = Path.home() / "Dados_Doc" / "resultados_modelos"

# Lista de arquivos
arquivos = sorted(base.glob("params_por_run_EXPbeta_Nc=Np_s_obs_*.csv"))

dados = []
L = 128
area = L**2

for arq in arquivos:
    # Extrair número de obstáculos
    m = re.search(r"s_obs_(\d+)", arq.name)
    if not m:
        continue
    n_obs = int(m.group(1))
    frac_obs = n_obs / area

    # Ler CSV
    df = pd.read_csv(arq)

    if "tau" in df.columns and "beta" in df.columns:
        tau_mean = df["tau"].mean()
        tau_std = df["tau"].std()
        beta_mean = df["beta"].mean()
        beta_std = df["beta"].std()
        dados.append({
            "frac_obs": frac_obs,
            "tau_mean": tau_mean, "tau_std": tau_std,
            "beta_mean": beta_mean, "beta_std": beta_std
        })

df_tau_beta = pd.DataFrame(dados).sort_values("beta_mean")

plt.figure(figsize=(10,7), dpi=150)

sc = plt.scatter(
    df_tau_beta["tau_mean"], df_tau_beta["beta_mean"],
    c=df_tau_beta["frac_obs"],          # cor ~ fração de obstáculos
    cmap="viridis",                     # colormap
    s=80,                               # tamanho dos pontos
    edgecolor="k"                       # borda preta
)

# barras de erro (sem marcador, só as barras)
plt.errorbar(
    df_tau_beta["tau_mean"], df_tau_beta["beta_mean"],
    #xerr=df_tau_beta["tau_std"], yerr=df_tau_beta["beta_std"],
    fmt="none", ecolor="gray", alpha=0.7, capsize=4
)

plt.xlabel(r"$\langle \tau \rangle$")
plt.ylabel(r"$\langle \beta \rangle$")
plt.title(r"Média de $\beta$ em função de $\tau$")

cbar = plt.colorbar(sc)
cbar.set_label("$\phi$")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tau_vs_beta_Nc=Np.pdf", dpi=300)
plt.show()
