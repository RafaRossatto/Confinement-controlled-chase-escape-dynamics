import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

# Caminho base
base = Path.home() / "Dados_Doc" / "Np=free*0.25"/ "resultados_modelos"

# Lista de arquivos
arquivos = sorted(base.glob("params_por_run_EXPbeta_Nc=Np_s_obs_*.csv"))

dados = []
L = 128
area = L**2  # 128^2

for arq in arquivos:
    # Extrair número de obstáculos
    m = re.search(r"s_obs_(\d+)", arq.name)
    if not m:
        continue
    n_obs = int(m.group(1))
    frac_obs = n_obs / area   # fração de obstáculos

    # Ler CSV
    df = pd.read_csv(arq)

    if "tau" in df.columns:
        tau_mean = df["tau"].mean()
        tau_std = df["tau"].std()
        dados.append({"frac_obs": frac_obs, "tau_mean": tau_mean, "tau_std": tau_std})

df_tau = pd.DataFrame(dados).sort_values("frac_obs")

plt.figure(figsize=(10,7), dpi=150)  # largura=10 pol, altura=7 pol, dpi=150

plt.errorbar(
    df_tau["frac_obs"], df_tau["tau_mean"],
    yerr=df_tau["tau_std"],
    fmt="o-", capsize=5, label="Nc=Np"
)

plt.axvline(x=0.59, color="red", linestyle="--", label=r"$\phi_c = 0.59$")

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\langle \tau \rangle$")
plt.title("")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tau_x_phi_Nc=Np.pdf", dpi=300)  # salva com qualidade alta
plt.show()