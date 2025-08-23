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

# === Plot Beta x Tau ===
plt.figure(figsize=(10,7), dpi=150)

plt.errorbar(
    df_tau_beta["beta_mean"],df_tau_beta["tau_mean"],
    xerr=df_tau_beta["beta_std"], yerr=df_tau_beta["tau_std"],
    fmt="o", capsize=5, label=r"Nc=Np"
)

plt.xlabel(r"$\langle \beta \rangle$")
plt.ylabel(r"$\langle \tau \rangle$")
plt.title(r"Média de $\tau$ em função de $\beta$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("tau_vs_beta_Nc=Np.pdf", dpi=300)
plt.show()
