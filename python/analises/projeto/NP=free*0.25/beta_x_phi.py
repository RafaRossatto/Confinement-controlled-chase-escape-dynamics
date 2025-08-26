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

    if "beta" in df.columns:
        beta_mean = df["beta"].mean()
        beta_std = df["beta"].std()
        dados.append({"frac_obs": frac_obs, "beta_mean": beta_mean, "beta_std": beta_std})

df_beta = pd.DataFrame(dados).sort_values("frac_obs")

plt.figure(figsize=(10,7), dpi=150)  # largura=10 pol, altura=7 pol, dpi=150

plt.errorbar(
    df_beta["frac_obs"], df_beta["beta_mean"],
    yerr=df_beta["beta_std"],
    fmt="o-", capsize=5, label="Nc=Np"
)

plt.axvline(x=0.59, color="red", linestyle="--", label=r"$\phi_c = 0.59$")

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\langle \beta \rangle$")
plt.title("")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("beta_x_phi_Nc=Np.pdf", dpi=300)  # salva com qualidade alta
plt.show()