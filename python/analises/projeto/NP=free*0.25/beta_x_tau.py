import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

# ========= escolha do cenário (um por vez) =========
NC_TAG    = "Nc=Np*0.5"          # "Nc=Np", "Nc=Np*0.8", "Nc=Np*0.5"
LABEL_TEX = r"$N^{C}_{0}=N^{E}_{0}*0.5$"
# ===================================================

# Caminho base
base = Path.home() / "Dados_Doc" / "Np=free*0.25" / "resultados_modelos"

# Lista de arquivos do cenário escolhido
arquivos = sorted(base.glob(f"params_por_run_EXPbeta_{NC_TAG}_s_obs_*.csv"))

dados = []
L = 128
area = L**2

for arq in arquivos:
    m = re.search(r"s_obs_(\d+)", arq.name)
    if not m:
        continue
    n_obs = int(m.group(1))
    frac_obs = n_obs / area

    df = pd.read_csv(arq)
    if {"tau","beta"}.issubset(df.columns):
        tau_mean  = df["tau"].mean()
        tau_std   = df["tau"].std(ddof=1) if len(df) > 1 else 0.0
        beta_mean = df["beta"].mean()
        beta_std  = df["beta"].std(ddof=1) if len(df) > 1 else 0.0
        dados.append({
            "frac_obs": frac_obs,
            "tau_mean": tau_mean,  "tau_std": tau_std,
            "beta_mean": beta_mean,"beta_std": beta_std
        })

if not dados:
    raise SystemExit(f"Nenhum arquivo válido encontrado para {NC_TAG}.")

df_tb = pd.DataFrame(dados).sort_values("frac_obs")

plt.figure(figsize=(10,7), dpi=150)

# Normalização divergente centrada em 0.60 (φ = fração de obstáculos)
phi = df_tb["frac_obs"].to_numpy()
vmin = float(np.nanmin(phi)); 
vmax = float(np.nanmax(phi))
VCENTER = 0.60
if not (vmin <= VCENTER <= vmax):
    eps = 1e-9
    vmin = min(vmin, VCENTER - eps)
    vmax = max(vmax, VCENTER + eps)
norm = TwoSlopeNorm(vmin=vmin, vcenter=VCENTER, vmax=vmax)

# Scatter τ × β (cor = φ)
sc = plt.scatter(
    df_tb["tau_mean"], df_tb["beta_mean"],
    c=df_tb["frac_obs"], cmap="coolwarm", norm=norm,
    s=80, edgecolor="k", linewidth=0.4
)

# Barras de erro (x: τ_std, y: β_std)
#plt.errorbar(
#    df_tb["tau_mean"], df_tb["beta_mean"],
#    xerr=df_tb["tau_std"], yerr=df_tb["beta_std"],
#    fmt="none", ecolor="gray", alpha=0.7, capsize=4
#)



plt.xlabel(r"$\langle \tau \rangle$")
plt.ylabel(r"$\langle \beta \rangle$")
plt.title(rf" {LABEL_TEX}")

# Colorbar para φ
cbar = plt.colorbar(sc, pad=0.02)
cbar.set_label(r"$\phi$")
# marca o centro na colorbar
cbar.ax.axhline(VCENTER, color="k", lw=1)
cbar.set_ticks([vmin, VCENTER, vmax])
cbar.set_ticklabels([f"{vmin:.2f}", f"{VCENTER:.2f}", f"{vmax:.2f}"])

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"tau_vs_beta_{NC_TAG}.pdf", dpi=300)
plt.show()
