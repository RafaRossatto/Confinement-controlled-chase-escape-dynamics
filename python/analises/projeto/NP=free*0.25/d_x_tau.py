from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ========= escolha do cenário (um por vez) =========
NC_TAG    = "Nc=Np"          # "Nc=Np", "Nc=Np*0.8", "Nc=Np*0.5"
LABEL_TEX = r"$N^{C}_{0}=N^{E}_{0}$"
# ===================================================

RES_DIR = Path.cwd()         # onde estão os CSVs gerados antes
L = 128
AREA = L**2

# arquivos esperados (gerados pelos scripts anteriores)
tau_csv = RES_DIR / f"tau_vs_phi_{NC_TAG}.csv"
d_csv   = RES_DIR / f"dist_minima_vs_phi_{NC_TAG}.csv"

if not tau_csv.exists() or not d_csv.exists():
    raise FileNotFoundError(f"Faltam CSVs: {tau_csv} ou {d_csv}")

# carregar
df_tau = pd.read_csv(tau_csv)  # colunas: phi, num_obs, n_runs, tau_mean, tau_std
df_d   = pd.read_csv(d_csv)    # colunas: phi, num_obs, n_runs_validos, d_mean, d_std, ...

# mesclar por num_obs (mais robusto que por phi)
df = pd.merge(df_tau, df_d, on="num_obs", suffixes=("_tau", "_d"))
if df.empty:
    raise SystemExit("[ERRO] Merge vazio. Verifique os CSVs.")

# garantir phi a partir de num_obs
df["phi"] = df["num_obs"] / AREA
df = df.sort_values("phi")

# ---------- colormap divergente em torno de 0.60 ----------
VCENTER = 0.60
VMIN = 0.00
# Use o máximo real dos dados (ou fixe um valor para comparar gráficos)
VMAX = float(df["phi"].max())            # opção dinâmica
# VMAX = 14745 / (128**2)                # opção fixa ≈ 0.90, boa p/ comparar
if VMAX < VCENTER:
    VMAX = VCENTER + 1e-9
norm = TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)

# --------------------- plot ---------------------
plt.figure(figsize=(10,7), dpi=150)
cmap_name = "coolwarm"   # escolha a paleta divergente
sc = plt.scatter(
    df["tau_mean"], df["d_mean"],
    c=df["phi"], cmap=cmap_name, norm=norm,
    s=80, marker="o", edgecolor="k", linewidth=0.4, zorder=3,
    label=LABEL_TEX
)


# barras de erro (como no exemplo)
#plt.errorbar(
 #   df["tau_mean"], df["d_mean"],
 #   xerr=df["tau_std"], yerr=df["d_std"],
 #   fmt="none", ecolor="gray", alpha=0.7, capsize=4, zorder=2
#)

plt.xlabel(r"$\langle \tau \rangle$")
plt.ylabel(r"$\langle d \rangle$")
plt.title(rf"{LABEL_TEX}")

# colorbar de φ (use valor do centro diretamente, não norm(...))
cbar = plt.colorbar(sc, pad=0.02)
cbar.set_label(r"$\phi$")
cbar.ax.axhline(VCENTER, color="k", lw=1)   # usa o valor direto
cbar.set_ticks([VMIN, VCENTER, VMAX])
cbar.set_ticklabels([f"{VMIN:.2f}", f"{VCENTER:.2f}", f"{VMAX:.2f}"])

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"d_vs_tau__{NC_TAG}.pdf", dpi=300)
plt.show()
