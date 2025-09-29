import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import re
from pathlib import Path

# -------- Config --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
FIT_DIR   = BASE_ROOT / "resultados_modelos" / "msd_fit"

# Lattice
L = 128
AREA = L**2

# Cores
cmap = colormaps.get_cmap("flag")

# Ler resultados
df = pd.read_csv(FIT_DIR / "msd_fit_results.csv")

# Extrair número de obstáculos
df["num_obs"] = df["obs"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
df["phi"] = df["num_obs"] / AREA

# Ordem desejada dos cenários
ordem = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]

plt.figure(figsize=(8,6))

for idx, conc in enumerate(ordem):
    sub = df[df["cenario"] == conc].sort_values("phi")
    if sub.empty:
        continue
    
    plt.errorbar(
        sub["phi"], sub["alpha_mean"],
        yerr=sub["alpha_std"],
        fmt="o-", capsize=4,
        color=cmap(idx),
        label=conc
    )

plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5, label=r"$\phi_c = 0.6$")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\alpha$")
plt.legend()
plt.grid()
plt.tight_layout()

# -------- Salvar figura --------
out_pdf = FIT_DIR / "alpha_vs_phi.pdf"
plt.savefig(out_pdf)
print(f">> Figura salva em:\n  - {out_pdf}")

plt.show()
