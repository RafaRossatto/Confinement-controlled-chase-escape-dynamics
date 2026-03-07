import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import re
from pathlib import Path

# -------- Config --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"
FIT_DIR   = BASE_ROOT / "resultados_modelos" / "paper_response"/ "msd_fit"

# Lattice
L = 128
AREA = L**2

# Cores
cmap = colormaps.get_cmap("flag")

# ALTERAÇÃO AQUI: Labels no mesmo estilo do primeiro programa
BASES = [
    (r"$N^{C}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}=N^{E}_{0}$",      "Nc=Np"),
]

# Ler resultados
df = pd.read_csv(FIT_DIR / "msd_fit_results.csv")

# Extrair número de obstáculos
df["num_obs"] = df["obs"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
df["phi"] = df["num_obs"] / AREA

# ALTERAÇÃO AQUI: Ordem usando os tags
ordem = [tag for label, tag in BASES]  # ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]

plt.figure(figsize=(8,6))

for idx, conc_tag in enumerate(ordem):
    # ALTERAÇÃO AQUI: Pega o label formatado correspondente ao tag
    conc_label = next((label for label, tag in BASES if tag == conc_tag), conc_tag)
    
    sub = df[df["cenario"] == conc_tag].sort_values("phi")
    if sub.empty:
        continue
    
    plt.errorbar(
        sub["phi"], sub["alpha_mean"],
        yerr=sub["alpha_std"],
        fmt="o-", capsize=4,
        color=cmap(idx),
        label=conc_label  # ALTERAÇÃO AQUI: Usa o label formatado
    )

plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5, label=r"$\phi_c = 0.60$")
plt.xlabel(r"$\phi$", fontsize=18)
plt.ylabel(r"$ \langle \alpha \rangle$", fontsize=18)
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

# -------- Salvar figura --------
out_pdf = FIT_DIR / "alpha_vs_phi.pdf"
plt.savefig(out_pdf)
print(f">> Figura salva em:\n  - {out_pdf}")

plt.show()