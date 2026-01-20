import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import re
from pathlib import Path

# -------- CONFIGURAÇÕES PRINCIPAIS --------
# Diferentes tamanhos de rede - ⬅️ ALTERE OS CAMINHOS AQUI
REDE_CONFIGS = [
    (64, "L_64", Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"), # ⬅️ ALTERE  
    (256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")  # ⬅️ ALTERE
]

# Cores - usando a mesma colormap do primeiro código
cmap = colormaps.get_cmap("flag")

# Apenas a proporção 0.5
FRAC_C = "Nc=Np*0.5"
FRAC_C_LABEL = r"$N^{C}_{0}=0.5\,N^{E}_{0}$"

# Diretórios
BASE_ROOT = Path.home() / "Dados_Doc" / "resultados_modelos"
FIT_DIR_BASE = BASE_ROOT / "msd_fit_multiple_L"  # Onde estão os CSVs do MSD
OUT_DIR = BASE_ROOT / "alpha_vs_phi_multiple_L"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- PLOT PRINCIPAL: α vs φ --------
plt.figure(figsize=(10, 7))

for idx, (L, rede_nome, base_path) in enumerate(REDE_CONFIGS):
    # Caminho para o CSV de resultados desta rede
    fit_csv_path = FIT_DIR_BASE / rede_nome / f"{rede_nome}_{FRAC_C}_msd_fit_results.csv"
    
    if not fit_csv_path.exists():
        print(f"[x] Arquivo não encontrado: {fit_csv_path}")
        continue

    # Ler resultados
    df = pd.read_csv(fit_csv_path)
    
    # Verificar se as colunas necessárias existem
    required_cols = ["obs", "alpha_mean", "alpha_std"]
    if not all(col in df.columns for col in required_cols):
        print(f"[x] Colunas esperadas não encontradas em {fit_csv_path}")
        continue

    # Extrair número de obstáculos e calcular phi
    area = L**2
    df["num_obs"] = df["obs"].apply(lambda x: int(re.findall(r"\d+", x)[0]))
    df["phi"] = df["num_obs"] / area

    # Ordenar por phi
    df_sorted = df.sort_values("phi")

    # Plot - CORREÇÃO AQUI: usando df_sorted em vez de sub
    cor = cmap(idx )
    plt.errorbar(
        df_sorted["phi"], 
        df_sorted["alpha_mean"],
        yerr=df_sorted["alpha_std"],
        fmt="o-", 
        capsize=4,
        color=cor,
        label=f"L = {L}",  # Label com tamanho da rede
        markersize=6,
        linewidth=2
    )

plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5, 
            label=r"$\phi_c = 0.60$")
plt.xlabel(r"$\phi$", fontsize=18)
plt.ylabel(r"$\langle \alpha \rangle$", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Salvar figura
out_pdf = OUT_DIR / "alpha_vs_phi_multiple_L.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
print(f">> Figura α vs φ salva em:\n  - {out_pdf}")

plt.show()

print(f"\nGrágico salvo em: {OUT_DIR}")