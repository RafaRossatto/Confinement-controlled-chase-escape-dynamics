import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
from matplotlib import colormaps

# -------------------------------
# 0. Configurações principais
# -------------------------------
BASE_ROOT   = Path.home() / "Dados_Doc" / "Np=free*0.25"
HAZARD_DIR  = BASE_ROOT / "resultados_modelos" / "hazard"
OUT_DIR     = BASE_ROOT / "resultados_modelos" / "hazard_max"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 128
AREA = L**2

SCENARIOS = [
    ("Nc=Np*0.5", r"$N^{C}_{0}=0.5\,N^{E}_{0}$"),
    ("Nc=Np*0.8", r"$N^{C}_{0}=0.8\,N^{E}_{0}$"),
    ("Nc=Np",     r"$N^{C}_{0}=N^{E}_{0}$"),   
]

# -------------------------------
# 1. Extrair máximo do hazard
# -------------------------------
resultados = []

for nc_tag, nc_label in SCENARIOS:
    files = sorted(HAZARD_DIR.glob(f"{nc_tag}_*_hazard_data.csv"))
    for f in files:
        match = re.search(r"ob(\d+)", f.stem)
        if match:
            obs_val = int(match.group(1))
        else:
            print(f"[Aviso] Não encontrei número de obstáculos em {f.name}")
            continue

        phi = obs_val / AREA
        df = pd.read_csv(f)
        if df.empty:
            continue

        # máximo do hazard
        idxmax = df["h(t)"].idxmax()
        h_max = df.loc[idxmax, "h(t)"]
        h_low = df.loc[idxmax, "h(t)-"]
        h_up  = df.loc[idxmax, "h(t)+"]

        # erro como intervalo (assimétrico possível)
        err_lower = abs(h_max - h_low)
        err_upper = abs(h_up - h_max)

        step_max = df.loc[idxmax, "step"]

        resultados.append({
            "cenario": nc_tag,
            "cenario_label": nc_label,
            "obs": obs_val,
            "phi": phi,
            "h_max": h_max,
            "err_lower": err_lower,
            "err_upper": err_upper,
            "step_max": step_max,
            "file": f.name
        })

df_res = pd.DataFrame(resultados)

if df_res.empty:
    print("[ERRO] Nenhum resultado encontrado, revise os nomes dos arquivos.")
else:
    df_res.to_csv(OUT_DIR / "hazard_max_summary.csv", index=False)
    print("[OK] Tabela resumo salva em hazard_max_summary.csv")

    # -------------------------------
    # 2. Plotar h_max vs phi com barras de erro
    # -------------------------------
    plt.figure(figsize=(8,6))
    cmap = colormaps.get_cmap("flag")

    for idx, (nc_tag, nc_label) in enumerate(SCENARIOS):
        subset = df_res[df_res["cenario"] == nc_tag].sort_values("phi")
        if subset.empty:
            continue

        plt.errorbar(
            subset["phi"], subset["h_max"],
            yerr=[subset["err_lower"], subset["err_upper"]],
            fmt="o-", capsize=4,
            color=cmap(idx),
            label=nc_label
        )
    plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5,
            label=r"$\phi = 0.60$")
    plt.xlabel(r"$\phi$", fontsize=16)
    plt.ylabel(r"$\max h(t)$", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hazard_max_vs_phi.pdf", dpi=300)
    plt.close()

    print("[OK] Figura salva em hazard_max_vs_phi.pdf")
