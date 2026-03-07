from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

# ========= lista de cenários =========
SCENARIOS = [
    ("Nc=Np",     r"$N^{C}_{0}=N^{E}_{0}$"),
    ("Nc=Np*0.8", r"$N^{C}_{0}=0.8\,N^{E}_{0}$"),
    ("Nc=Np*0.5", r"$N^{C}_{0}=0.5\,N^{E}_{0}$"),
]
# =====================================

L = 128
AREA = L**2
BASE_ROOT   = Path.home() / "Dados_Doc" / "Np=free*0.25"
HAZARD_DIR  = BASE_ROOT / "resultados_modelos" / "hazard"
DIST_DIR    = BASE_ROOT / "resultados_modelos" / "dist_min"
OUT_DIR     = BASE_ROOT / "resultados_modelos" / "tau_x_d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- colormap divergente ----------
VCENTER = 0.60
VMIN = 0.0

for NC_TAG, LABEL_TEX in SCENARIOS:
    print(f"[INFO] Processando {NC_TAG}...")

    # ---------- carregar τ ----------
    tau_file = HAZARD_DIR / f"{NC_TAG}_fit_results.csv"
    if not tau_file.exists():
        print(f"[x] Arquivo não encontrado: {tau_file}")
        continue

    df_tau = pd.read_csv(tau_file)
    if not {"obs", "tau", "tau_err"}.issubset(df_tau.columns):
        print(f"[x] Colunas τ não encontradas em {tau_file}")
        continue

    df_tau["num_obs"] = df_tau["obs"].str.replace("s_obs_", "", regex=False).astype(int)
    df_tau["phi"]     = df_tau["num_obs"] / AREA
    df_tau = df_tau[["phi", "tau", "tau_err"]]

    # ---------- carregar d ----------
    d_file = DIST_DIR / f"dist_min_agregado_{NC_TAG}.csv"
    if not d_file.exists():
        print(f"[x] Arquivo não encontrado: {d_file}")
        continue

    df_d = pd.read_csv(d_file)
    if "dist_mean_avg" not in df_d.columns:
        print(f"[x] Coluna 'dist_mean_avg' não encontrada em {d_file}")
        continue

    df_d = df_d.rename(columns={"dist_mean_avg": "d_mean"})
    if "dist_std_avg" in df_d.columns:
        df_d = df_d.rename(columns={"dist_std_avg": "d_std"})
    else:
        df_d["d_std"] = np.nan

    # garantir que tenha phi
    if "phi" not in df_d.columns:
        print(f"[x] Coluna 'phi' não encontrada em {d_file}")
        continue

    # ---------- merge usando phi ----------
    df = pd.merge(df_tau, df_d, on="phi", how="inner")
    if df.empty:
        print(f"[ERRO] Merge vazio para {NC_TAG}")
        continue

    df = df.sort_values("phi")
    VMAX = float(df["phi"].max())
    if VMAX < VCENTER:
        VMAX = VCENTER + 1e-9
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)

    # ---------- plot ----------
    plt.figure(figsize=(10,7), dpi=150)
    sc = plt.scatter(
        df["tau"], df["d_mean"],
        c=df["phi"], cmap="coolwarm", norm=norm,
        s=80, edgecolor="k", linewidth=0.4, zorder=3, label=LABEL_TEX
    )

    # marcador em phi ~ 0.60
    target_phi = 0.60
    nearest_idx = (df["phi"] - target_phi).abs().idxmin()
    tau_060 = df.loc[nearest_idx, "tau"]
    d_060   = df.loc[nearest_idx, "d_mean"]

    plt.scatter(
        tau_060, d_060,
        marker="*", s=300, color="gold", edgecolor="k",
        zorder=5, label=r"$\phi \approx 0.60$"
    )

    plt.xlabel(r"$\tau$", fontsize=22)
    plt.ylabel(r"$\langle d \rangle$", fontsize=22)
    plt.title(LABEL_TEX, fontsize=18)

    # colorbar
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label(r"$\phi$")
    cbar.ax.axhline(VCENTER, color="k", lw=1)
    cbar.set_ticks([VMIN, VCENTER, VMAX])
    cbar.set_ticklabels([f"{VMIN:.2f}", f"{VCENTER:.2f}", f"{VMAX:.2f}"])
    # limitar eixo y
    plt.ylim(1.3, 3.1) 
    plt.xlim(1.5, 11.0)  
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # salvar
    out_file = OUT_DIR / f"tau_vs_d_{NC_TAG}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figura salva em {out_file}")
