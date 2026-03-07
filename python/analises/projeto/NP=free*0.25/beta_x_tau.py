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
BASE_ROOT  = Path.home() / "Dados_Doc" / "Np=free*0.25"
HAZARD_DIR = BASE_ROOT / "resultados_modelos" / "hazard"
OUT_DIR    = BASE_ROOT / "resultados_modelos" / "beta_x_tau"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- colormap divergente ----------
VCENTER = 0.60
VMIN = 0.0

for NC_TAG, LABEL_TEX in SCENARIOS:
    print(f"[INFO] Processando {NC_TAG}...")

    # ---------- carregar τ e β ----------
    fit_file = HAZARD_DIR / f"{NC_TAG}_fit_results.csv"
    if not fit_file.exists():
        print(f"[x] Arquivo não encontrado: {fit_file}")
        continue

    df = pd.read_csv(fit_file)
    if not {"obs", "tau", "beta", "tau_err", "beta_err"}.issubset(df.columns):
        print(f"[x] Colunas necessárias não encontradas em {fit_file}")
        continue

    df["num_obs"] = df["obs"].str.replace("s_obs_", "", regex=False).astype(int)
    df["phi"]     = df["num_obs"] / AREA
    df = df[["phi", "tau", "tau_err", "beta", "beta_err"]].sort_values("phi")

    VMAX = float(df["phi"].max())
    if VMAX < VCENTER:
        VMAX = VCENTER + 1e-9
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)

    # ---------- plot ----------
    plt.figure(figsize=(10,7), dpi=150)
    sc = plt.scatter(
        df["tau"], df["beta"],
        c=df["phi"], cmap="coolwarm", norm=norm,
        s=80, edgecolor="k", linewidth=0.4, zorder=3, label=LABEL_TEX
    )

    # marcador em phi ~ 0.60
    target_phi = 0.60
    nearest_idx = (df["phi"] - target_phi).abs().idxmin()
    tau_060  = df.loc[nearest_idx, "tau"]
    beta_060 = df.loc[nearest_idx, "beta"]

    plt.scatter(
        tau_060, beta_060,
        marker="*", s=300, color="gold", edgecolor="k",
        zorder=4, label=r"$\phi \approx 0.60$"
    )

    plt.xlabel(r"$\tau$", fontsize=22)
    plt.ylabel(r"$\beta$", fontsize=22)
    plt.title(LABEL_TEX, fontsize=18)

    # colorbar
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label(r"$\phi$")
    cbar.ax.axhline(VCENTER, color="k", lw=1)
    cbar.set_ticks([VMIN, VCENTER, VMAX])
    cbar.set_ticklabels([f"{VMIN:.2f}", f"{VCENTER:.2f}", f"{VMAX:.2f}"])
    plt.xlim(1.5, 11.0)  
    plt.ylim(0.6, 1.20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # salvar
    out_file = OUT_DIR / f"beta_vs_tau_{NC_TAG}.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figura salva em {out_file}")
