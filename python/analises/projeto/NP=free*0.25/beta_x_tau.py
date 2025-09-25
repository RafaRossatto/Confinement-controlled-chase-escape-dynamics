import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
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
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
HAZARD_DIR = BASE_ROOT / "resultados_modelos" / "hazard"
OUT_DIR = BASE_ROOT / "resultados_modelos" / "beta_x_tau"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for NC_TAG, LABEL_TEX in SCENARIOS:
    fpath = HAZARD_DIR / f"{NC_TAG}_fit_results.csv"
    if not fpath.exists():
        print(f"[x] Arquivo não encontrado: {fpath}")
        continue

    df = pd.read_csv(fpath).copy()

    # checar colunas necessárias
    expected_cols = {"obs", "tau", "tau_err", "beta", "beta_err"}
    if not expected_cols.issubset(df.columns):
        print(f"[x] Colunas {expected_cols} não encontradas em {fpath}")
        continue

    # extrair φ a partir do campo "obs"
    df["n_obs"] = df["obs"].str.replace("s_obs_", "", regex=False).astype(int)
    df["phi"] = df["n_obs"] / AREA
    df_sorted = df.sort_values("phi")

    # ======================== Plot ========================
    plt.figure(figsize=(10,7), dpi=150)

    phi = df_sorted["phi"].to_numpy()
    vmin = float(np.nanmin(phi))
    vmax = float(np.nanmax(phi))
    VCENTER = 0.60
    if not (vmin <= VCENTER <= vmax):
        eps = 1e-9
        vmin = min(vmin, VCENTER - eps)
        vmax = max(vmax, VCENTER + eps)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=VCENTER, vmax=vmax)

    # scatter τ × β (cor = φ)
    sc = plt.scatter(
        df_sorted["tau"], df_sorted["beta"],
        c=df_sorted["phi"], cmap="coolwarm", norm=norm,
        s=80, edgecolor="k", linewidth=0.4
    )

    plt.xlabel(r"$\tau$", fontsize=22)
    plt.ylabel(r"$\beta$", fontsize=22)
    plt.title(rf"{LABEL_TEX}", fontsize=18)

    # colorbar para φ
    cbar = plt.colorbar(sc, pad=0.02)
    cbar.set_label(r"$\phi$")
    cbar.ax.axhline(VCENTER, color="k", lw=1)
    cbar.set_ticks([vmin, VCENTER, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{VCENTER:.2f}", f"{vmax:.2f}"])

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # salvar
    out_file = OUT_DIR / f"beta_vs_tau_{NC_TAG}.pdf"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[OK] Figura salva em {out_file}")
