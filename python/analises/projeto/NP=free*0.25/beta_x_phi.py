from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

# ---------------------- parâmetros ----------------------
L = 128
AREA = L**2
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
HAZARD_DIR = BASE_ROOT / "resultados_modelos" / "hazard"
OUT_DIR = BASE_ROOT / "resultados_modelos" / "beta_x_phi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}_{0}=N^{E}_{0}$",      "Nc=Np"),
]

# ---------------------- plot ----------------------
plt.figure(figsize=(10,6), dpi=150)
cmap = colormaps.get_cmap("flag")

for idx, (label_tex, nc_tag) in enumerate(bases):
    fpath = HAZARD_DIR / f"{nc_tag}_fit_results.csv"
    if not fpath.exists():
        print(f"[x] Arquivo não encontrado: {fpath}")
        continue

    df = pd.read_csv(fpath).copy()
    if not {"obs","beta","beta_err"}.issubset(df.columns):
        print(f"[x] Colunas esperadas não encontradas em {fpath}")
        continue

    # extrair número de obstáculos do campo "obs" -> fração φ
    df["n_obs"] = df["obs"].str.replace("s_obs_", "", regex=False).astype(int)
    df["phi"] = df["n_obs"] / AREA

    df_sorted = df.sort_values("phi")

    # plota linha com pontos + barras de erro
    plt.errorbar(
        df_sorted["phi"], df_sorted["beta"],
        yerr=df_sorted["beta_err"],
        fmt="o-", capsize=4,
        color=cmap(idx),
        label=label_tex
    )

# ---------------------- decoração ----------------------
plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5,
            label=r"$\phi = 0.60$")

plt.xlabel(r"$\phi$", fontsize=22)
plt.ylabel(r"$\beta$", fontsize=22)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

# salvar
out_file = OUT_DIR / "beta_vs_phi.pdf"
plt.savefig(out_file, bbox_inches="tight")
plt.show()

print(f"[OK] Gráfico salvo em {out_file}")
