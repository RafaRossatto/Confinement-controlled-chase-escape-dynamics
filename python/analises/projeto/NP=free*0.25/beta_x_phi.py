from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- parâmetros ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
            "obs_8192", "obs_9830", "obs_11468", "obs_13107", "obs_14745"]

bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}_{0}=N^{E}_{0}$",      "Nc=Np"),
]

L = 128
area = L**2
BASE_RES = Path.home() / "Dados_Doc" / "Np=free*0.25" / "resultados_modelos"
MODEL_TAG = "EXPbeta"

# Paleta
from matplotlib import colormaps
cmap_flag = colormaps.get_cmap("flag")

# ---------------------- funções auxiliares ----------------------
def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / area

def symmetric_offsets(n_series: int, delta: float = 0.25):
    """Offsets simétricos: n=3 -> [-d, 0, +d]; n=2 -> [-d/2, +d/2]; n=4 -> [-1.5d,-0.5d,+0.5d,+1.5d]."""
    idx = np.arange(n_series)
    if n_series % 2 == 1:
        center = n_series // 2
        return (idx - center) * delta
    else:
        return (idx - (n_series - 1)/2) * delta

def add_phi_separators_and_phi60(ax, phi_sorted, tick_positions):
    # separadores em 0.05, 0.15, ..., até <0.90
    for phi_sep in np.arange(0.05, min(0.90, max(phi_sorted)) + 1e-12, 0.10):
        if np.isclose(phi_sep, 0.90):
            continue
        x_sep = np.interp(phi_sep, phi_sorted, tick_positions)
        ax.axvline(x=x_sep, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    # linha especial em 0.60
    x_phi60 = np.interp(0.60, phi_sorted, tick_positions)
    ax.axvline(x=x_phi60, color="black", linestyle="--", linewidth=1.5, label=r"$\phi = 0.60$")

def put_phi60_first_in_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    linha60, outros = [], []
    for h, l in zip(handles, labels):
        if r"$\phi = 0.60$" in l:
            linha60.append((h, l))
        else:
            outros.append((h, l))
    if linha60:
        new_handles, new_labels = zip(*(linha60 + outros))
        ax.legend(new_handles, new_labels, frameon=False)
    else:
        ax.legend(frameon=False)

# ---------------------- MAIN ----------------------
plt.figure(figsize=(12, 6), dpi=150)

rows_all = []          # CSV combinado (todas as bases, todos os runs)
phi_labels = []        # valores únicos de phi
phi_to_index = {}      # mapeia phi -> índice categórico

OFFSETS = symmetric_offsets(len(bases), delta=0.25)

for base_idx, (label_tex, nc_tag) in enumerate(bases):
    phi_vals, beta_groups = [], []

    for obs_folder in obs_list:
        n_obs = extrai_num_obs(obs_folder)
        phi = densidade_obs(n_obs)

        # arquivo esperado
        fname = f"params_por_run_{MODEL_TAG}_{nc_tag}_s_obs_{n_obs:02d}.csv"
        fpath = BASE_RES / fname
        if not fpath.exists():
            print(f"[WARN] não encontrei: {fpath.name}")
            continue

        df = pd.read_csv(fpath)
        if "beta" not in df.columns:
            print(f"[WARN] sem coluna 'beta' em: {fpath.name}")
            continue

        beta_values = df["beta"].dropna().values
        if len(beta_values) == 0:
            continue

        if phi not in phi_to_index:
            phi_to_index[phi] = len(phi_labels)
            phi_labels.append(phi)

        phi_vals.append(phi_to_index[phi])
        beta_groups.append(beta_values)

        # CSV combinado com valores individuais
        for v in beta_values:
            rows_all.append({
                "scenario": nc_tag,
                "phi": phi,
                "num_obs": n_obs,
                "beta_run": float(v)
            })

    if beta_groups:
        # ordenar por índice de phi e posicionar com offsets simétricos
        phi_vals, beta_groups = zip(*sorted(zip(phi_vals, beta_groups)))
        positions = np.array(phi_vals, dtype=float) + OFFSETS[base_idx]

        plt.boxplot(
            beta_groups,
            positions=positions,
            widths=0.2,
            patch_artist=True,
            boxprops=dict(facecolor=cmap_flag(base_idx), alpha=0.5),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=cmap_flag(base_idx)),
            capprops=dict(color=cmap_flag(base_idx)),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor=cmap_flag(base_idx), markeredgecolor="none")
        )
        # “handle” para legenda
        plt.plot([], [], color=cmap_flag(base_idx), label=label_tex)

# ---------------------- salvar CSV combinado ----------------------
if rows_all:
    df_all = pd.DataFrame(rows_all).sort_values(["scenario", "phi"])
    df_all.to_csv("beta_vs_phi__todas_bases__runs.csv", index=False)
    print("[OK] CSV salvo: beta_vs_phi__todas_bases__runs.csv")

# ---------------------- decoração ----------------------
phi_sorted = sorted(phi_labels)
tick_positions = np.arange(len(phi_sorted))
tick_labels = [f"{phi:.3f}" for phi in phi_sorted]
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\beta$")
plt.grid(axis="y", linestyle="--", alpha=0.35)

# separadores e linha 0.60
ax = plt.gca()
add_phi_separators_and_phi60(ax, phi_sorted, tick_positions)
put_phi60_first_in_legend(ax)

plt.tight_layout()
plt.savefig("beta_vs_phi_boxplot_sep.pdf", dpi=200)
plt.show()
