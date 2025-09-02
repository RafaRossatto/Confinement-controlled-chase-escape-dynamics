from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

# ---------------------- parâmetros ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
            "obs_8192", "obs_9830", "obs_11468", "obs_13107"]

bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}_{0}=N^{E}_{0}$",      "Nc=Np"),
]

L = 128
area = L**2
BASE_RES = Path.home() / "Dados_Doc" / "Np=free*0.25" / "resultados_modelos"
MODEL_TAG = "EXPbeta"
PHI_CRIT_OBST = 0.60

cmap = plt.get_cmap("flag")

plt.figure(figsize=(12, 6), dpi=150)
rows_all = []

phi_labels = []
phi_to_index = {}
phi_to_nobs = {}

# --------- offsets simétricos ---------
def symmetric_offsets(n_series: int, delta: float = 0.25):
    idx = np.arange(n_series)
    if n_series % 2 == 1:
        center = n_series // 2
        return (idx - center) * delta
    else:
        return (idx - (n_series - 1)/2) * delta

OFFSETS = symmetric_offsets(len(bases), delta=0.25)

# ---------------------- loop principal ----------------------
for base_idx, (label_tex, nc_tag) in enumerate(bases):
    phi_vals, tau_groups = [], []

    for obs_folder in obs_list:
        n_obs = int(obs_folder.split("_")[1])
        phi = n_obs / area

        fname = f"params_por_run_{MODEL_TAG}_{nc_tag}_s_obs_{n_obs:02d}.csv"
        fpath = BASE_RES / fname
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)
        if "tau" not in df.columns:
            continue

        tau_values = df["tau"].dropna().values
        if len(tau_values) == 0:
            continue

        if phi not in phi_to_index:
            phi_to_index[phi] = len(phi_labels)
            phi_labels.append(phi)
            phi_to_nobs[phi] = n_obs

        phi_vals.append(phi_to_index[phi])
        tau_groups.append(tau_values)

        for v in tau_values:
            rows_all.append({
                "scenario": nc_tag,
                "phi": phi,
                "num_obs": n_obs,
                "tau_run": float(v)
            })

    if tau_groups:
        phi_vals, tau_groups = zip(*sorted(zip(phi_vals, tau_groups)))
        positions = np.array(phi_vals, dtype=float) + OFFSETS[base_idx]

        plt.boxplot(
            tau_groups,
            positions=positions,
            widths=0.2,
            patch_artist=True,
            boxprops=dict(facecolor=cmap(base_idx), alpha=0.5),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=cmap(base_idx)),
            capprops=dict(color=cmap(base_idx)),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor=cmap(base_idx), markeredgecolor="none")
        )
        plt.plot([], [], color=cmap(base_idx), label=label_tex)

# ---------------------- CSV combinado ----------------------
if rows_all:
    df_all = pd.DataFrame(rows_all).sort_values(["scenario", "phi"])
    df_all.to_csv("tau_vs_phi__todas_bases__runs.csv", index=False)
    print("[OK] CSV salvo: tau_vs_phi__todas_bases__runs.csv")

# ---------------------- decoração ----------------------
phi_sorted = sorted(phi_labels)
tick_positions = np.arange(len(phi_sorted))
tick_labels = [f"{phi:.1f}" for phi in phi_sorted]
plt.xticks(tick_positions, tick_labels, rotation=0)

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\tau$")
plt.grid(axis="y", linestyle="--", alpha=0.25)

# --- linhas verticais em 0.05, 0.15, 0.25, ... ---
for phi_sep in np.arange(0.05, max(phi_sorted), 0.10):
    # converter o valor real de phi para posição no eixo categórico
    x_sep = np.interp(phi_sep, phi_sorted, tick_positions)
    plt.axvline(x=x_sep, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    # --- linha especial em phi = 0.60 ---
x_phi60 = np.interp(0.60, phi_sorted, tick_positions)
plt.axvline(x=x_phi60, color="black", linestyle="--", linewidth=1.5,
            label=r"$\phi = 0.60$")

# pegar handles e labels
handles, labels = plt.gca().get_legend_handles_labels()

# separar a linha phi=0.60
linha60 = []
outros = []
for h, l in zip(handles, labels):
    if r"$\phi = 0.60$" in l:   # detecta o label da linha
        linha60.append((h, l))
    else:
        outros.append((h, l))

# reordenar: linha 0.60 primeiro
new_handles, new_labels = zip(*(linha60 + outros))
plt.legend(new_handles, new_labels)
plt.tight_layout()
plt.savefig("tau_vs_phi.pdf", dpi=200)
plt.show()
