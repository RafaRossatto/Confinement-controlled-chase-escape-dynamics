from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import csv

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
MODEL_TAG = "EXPbeta"  # mantendo o mesmo modelo para consistência

# Se o eixo é fração de OBSTÁCULOS, usar ~0.593; se for fração LIVRE, usar ~0.407
PHI_CRIT = 0.592746

# Paleta discreta (cores bem distintas) a partir do tab20c
cmap = plt.get_cmap("flag")
markers = ["o", "o", "o"]

# --------------------------------------------------------

def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / area

plt.figure(figsize=(10, 6), dpi=150)

# também vamos montar um CSV combinado (todas as bases, todos os runs)
rows_all = []
agg_rows = []

for base_idx, (label_tex, nc_tag) in enumerate(bases):
    xs_phi, beta_mean, beta_std = [], [], []
    rows_csv = []

    for obs_folder in obs_list:
        n_obs = extrai_num_obs(obs_folder)
        phi = densidade_obs(n_obs)

        # arquivo esperado: params_por_run_EXPbeta_{NcTag}_s_obs_{N}.csv
        fname = f"params_por_run_{MODEL_TAG}_{nc_tag}_s_obs_{n_obs:02d}.csv"
        fpath = BASE_RES / fname
        if not fpath.exists():
            print(f"[WARN] não encontrei: {fpath.name}")
            continue

        df = pd.read_csv(fpath)
        if "beta" not in df.columns:
            print(f"[WARN] sem coluna 'beta' em: {fpath.name}")
            continue

        m = float(df["beta"].mean())
        s = float(df["beta"].std(ddof=1)) if len(df) > 1 else 0.0
        n_runs = int(len(df))
        sem = s / np.sqrt(n_runs) if n_runs > 0 else 0.0

        agg_rows.append({
            "cenario_label": label_tex,
            "cenario_tag": nc_tag,
            "phi": phi,
            "num_obs": n_obs,
            "n_runs": n_runs,
            "beta_mean": m,
            "beta_std": s,
            "beta_sem": sem,
        })

        xs_phi.append(phi)
        beta_mean.append(m)
        beta_std.append(s)

        rows_csv.append({
            "phi": phi,
            "num_obs": n_obs,
            "n_runs": int(len(df)),
            "beta_mean": m,
            "beta_std": s,
        })

        # também preencher CSV combinado com valores individuais (opcional para análises futuras)
        for v in df["beta"].values:
            rows_all.append({
                "scenario": nc_tag,
                "phi": phi,
                "num_obs": n_obs,
                "beta_run": float(v)
            })

    if xs_phi:
        xs_phi, beta_mean, beta_std = zip(*sorted(zip(xs_phi, beta_mean, beta_std)))
        plt.errorbar(
            xs_phi, beta_mean, yerr=beta_std,
            fmt=markers[base_idx % len(markers)]+"-",
            capsize=5, markersize=5,
            color=cmap(base_idx),
            label=label_tex
        )

        # salvar CSV por cenário (ordenado por phi)
        if rows_csv:
            out_csv = f"beta_vs_phi_{nc_tag}.csv"
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
                w.writeheader()
                for r in sorted(rows_csv, key=lambda r: r["phi"]):
                    w.writerow(r)
            print(f"[OK] CSV salvo: {out_csv}")

# salva CSV combinado (todas as bases, todos os runs)
if rows_all:
    df_all = pd.DataFrame(rows_all).sort_values(["scenario", "phi"])
    df_all.to_csv("beta_vs_phi__todas_bases__runs.csv", index=False)
    print("[OK] CSV salvo: beta_vs_phi__todas_bases__runs.csv")

# ====================== PLOTS: 3 figuras (mean±std, mean, dispersão) ======================
from matplotlib import colormaps
cmap_flag = colormaps.get_cmap("flag")

if not agg_rows:
    raise SystemExit("[!] Sem dados agregados para plotar.")

df_agg = pd.DataFrame(agg_rows).sort_values(["cenario_label", "phi"]).reset_index(drop=True)

# CSV com estatísticas agregadas (útil p/ checar depois)
df_agg.to_csv("beta_vs_phi__agg_stats.csv", index=False)
print("[OK] CSV salvo: beta_vs_phi__agg_stats.csv")

# Opções
DISP_METRIC = "std"   # "std" (desvio-padrão) ou "sem" (erro-padrão da média)
CLIP_NONNEG_ERR = False  # β costuma ser ≥0 pelo fit; ative True se quiser cortar barra inferior em 0

# --- FIGURA 1: média ± desvio-padrão ---
plt.figure(figsize=(10, 6), dpi=150)
ax = plt.gca()

for base_idx, (label_tex, _nc_tag) in enumerate(bases):
    sub = df_agg[df_agg["cenario_label"] == label_tex].sort_values("phi")
    if sub.empty: 
        continue
    x = sub["phi"].to_numpy()
    y = sub["beta_mean"].to_numpy()
    base = sub["beta_std"].to_numpy()

    if CLIP_NONNEG_ERR:
        yerr_lower = np.minimum(y, base)  # garante y - err >= 0
        yerr_upper = base
        yerr = np.vstack([yerr_lower, yerr_upper])
    else:
        yerr = base  # barras simétricas

    ax.errorbar(
        x, y, yerr=yerr, fmt="o-", capsize=5, linewidth=1.6, markersize=5,
        label=label_tex, color=cmap_flag(base_idx)
    )

ax.axvline(PHI_CRIT, color="black", linestyle="--", linewidth=1.5, label=rf"$\phi_c \approx {PHI_CRIT:.2f}$")
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"Média de $\beta$ (barras $=\ \sigma$)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=False)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("beta_mean_std_vs_phi.pdf", dpi=200, bbox_inches="tight")
print("[OK] Figura: beta_mean_std_vs_phi.pdf")
plt.show()

# --- FIGURA 2: somente a média ---
plt.figure(figsize=(10, 6), dpi=150)
ax = plt.gca()
ax.axvline(PHI_CRIT, color="black", linestyle="--", linewidth=1.5, label=rf"$\phi_c \approx {PHI_CRIT:.2f}$")
for base_idx, (label_tex, _nc_tag) in enumerate(bases):
    sub = df_agg[df_agg["cenario_label"] == label_tex].sort_values("phi")
    if sub.empty: 
        continue
    ax.plot(
        sub["phi"], sub["beta_mean"],
        "o-", linewidth=1.6, markersize=5,
        label=label_tex, color=cmap_flag(base_idx)
    )

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$ \langle \beta \rangle $")
ax.set_ylim(0.59,1.0)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=False)
#ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("beta_mean_vs_phi.pdf", dpi=200, bbox_inches="tight")
print("[OK] Figura: beta_mean_vs_phi.pdf")
plt.show()

# --- FIGURA 3: dispersão (std/sem) vs phi ---
plt.figure(figsize=(10, 6), dpi=150)
ax = plt.gca()

for base_idx, (label_tex, _nc_tag) in enumerate(bases):
    sub = df_agg[df_agg["cenario_label"] == label_tex].sort_values("phi")
    if sub.empty: 
        continue
    if DISP_METRIC == "sem":
        y = sub["beta_sem"].to_numpy()
        ylabel = r"Dispersão de $\beta$ (SEM)"
    else:
        y = sub["beta_std"].to_numpy()
        ylabel = r"Dispersão de $\beta$ ($\sigma$)"

    ax.plot(
        sub["phi"], y,
        "o-", linewidth=1.6, markersize=5,
        label=label_tex, color=cmap_flag(base_idx)
    )

ax.axvline(PHI_CRIT, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\sigma$")
ax.grid(True, linestyle="--", alpha=0.6)
#ax.legend(frameon=False)
ax.set_ylim(bottom=0)
plt.tight_layout()
suffix = DISP_METRIC
plt.savefig(f"beta_disp_{suffix}_vs_phi.pdf", dpi=200, bbox_inches="tight")
print(f"[OK] Figura: beta_disp_{suffix}_vs_phi.pdf")
plt.show()

