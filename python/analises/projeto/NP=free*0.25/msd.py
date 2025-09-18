#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TA-MSD (hunters) com TrajPy:
- percorre todos *_hunter_trajectories.csv na pasta base
- unwrapping 2D para CPC (Lx,Ly)
- TA-MSD via TrajPy + correção de viés N/(N-τ)
- ensemble (média, std, n_contrib)
- gráficos com faixa sombreada ± std

Requisitos:
  pip install trajpy pandas numpy matplotlib
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trajpy.trajpy as tj

# -------- Config --------
base = Path("/home/rrossatto/Dados_Doc/Np=free*0.25/Nc=Np*0.5/s_obs_00")
pattern = "*_hunter_trajectories.csv"
Lx = Ly = 128           # tamanho da caixa periódica
min_contrib_plot = 10   # mínimo para exibir no gráfico
out_prefix = "msd_hunters_all_runs"  # prefixo de saída (csv/png)

# -------- Funções --------
def unwrap_1d(x, L):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    dx -= np.round(dx / L) * L
    xu = np.empty_like(x, dtype=float)
    xu[0] = x[0]
    xu[1:] = xu[0] + np.cumsum(dx)
    return xu

def unwrap_xy(x, y, Lx, Ly):
    return unwrap_1d(x, Lx), unwrap_1d(y, Ly)

# -------- Coleta de arquivos --------
arquivos = sorted(base.glob(pattern))
if not arquivos:
    raise FileNotFoundError(f"Nenhum arquivo encontrado com padrão {pattern} em {base}")

# -------- Loop em todos os runs --------
all_msd = []
max_len = 0


# --- CÓDIGO DE VERIFICAÇÃO (coloque antes do loop 'for arq in arquivos:') ---
# Gere uma trajetória de teste simples: 5 passos para a direita
t_test = np.array([0, 1, 2, 3, 4])
x_test = np.array([0, 1, 2, 3, 4])  # Move 1 unidade para a direita a cada passo
y_test = np.array([0, 0, 0, 0, 0])
traj_test = np.column_stack([t_test, x_test, y_test])

print("=== TRAJETÓRIA DE TESTE ===")
print("t, x, y")
print(traj_test)

print("\n=== CÁLCULO MANUAL do MSD(τ) ===")
# Cálculo manual CORRETO: MSD(τ) = média sobre t de [ (r(t+τ) - r(t))² ]
for tau in [1, 2, 3]:
    msd_vals = []
    for t_start in [0, 1, 2]: # Todos os 't' válidos para este tau
        if t_start + tau < len(traj_test):
            dr = traj_test[t_start + tau, 1:3] - traj_test[t_start, 1:3]
            dr_sq = np.dot(dr, dr)
            msd_vals.append(dr_sq)
            print(f"Para τ={tau}, t={t_start}: dr = {dr}, dr² = {dr_sq}")
    if msd_vals:
        print(f"MSD_manual(τ={tau}) = {np.mean(msd_vals):.4f}\n")

print("=== CÁLCULO TRAJPY do MSD(τ) ===")
# O que o trajpy calcula?
try:
    msd_trajpy = np.asarray(tj.Trajectory.msd_time_averaged_(traj_test, [1, 2, 3]), float).ravel()
    print(f"MSD_trajpy(τ=[1,2,3]) = {msd_trajpy}")
except Exception as e:
    print(f"Erro no trajpy: {e}")

# Se os resultados forem diferentes, sabemos a causa.
input("Pressione Enter para continuar com a análise principal...")
# --- FIM DO CÓDIGO DE VERIFICAÇÃO ---


for arq in arquivos:
    df = pd.read_csv(arq, usecols=["timestep", "cell_id", "x", "y"])
    df = df.sort_values(["cell_id", "timestep"])

    for cid, g in df.groupby("cell_id", sort=True):
        t = g["timestep"].to_numpy(float)
        x = g["x"].to_numpy(float)
        y = g["y"].to_numpy(float)

        # unwrapping CPC
        x, y = unwrap_xy(x, y, Lx, Ly)

        traj = np.column_stack([t, x, y])  # TrajPy espera [t,x,y]
        N = len(traj)
        if N < 2:
            continue

        taus = np.arange(1, N, dtype=int)
        # TA-MSD pelo TrajPy
        try:
            msd = np.asarray(tj.Trajectory.msd_time_averaged_(traj, taus), float).ravel()
        except Exception:
            msd = np.array([tj.Trajectory.msd_time_averaged_(traj, int(tau)) for tau in taus], float)

        # correção de viés de tempo finito
        msd *= (N / (N - taus))

        all_msd.append(msd)
        max_len = max(max_len, msd.size)

if not all_msd:
    raise RuntimeError("Nenhuma trajetória válida encontrada nos arquivos selecionados.")

# -------- Ensemble: alinhar por τ e agregar --------
mat = np.full((len(all_msd), max_len), np.nan, dtype=float)
for i, arr in enumerate(all_msd):
    mat[i, :arr.size] = arr

taus = np.arange(1, max_len + 1, dtype=int)
msd_mean = np.nanmean(mat, axis=0)

# n_contrib por τ
n_contrib = np.sum(~np.isnan(mat), axis=0).astype(int)

# std amostral apenas onde há pelo menos 2 amostras
msd_std = np.full_like(msd_mean, np.nan)
valid_cols = np.where(n_contrib > 1)[0]
for j in valid_cols:
    col = mat[:, j]
    col = col[~np.isnan(col)]
    if col.size > 1:
        msd_std[j] = col.std(ddof=1)

out = pd.DataFrame({
    "tau": taus,
    "msd_mean": msd_mean,
    "msd_std": msd_std,
    "n_contrib": n_contrib
})
out.to_csv(f"{out_prefix}.csv", index=False)



# ===== Difusão: estimativa de D a partir do MSD =====
# Parâmetros do ajuste
dt = 1.0                 # seu passo de tempo físico por frame
tau_fit_min = 1          # menor tau no fit (frames)
tau_fit_max_frac = 0.15  # usar até ~15% do tamanho mediano das trajetórias

# Seleção de pontos para plot e ajuste
min_contrib = min_contrib_plot
mask_base = (out["n_contrib"] >= min_contrib) & np.isfinite(out["msd_mean"])

# Limite superior de tau para o ajuste (curto tempo)
N_med_fit = np.nan  # se você já tiver sizes, use a mediana deles
# se não tiver sizes, aproxime pelo ponto onde ainda há muitas contribuições
if np.isnan(N_med_fit):
    # heurística: pegar o maior tau com >= 80% do n_contrib máximo
    nmax = out["n_contrib"].max()
    tau_fit_max = int(np.ceil(out.loc[out["n_contrib"] >= 0.8*nmax, "tau"].max() * tau_fit_max_frac)) or 5
else:
    tau_fit_max = max(3, int(tau_fit_max_frac * N_med_fit))

mask_fit = mask_base & (out["tau"] >= tau_fit_min) & (out["tau"] <= tau_fit_max)

tau_plot = out.loc[mask_base, "tau"].to_numpy()
mean_plot = out.loc[mask_base, "msd_mean"].to_numpy()
std_plot  = out.loc[mask_base, "msd_std"].to_numpy()
std_plot  = np.where(np.isfinite(std_plot), std_plot, 0.0)

# Dados do ajuste (linear: MSD = slope * t, t = tau*dt)
tau_fit = out.loc[mask_fit, "tau"].to_numpy()
t_fit = tau_fit * dt
y_fit = out.loc[mask_fit, "msd_mean"].to_numpy()

# Ajuste com intercepto 0 (OLS): slope = (x·y)/(x·x)
if t_fit.size >= 2:
    xy = np.dot(t_fit, y_fit)
    xx = np.dot(t_fit, t_fit)
    slope = xy / xx
    # erro padrão da inclinação (assumindo homoscedasticidade)
    resid = y_fit - slope * t_fit
    dof = max(1, t_fit.size - 1)
    sigma2 = np.dot(resid, resid) / dof
    slope_se = np.sqrt(sigma2 / xx)
    D = slope / 4.0
    D_se = slope_se / 4.0
else:
    slope = np.nan; slope_se = np.nan; D = np.nan; D_se = np.nan

# Estimativa do expoente alfa (log-log): MSD ~ A * t^alpha
alpha = np.nan; alpha_se = np.nan
if t_fit.size >= 2 and np.all(y_fit > 0):
    X = np.log(t_fit)
    Y = np.log(y_fit)
    # regressão linear Y = a0 + alpha * X
    A = np.vstack([np.ones_like(X), X]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    a0, alpha = coeff
    # erro padrão aproximado de alpha
    Y_hat = A @ coeff
    resid = Y - Y_hat
    dof = max(1, X.size - 2)
    s2 = np.dot(resid, resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    alpha_se = np.sqrt(cov[1,1])

# Salvar resumo
with open(f"{out_prefix}_fit_summary.txt", "w") as f:
    f.write(f"dt = {dt}\n")
    f.write(f"tau_fit_min = {tau_fit_min}, tau_fit_max = {tau_fit_max}\n")
    f.write(f"points_used = {t_fit.size}\n")
    f.write(f"slope (MSD vs t) = {slope:.6g} ± {slope_se:.3g}\n")
    f.write(f"D (2D, MSD≈4Dt)   = {D:.6g} ± {D_se:.3g}\n")
    f.write(f"alpha (MSD~t^alpha) = {alpha:.4g} ± {alpha_se:.2g}\n")

print(f"[OK] D ≈ {D:.6g} ± {D_se:.3g} (unid: lattice^2 / tempo), alpha ≈ {alpha:.3g}")

# ===== Plots com faixa sombreada e reta de ajuste =====
# Linear
plt.figure()
plt.plot(tau_plot, mean_plot, label="MSD médio")
plt.fill_between(tau_plot, mean_plot - std_plot, mean_plot + std_plot, alpha=0.25, label="± std")

# reta de ajuste (em unidades de tau): MSD ≈ 4 D (tau*dt)
if np.isfinite(D):
    tau_line = np.linspace(tau_fit.min(), tau_fit.max(), 100)
    msd_line = 4.0 * D * (tau_line * dt)
    plt.plot(tau_line, msd_line, linestyle="--", label=f"ajuste curto-tempo: D≈{D:.3g}")

plt.xlabel("tau (frames)")
plt.ylabel("MSD")
plt.title("TA-MSD (ensemble) — média ± std (linear)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_prefix}_linear.png", dpi=150)

# Log-log
eps = 1e-12
lower = np.clip(mean_plot - std_plot, eps, None)
upper = np.clip(mean_plot + std_plot, eps, None)

plt.figure()
plt.loglog(tau_plot, mean_plot, label="MSD médio")
plt.fill_between(tau_plot, lower, upper, alpha=0.25, label="± std")

if np.isfinite(D):
    tau_line = np.logspace(np.log10(max(1, tau_fit.min())), np.log10(tau_fit.max()), 100)
    msd_line = 4.0 * D * (tau_line * dt)
    plt.loglog(tau_line, msd_line, linestyle="--", label=f"ajuste: D≈{D:.3g}, α≈{alpha:.2g}")

plt.xlabel("tau (frames)")
plt.ylabel("MSD")
plt.title("TA-MSD (ensemble) — média ± std (log-log)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_prefix}_loglog.png", dpi=150)
