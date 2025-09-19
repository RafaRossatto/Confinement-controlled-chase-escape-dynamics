#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== PARÂMETROS (edite aqui) ==================
CSV_PATH      = Path("msd_ensemble_trajpy_results.csv")  # caminho do CSV
OUT_PREFIX    = f"{CSV_PATH.stem}__fit2windows"
MIN_CONTRIB   = 10       # mínimo de contribuições por lag
USE_WEIGHTS   = True     # True -> pesos = sqrt(n_contrib); False -> sem pesos

# JANELA DO AJUSTE 1
TMIN1         = 1        # tau mínimo do 1º fit (None => usa mínimo válido)
TMAX1         = 27     # tau máximo do 1º fit (None => usa máximo válido)

# JANELA DO AJUSTE 2
TMIN2         = None        # tau mínimo do 2º fit (None => usa mínimo válido)
TMAX2         = None     # tau máximo do 2º fit (None => usa máximo válido)
# =============================================================

# ---------- funções utilitárias ----------
def preparar_dados(csv_path, min_contrib):
    df = pd.read_csv(csv_path)
    required = {"tau", "msd_ensemble", "n_contrib"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa conter as colunas: {required}")
    tau = df["tau"].to_numpy(float)
    msd = df["msd_ensemble"].to_numpy(float)
    nc  = df["n_contrib"].to_numpy(float)

    mask_all = np.isfinite(tau) & np.isfinite(msd) & (tau > 0) & (msd > 0) & (nc >= min_contrib)
    tau_v, msd_v, nc_v = tau[mask_all], msd[mask_all], nc[mask_all]
    if tau_v.size < 3:
        raise RuntimeError("Poucos pontos válidos após filtro básico.")
    return tau_v, msd_v, nc_v

def ajustar_loglog_janela(tau_v, msd_v, nc_v, tmin, tmax, use_weights=True):
    # define limites efetivos
    tmin_eff = tau_v.min() if tmin is None else float(tmin)
    tmax_eff = tau_v.max() if tmax is None else float(tmax)
    if tmax_eff < tmin_eff:
        tmin_eff, tmax_eff = tmax_eff, tmin_eff

    mask_fit = (tau_v >= tmin_eff) & (tau_v <= tmax_eff)
    tau_f, msd_f, nc_f = tau_v[mask_fit], msd_v[mask_fit], nc_v[mask_fit]
    if tau_f.size < 3:
        return {
            "ok": False, "msg": "Janela com menos de 3 pontos.",
            "tmin": tmin_eff, "tmax": tmax_eff, "npts": int(tau_f.size)
        }

    # log-transform
    x = np.log(tau_f)
    y = np.log(msd_f)
    w = np.sqrt(nc_f) if use_weights else None

    # ajuste linear: y = a1*x + a0
    if w is None:
        a1, a0 = np.polyfit(x, y, 1)
        yhat = a1*x + a0
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        # covariância (OLS)
        n = x.size
        X = np.column_stack([x, np.ones_like(x)])
        sigma2 = ss_res / max(n - 2, 1)
        cov = sigma2 * np.linalg.inv(X.T @ X)
    else:
        a1, a0 = np.polyfit(x, y, 1, w=w)
        yhat = a1*x + a0
        # R² ponderado (WLS)
        W = np.diag(w**2)
        res_w = (y - yhat) * w
        ss_res = float(res_w @ res_w)  # soma dos quadrados ponderada
        wy_sum = float((w**2) @ y)
        w_sum  = float(np.sum(w**2))
        ybar_w = wy_sum / w_sum
        ss_tot = float(np.sum((w * (y - ybar_w))**2))
        R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
        # covariância (WLS)
        n = x.size
        X = np.column_stack([x, np.ones_like(x)])
        XtWX = X.T @ W @ X
        sigma2 = ss_res / max(n - 2, 1)
        cov = sigma2 * np.linalg.inv(XtWX)

    alpha = float(a1)
    K     = float(np.exp(a0))
    se_alpha = float(np.sqrt(cov[0, 0]))
    se_lnK   = float(np.sqrt(cov[1, 1]))
    z = 1.96
    alpha_ci = (alpha - z*se_alpha, alpha + z*se_alpha)
    K_ci     = (np.exp(a0 - z*se_lnK), np.exp(a0 + z*se_lnK))

    return {
        "ok": True,
        "tmin": tmin_eff, "tmax": tmax_eff, "npts": int(tau_f.size),
        "alpha": alpha, "K": K, "R2": R2, "SSE": ss_res,
        "se_alpha": se_alpha, "se_lnK": se_lnK,
        "alpha_ci": alpha_ci, "K_ci": K_ci,
        "tau_f": tau_f, "msd_f": msd_f,
    }

# ---------- pipeline ----------
tau_v, msd_v, nc_v = preparar_dados(CSV_PATH, MIN_CONTRIB)

# Ajuste 1
fit1 = ajustar_loglog_janela(tau_v, msd_v, nc_v, TMIN1, TMAX1, USE_WEIGHTS)
# Ajuste 2
fit2 = ajustar_loglog_janela(tau_v, msd_v, nc_v, TMIN2, TMAX2, USE_WEIGHTS)

# ---------- plot ----------
plt.figure(figsize=(8.5, 6))

# todos os pontos válidos (cinza claro)
plt.loglog(tau_v, msd_v, 'o', ms=3, alpha=0.30, label="MSD (válidos)")

# plot do fit 1 (se ok)
if fit1["ok"]:
    plt.loglog(fit1["tau_f"], fit1["msd_f"], 'o-', ms=4, label=f"Janela 1 [{int(fit1['tmin'])},{int(fit1['tmax'])}]")
    tau_fit = np.array([fit1["tau_f"].min(), fit1["tau_f"].max()], float)
    msd_fit = fit1["K"] * tau_fit**fit1["alpha"]
    plt.loglog(tau_fit, msd_fit, '--', label=rf"Fit 1: $\alpha={fit1['alpha']:.3f}\ \pm\ {fit1['se_alpha']:.3f}$, $R^2={fit1['R2']:.3f}$")
    plt.axvline(fit1["tmin"], linestyle=":", alpha=0.5, color="C0")
    plt.axvline(fit1["tmax"], linestyle=":", alpha=0.5, color="C0")
else:
    print(f"[Ajuste 1] não realizado: {fit1['msg']} (janela [{fit1['tmin']},{fit1['tmax']}])")

# plot do fit 2 (se ok)
if fit2["ok"]:
    # usar cor diferente (C1)
    plt.loglog(fit2["tau_f"], fit2["msd_f"], 'o-', ms=4, label=f"Janela 2 [{int(fit2['tmin'])},{int(fit2['tmax'])}]")
    tau_fit2 = np.array([fit2["tau_f"].min(), fit2["tau_f"].max()], float)
    msd_fit2 = fit2["K"] * tau_fit2**fit2["alpha"]
    plt.loglog(tau_fit2, msd_fit2, '--', label=rf"Fit 2: $\alpha={fit2['alpha']:.3f}\ \pm\ {fit2['se_alpha']:.3f}$, $R^2={fit2['R2']:.3f}$")
    plt.axvline(fit2["tmin"], linestyle=":", alpha=0.5, color="C1")
    plt.axvline(fit2["tmax"], linestyle=":", alpha=0.5, color="C1")
else:
    print(f"[Ajuste 2] não realizado: {fit2['msg']} (janela [{fit2['tmin']},{fit2['tmax']}])")

plt.xlabel("Time Lag τ (frames)")
plt.ylabel("MSD(τ)")
plt.title("MSD (Log–Log) • dois ajustes por janelas escolhidas")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

fig_path = f"{OUT_PREFIX}_loglog_fit2.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")

# ---------- resumo ----------
txt_path = f"{OUT_PREFIX}_summary.txt"
with open(txt_path, "w") as f:
    f.write("=== LOG-LOG MSD: DOIS AJUSTES POR JANELAS ===\n\n")
    f.write(f"Arquivo CSV: {CSV_PATH}\n")
    f.write(f"Pontos válidos (após filtro): {tau_v.size}  | MIN_CONTRIB={MIN_CONTRIB}\n")
    f.write(f"Pesos: {'sqrt(n_contrib)' if USE_WEIGHTS else 'nenhum'}\n\n")

    f.write("Ajuste 1:\n")
    f.write(f"  Janela: [{fit1['tmin']}, {fit1['tmax']}]  | n={fit1['npts']}\n")
    if fit1["ok"]:
        f.write(f"  alpha = {fit1['alpha']:.6f} ± {fit1['se_alpha']:.6f}  (95% CI: {fit1['alpha_ci'][0]:.6f} .. {fit1['alpha_ci'][1]:.6f})\n")
        f.write(f"  K     = {fit1['K']:.6g}      (95% CI: {fit1['K_ci'][0]:.6g} .. {fit1['K_ci'][1]:.6g})\n")
        f.write(f"  R^2   = {fit1['R2']:.6f}     | SSE={fit1['SSE']:.6g}\n\n")
    else:
        f.write(f"  [não realizado] {fit1['msg']}\n\n")

    f.write("Ajuste 2:\n")
    f.write(f"  Janela: [{fit2['tmin']}, {fit2['tmax']}]  | n={fit2['npts']}\n")
    if fit2["ok"]:
        f.write(f"  alpha = {fit2['alpha']:.6f} ± {fit2['se_alpha']:.6f}  (95% CI: {fit2['alpha_ci'][0]:.6f} .. {fit2['alpha_ci'][1]:.6f})\n")
        f.write(f"  K     = {fit2['K']:.6g}      (95% CI: {fit2['K_ci'][0]:.6g} .. {fit2['K_ci'][1]:.6g})\n")
        f.write(f"  R^2   = {fit2['R2']:.6f}     | SSE={fit2['SSE']:.6g}\n\n")
    else:
        f.write(f"  [não realizado] {fit2['msg']}\n\n")

    f.write(f"Figura: {fig_path}\n")

print(f"[OK] Figura:  {fig_path}")
print(f"[OK] Resumo:  {txt_path}")
if fit1["ok"]:
    print(f"[OK] Fit 1: alpha={fit1['alpha']:.6f} ± {fit1['se_alpha']:.6f} | R^2={fit1['R2']:.3f}  (janela [{fit1['tmin']},{fit1['tmax']}])")
if fit2["ok"]:
    print(f"[OK] Fit 2: alpha={fit2['alpha']:.6f} ± {fit2['se_alpha']:.6f} | R^2={fit2['R2']:.3f}  (janela [{fit2['tmin']},{fit2['tmax']}])")
