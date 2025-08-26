#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit

# ============================
# Modelos
# ============================

EPS = 1e-12  # evita z=0 em potências

def exp_offset(z: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """y = A * exp(-z / tau) + C"""
    return A * np.exp(-z / tau) + C

def exp_power(z: np.ndarray, A: float, tau: float, beta: float, C: float) -> np.ndarray:
    """y = A * exp(-(z / tau)**beta) + C  (KWW / stretched exponential)"""
    zc = np.maximum(z, 0.0)  # z >= 0 por construção (clamp)
    return A * np.exp(- (zc / tau)**beta ) + C

def power_law(z: np.ndarray, k: float, beta: float, C: float) -> np.ndarray:
    """y = k * z**beta + C"""
    zc = np.maximum(z, EPS)  # evita 0**beta (beta<0) -> inf
    return k * (zc**beta) + C

def power_shift(z: np.ndarray, k: float, alpha: float, z0: float, C: float) -> np.ndarray:
    """y = k * (z + z0)^(-alpha) + C  (z0>0 evita singularidade)"""
    zc = np.maximum(z + z0, EPS)
    return k * (zc**(-alpha)) + C

# ============================
# Utilidades de ajuste e métricas
# ============================

def fit_model(path: Path,
              model: str,
              C: Optional[float] = None,
              tau_factor: float = 20.0,
              maxfev: int = 20000,
              fix_A: bool = False):
    """
    Ajusta um dos modelos aos pontos com 'presas_vivas' > 0:
      - "exp"        : A * exp(-(t-t0)/tau) + C
      - "exp_power"  : A * exp(-((t-t0)/tau)**beta) + C
      - "power_shift": k * (t-t0 + z0)^(-alpha) + C

    C é fixo (último valor do CSV se não fornecido).
    Se fix_A=True (apenas exp e exp_power): A = N_inicial - C (clamp >= 1e-9).
    Retorna (params_tuple, t0, C).
    """
    df_all = pd.read_csv(path, usecols=["passo", "presas_vivas"]).sort_values("passo")
    if C is None:
        C = float(df_all["presas_vivas"].iloc[-1])

    # N0: número inicial de presas (no menor "passo" do CSV, tipicamente t=0)
    N0 = float(df_all["presas_vivas"].iloc[0])
    A_fix = max(N0 - C, 1e-9) if fix_A and model in ("exp", "exp_power") else None

    # usa só >0 para ajuste
    df_pos = df_all[df_all["presas_vivas"] > 0]
    x = df_pos["passo"].to_numpy(float)
    y = df_pos["presas_vivas"].to_numpy(float)
    if x.size < 3:
        t0 = float(df_all["passo"].min())
        L = 1.0
        if model == "exp":
            if A_fix is not None:
                tau0 = 1.0
                return (A_fix, tau0, C), t0, C
            A = max(float(df_pos["presas_vivas"].mean() - C), 0.0) if x.size else max(N0 - C, 0.0)
            tau = 1.0
            return (A, tau, C), t0, C

        elif model == "exp_power":
            if A_fix is not None:
                return (A_fix, 1.0, 1.0, C), t0, C
            return (max(float(df_pos["presas_vivas"].mean() - C), 0.0) if x.size else max(N0 - C, 0.0),
                    1.0, 1.0, C), t0, C

        elif model == "power_shift":
            return (max(float(df_pos["presas_vivas"].mean() - C), 0.0) if x.size else max(N0 - C, 0.0),
                    1.0, 0.1, C), t0, C
        else:
            raise ValueError(f"Modelo desconhecido: {model}")

    # referência temporal
    t0 = float(x.min())
    z = x - t0
    L = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0
    tau_max = max(1e-6, tau_factor * L)

    # --- checagens comuns (degradação quando há pouca informação) ---
    min_pts = {"exp": 3, "exp_power": 6, "power_shift": 6}
    dr = float(y.max() - y.min())  # amplitude útil
    if x.size < min_pts.get(model, 3) or dr <= 1e-9:
        if model == "exp":
            if A_fix is not None:
                tau0 = max(L/2, 1e-3)
                try:
                    popt, _ = curve_fit(lambda zz, tau: exp_offset(zz, A_fix, tau, C),
                                        z, y, p0=(tau0,),
                                        bounds=([1e-6], [tau_max]),
                                        maxfev=maxfev)
                    tau = popt[0]
                except Exception:
                    tau = tau0
                return (A_fix, tau, C), t0, C
            else:
                A = max(float(y.mean() - C), 0.0)
                tau = max(L/2, 1e-3)
                return (A, tau, C), t0, C

        elif model == "exp_power":
            # degrada para exponencial (β=1)
            tau0 = max(L/2, 1e-3)
            if A_fix is not None:
                try:
                    popt, _ = curve_fit(lambda zz, tau: exp_offset(zz, A_fix, tau, C),
                                        z, y, p0=(tau0,),
                                        bounds=([1e-6], [tau_max]),
                                        maxfev=maxfev)
                    tau = popt[0]
                except Exception:
                    tau = tau0
                return (A_fix, tau, 1.0, C), t0, C
            else:
                A0 = max(y[0] - C, 1e-9)
                try:
                    popt, _ = curve_fit(lambda zz, A, tau: exp_offset(zz, A, tau, C),
                                        z, y, p0=(A0, tau0),
                                        bounds=([0, 1e-6], [10*y.max(), tau_max]),
                                        maxfev=maxfev)
                    A, tau = popt
                except Exception:
                    A, tau = A0, tau0
                return (A, tau, 1.0, C), t0, C

        elif model == "power_shift":
            z0 = max(0.1 * L, 1e-3)
            alpha0 = 1.0
            k0 = max((y[0] - C) * (z0**alpha0), 1e-9)
            try:
                popt, _ = curve_fit(lambda zz, k, alpha: power_shift(zz, k, alpha, z0, C),
                                    z, y, p0=(k0, alpha0),
                                    bounds=([0.0, 0.05], [np.inf, 5.0]),
                                    maxfev=maxfev)
                k, alpha = popt
            except Exception:
                k, alpha = k0, alpha0
            return (k, alpha, z0, C), t0, C

    # =========================
    # ramos NORMAIS (com retorno!)
    # =========================
    if model == "exp":
        tau0 = L/2
        if A_fix is not None:
            try:
                popt, _ = curve_fit(lambda zz, tau: exp_offset(zz, A_fix, tau, C),
                                    z, y, p0=(tau0,),
                                    bounds=([1e-6], [tau_max]),
                                    maxfev=maxfev)
                tau = popt[0]
            except Exception:
                tau = max(tau0, 1e-3)
            return (A_fix, tau, C), t0, C
        else:
            A0 = max(y[0] - C, 1e-9)
            try:
                popt, _ = curve_fit(lambda zz, A, tau: exp_offset(zz, A, tau, C),
                                    z, y, p0=(A0, tau0),
                                    bounds=([0, 1e-6], [10*y[0], tau_max]),
                                    maxfev=maxfev)
                A, tau = popt
            except Exception:
                A, tau = A0, max(tau0, 1e-3)
            return (A, tau, C), t0, C

    elif model == "exp_power":
        # reescala z para [0,1] para melhorar condicionamento
        z_s = np.maximum(z / L, 0.0)
        tau0_s = 0.5
        beta0  = 1.0
        if A_fix is not None:
            lb = (1e-3, 0.2)     # tau_s, beta
            ub = (5.0, 4.0)
            def f_scaled_fixA(zz_s, tau_s, beta):
                return exp_power(zz_s * L, A_fix, tau_s * L, beta, C)
            try:
                popt, _ = curve_fit(f_scaled_fixA, z_s, y, p0=(tau0_s, beta0),
                                    bounds=(lb, ub), maxfev=maxfev)
                tau_s, beta = popt
            except Exception:
                tau_s, beta = tau0_s, beta0
            tau = tau_s * L
            return (A_fix, tau, beta, C), t0, C
        else:
            A0 = max(y[0] - C, 1e-9)
            lb = (0.0, 1e-3, 0.2)   # A, tau_s, beta
            ub = (10*y.max(), 5.0, 4.0)
            def f_scaled(zz_s, A, tau_s, beta):
                return exp_power(zz_s * L, A, tau_s * L, beta, C)
            try:
                popt, _ = curve_fit(f_scaled, z_s, y, p0=(A0, tau0_s, beta0),
                                    bounds=(lb, ub), maxfev=maxfev)
                A, tau_s, beta = popt
            except Exception:
                A, tau_s, beta = A0, tau0_s, beta0
            tau = tau_s * L
            return (A, tau, beta, C), t0, C

    elif model == "power_shift":
        z_s = np.maximum(z / L, 0.0)
        alpha0 = 1.0
        z0_s0  = 0.1
        k0     = max((y[0] - C) * (z0_s0 * L)**alpha0, 1e-9)
        lb = (0.0, 0.05, 1e-3)   # k, alpha, z0_s
        ub = (np.inf, 5.0, 1.0)
        def f_scaled(zz_s, k, alpha, z0_s):
            return power_shift(zz_s * L, k, alpha, z0_s * L, C)
        try:
            popt, _ = curve_fit(f_scaled, z_s, y, p0=(k0, alpha0, z0_s0),
                                bounds=(lb, ub), maxfev=maxfev)
            k, alpha, z0_s = popt
        except Exception:
            k, alpha, z0_s = k0, alpha0, z0_s0
        z0 = z0_s * L
        return (k, alpha, z0, C), t0, C

    else:
        raise ValueError(f"Modelo desconhecido: {model}")


def predict_full_on_csv(path: Path, model: str, params, t0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prediz em TODOS os passos do CSV (incluindo zeros).
    Retorna (t_full, y_true_full, y_pred_full).
    """
    df_ref = pd.read_csv(path, usecols=["passo", "presas_vivas"]).sort_values("passo")
    t_full = df_ref["passo"].to_numpy(float)
    y_true = df_ref["presas_vivas"].to_numpy(float)
    z_full = t_full - t0

    if model == "exp":
        A, tau, C = params
        y_pred = exp_offset(np.maximum(z_full, 0.0), A, tau, C)
    elif model == "exp_power":
        A, tau, beta, C = params
        y_pred = exp_power(np.maximum(z_full, 0.0), A, tau, beta, C)
    elif model == "power_shift":
        k, alpha, z0, C = params
        y_pred = power_shift(z_full, k, alpha, z0, C)
    else:
        raise ValueError(f"Modelo desconhecido: {model}")

    return t_full, y_true, y_pred

def metrics(y_true: np.ndarray, y_pred: np.ndarray, k_params: int):
    """
    Retorna RSS, R2, AIC, BIC.
    k_params = nº de parâmetros AJUSTADOS (não conte C se foi fixo).
    """
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[m]; yh = y_pred[m]
    n = y.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    rss = float(np.sum((y - yh)**2))
    tss = float(np.sum((y - np.mean(y))**2))
    r2  = float(1.0 - rss/tss) if tss > 0 else np.nan
    if rss <= 0:
        aic = bic = np.nan
    else:
        aic = float(n*np.log(rss/n) + 2*k_params)
        bic = float(n*np.log(rss/n) + k_params*np.log(n))
    return rss, r2, aic, bic

# ============================
# I/O
# ============================

def collect_runs(scen_dir: Path) -> List[Path]:
    return sorted([p for p in scen_dir.rglob("*_run_*_presas_por_passo.csv") if p.is_file()])

# ============================
# Pipeline
# ============================

def analisar_todos(
    base: Path,
    SCENARIO: str,
    n_obs: int,
    salvar_csv: bool = True,
    salvar_fig: bool = True,
    out_dir: Optional[Path] = None,
    overlay_runs: bool = False,
    fix_A: bool = True,
    salvar_series_ajuste: bool = True,
):
    """
    Ajusta 3 modelos em todos os runs:
      1) Exp:       A, tau (+C fixo)   [A pode ser fixo em N0 - C]
      2) Exp^beta:  A, tau, beta (+C)  [A pode ser fixo em N0 - C]
      3) PowerShift:k, alpha, z0 (+C)

    Agrega estatísticas por modelo (medianas) e plota média±1σ + curvas medianas.
    Também salva, por run, os pontos da curva prevista para facilitar plots.
    """
    if out_dir is None:
        out_dir = base / "resultados_modelos"
    out_dir.mkdir(parents=True, exist_ok=True)
    series_dir = out_dir / "curvas_por_run"
    if salvar_series_ajuste:
        series_dir.mkdir(parents=True, exist_ok=True)

    scen_dir = base / SCENARIO / (f"s_obs_{n_obs}" if n_obs != 0 else "s_obs_00")
    if not scen_dir.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {scen_dir}")

    arquivos = collect_runs(scen_dir)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV *_run_*_presas_por_passo.csv em {scen_dir}")

    rows_exp = []
    rows_ep  = []
    rows_ps  = []
    series_list = []

    for i, path in enumerate(arquivos, 1):
        try:
            df_ref = pd.read_csv(path, usecols=["passo", "presas_vivas"]).sort_values("passo")
            # guarda série (>0) para média±std
            series_list.append(df_ref[df_ref["presas_vivas"] > 0]
                               .set_index("passo")["presas_vivas"])

            C_run = float(df_ref["presas_vivas"].iloc[-1])

            # --- EXP ---
            params_exp, t0_exp, _ = fit_model(path, "exp", C=C_run, fix_A=fix_A)
            t_full, y_true, y_pred_exp = predict_full_on_csv(path, "exp", params_exp, t0_exp)
            rss, r2, aic, bic = metrics(y_true, y_pred_exp, k_params=1 if fix_A else 2)  # tau (ou A+tau)
            A, tau, C = params_exp
            rows_exp.append({"arquivo": path.name, "t0": t0_exp, "A": A, "tau": tau, "C": C,
                             "RSS": rss, "R2": r2, "AIC": aic, "BIC": bic})

            # --- EXP^BETA (KWW) ---
            params_ep, t0_ep, _ = fit_model(path, "exp_power", C=C_run, fix_A=fix_A)
            _, _, y_pred_ep = predict_full_on_csv(path, "exp_power", params_ep, t0_ep)
            kparams_ep = 2 if fix_A else 3  # (tau,beta) ou (A,tau,beta)
            rss, r2, aic, bic = metrics(y_true, y_pred_ep, k_params=kparams_ep)
            A2, tau2, beta2, C2 = params_ep
            rows_ep.append({"arquivo": path.name, "t0": t0_ep, "A": A2, "tau": tau2, "beta": beta2, "C": C2,
                            "RSS": rss, "R2": r2, "AIC": aic, "BIC": bic})

            # --- POWER SHIFT ---
            params_ps, t0_ps, _ = fit_model(path, "power_shift", C=C_run, fix_A=False)
            _, _, y_pred_ps = predict_full_on_csv(path, "power_shift", params_ps, t0_ps)
            rss, r2, aic, bic = metrics(y_true, y_pred_ps, k_params=3)  # k, alpha, z0
            kS, alphaS, z0S, CS = params_ps
            rows_ps.append({"arquivo": path.name, "t0": t0_ps, "k": kS, "alpha": alphaS, "z0": z0S, "C": CS,
                            "RSS": rss, "R2": r2, "AIC": aic, "BIC": bic})

            # --- Série por run para exportar ---
            if salvar_series_ajuste:
                df_out = pd.DataFrame({
                    "passo": t_full,
                    "dado": y_true,
                    "y_exp": y_pred_exp,
                    "y_expbeta": y_pred_ep,
                    "y_powershift": y_pred_ps,
                })
                out_name = series_dir / f"{path.stem}_ajuste.csv"
                df_out.to_csv(out_name, index=False)

        except Exception as e:
            print(f"[ERRO] {path.name}: {e}")
            continue

    if not rows_exp:
        raise RuntimeError("Nenhum run válido após filtros.")

    df_exp = pd.DataFrame(rows_exp).sort_values("arquivo")
    df_ep  = pd.DataFrame(rows_ep).sort_values("arquivo") if rows_ep else pd.DataFrame()
    df_ps  = pd.DataFrame(rows_ps).sort_values("arquivo") if rows_ps else pd.DataFrame()

    # Média±std dos dados (>0)
    df_all = pd.concat(series_list, axis=1)
    y_mean = df_all.mean(axis=1)
    y_std  = df_all.std(axis=1)
    x_vals = y_mean.index.values.astype(float)

    # Parâmetros "centrais" (medianas) por modelo
    A_med, tau_med, C_med, t0_exp_med = (np.nanmedian(df_exp["A"]),
                                         np.nanmedian(df_exp["tau"]),
                                         np.nanmedian(df_exp["C"]),
                                         np.nanmedian(df_exp["t0"]))

    if not df_ep.empty:
        A_ep_med   = float(np.nanmedian(df_ep["A"]))
        tau_ep_med = float(np.nanmedian(df_ep["tau"]))
        beta_ep_med= float(np.nanmedian(df_ep["beta"]))
        C_ep_med   = float(np.nanmedian(df_ep["C"]))
        t0_ep_med  = float(np.nanmedian(df_ep["t0"]))
    else:
        A_ep_med = tau_ep_med = beta_ep_med = C_ep_med = t0_ep_med = np.nan

    if not df_ps.empty:
        k_ps_med    = float(np.nanmedian(df_ps["k"]))
        alpha_ps_med= float(np.nanmedian(df_ps["alpha"]))
        z0_ps_med   = float(np.nanmedian(df_ps["z0"]))
        C_ps_med    = float(np.nanmedian(df_ps["C"]))
        t0_ps_med   = float(np.nanmedian(df_ps["t0"]))
    else:
        k_ps_med = alpha_ps_med = z0_ps_med = C_ps_med = t0_ps_med = np.nan

    # -------- Plot --------
    plt.figure(figsize=(9, 6))
    plt.errorbar(x_vals, y_mean.values, yerr=y_std.values,
                 fmt='o', capsize=3, markersize=3, alpha=0.8,
                 label="Dados médios ±1σ")

    x_fit = np.linspace(float(x_vals.min()), float(x_vals.max()), 600)

    # Exp
    z_fit_exp = np.maximum(x_fit - t0_exp_med, 0.0)
    y_fit_exp = exp_offset(z_fit_exp, A_med, tau_med, C_med)
    plt.plot(x_fit, y_fit_exp, '-', label=f"Exp: A={A_med:.2g}, τ={tau_med:.2g}, C={C_med:.2g}")

    # Exp^beta
    if not np.isnan(A_ep_med):
        z_fit_ep = np.maximum(x_fit - t0_ep_med, 0.0)
        y_fit_ep = exp_power(z_fit_ep, A_ep_med, tau_ep_med, beta_ep_med, C_ep_med)
        plt.plot(x_fit, y_fit_ep, '--', label=f"Exp^β: A={A_ep_med:.2g}, τ={tau_ep_med:.2g}, β={beta_ep_med:.2g}, C={C_ep_med:.2g}")

    # Power shift
    if not np.isnan(k_ps_med):
        z_fit_ps = x_fit - t0_ps_med
        y_fit_ps = power_shift(z_fit_ps, k_ps_med, alpha_ps_med, z0_ps_med, C_ps_med)
        plt.plot(x_fit, y_fit_ps, ':', label=f"Power shift: k={k_ps_med:.2g}, α={alpha_ps_med:.2g}, z0={z0_ps_med:.2g}, C={C_ps_med:.2g}")

    # Overlay opcional (curvas fininhas) — aqui só para EXP para não poluir
    if overlay_runs:
        for _, r in df_exp.iterrows():
            z_i = np.maximum(x_fit - r["t0"], 0.0)
            yi  = exp_offset(z_i, r["A"], r["tau"], r["C"])
            plt.plot(x_fit, yi, lw=0.6, alpha=0.2)

    plt.xlabel("steps")
    plt.ylabel("preys alives")
    plt.title(f"{SCENARIO} • s_obs_{n_obs if n_obs != 0 else '00'}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if salvar_fig:
        fig_base = f"fits_media_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}"
        plt.savefig(out_dir / f"{fig_base}.pdf", bbox_inches="tight")
    plt.show()

    # CSVs com parâmetros por run
    if salvar_csv:
        df_exp.to_csv(out_dir / f"params_por_run_EXP_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}.csv", index=False)
        if not df_ep.empty:
            df_ep.to_csv(out_dir / f"params_por_run_EXPbeta_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}.csv", index=False)
        if not df_ps.empty:
            df_ps.to_csv(out_dir / f"params_por_run_POWERSHIFT_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}.csv", index=False)

    # Resumos (medianas). Use mean se preferir.
    def med(df, col): return float(np.nanmedian(df[col])) if col in df else np.nan
    print("\n=== RESUMO por modelo (medianas) ===")
    print(f"EXP:      R2={med(df_exp,'R2'):.4f}, AIC={med(df_exp,'AIC'):.2f}, BIC={med(df_exp,'BIC'):.2f}")
    if not df_ep.empty:
        print(f"EXP^β:    R2={med(df_ep,'R2'):.4f}, AIC={med(df_ep,'AIC'):.2f}, BIC={med(df_ep,'BIC'):.2f}")
    if not df_ps.empty:
        print(f"POWERSHIFT: R2={med(df_ps,'R2'):.4f}, AIC={med(df_ps,'AIC'):.2f}, BIC={med(df_ps,'BIC'):.2f}")

    return {
        "df_exp": df_exp,
        "df_exp_power": df_ep,
        "df_power_shift": df_ps,
        "x_mean": x_vals,
        "y_mean": y_mean.values,
        "y_std": y_std.values,
        "params_median": {
            "exp":     {"A": A_med, "tau": tau_med, "C": C_med, "t0": t0_exp_med},
            "expbeta": {"A": A_ep_med, "tau": tau_ep_med, "beta": beta_ep_med, "C": C_ep_med, "t0": t0_ep_med},
            "powershift": {"k": k_ps_med, "alpha": alpha_ps_med, "z0": z0_ps_med, "C": C_ps_med, "t0": t0_ps_med},
        }
    }

# ============================
# Execução
# ============================

if __name__ == "__main__":
    base = Path.home() / "Dados_Doc/Np=free*0.25"
    SCENARIO = "Nc=Np*0.5"    # ex.: "Nc=Np" ou "Nc=Np*0.5"
    n_obs = 14745 # 0, 1638, 3276, 4915, 6553, 8192, 9830, 11468, 13107, 14745

    _ = analisar_todos(
        base=base,
        SCENARIO=SCENARIO,
        n_obs=n_obs,
        salvar_csv=True,
        salvar_fig=True,
        out_dir=base / "resultados_modelos",
        overlay_runs=False,
        fix_A=True,                   # << usar A = N0 - C
        salvar_series_ajuste=True,    # << salva CSV por run com dados e curvas
    )
