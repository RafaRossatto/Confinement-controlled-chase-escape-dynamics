#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit
import re

# ============================
# Modelo
# ============================

EPS = 1e-12  # evita z=0 em potências

def exp_offset(z: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """y = A * exp(-z / tau) + C"""
    return A * np.exp(-z / tau) + C

def exp_power(z: np.ndarray, A: float, tau: float, beta: float, C: float) -> np.ndarray:
    """y = A * exp(-(z / tau)**beta) + C  (KWW / stretched exponential)"""
    zc = np.maximum(z, 0.0)  # z >= 0 por construção (clamp)
    return A * np.exp(- (zc / tau)**beta ) + C

# ============================
# Utilidades de ajuste e métricas
# ============================

def fit_expbeta(
    path: Path,
    C: Optional[float] = None,
    tau_factor: float = 20.0,
    maxfev: int = 20000,
    fix_A: bool = True,
    max_step_fit: Optional[float] = None,
):
    """
    Ajusta SOMENTE o modelo exponencial esticado (KWW):
        y = A * exp(-((t-t0)/tau)**beta) + C

    Parâmetros principais:
      - C é fixo (último valor do CSV se não fornecido)
      - Se fix_A=True: A = N0 - C (clamp >= 1e-9)
      - max_step_fit: se definido, IGNORA pontos com passo > max_step_fit ao fazer o *fit*.

    Retorna (params_tuple=(A,tau,beta,C), t0, C).
    """
    df_all = pd.read_csv(path, usecols=["step", "living_prey"]).sort_values("step")

    if C is None:
        C = float(df_all["living_prey"].iloc[-1])

    # N0: número inicial de presas (no menor "passo" do CSV, tipicamente t=0)
    N0 = float(df_all["living_prey"].iloc[0])
    A_fix = max(N0 - C, 1e-9) if fix_A else None

    # usa só >0 para ajuste; aplica corte no tempo, se pedido
    m_pos = df_all["living_prey"] > 0
    if max_step_fit is not None:
        m_pos &= df_all["step"] <= float(max_step_fit)
    df_pos = df_all[m_pos]

    x = df_pos["step"].to_numpy(float)
    y = df_pos["living_prey"].to_numpy(float)

    # referência temporal
    t0 = float(df_all["step"].min()) if x.size == 0 else float(x.min())

    # poucos pontos → valores seguros/degradação
    if x.size < 3:
        if A_fix is not None:
            return (A_fix, 1.0, 1.0, C), t0, C
        A0 = max((y.mean() - C) if x.size else (N0 - C), 1e-9)
        return (A0, 1.0, 1.0, C), t0, C

    z = x - t0
    L = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0
    _ = max(1e-6, tau_factor * L)  # bounds tratam implicitamente

    # reescala z para [0,1] para melhorar condicionamento
    z_s = np.maximum(z / L, 0.0)
    tau0_s = 0.5
    beta0 = 1.0

    if A_fix is not None:
        lb = (1e-3, 0.2)     # tau_s, beta
        ub = (5.0, 4.0)

        def f_scaled_fixA(zz_s, tau_s, beta):
            return exp_power(zz_s * L, A_fix, tau_s * L, beta, C)

        try:
            popt, _ = curve_fit(
                f_scaled_fixA, z_s, y, p0=(tau0_s, beta0), bounds=(lb, ub), maxfev=maxfev
            )
            tau_s, beta = popt
        except Exception:
            tau_s, beta = tau0_s, beta0
        tau = tau_s * L
        return (A_fix, tau, beta, C), t0, C

    else:
        A0 = max(y[0] - C, 1e-9)
        lb = (0.0, 1e-3, 0.2)   # A, tau_s, beta
        ub = (10*max(y.max(), 1.0), 5.0, 4.0)

        def f_scaled(zz_s, A, tau_s, beta):
            return exp_power(zz_s * L, A, tau_s * L, beta, C)

        try:
            popt, _ = curve_fit(
                f_scaled, z_s, y, p0=(A0, tau0_s, beta0), bounds=(lb, ub), maxfev=maxfev
            )
            A, tau_s, beta = popt
        except Exception:
            A, tau_s, beta = A0, tau0_s, beta0
        tau = tau_s * L
        return (A, tau, beta, C), t0, C

def predict_full_on_csv_expbeta(
    path: Path, params, t0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediz em TODOS os passos do CSV (incluindo zeros) para o modelo exp^beta."""
    df_ref = pd.read_csv(path, usecols=["step", "living_prey"]).sort_values("step")
    t_full = df_ref["step"].to_numpy(float)
    y_true = df_ref["living_prey"].to_numpy(float)
    z_full = np.maximum(t_full - t0, 0.0)

    A, tau, beta, C = params
    y_pred = exp_power(z_full, A, tau, beta, C)
    return t_full, y_true, y_pred

def metrics(y_true: np.ndarray, y_pred: np.ndarray, k_params: int):
    """
    Retorna RSS, R2, AIC, BIC.
    k_params = nº de parâmetros AJUSTADOS (não conte C nem A se estiverem fixos).
    """
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[m]
    yh = y_pred[m]
    n = y.size
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    rss = float(np.sum((y - yh) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan
    if rss <= 0:
        aic = bic = np.nan
    else:
        aic = float(n * np.log(rss / n) + 2 * k_params)
        bic = float(n * np.log(rss / n) + k_params * np.log(n))
    return rss, r2, aic, bic

# ============================
# I/O
# ============================

def collect_runs(scen_dir: Path) -> List[Path]:
    return sorted([p for p in scen_dir.rglob("*_run_*_prey_per_step.csv") if p.is_file()])

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
    max_step_fit: Optional[float] = None,
    mostrar_fit_media: bool = True,   # << controla a 2ª curva (fit direto na média)
    label_latex: bool = True,
    incluir_A_na_legenda: bool = False,
):
    """
    Ajusta APENAS o modelo exponencial esticado (KWW) em todos os runs.

    - max_step_fit: se definido, corta pontos com passo > max_step_fit **somente** para o *fit*.
      (As previsões e as médias/±1σ continuam sendo computadas em todos os passos disponíveis.)

    Plota média±1σ + curva mediana (parâmetros mediana dos runs).
    Opcionalmente:
      • plota o ajuste direto na média (mostrar_fit_media=True)
    Também salva, por run, os pontos previstos para facilitar plots.
    """
    if out_dir is None:
        out_dir = base / "resultados_modelos"
    out_dir.mkdir(parents=True, exist_ok=True)
    series_dir = out_dir / "curvas_por_run"
    if salvar_series_ajuste:
        series_dir.mkdir(parents=True, exist_ok=True)

    # pasta para salvar parâmetros por run (um arquivo por run)
    params_dir = out_dir / "params_por_run_detalhe"
    if salvar_csv:
        params_dir.mkdir(parents=True, exist_ok=True)

    scen_dir = base / SCENARIO / (f"s_obs_{n_obs}" if n_obs != 0 else "s_obs_00")
    if not scen_dir.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {scen_dir}")

    arquivos = collect_runs(scen_dir)
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV *_run_*_presas_por_passo.csv em {scen_dir}"
        )

    rows_ep = []
    series_list = []

    for path in arquivos:
        try:
            df_ref = (
                pd.read_csv(path, usecols=["step", "living_prey"])\
                  .sort_values("step")
            )
            # guarda série (>0) para média±std (sem corte; apenas >0)
            series_list.append(
                df_ref[df_ref["living_prey"] > 0].set_index("step")["living_prey"]
            )

            C_run = float(df_ref["living_prey"].iloc[-1])

            # --- EXP^BETA (KWW) --- (com corte opcional no *fit*)
            params_ep, t0_ep, _ = fit_expbeta(
                path,
                C=C_run,
                fix_A=fix_A,
                max_step_fit=max_step_fit,
            )
            t_full, y_true, y_pred_ep = predict_full_on_csv_expbeta(path, params_ep, t0_ep)
            kparams_ep = 2 if fix_A else 3  # (tau,beta) ou (A,tau,beta)
            rss, r2, aic, bic = metrics(y_true, y_pred_ep, k_params=kparams_ep)
            A2, tau2, beta2, C2 = params_ep
            rows_ep.append(
                {
                    "arquivo": path.name,
                    "t0": t0_ep,
                    "A": A2,
                    "tau": tau2,
                    "beta": beta2,
                    "C": C2,
                    "RSS": rss,
                    "R2": r2,
                    "AIC": aic,
                    "BIC": bic,
                }
            )

            # --- Salvar parâmetros por run (CSV pequeno) ---
            if salvar_csv:
                pd.DataFrame([
                    {
                        "arquivo": path.name,
                        "t0": t0_ep,
                        "A": A2,
                        "tau": tau2,
                        "beta": beta2,
                        "C": C2,
                        "RSS": rss,
                        "R2": r2,
                        "AIC": aic,
                        "BIC": bic,
                        "max_step_fit": max_step_fit if max_step_fit is not None else np.nan,
                        "fix_A": bool(fix_A),
                    }
                ]).to_csv(params_dir / f"{path.stem}_params.csv", index=False)

            # --- Série por run para exportar ---
            if salvar_series_ajuste:
                df_out = pd.DataFrame(
                    {
                        "step": t_full,
                        "dado": y_true,
                        "y_expbeta": y_pred_ep,
                    }
                )
                out_name = series_dir / f"{path.stem}_ajuste.csv"
                df_out.to_csv(out_name, index=False)

        except Exception as e:
            print(f"[ERRO] {path.name}: {e}")
            continue

    if not rows_ep:
        raise RuntimeError("Nenhum run válido após filtros.")

    df_ep = pd.DataFrame(rows_ep).sort_values("arquivo")

    # Média±std dos dados (>0)
    df_all = pd.concat(series_list, axis=1)
    y_mean = df_all.mean(axis=1)
    y_std = df_all.std(axis=1)
    x_vals = y_mean.index.values.astype(float)

    # Parâmetros "centrais" (medianas)
    A_ep_med = float(np.nanmedian(df_ep["A"]))
    tau_ep_med = float(np.nanmedian(df_ep["tau"]))
    beta_ep_med = float(np.nanmedian(df_ep["beta"]))
    C_ep_med = float(np.nanmedian(df_ep["C"]))
    t0_ep_med = float(np.nanmedian(df_ep["t0"]))

    # --- Ajuste direto na média (opcional) ---
    params_mean = None
    r2_med_vs_mean = aic_med_vs_mean = bic_med_vs_mean = np.nan
    r2_mean_vs_mean = aic_mean_vs_mean = bic_mean_vs_mean = np.nan

    if mostrar_fit_media:
        m_fit_mean = x_vals <= (max_step_fit if max_step_fit is not None else x_vals.max())
        x_mean_fit = x_vals[m_fit_mean]
        y_mean_fit = y_mean.values[m_fit_mean]
        if x_mean_fit.size >= 3:
            t0_m = float(x_mean_fit.min())
            z = x_mean_fit - t0_m
            Lm = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0
            z_s = np.maximum(z / Lm, 0.0)
            tau0_s = 0.5
            beta0 = 1.0
            C_m = float(y_mean_fit[-1])
            if fix_A:
                A_m = max(y_mean_fit[0] - C_m, 1e-9)
                lb = (1e-3, 0.2)
                ub = (5.0, 4.0)
                def f_mean(zz_s, tau_s, beta):
                    return exp_power(zz_s * Lm, A_m, tau_s * Lm, beta, C_m)
                try:
                    popt, _ = curve_fit(f_mean, z_s, y_mean_fit, p0=(tau0_s, beta0), bounds=(lb, ub), maxfev=20000)
                    tau_s, beta = popt
                except Exception:
                    tau_s, beta = tau0_s, beta0
                tau_m = tau_s * Lm
                params_mean = (A_m, tau_m, beta, C_m, t0_m)
            else:
                A0 = max(y_mean_fit[0] - C_m, 1e-9)
                lb = (0.0, 1e-3, 0.2)
                ub = (10*max(y_mean_fit.max(),1.0), 5.0, 4.0)
                def f_mean(zz_s, A, tau_s, beta):
                    return exp_power(zz_s * Lm, A, tau_s * Lm, beta, C_m)
                try:
                    popt, _ = curve_fit(f_mean, z_s, y_mean_fit, p0=(A0, tau0_s, beta0), bounds=(lb, ub), maxfev=20000)
                    A_m, tau_s, beta = popt
                except Exception:
                    A_m, tau_s, beta = A0, tau0_s, beta0
                tau_m = tau_s * Lm
                params_mean = (A_m, tau_m, beta, C_m, t0_m)

    # métricas vs média para a curva mediana
    y_fit_med_on_x = exp_power(np.maximum(x_vals - t0_ep_med, 0.0), A_ep_med, tau_ep_med, beta_ep_med, C_ep_med)
    rss_med, r2_med_vs_mean, aic_med_vs_mean, bic_med_vs_mean = metrics(y_mean.values, y_fit_med_on_x, k_params=(2 if fix_A else 3))

    if mostrar_fit_media and params_mean is not None:
        A_m, tau_m, beta_m, C_m, t0_m = params_mean
        y_fit_mean_on_x = exp_power(np.maximum(x_vals - t0_m, 0.0), A_m, tau_m, beta_m, C_m)
        rss_mean, r2_mean_vs_mean, aic_mean_vs_mean, bic_mean_vs_mean = metrics(y_mean.values, y_fit_mean_on_x, k_params=(2 if fix_A else 3))

    # -------- Plot --------
    plt.figure(figsize=(9, 6))

    # --- Rótulo dos dados conforme SCENARIO ---
    # Se SCENARIO="Nc=Np*0.5" → f=0.5 → label "N^C_{0}=0.5 N^E_{0}"
    # Se SCENARIO="Nc=Np"     → f=1   → label "⟨N^E⟩ ± σ"
    try:
        txt = (SCENARIO or "").replace(" ", "")
        m = re.search(r"Nc\s*=\s*Np(?:\*([0-9]*\.?[0-9]+))?$", txt, flags=re.IGNORECASE)
        f_nc = 1.0 if (m and m.group(1) is None) else (float(m.group(1)) if m else None)
    except Exception:
        f_nc = None

    if label_latex:
        if (f_nc is not None) and (abs(f_nc - 1.0) > 1e-12):
            dados_label = rf"$N^C_{{0}}={f_nc:g}\,N^E_{{0}}$"
        else:
            dados_label = r"$ N^C_{0} = N^E_{0}$"
    else:
        if (f_nc is not None) and (abs(f_nc - 1.0) > 1e-12):
            dados_label = f"N^C_0={f_nc:g}·N^E_0"
        else:
            dados_label = "Dados médios ±1σ"

    plt.errorbar(
        x_vals,
        y_mean.values,
        yerr=y_std.values,
        fmt='o',
        capsize=3,
        markersize=3,
        alpha=0.8,
        label=dados_label,
    )

    x_fit = np.linspace(float(x_vals.min()), float(x_vals.max()), 600)

    # Exp^beta mediano
    z_fit_ep = np.maximum(x_fit - t0_ep_med, 0.0)
    y_fit_ep = exp_power(z_fit_ep, A_ep_med, tau_ep_med, beta_ep_med, C_ep_med)
    if incluir_A_na_legenda:
        lbl_med = rf"$\mathrm{{Exp}}^{{\beta}}_{{med}}:~A={A_ep_med:.3g},~\tau={tau_ep_med:.3g},~\beta={beta_ep_med:.3g},~C={C_ep_med:.3g}$"
    else:
        lbl_med = rf"$\mathrm{{Exp}}^{{\beta}}_{{med}}:~\tau={tau_ep_med:.3g},~\beta={beta_ep_med:.3g},~C={C_ep_med:.3g}$"
    plt.plot(x_fit, y_fit_ep, '--', label=lbl_med)

    # 2ª curva: Fit direto na média (opcional)
    if mostrar_fit_media and params_mean is not None:
        A_m, tau_m, beta_m, C_m, t0_m = params_mean
        z_fit_m = np.maximum(x_fit - t0_m, 0.0)
        y_fit_m = exp_power(z_fit_m, A_m, tau_m, beta_m, C_m)
        if incluir_A_na_legenda:
            lbl_mean = rf"$\mathrm{{Exp}}^{{\beta}}_{{mean}}:~A={A_m:.3g},~\tau={tau_m:.3g},~\beta={beta_m:.3g},~C={C_m:.3g}$"
        else:
            lbl_mean = rf"$\mathrm{{Exp}}^{{\beta}}_{{mean}}:~\tau={tau_m:.3g},~\beta={beta_m:.3g},~C={C_m:.3g}$"
        plt.plot(x_fit, y_fit_m, '-', label=lbl_mean)

    # Overlay opcional: curvas finas por run
    if overlay_runs:
        for _, r in df_ep.iterrows():
            zi = np.maximum(x_fit - r["t0"], 0.0)
            yi = exp_power(zi, r["A"], r["tau"], r["beta"], r["C"])
            plt.plot(x_fit, yi, lw=0.6, alpha=0.2)

    plt.xlabel("steps")
    plt.ylabel(r"$N^E$" if label_latex else "N^E")
    # plt.title(f"{SCENARIO} • s_obs_{n_obs if n_obs != 0 else '00'}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if salvar_fig:
        fig_base = f"fits_media_{SCENARIO.replace('/', '-')}_s_obs_{n_obs if n_obs!=0 else '00'}"
        plt.savefig(out_dir / f"{fig_base}.pdf", bbox_inches="tight")
    plt.show()

    # CSV com parâmetros por run (apenas EXP^beta)
    if salvar_csv:
        df_ep.to_csv(
            out_dir
            / f"params_por_run_EXPbeta_{SCENARIO.replace('/', '-')}_s_obs_{n_obs if n_obs!=0 else '00'}.csv",
            index=False,
        )

    # Resumo (medianas). Use mean se preferir.
    def med(df, col):
        return float(np.nanmedian(df[col])) if col in df else np.nan

    print("=== RESUMO (medianas) — EXP^β ===")
    print(f"R2={med(df_ep,'R2'):.4f}, AIC={med(df_ep,'AIC'):.2f}, BIC={med(df_ep,'BIC'):.2f}")

    print("=== COMPARAÇÃO exp^β (mediana por run vs fit direto na média) ===")
    print(f"Mediana: τ={tau_ep_med:.4g}, β={beta_ep_med:.4g}, C={C_ep_med:.4g} | R2_vs_média={r2_med_vs_mean:.4f}, AIC_vs_média={aic_med_vs_mean:.2f}")
    if mostrar_fit_media and params_mean is not None:
        print(f"Média  : τ={tau_m:.4g}, β={beta_m:.4g}, C={C_m:.4g} | R2_vs_média={r2_mean_vs_mean:.4f}, AIC_vs_média={aic_mean_vs_mean:.2f}")
        if (tau_m > 0) and (beta_m > 0):
            d_tau = (tau_ep_med - tau_m) / tau_m
            d_beta = (beta_ep_med - beta_m) / beta_m
            print(f"Δrel τ={d_tau:.2%}, Δrel β={d_beta:.2%}")

    retorno = {
        "df_exp_power": df_ep,
        "x_mean": x_vals,
        "y_mean": y_mean.values,
        "y_std": y_std.values,
        "params_median": {
            "expbeta": {
                "A": A_ep_med,
                "tau": tau_ep_med,
                "beta": beta_ep_med,
                "C": C_ep_med,
                "t0": t0_ep_med,
            }
        },
    }
    if mostrar_fit_media and params_mean is not None:
        retorno["params_fit_mean"] = {"A": A_m, "tau": tau_m, "beta": beta_m, "C": C_m, "t0": t0_m}
    return retorno

# ============================
# Execução
# ============================

if __name__ == "__main__":
    base = Path.home() / "Dados_Doc/Np=free*0.25/L_128"
    SCENARIO = "Nc=Np*0.5"    # ex.: "Nc=Np" ou "Nc=Np*0.5"
    n_obs = 0 # 0, 1638, 3276, 4915, 6553, 8192, 9830, 11468, 13107, 14745

    _ = analisar_todos(
        base=base,
        SCENARIO=SCENARIO,
        n_obs=n_obs,
        salvar_csv=True,
        salvar_fig=True,
        out_dir=base / "resultados_modelos",
        overlay_runs=False,
        fix_A=True,                # usa A = N0 - C
        salvar_series_ajuste=True,
        max_step_fit=1500,         # corte no *fit*
        mostrar_fit_media=False,    # << liga/desliga a 2ª curva (fit na média)
        label_latex=True,          # rótulos em LaTeX
        incluir_A_na_legenda=True,
    )
