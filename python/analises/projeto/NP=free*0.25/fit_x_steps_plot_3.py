#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit
import re

# --- Configuração global de fonte nos eixos e legenda ---
plt.rcParams.update({
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14
})

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
    df_all = pd.read_csv(path, usecols=["passo", "presas_vivas"]).sort_values("passo")

    if C is None:
        C = float(df_all["presas_vivas"].iloc[-1])

    # N0: número inicial de presas (no menor "passo" do CSV, tipicamente t=0)
    N0 = float(df_all["presas_vivas"].iloc[0])
    A_fix = max(N0 - C, 1e-9) if fix_A else None

    # usa só >0 para ajuste; aplica corte no tempo, se pedido
    m_pos = df_all["presas_vivas"] > 0
    if max_step_fit is not None:
        m_pos &= df_all["passo"] <= float(max_step_fit)
    df_pos = df_all[m_pos]

    x = df_pos["passo"].to_numpy(float)
    y = df_pos["presas_vivas"].to_numpy(float)

    # referência temporal
    t0 = float(df_all["passo"].min()) if x.size == 0 else float(x.min())

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
    df_ref = pd.read_csv(path, usecols=["passo", "presas_vivas"]).sort_values("passo")
    t_full = df_ref["passo"].to_numpy(float)
    y_true = df_ref["presas_vivas"].to_numpy(float)
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
    return sorted(
        [p for p in scen_dir.rglob("*_run_*_presas_por_passo.csv") if p.is_file()]
    )

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
    mostrar_fit_media: bool = True,
    label_latex: bool = True,
    incluir_A_na_legenda: bool = False,
    plot_individual: bool = False,   # <<< controla se plota ou só retorna dados
):
    """
    Ajusta o modelo exponencial esticado (KWW): y = A * exp(-(z/tau)^beta) + C
    para TODOS os runs de um cenário (SCENARIO) em um dado número de obstáculos (n_obs).

    - O fit por run respeita 'max_step_fit' (corte só no ajuste, não nas séries salvas).
    - Retorna média ±1σ de N^E(t) (>0), e parâmetros medianos para compor o plot fora.
    """
    # ---------------- I/O e paths ----------------
    if out_dir is None:
        out_dir = base / "resultados_modelos"
    out_dir.mkdir(parents=True, exist_ok=True)

    series_dir = out_dir / "curvas_por_run"
    if salvar_series_ajuste:
        series_dir.mkdir(parents=True, exist_ok=True)

    params_dir = out_dir / "params_por_run_detalhe"
    if salvar_csv:
        params_dir.mkdir(parents=True, exist_ok=True)

    scen_dir = base / SCENARIO / (f"s_obs_{n_obs}" if n_obs != 0 else "s_obs_00")
    if not scen_dir.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {scen_dir}")

    arquivos = collect_runs(scen_dir)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV *_run_*_presas_por_passo.csv em {scen_dir}")

    # ---------------- Loop de runs ----------------
    rows_ep = []
    series_list = []
    for path in arquivos:
        try:
            df_ref = (
                pd.read_csv(path, usecols=["passo", "presas_vivas"])
                .sort_values("passo")
            )

            # série (>0) para média±std (mantém todos os passos, só filtra N^E>0)
            series_list.append(
                df_ref[df_ref["presas_vivas"] > 0].set_index("passo")["presas_vivas"]
            )

            # C fixo por run = último valor observado
            C_run = float(df_ref["presas_vivas"].iloc[-1])

            # fit exp^beta com corte opcional
            params_ep, t0_ep, _ = fit_expbeta(
                path,
                C=C_run,
                fix_A=fix_A,
                max_step_fit=max_step_fit,
            )

            # previsão completa no domínio do CSV
            t_full, y_true, y_pred_ep = predict_full_on_csv_expbeta(path, params_ep, t0_ep)

            # métricas (k_params depende de fix_A)
            kparams_ep = 2 if fix_A else 3
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

            # salva série prevista por run (opcional)
            if salvar_series_ajuste:
                pd.DataFrame(
                    {"passo": t_full, "dado": y_true, "y_expbeta": y_pred_ep}
                ).to_csv(series_dir / f"{path.stem}_ajuste.csv", index=False)

            # salva params por run (opcional)
            if salvar_csv:
                pd.DataFrame(
                    [
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
                    ]
                ).to_csv(params_dir / f"{path.stem}_params.csv", index=False)

        except Exception as e:
            print(f"[ERRO] {path.name}: {e}")
            continue

    if not rows_ep:
        # nada válido após filtros → retorna estrutura mínima
        return {
            "df_exp_power": pd.DataFrame(),
            "x_mean": np.array([]),
            "y_mean": np.array([]),
            "y_std":  np.array([]),
            "params_median": {"expbeta": {"A": np.nan, "tau": np.nan, "beta": np.nan, "C": np.nan, "t0": np.nan}},
        }

    # ---------------- Agrega por cenário ----------------
    df_ep = pd.DataFrame(rows_ep).sort_values("arquivo")

    # média e desvio padrão por tempo (somente valores >0, já filtrados em series_list)
    df_all = pd.concat(series_list, axis=1)
    y_mean = df_all.mean(axis=1)
    y_std = df_all.std(axis=1)
    x_vals = y_mean.index.values.astype(float)

    # parâmetros medianos
    A_ep_med   = float(np.nanmedian(df_ep["A"]))
    tau_ep_med = float(np.nanmedian(df_ep["tau"]))
    beta_ep_med= float(np.nanmedian(df_ep["beta"]))
    C_ep_med   = float(np.nanmedian(df_ep["C"]))
    t0_ep_med  = float(np.nanmedian(df_ep["t0"]))

    # ---------------- Fit direto na média (opcional) ----------------
    params_mean = None
    if mostrar_fit_media and x_vals.size >= 3:
        # aplica o mesmo corte no *fit* da média (se houver)
        x_fit_mask = x_vals <= (max_step_fit if max_step_fit is not None else x_vals.max())
        x_mean_fit = x_vals[x_fit_mask]
        y_mean_fit = y_mean.values[x_fit_mask]

        if x_mean_fit.size >= 3:
            t0_m = float(x_mean_fit.min())
            z = x_mean_fit - t0_m
            Lm = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0
            z_s = np.maximum(z / Lm, 0.0)
            tau0_s, beta0 = 0.5, 1.0
            C_m = float(y_mean_fit[-1])

            if fix_A:
                A_m = max(y_mean_fit[0] - C_m, 1e-9)
                lb, ub = (1e-3, 0.2), (5.0, 4.0)

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
                lb, ub = (0.0, 1e-3, 0.2), (10*max(y_mean_fit.max(),1.0), 5.0, 4.0)

                def f_mean(zz_s, A, tau_s, beta):
                    return exp_power(zz_s * Lm, A, tau_s * Lm, beta, C_m)

                try:
                    popt, _ = curve_fit(f_mean, z_s, y_mean_fit, p0=(A0, tau0_s, beta0), bounds=(lb, ub), maxfev=20000)
                    A_m, tau_s, beta = popt
                except Exception:
                    A_m, tau_s, beta = A0, tau0_s, beta0
                tau_m = tau_s * Lm
                params_mean = (A_m, tau_m, beta, C_m, t0_m)

    # ---------------- PLOT (apenas se plot_individual=True) ----------------
    if plot_individual:
        plt.figure(figsize=(9, 6))

        # rótulo dos dados conforme SCENARIO
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
                dados_label = r"$N^C_{0}=N^E_{0}$"
        else:
            if (f_nc is not None) and (abs(f_nc - 1.0) > 1e-12):
                dados_label = f"N^C_0={f_nc:g}·N^E_0"
            else:
                dados_label = "Dados médios ±1σ"

        # pontos: média ±1σ
        plt.errorbar(
            x_vals, y_mean.values, yerr=y_std.values,
            fmt='o', capsize=3, markersize=3, alpha=0.85, label=dados_label,
        )

        # curva: parâmetros medianos
        x_fit = np.linspace(float(x_vals.min()), float(x_vals.max()), 600) if x_vals.size else np.array([])
        if x_fit.size:
            z_fit_ep = np.maximum(x_fit - t0_ep_med, 0.0)
            y_fit_ep = exp_power(z_fit_ep, A_ep_med, tau_ep_med, beta_ep_med, C_ep_med)
            if incluir_A_na_legenda:
                lbl_med = rf"$\mathrm{{Exp}}^{{\beta}}_{{med}}:~A={A_ep_med:.3g},~\tau={tau_ep_med:.3g},~\beta={beta_ep_med:.3g},~C={C_ep_med:.3g}$"
            else:
                lbl_med = rf"$\mathrm{{Exp}}^{{\beta}}_{{med}}:~\tau={tau_ep_med:.3g},~\beta={beta_ep_med:.3g},~C={C_ep_med:.3g}$"
            plt.plot(x_fit, y_fit_ep, '--', label=lbl_med, lw=1.6)

        # 2ª curva: fit direto na média (se calculado)
        if mostrar_fit_media and (params_mean is not None) and x_fit.size:
            A_m, tau_m, beta_m, C_m, t0_m = params_mean
            z_fit_m = np.maximum(x_fit - t0_m, 0.0)
            y_fit_m = exp_power(z_fit_m, A_m, tau_m, beta_m, C_m)
            if incluir_A_na_legenda:
                lbl_mean = rf"$\mathrm{{Exp}}^{{\beta}}_{{mean}}:~A={A_m:.3g},~\tau={tau_m:.3g},~\beta={beta_m:.3g},~C={C_m:.3g}$"
            else:
                lbl_mean = rf"$\mathrm{{Exp}}^{{\beta}}_{{mean}}:~\tau={tau_m:.3g},~\beta={beta_m:.3g},~C={C_m:.3g}$"
            plt.plot(x_fit, y_fit_m, '-', label=lbl_mean, lw=1.6)

        if overlay_runs and x_fit.size:
            for _, r in df_ep.iterrows():
                zi = np.maximum(x_fit - r["t0"], 0.0)
                yi = exp_power(zi, r["A"], r["tau"], r["beta"], r["C"])
                plt.plot(x_fit, yi, lw=0.6, alpha=0.2)

        plt.xlabel("steps")
        plt.ylabel(r"$N^E$" if label_latex else "N^E")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if salvar_fig:
            fig_base = f"fits_media_{SCENARIO.replace('/', '-')}_s_obs_{n_obs if n_obs!=0 else '00'}"
            plt.savefig(out_dir / f"{fig_base}.pdf", bbox_inches="tight")
        plt.show()

    # ---------------- Retorno ----------------
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
    if params_mean is not None:
        A_m, tau_m, beta_m, C_m, t0_m = params_mean
        retorno["params_fit_mean"] = {"A": A_m, "tau": tau_m, "beta": beta_m, "C": C_m, "t0": t0_m}

    return retorno

# ============================
# Plot conjunto: Nc=Np*0.5, Nc=Np*0.8, Nc=Np
# ============================

def plot_tres_cenarios(
    base: Path,
    n_obs: int,
    max_step_fit: int = 1500,
    fix_A: bool = True,
    incluir_A_na_legenda: bool = False,
    salvar_fig: bool = True,
    out_dir: Optional[Path] = None,
    # --- parâmetros da curva de referência:
    A_ref: float = 1.0,
    tau_ref: float = 1.0,
):
    """
    Plota em UM ÚNICO gráfico os três cenários:
      Nc=Np*0.5, Nc=Np*0.8 e Nc=Np
    mostrando:
      • média ±1σ
      • ajuste Exp^β (mediano)
      • curva de referência global y = A_ref * exp(-t/tau_ref)
    """
    if out_dir is None:
        out_dir = base / "resultados_modelos"
    out_dir.mkdir(parents=True, exist_ok=True)

    cenarios = [
        ("Nc=Np*0.5", 0.5, {"fmt": "o", "ls": "--", "color": "tab:blue"}),
        ("Nc=Np*0.8", 0.8, {"fmt": "s", "ls": "-.", "color": "tab:orange"}),
        ("Nc=Np"    , 1.0, {"fmt": "D", "ls": "-",  "color": "tab:green"}),
    ]

    plt.figure(figsize=(9, 6))

    max_x_global = 0.0

    for scen, f_nc, style in cenarios:
        ret = analisar_todos(
            base=base,
            SCENARIO=scen,
            n_obs=n_obs,
            salvar_csv=False,
            salvar_fig=False,
            out_dir=out_dir,
            overlay_runs=False,
            fix_A=fix_A,
            salvar_series_ajuste=False,
            max_step_fit=max_step_fit,
            mostrar_fit_media=False,
            label_latex=True,
            incluir_A_na_legenda=incluir_A_na_legenda,
            plot_individual=False,
        )

        # pontos: média ±1σ
        x = ret["x_mean"]
        y = ret["y_mean"]
        yerr = ret["y_std"]

        if x.size:
            max_x_global = max(max_x_global, float(np.max(x)))

        if abs(f_nc - 1.0) < 1e-12:
            dados_label = r"$N^C_{0}=N^E_{0}$"
        else:
            dados_label = rf"$N^C_{{0}}={f_nc:g}\,N^E_{{0}}$"

        plt.errorbar(
            x, y, yerr=yerr,
            fmt=style["fmt"], color=style["color"],
            markersize=3, capsize=3, alpha=0.9,
            label=dados_label,
        )

        # curva exp^β (KWW)
        p = ret["params_median"]["expbeta"]
        A, tau, beta, C, t0 = p["A"], p["tau"], p["beta"], p["C"], p["t0"]

        if x.size:
            x_fit = np.linspace(float(x.min()), float(x.max()), 600)
            z_fit = np.maximum(x_fit - t0, 0.0)
            y_fit_beta = exp_power(z_fit, A, tau, beta, C)
            lbl_fit_beta = rf"$\tau={tau:.3g}, \beta={beta:.3g}$"
            plt.plot(x_fit, y_fit_beta, style["ls"], color=style["color"], lw=1.6, label=lbl_fit_beta)

    # --- curva de referência global ---
        # --- curva de referência global ---
    if max_x_global > 0:
        A_ref = 4100
        x_zero = 35
        tau_ref = x_zero / np.log(A_ref)   # ajusta tau para que y(x_zero) ~ 1

        x_ref = np.linspace(0, max_x_global, 600)
        y_ref = A_ref * np.exp(-x_ref / tau_ref)

        plt.plot(
            x_ref, y_ref, 'k--', lw=2,
            label=rf"Ref: $e^{{-t/\tau}}$"
        )

    plt.xlabel("steps",fontsize=22)
    plt.ylabel(r"$N^E$",fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if salvar_fig:
        fname = f"tres_cenarios_s_obs_{n_obs if n_obs!=0 else '00'}_com_ref.pdf"
        plt.savefig(out_dir / fname, bbox_inches="tight")

    plt.show()



# ============================
# Execução
# ============================

if __name__ == "__main__":
    base = Path.home() / "Dados_Doc/Np=free*0.25"
    n_obs = 0  # mude para 1638, 3276, 4915, 6553, 8192, 9830, 11468, 13107, 14745

    # Se quiser ver plots individuais por cenário, chame assim (opcional):
    # _ = analisar_todos(base, "Nc=Np*0.5", n_obs, plot_individual=True)
    # _ = analisar_todos(base, "Nc=Np*0.8", n_obs, plot_individual=True)
    # _ = analisar_todos(base, "Nc=Np",     n_obs, plot_individual=True)

    # Gráfico único com as 3 curvas:
    plot_tres_cenarios(
        base=base,
        n_obs=n_obs,
        max_step_fit=1500,
        fix_A=True,
        incluir_A_na_legenda=True,
        salvar_fig=True,
        out_dir=base / "resultados_modelos",
    )
