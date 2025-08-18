#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.optimize import curve_fit, OptimizeWarning

# ============================
# Modelo e ajuste (simples e robusto)
# ============================

def exp_offset(z: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """y = A * exp(-z / tau) + C, com z = t - t0 (t0 = t.min() do run)."""
    return A * np.exp(-z / tau) + C

def fit_exp_plus_c(x: np.ndarray,
                   y: np.ndarray,
                   tau_factor: float = 20.0,
                   maxfev: int = 20000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Ajusta y = A*exp(-(t - t0)/tau) + C para um run.
    - Recentra o tempo: z = t - t0 (t0 = x.min()) e guarda t0 para plot posterior.
    - Multi-start leve em C, com bounds simples; fallback 2p (C fixo) se necessário.
    Retorna (params[A,tau,C], yhat_no_domínio_de_x, t0).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[m]; y = y[m]
    if x.size < 3:
        raise ValueError("Poucos pontos (>0) após filtro.")

    # Recentrar
    t0 = float(x.min())
    z = x - t0
    L = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0

    y_min, y_max = float(np.min(y)), float(np.max(y))
    tau_max = max(1e-6, tau_factor * L)

    # Bounds
    lb3 = np.array([0.0, 1e-6, 0.0], float)                         # A>=0, tau>0, C>=0
    ub3 = np.array([10.0*y_max if y_max>0 else 1.0, tau_max, 1.1*y_max], float)

    # Candidatos para C (platô)
    k_last = max(3, int(0.1 * len(y)))
    c_cands = np.array([0.0, y_min, np.percentile(y, 10),
                        float(np.mean(y[-k_last:])), y[-1]], float)
    c_cands = np.clip(np.unique(np.round(c_cands, 10)), 0.0, ub3[2])

    def at_bounds(p):
        return (np.isclose(p[1], ub3[1]) or np.isclose(p[2], lb3[2]) or np.isclose(p[2], ub3[2]))

    best = None
    had_warning = False

    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always", OptimizeWarning)
        for c0 in c_cands:
            # chutes simples dependentes de C
            A0 = float(max(y[0] - c0, 1e-9))
            # tau por razão (protetor)
            num = max(y[0] - c0, 1e-12)
            den = max(y[-1] - c0, 1e-12)
            ratio = max(num / den, 1.000001)
            tau0 = float(np.clip(L / np.log(ratio), lb3[1], ub3[1]))
            p0 = np.array([A0, tau0, c0], float)
            try:
                popt, _ = curve_fit(exp_offset, z, y, p0=p0, bounds=(lb3, ub3), maxfev=maxfev)
                yhat = exp_offset(z, *popt)
                rss = float(np.sum((y - yhat)**2))
                if (best is None) or (rss < best[0]):
                    best = (rss, popt, yhat)
            except Exception:
                continue
        had_warning = any(isinstance(w.message, OptimizeWarning) for w in wlog)

    # Se ficou mal condicionado, tenta fallback 2p (C fixo)
    if (best is None) or had_warning or at_bounds(best[1]):
        lb2 = np.array([0.0, 1e-6], float)
        ub2 = np.array([ub3[0], ub3[1]], float)
        best_fb = None
        for c0 in c_cands:
            y_adj = y - c0
            if np.all(y_adj <= 0):
                continue
            A0 = float(max(np.max(y_adj), 1e-9))
            tau0 = float(max(L/2.0, 1e-6))
            try:
                popt2, _ = curve_fit(lambda zz, AA, TT: AA*np.exp(-zz/TT),
                                     z, y_adj, p0=(A0, tau0), bounds=(lb2, ub2), maxfev=maxfev)
                yhat2 = popt2[0]*np.exp(-z/popt2[1]) + c0
                rss2 = float(np.sum((y - yhat2)**2))
                cand = (rss2, np.array([popt2[0], popt2[1], c0], float), yhat2)
                if (best_fb is None) or (rss2 < best_fb[0]):
                    best_fb = cand
            except Exception:
                continue
        if best is None:
            best = best_fb
        elif best_fb is not None and best_fb[0] < best[0]:
            best = best_fb

    if best is None:
        raise RuntimeError("Falha no ajuste exponencial para este run.")

    params, yhat = best[1], best[2]
    return params, yhat, t0

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    rss = float(np.sum((y - yhat)**2))
    tss = float(np.sum((y - np.mean(y))**2))
    return float(1.0 - rss / tss) if tss > 0 else np.nan

# ============================
# I/O dos dados
# ============================

def collect_runs(scen_dir: Path) -> List[Path]:
    return sorted([p for p in scen_dir.rglob("*_run_*_presas_por_passo.csv") if p.is_file()])

def load_xy_from_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).sort_values("passo")
    if not {"passo", "presas_vivas"} <= set(df.columns):
        raise ValueError(f"{path.name} precisa das colunas 'passo' e 'presas_vivas'.")
    c = df["presas_vivas"].to_numpy(float)
    t = df["passo"].to_numpy(float)
    m = np.isfinite(c) & np.isfinite(t) & (c > 0)
    x = t[m]; y = c[m]
    if x.size < 3:
        raise ValueError(f"Pontos insuficientes (>0) em {path.name}.")
    return x, y

# ============================
# Pipeline multi-runs + plot
# ============================

def analisar_todos_exp(
    base: Path,
    SCENARIO: str,
    n_obs: int,
    salvar_csv: bool = True,
    salvar_fig: bool = True,
    out_dir: Optional[Path] = None,
    overlay_runs: bool = False,
):
    """
    Percorre todos os *_run_*_presas_por_passo.csv em base/SCENARIO/s_obs_{n_obs},
    ajusta y = A*exp(-(t - t0)/tau) + C em cada run, agrega estatísticas e plota
    a curva média ±1σ com o ajuste exponencial usando parâmetros médios.
    """
    if out_dir is None:
        out_dir = base / "resultados_exp"
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_dir = base / SCENARIO / (f"s_obs_{n_obs}" if n_obs != 0 else "s_obs_00")
    if not scen_dir.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {scen_dir}")

    arquivos = collect_runs(scen_dir)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV *_run_*_presas_por_passo.csv em {scen_dir}")

    rows = []
    series_list = []

    for path in arquivos:
        try:
            x, y = load_xy_from_csv(path)
            (A, tau, C), yhat, t0 = fit_exp_plus_c(x, y)
            R2 = r2_score(y, yhat)

            rows.append({"arquivo": path.name, "t0": t0, "A": A, "tau": tau, "C": C, "R2": R2})

            # Série para média (index = passo)
            df_tmp = pd.read_csv(path, usecols=["passo", "presas_vivas"])
            df_tmp = df_tmp[df_tmp["presas_vivas"] > 0].set_index("passo").sort_index()
            series_list.append(df_tmp["presas_vivas"])
        except Exception:
            continue

    if not rows or not series_list:
        raise RuntimeError("Nenhum run válido após filtros.")

    df_runs = pd.DataFrame(rows).sort_values("arquivo")

    # Parâmetros médios
    A_med  = float(np.nanmean(df_runs["A"]))
    tau_med= float(np.nanmean(df_runs["tau"]))
    C_med  = float(np.nanmean(df_runs["C"]))
    R2_med = float(np.nanmean(df_runs["R2"]))

    # Curva média real e std ao longo dos passos
    df_all = pd.concat(series_list, axis=1)
    y_mean = df_all.mean(axis=1)
    y_std  = df_all.std(axis=1)
    x_vals = y_mean.index.values.astype(float)

    # -------- Plot --------
    plt.figure(figsize=(9, 6))
    plt.errorbar(x_vals, y_mean.values, yerr=y_std.values,
                 fmt='o', capsize=3, markersize=3, alpha=0.8,
                 label="Dados médios ±1σ")

    # Ajuste médio (usar t0 médio para a translação)
    t0_med = float(np.nanmean(df_runs["t0"]))
    x_fit = np.linspace(float(x_vals.min()), float(x_vals.max()), 400)
    z_fit = x_fit - t0_med
    z_fit[z_fit < 0] = 0  # evita valores antes do t0 médio
    y_fit = exp_offset(z_fit, A_med, tau_med, C_med)

    plt.plot(x_fit, y_fit, '-',
             label=f"Exp (médio): A={A_med:.2g}, τ={tau_med:.2g}, C={C_med:.2g} | R²≈{R2_med:.3f}")

    if overlay_runs:
        # Desenha as curvas ajustadas de cada run (fininhas)
        for _, r in df_runs.iterrows():
            t0i, Ai, taui, Ci = r["t0"], r["A"], r["tau"], r["C"]
            msk = x_fit >= t0i
            if not np.any(msk):
                continue
            yi = exp_offset(x_fit[msk] - t0i, Ai, taui, Ci)
            plt.plot(x_fit[msk], yi, lw=0.7, alpha=0.25)

    plt.xlabel("Passo")
    plt.ylabel("Presas vivas")
    plt.title(f"{SCENARIO} • s_obs_{n_obs if n_obs != 0 else '00'}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if salvar_fig:
        fig_base = f"exp_media_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}"
        plt.savefig(out_dir / f"{fig_base}.pdf", bbox_inches="tight")
        plt.savefig(out_dir / f"{fig_base}.png", bbox_inches="tight", dpi=300)
    plt.show()

    if salvar_csv:
        csv_name = f"params_por_run_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}.csv"
        df_runs.to_csv(out_dir / csv_name, index=False)

    # Resumo
    print("\n=== Resumo do ajuste exponencial (todos os runs) ===")
    print(f"A_med  = {A_med:.6g}")
    print(f"tau_med= {tau_med:.6g}")
    print(f"C_med  = {C_med:.6g}")
    print(f"R2_med = {R2_med:.6g}")
    print(f"Arquivo de parâmetros por run salvo em: {out_dir}")

    return {
        "df_runs": df_runs,
        "x_mean": x_vals,
        "y_mean": y_mean.values,
        "y_std": y_std.values,
        "params_mean": {"A": A_med, "tau": tau_med, "C": C_med, "t0": t0_med},
        "R2_med": R2_med
    }

# ============================
# Execução
# ============================

if __name__ == "__main__":
    base = Path.home() / "Dados_Doc"
    SCENARIO = "Nc=Np*0.5"    # ex.: "Nc=Np" ou "Nc=Np*0.5"
    n_obs = 0             # 0 -> s_obs_00; caso contrário s_obs_{n_obs}

    _ = analisar_todos_exp(
        base=base,
        SCENARIO=SCENARIO,
        n_obs=n_obs,
        salvar_csv=True,
        salvar_fig=True,
        out_dir=base / "resultados_exp",
        overlay_runs=True,   # True para visualizar as curvas de cada run
    )
