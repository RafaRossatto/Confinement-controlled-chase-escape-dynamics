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

def fit_exp_c_last(path: Path,
                   tau_factor: float = 20.0,
                   maxfev: int = 20000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Ajusta y = A*exp(-(t - t0)/tau) + C para um run.

    - Lê o CSV diretamente.
    - C é sempre o último valor real de 'presas_vivas' no arquivo (mesmo que seja 0).
    - Depois remove zeros apenas para facilitar o ajuste.
    - Retorna (params[A,tau,C], yhat_completo, t0), onde
      yhat_completo inclui também o ponto final C.
    """
    # Lê o CSV inteiro
    df = pd.read_csv(path, usecols=["passo", "presas_vivas"]).sort_values("passo")

    # Pega o último valor bruto como C (sem filtro)
    C = float(df["presas_vivas"].iloc[-1])

    # Remove zeros apenas para o ajuste (mas sem afetar o C)
    df_pos = df[df["presas_vivas"] > 0]
    x = df_pos["passo"].to_numpy(float)
    y = df_pos["presas_vivas"].to_numpy(float)
    if x.size < 3:
        raise ValueError(f"Pontos insuficientes (>0) em {path.name}.")

    # Recentrar tempo
    t0 = float(x.min())
    z = x - t0
    L = float(np.ptp(z)) if np.ptp(z) > 0 else 1.0
    tau_max = max(1e-6, tau_factor * L)

    # Chute inicial
    A0 = max(y[0] - C, 1e-9)
    tau0 = L / 2.0

    # Ajuste com A e tau livres, C fixo
    popt, _ = curve_fit(lambda zz, AA, TT: AA * np.exp(-zz / TT) + C,
                        z, y, p0=(A0, tau0),
                        bounds=([0, 1e-6], [10 * y[0], tau_max]),
                        maxfev=maxfev)
    A, tau = popt

    # Predição apenas nos pontos >0
    yhat = A * np.exp(-z / tau) + C

    # Constrói yhat completo (mesmo tamanho do CSV original)
    yhat_completo = np.concatenate([yhat, [C]])

    return (A, tau, C), yhat_completo, t0


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

        # Primeiro: descobrir o maior tamanho entre todos os runs
    max_len = 0
    for path in arquivos:
        n_linhas = len(pd.read_csv(path))
        if n_linhas > max_len:
            max_len = n_linhas

    for path in arquivos:
        print(f"[debug] Total de arquivos encontrados: {len(arquivos)}")
        for p in arquivos:
            print(" -", p.name)
        try:
            params, yhat, t0 = fit_exp_c_last(path)
            A, tau, C = params   # <<< extrai os parâmetros

            # --- Lê os dados reais ---
            df_ref = pd.read_csv(path, usecols=["passo", "presas_vivas"])
            y_true = df_ref["presas_vivas"].to_numpy(float)

            # --- Inclui o ponto final C no predito ---
            y_pred = np.concatenate([yhat, [C]])

            # --- Ajuste de comprimento (padding com último valor) ---
            max_len = max(len(y_true), len(y_pred))
            if len(y_true) < max_len:
                y_true = np.pad(y_true, (0, max_len - len(y_true)),
                                mode="edge")  # repete último valor
            if len(y_pred) < max_len:
                y_pred = np.pad(y_pred, (0, max_len - len(y_pred)),
                                mode="edge")

            # --- Calcula R² ---
            R2 = r2_score(y_true, y_pred)

            rows.append({"arquivo": path.name, "t0": t0,
                        "A": A, "tau": tau, "C": C, "R2": R2})

            # Mantém a série com presas > 0
            df_tmp = df_ref[df_ref["presas_vivas"] > 0].set_index("passo").sort_index()
            series_list.append(df_tmp["presas_vivas"])

        except Exception as e:
            print(f"[ERRO] {path.name}: {e}")
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

    plt.xlabel("steps")
    plt.ylabel("preys alives")
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

    # Mostra todos os valores de C de cada run
    print("\n=== Valores individuais de C (um por run) ===")
    print(df_runs[["arquivo", "C"]])

    # Mostra a média e desvio padrão sem arredondar
    print("\n=== Estatísticas de C ===")
    print("Média bruta:", df_runs["C"].mean())
    print("Desvio padrão:", df_runs["C"].std())
    print("Valores únicos:", sorted(df_runs["C"].unique()))
    # Resumo
    print("\n=== Resumo do ajuste exponencial (todos os runs) ===")
    print(f"A_med  = {A_med:.6g}")
    print(f"tau_med= {tau_med:.6g}")
    print(f"C_med  = {C_med:.2f}")
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
    SCENARIO = "Nc=Np"    # ex.: "Nc=Np" ou "Nc=Np*0.5"
    n_obs = 14745  # 0, 1638, 3276, 4915, 6553, 8192, 9830, 11468, 13107, 14745

    _ = analisar_todos_exp(
        base=base,
        SCENARIO=SCENARIO,
        n_obs=n_obs,
        salvar_csv=True,
        salvar_fig=True,
        out_dir=base / "resultados_exp",
        overlay_runs=False,   # True para visualizar as curvas de cada run
    )
