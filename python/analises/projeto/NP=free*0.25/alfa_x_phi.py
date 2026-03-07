#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

# -------- Configurações --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"
MSD_DIR   = BASE_ROOT / "resultados_modelos" / "paper_response"/"msd_trajPy"
OUT_DIR   = BASE_ROOT / "resultados_modelos" / "paper_response"/ "msd_fit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Concentrações e obstáculos
CONCENTRACOES = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]
OBSTACULOS = [
    "s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915",
    "s_obs_6553","s_obs_8028", "s_obs_8192","s_obs_8355","s_obs_9666", "s_obs_9830","s_obs_9994",
    "s_obs_11468", "s_obs_13107"
]

# -------- Definição de janelas --------
# Você pode ajustar individualmente depois
FIT_WINDOWS = {
    "Nc=Np*0.5": 
    {
        "s_obs_00": (1, 13),
        "s_obs_1638": (1, 16),
        "s_obs_3276": (1, 16),
        "s_obs_4915": (1, 18),
        "s_obs_6553": (1, 18),
        "s_obs_8028": (1, 25),
        "s_obs_8192": (1, 22),
        "s_obs_8355": (1, 25),
        "s_obs_9666": (1, 22),
        "s_obs_9830": (1, 22),
        "s_obs_9994": (1, 20),
        "s_obs_11468": (1, 10),
        "s_obs_13107": (1, 5)
    },
    "Nc=Np*0.8": 
    {
        "s_obs_00": (1, 7),
        "s_obs_1638": (1, 9),
        "s_obs_3276": (1, 11),
        "s_obs_4915": (1, 12),
        "s_obs_6553": (1, 13),
        "s_obs_8028": (1, 11),
        "s_obs_8192": (1, 15),
        "s_obs_8355": (1, 13),
        "s_obs_9666": (1, 14),
        "s_obs_9830": (1, 14),
        "s_obs_9994": (1, 10),
        "s_obs_11468": (1, 10),
        "s_obs_13107": (1, 5)
    },
    "Nc=Np": 
    {
        "s_obs_00": (1, 5),
        "s_obs_1638": (1, 6),
        "s_obs_3276": (1, 6),
        "s_obs_4915": (1, 8),
        "s_obs_6553": (1, 9),
        "s_obs_8028": (1, 10),
        "s_obs_8192": (1, 8),
        "s_obs_8355": (1, 10),
        "s_obs_9666": (1, 8),
        "s_obs_9830": (1, 10),
        "s_obs_9994": (1, 8),
        "s_obs_11468": (1, 6),
        "s_obs_13107": (1, 5)
    }
}

# -------- Função para ajuste --------
def fit_power_law(df, fit_range):
    """Faz o fit MSD ~ t^alpha dentro da janela definida."""
    t = df["tau"].values
    msd = df["msd_ensemble"].values

    mask = (t >= fit_range[0]) & (t <= fit_range[1])
    t_fit = t[mask]
    msd_fit = msd[mask]

    if len(t_fit) < 2:
        raise ValueError("Poucos pontos na janela para ajuste.")

    log_t = np.log(t_fit)
    log_msd = np.log(msd_fit)

    slope, intercept, r_value, p_value, std_err = linregress(log_t, log_msd)
    return slope, intercept, r_value**2, std_err

# -------- Loop principal --------
resultados = []

for conc in CONCENTRACOES:
    for obs in OBSTACULOS:
        obs_dir = MSD_DIR / conc / obs
        if not obs_dir.exists():
            print(f"[WARN] Pasta não encontrada: {obs_dir}")
            continue

        fit_range = FIT_WINDOWS.get(conc, {}).get(obs, (10, 1000))
        alphas = []

        for msd_file in obs_dir.glob("*_msd.csv"):
            df = pd.read_csv(msd_file)
            try:
                alpha, intercept, r2, err = fit_power_law(df, fit_range)
                alphas.append(alpha)
            except Exception as e:
                print(f"[ERRO] Fit falhou para {msd_file}: {e}")

        if alphas:
            alpha_mean = np.mean(alphas)
            alpha_std = np.std(alphas, ddof=1)  # desvio padrão amostral
            resultados.append({
                "cenario": conc,
                "obs": obs,
                "t_min": fit_range[0],
                "t_max": fit_range[1],
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
                "n_runs": len(alphas)
            })

# Salva resultados
df_out = pd.DataFrame(resultados)
df_out.to_csv(OUT_DIR / "msd_fit_results.csv", index=False)
print(">> Resultados salvos em:", OUT_DIR / "msd_fit_results.csv")
