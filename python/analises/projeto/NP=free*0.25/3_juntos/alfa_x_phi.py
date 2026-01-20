#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress

# -------- CONFIGURAÇÕES PRINCIPAIS --------
# Diferentes tamanhos de rede - ⬅️ ALTERE OS CAMINHOS AQUI
REDE_CONFIGS = [
    (64,  "L_64",  Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"),   # ⬅️ ALTERE  
    (256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")    # ⬅️ ALTERE
]

# Apenas a proporção 0.5
FRAC_C = "Nc=Np*0.5"

# Lista de obstáculos para cada tamanho de rede (ajuste conforme necessário)
OBS_TEMPLATES = {
    64: ["s_obs_00", "s_obs_409", "s_obs_819", "s_obs_1228", "s_obs_1638", "s_obs_2048", "s_obs_2416","s_obs_2457","s_obs_2498","s_obs_2867","s_obs_3276"],
    128: ["s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", "s_obs_6553","s_obs_8192","s_obs_9666", "s_obs_9830","s_obs_9994", "s_obs_11468", "s_obs_13107"],
    256: ["s_obs_00", "s_obs_6553", "s_obs_13107", "s_obs_19660", "s_obs_26214", "s_obs_32768","s_obs_38666","s_obs_39321","s_obs_39976","s_obs_45875","s_obs_52428"]
}

# Diretórios
BASE_ROOT = Path.home() / "Dados_Doc" / "resultados_modelos"
MSD_DIR_BASE = BASE_ROOT / "msd_trajPy_multiple_L"  # Onde estão os dados MSD
OUT_DIR_BASE = BASE_ROOT / "msd_fit_multiple_L"     # Onde salvar os resultados
OUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# -------- DEFINIÇÃO DE JANELAS DE AJUSTE --------
# Você pode ajustar individualmente para cada tamanho de rede
FIT_WINDOWS = {
    # Para L64
    64: {
        "s_obs_00": (1, 11),
        "s_obs_409": (1, 14),
        "s_obs_819": (1, 15),
        "s_obs_1228": (1, 15),
        "s_obs_1638": (1, 16),
        "s_obs_2048": (1, 16),
        "s_obs_2416": (1, 14),
        "s_obs_2457": (1, 14),
        "s_obs_2498": (1, 14),
        "s_obs_2867": (1, 7),
        "s_obs_3276": (1, 5)
    },
    # Para L128 (valores originais adaptados)
    128: {
        "s_obs_00": (1, 13),
        "s_obs_1638": (1, 16),
        "s_obs_3276": (1, 16),
        "s_obs_4915": (1, 18),
        "s_obs_6553": (1, 18),
        "s_obs_8192": (1, 22),
        "s_obs_9666": (1, 22),
        "s_obs_9830": (1, 22),
        "s_obs_9994": (1, 20),
        "s_obs_11468": (1, 10),
        "s_obs_13107": (1, 5)
    },
    # Para L256 (valores estimados - ajuste conforme necessário)
    256: {
        "s_obs_00": (1, 18),
        "s_obs_6553": (1, 20),
        "s_obs_13107": (1, 20),
        "s_obs_19660": (1, 24),
        "s_obs_26214": (1, 28),
        "s_obs_32768": (1, 35),
        "s_obs_38666": (1, 35),
        "s_obs_39321": (1, 25),
        "s_obs_39976": (1, 30),
        "s_obs_45875": (1, 15),
        "s_obs_52428": (1, 7)
    }
}

# -------- FUNÇÃO PARA AJUSTE (MANTIDA) --------
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
    
    # Calcular também o coeficiente de difusão D = exp(intercept) / 4 para 2D
    D = np.exp(intercept) / 4.0
    
    return slope, intercept, r_value**2, std_err, D

# -------- LOOP PRINCIPAL MODIFICADO --------
def processar_rede(L, rede_nome, base_path):
    """Processa uma rede específica."""
    print(f"\n🎯 Processando rede: {rede_nome} (L={L})")
    
    # Diretório de saída para esta rede
    out_dir_rede = OUT_DIR_BASE / rede_nome
    out_dir_rede.mkdir(parents=True, exist_ok=True)
    
    # Lista de obstáculos para este L
    obs_list = OBS_TEMPLATES.get(L, [f"s_obs_{i}" for i in range(0, L**2, L**2//8)])
    
    # Janelas de ajuste para este L
    fit_windows_rede = FIT_WINDOWS.get(L, {})
    
    resultados = []
    
    for obs in obs_list:
        # Diretório dos dados MSD: MSD_DIR_BASE / rede_nome / FRAC_C / obs
        obs_dir = MSD_DIR_BASE / rede_nome / FRAC_C / obs
        
        if not obs_dir.exists():
            print(f"[WARN] Pasta não encontrada: {obs_dir}")
            continue

        # Usa janela específica ou padrão
        fit_range = fit_windows_rede.get(obs, (1, min(50, L//2)))  # padrão conservador
        
        alphas = []
        Ds = []
        r2s = []

        for msd_file in obs_dir.glob("*_msd.csv"):
            try:
                df = pd.read_csv(msd_file)
                alpha, intercept, r2, err, D = fit_power_law(df, fit_range)
                alphas.append(alpha)
                Ds.append(D)
                r2s.append(r2)
            except Exception as e:
                print(f"[ERRO] Fit falhou para {msd_file}: {e}")
                continue

        if alphas:
            alpha_mean = np.mean(alphas)
            alpha_std = np.std(alphas, ddof=1)
            D_mean = np.mean(Ds)
            D_std = np.std(Ds, ddof=1)
            r2_mean = np.mean(r2s)
            
            resultados.append({
                "rede": rede_nome,
                "L": L,
                "cenario": FRAC_C,
                "obs": obs,
                "t_min": fit_range[0],
                "t_max": fit_range[1],
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
                "D_mean": D_mean,
                "D_std": D_std,
                "r2_mean": r2_mean,
                "n_runs": len(alphas)
            })
            
            print(f"  ✅ {obs}: α = {alpha_mean:.3f} ± {alpha_std:.3f} (n={len(alphas)})")
        else:
            print(f"  ❌ {obs}: Nenhum ajuste válido")

    return resultados

# -------- EXECUÇÃO PRINCIPAL --------
if __name__ == "__main__":
    todos_resultados = []
    
    for L, rede_nome, base_path in REDE_CONFIGS:
        # Verifica se a base existe
        if not base_path.exists():
            print(f"❌ ERRO: Pasta base não encontrada: {base_path}")
            print(f"   Por favor, ajuste o caminho em REDE_CONFIGS")
            continue
            
        # Processa esta rede
        resultados_rede = processar_rede(L, rede_nome, base_path)
        
        if resultados_rede:
            # Salva resultados individuais da rede
            df_rede = pd.DataFrame(resultados_rede)
            csv_rede = OUT_DIR_BASE / rede_nome / f"{rede_nome}_{FRAC_C}_msd_fit_results.csv"
            df_rede.to_csv(csv_rede, index=False)
            print(f"💾 Resultados de {rede_nome} salvos em: {csv_rede}")
            
            todos_resultados.extend(resultados_rede)
    
    # Salva todos os resultados combinados
    if todos_resultados:
        df_todos = pd.DataFrame(todos_resultados)
        csv_todos = OUT_DIR_BASE / f"TODOS_{FRAC_C}_msd_fit_results.csv"
        df_todos.to_csv(csv_todos, index=False)
        print(f"\n💾 Todos os resultados salvos em: {csv_todos}")
    
    print(f"\n✅ Processamento concluído! Resultados em: {OUT_DIR_BASE}")