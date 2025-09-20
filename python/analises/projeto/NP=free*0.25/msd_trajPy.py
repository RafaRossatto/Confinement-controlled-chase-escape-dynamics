#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD Ensemble-Averaged - Com janela de fit opcional
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

# -------- Configurações --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
base = BASE_ROOT/"Nc=Np*0.5/s_obs_13107"
pattern = "*_hunter_trajectories.csv"
Lx = Ly = 128
min_traj_length = 5
out_prefix = "msd_ensemble_hunter_obs_13107_Nc=NP*0.5"

# JANELA DE FIT - MODIFICAR AQUI!
# Se None, usa todos os pontos válidos. Se definido, usa [tau_min, tau_max]
JANELA_FIT = None  # Exemplos: None, [5, 30], [1, 20]

# Saídas
OUT_DIR = BASE_ROOT / "resultados_modelos" / "msd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Funções de Unwrapping --------
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

# -------- Função para calcular MSD por arquivo --------
def calcular_msd_por_arquivo(trajetorias, max_tau):
    """
    Calcula MSD ensemble para um conjunto de trajetórias (um arquivo)
    """
    msd_ensemble = np.zeros(max_tau)
    n_contrib = np.zeros(max_tau, dtype=int)
    
    for tau in range(1, max_tau + 1):
        msd_values = []
        for traj in trajetorias:
            if len(traj) > tau:
                dr = traj[tau] - traj[0]
                msd_val = np.sum(dr**2)
                msd_values.append(msd_val)
        
        if msd_values:
            msd_ensemble[tau-1] = np.mean(msd_values)
            n_contrib[tau-1] = len(msd_values)
        else:
            msd_ensemble[tau-1] = np.nan
            n_contrib[tau-1] = 0
    
    return msd_ensemble, n_contrib

# -------- Função para calcular α com janela opcional --------
def calcular_alpha(taus, msd, janela_fit=None):
    """
    Calcula expoente α a partir de MSD vs tau
    janela_fit: [tau_min, tau_max] ou None para usar todos os pontos
    """
    # Primeiro filtro: valores válidos
    mask = (msd > 0) & np.isfinite(msd)
    
    # Segundo filtro: janela temporal se especificada
    if janela_fit is not None:
        tau_min, tau_max = janela_fit
        mask_janela = (taus >= tau_min) & (taus <= tau_max)
        mask = mask & mask_janela
        #print(f"  Janela de fit: τ = [{tau_min}, {tau_max}]")
    #else:
     #   print(f"  Janela de fit: todos os pontos válidos")
    
    if np.sum(mask) < 3:
        print(f"  Pontos insuficientes para fit: {np.sum(mask)}")
        return np.nan, np.nan, np.nan, mask
    
    try:
        x = np.log(taus[mask])
        y = np.log(msd[mask])
        a1, a0 = np.polyfit(x, y, 1)
        
        # Calcula R²
        y_pred = a1 * x + a0
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        #print(f"  α = {a1:.3f}, R² = {r2:.3f}, pontos = {len(x)}")
        return a1, np.exp(a0), r2, mask  # α, K, R², máscara usada
        
    except Exception as e:
        print(f"  Erro no fit: {e}")
        return np.nan, np.nan, np.nan, mask

# -------- Função Principal --------
def main():
    # Configurações
    max_tau = 50  # Número máximo de pontos para MSD
    
    print(f" Configuração da janela de fit: {JANELA_FIT}")
    
    # Coleta arquivos
    arquivos = sorted(base.glob(pattern))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {base}")
    
    print(f"Encontrados {len(arquivos)} arquivos")
    
    # Dados por arquivo
    dados_por_arquivo = []
    todos_msd_por_arquivo = []
    
    for arq in arquivos:
        #print(f"\n Processando: {arq.name}")
        
        try:
            df = pd.read_csv(arq, usecols=["timestep", "cell_id", "x", "y"])
            df = df.sort_values(["cell_id", "timestep"])
            
            # Coleta todas as trajetórias deste arquivo
            trajetorias_arquivo = []
            comprimentos = []
            
            for cid, g in df.groupby("cell_id"):
                t = g["timestep"].to_numpy(float)
                x = g["x"].to_numpy(float)
                y = g["y"].to_numpy(float)
                
                # Unwrapping
                x, y = unwrap_xy(x, y, Lx, Ly)
                
                # Cria array no formato [x, y]
                traj = np.column_stack([x, y])
                
                if len(traj) >= min_traj_length:
                    trajetorias_arquivo.append(traj)
                    comprimentos.append(len(traj))
            
            if not trajetorias_arquivo:
                print(f"  Nenhuma trajetória válida")
                continue
            
            #print(f"  Trajetórias: {len(trajetorias_arquivo)}, comprimento médio: {np.mean(comprimentos):.1f}")
            
            # Calcula MSD ensemble para este arquivo
            msd_ensemble, n_contrib = calcular_msd_por_arquivo(trajetorias_arquivo, max_tau)
            taus = np.arange(1, max_tau + 1)
            
            # Filtra valores válidos
            mask = (n_contrib >= 5) & np.isfinite(msd_ensemble)
            taus_valid = taus[mask]
            msd_valid = msd_ensemble[mask]
            
            # Calcula α para este arquivo com janela opcional
            alpha, K, r2, mask_fit = calcular_alpha(taus_valid, msd_valid, JANELA_FIT)
            
            dados_arquivo = {
                'arquivo': arq.name,
                'n_trajetorias': len(trajetorias_arquivo),
                'comprimento_medio': np.mean(comprimentos),
                'alpha': alpha,
                'K': K,
                'r2': r2,
                'pontos_fit': np.sum(mask_fit) if mask_fit is not None else np.sum(mask)
            }
            
            dados_por_arquivo.append(dados_arquivo)
            todos_msd_por_arquivo.append((taus_valid, msd_valid, alpha, arq.name, mask_fit))
            
        except Exception as e:
            print(f"Erro processando {arq.name}: {e}")
            continue
    
    if not dados_por_arquivo:
        raise RuntimeError("Nenhum dado válido encontrado")
    
    print(f" Total de arquivos processados: {len(dados_por_arquivo)}")
    
    # -------- Salva dados por arquivo em CSV --------
    df_arquivos = pd.DataFrame(dados_por_arquivo)
    arquivos_csv_path = OUT_DIR / f"{out_prefix}.csv"
    df_arquivos.to_csv(arquivos_csv_path, index=False, float_format='%.6f')
    
    alphas_valid = df_arquivos['alpha'].dropna()
    """
    if len(alphas_valid) > 0:
        print(f"Média: {alphas_valid.mean():.4f} ± {alphas_valid.std():.4f}")
        print(f"Mínimo: {alphas_valid.min():.4f}")
        print(f"Máximo: {alphas_valid.max():.4f}")
        print(f"Mediana: {alphas_valid.median():.4f}")
        print(f"Número de arquivos com fit válido: {len(alphas_valid)}/{len(dados_por_arquivo)}")
    else:
        print("Nenhum fit válido encontrado!")
        """
    
    # -------- Plot dos MSDs por Arquivo --------
    
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. MSDs por arquivo em escala linear com região de fit destacada
    colors = plt.cm.viridis(np.linspace(0, 1, len(todos_msd_por_arquivo)))
    
    for i, (taus, msd, alpha, nome_arquivo, mask_fit) in enumerate(todos_msd_por_arquivo):
        color = colors[i]
        alpha_val = 0.6
        linewidth = 1.5
        
        # Plota toda a curva
        ax1.plot(taus, msd, color=color, alpha=alpha_val, linewidth=linewidth)
        
        # Destaca região do fit
        if mask_fit is not None and np.any(mask_fit):
            ax1.plot(taus[mask_fit], msd[mask_fit], color=color, 
                    alpha=0.8, linewidth=linewidth+1)
    
    ax1.set_xlabel('Time Lag τ (frames)')
    ax1.set_ylabel('MSD(τ)')
    ax1.set_title(f'MSD por Arquivo ({len(todos_msd_por_arquivo)} arquivos)\nVermelho: região do fit')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # 3. Histograma dos expoentes α por arquivo
    if len(alphas_valid) > 0:
        ax3.hist(alphas_valid, bins=30, alpha=0.7, edgecolor='black', density=True)
        #ax3.axvline(1.0, color='red', linestyle='--', label='α=1 (Difusão normal)', linewidth=2)
        ax3.axvline(alphas_valid.mean(), color='blue', linestyle='-', 
                   label=f'Média: α={alphas_valid.mean():.3f}', linewidth=2)
        
        ax3.set_xlabel('Expoente α')
        ax3.set_ylabel('Densidade de probabilidade')
        ax3.set_title(f'Distribuição dos Expoentes α\n(Janela: {JANELA_FIT})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Nenhum fit válido', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
    
    plt.tight_layout()
    plot_path = OUT_DIR / f"{out_prefix}_por_arquivo.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
 
if __name__ == "__main__":
    main()