#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD Ensemble-Averaged - Loop para todas as concentrações e obstáculos
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors


# -------- Configurações Gerais --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
pattern = "*_hunter_trajectories.csv"
Lx = Ly = 128
min_traj_length = 5

# JANELA DE FIT
JANELA_FIT = None  # Exemplos: None, [5, 30], [1, 20]

# Saídas
OUT_DIR = BASE_ROOT / "resultados_modelos" / "msd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Lista de Concentrações e Obstáculos --------
CONCENTRACOES = [
    "Nc=Np*0.5",
    "Nc=Np*0.8", 
    "Nc=Np"
]

# -------- Lista de Obstáculos para Processar --------
# Defina aqui todos os diretórios de obstáculos que você quer processar
OBSTACULOS = [
    "s_obs_00",
    "s_obs_1638", 
    "s_obs_3276",
    "s_obs_4915",
    "s_obs_6553",
    "s_obs_8192",
    "s_obs_9830",
    "s_obs_11468",
    "s_obs_13107"
]

# -------- Funções (mantidas do código anterior) --------
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

def calcular_msd_por_arquivo(trajetorias, max_tau):
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

def calcular_alpha(taus, msd, janela_fit=None):
    mask = (msd > 0) & np.isfinite(msd)
    
    if janela_fit is not None:
        tau_min, tau_max = janela_fit
        mask_janela = (taus >= tau_min) & (taus <= tau_max)
        mask = mask & mask_janela
    
    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, mask
    
    try:
        x = np.log(taus[mask])
        y = np.log(msd[mask])
        a1, a0 = np.polyfit(x, y, 1)
        
        y_pred = a1 * x + a0
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        return a1, np.exp(a0), r2, mask
        
    except Exception as e:
        return np.nan, np.nan, np.nan, mask

# -------- Função para processar uma concentração e obstáculo --------
def processar_concentracao_obstaculo(concentracao, obstaculo_dir, janela_fit=None):
    """
    Processa um par concentração/obstáculo
    """
    base = BASE_ROOT / concentracao / obstaculo_dir
    out_prefix = f"msd_ensemble_hunter_{obstaculo_dir}_{concentracao.replace('*', '_')}"
    
    print(f"\n{'='*60}")
    print(f" PROCESSANDO: {concentracao} / {obstaculo_dir}")
    print(f"{'='*60}")
    
    # Verifica se o diretório existe
    if not base.exists():
        print(f" Diretório não encontrado: {base}")
        return None
    
    # Coleta arquivos
    arquivos = sorted(base.glob(pattern))
    if not arquivos:
        print(f" Nenhum arquivo encontrado em {base}")
        return None
    
    print(f" Encontrados {len(arquivos)} arquivos")
    
    # Dados por arquivo
    dados_por_arquivo = []
    todos_msd_por_arquivo = []
    
    for arq in arquivos:
        try:
            df = pd.read_csv(arq, usecols=["timestep", "cell_id", "x", "y"])
            df = df.sort_values(["cell_id", "timestep"])
            
            trajetorias_arquivo = []
            comprimentos = []
            
            for cid, g in df.groupby("cell_id"):
                t = g["timestep"].to_numpy(float)
                x = g["x"].to_numpy(float)
                y = g["y"].to_numpy(float)
                
                x, y = unwrap_xy(x, y, Lx, Ly)
                traj = np.column_stack([x, y])
                
                if len(traj) >= min_traj_length:
                    trajetorias_arquivo.append(traj)
                    comprimentos.append(len(traj))
            
            if not trajetorias_arquivo:
                continue
            
            # Calcula MSD ensemble
            max_tau = 50
            msd_ensemble, n_contrib = calcular_msd_por_arquivo(trajetorias_arquivo, max_tau)
            taus = np.arange(1, max_tau + 1)
            
            mask = (n_contrib >= 5) & np.isfinite(msd_ensemble)
            taus_valid = taus[mask]
            msd_valid = msd_ensemble[mask]
            
            # Calcula α
            alpha, K, r2, mask_fit = calcular_alpha(taus_valid, msd_valid, janela_fit)
            
            dados_arquivo = {
                'concentracao': concentracao,
                'obstaculo': obstaculo_dir,
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
            print(f" Erro processando {arq.name}: {e}")
            continue
    
    if not dados_por_arquivo:
        print(f" Nenhum dado válido encontrado para {concentracao}/{obstaculo_dir}")
        return None
    
    # Salva dados
    df_arquivos = pd.DataFrame(dados_por_arquivo)
    arquivos_csv_path = OUT_DIR / f"{out_prefix}.csv"
    df_arquivos.to_csv(arquivos_csv_path, index=False, float_format='%.6f')
    
    alphas_valid = df_arquivos['alpha'].dropna()
       
    # Plot individual (opcional - comente se não quiser plots individuais)
    if len(alphas_valid) > 0:
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(todos_msd_por_arquivo)))
        
        for i, (taus, msd, alpha, nome_arquivo, mask_fit) in enumerate(todos_msd_por_arquivo):
            color = colors[i]
            ax1.plot(taus, msd, color=color, alpha=0.6, linewidth=1.5)
            
            if mask_fit is not None and np.any(mask_fit):
                ax1.plot(taus[mask_fit], msd[mask_fit], color=color, alpha=0.9, linewidth=2)
        
        ax1.set_xlabel('Time Lag τ (frames)')
        ax1.set_ylabel('MSD(τ)')
        ax1.set_title(f'MSD - {concentracao}/{obstaculo_dir}\n({len(todos_msd_por_arquivo)} arquivos)')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        ax3.hist(alphas_valid, bins=30, alpha=0.7, edgecolor='black', density=True)
        ax3.axvline(alphas_valid.mean(), color='blue', linestyle='-', 
                   label=f'Média: α={alphas_valid.mean():.3f}', linewidth=2)
        
        ax3.set_xlabel('Expoente α')
        ax3.set_ylabel('Densidade')
        ax3.set_title(f'Distribuição α - {concentracao}/{obstaculo_dir}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUT_DIR / f"{out_prefix}_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico salvo: {plot_path}")
    
    return {
        'concentracao': concentracao,
        'obstaculo': obstaculo_dir,
        'media_alpha': alphas_valid.mean() if len(alphas_valid) > 0 else np.nan,
        'std_alpha': alphas_valid.std() if len(alphas_valid) > 0 else np.nan,
        'n_arquivos': len(dados_por_arquivo),
        'n_validos': len(alphas_valid)
    }

# -------- Função Principal --------
def main():
    print(" INICIANDO PROCESSAMENTO DE TODAS AS CONCENTRAÇÕES E OBSTÁCULOS")
    print(f" Diretório base: {BASE_ROOT}")
    print(f" Janela de fit: {JANELA_FIT}")
    print(f" Saída: {OUT_DIR}")
    
    resultados_gerais = []
    
    # Loop por todas as concentrações e obstáculos
    for concentracao in CONCENTRACOES:
        for obstaculo in OBSTACULOS:
            resultado = processar_concentracao_obstaculo(concentracao, obstaculo, JANELA_FIT)
            if resultado is not None:
                resultados_gerais.append(resultado)
    
    # Salva resumo geral
    if resultados_gerais:
        df_resumo = pd.DataFrame(resultados_gerais)
        resumo_path = OUT_DIR / "resumo_geral_completo.csv"
        df_resumo.to_csv(resumo_path, index=False, float_format='%.6f')
               
        # Plot comparativo por concentração
        fig, axes = plt.subplots(1, len(CONCENTRACOES), figsize=(18, 6))
        if len(CONCENTRACOES) == 1:
            axes = [axes]
        
        for i, conc in enumerate(CONCENTRACOES):
            resultados_conc = [r for r in resultados_gerais if r['concentracao'] == conc]
            
            if resultados_conc:
                obstaculos = [r['obstaculo'] for r in resultados_conc]
                medias = [r['media_alpha'] for r in resultados_conc]
                stds = [r['std_alpha'] for r in resultados_conc]
                
                x_pos = np.arange(len(obstaculos))
                bars = axes[i].bar(x_pos, medias, yerr=stds, alpha=0.7, capsize=5)
                
                axes[i].set_xlabel('Obstáculo')
                axes[i].set_ylabel('Expoente α médio')
                axes[i].set_title(f'Concentração: {conc}')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(obstaculos, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3, axis='y')
                
                for bar, media in zip(bars, medias):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{media:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        comparativo_path = OUT_DIR / "comparativo_completo.png"
        plt.savefig(comparativo_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico comparativo salvo: {comparativo_path}")
    
    print(f"\n PROCESSAMENTO CONCLUÍDO!")

if __name__ == "__main__":
    main()