#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import logging
import re

# ---------------------- Configuração de Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------- Config ----------------------
class Config:
    BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128"
    OUT_DIR = BASE_ROOT / "resultados_modelos" / "custo_dinamico"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Cenários (subpastas principais)
    CENARIOS = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]
    
    # Lista de pastas de observação (dentro de cada cenário)
    OBS_FOLDERS = [
        "s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", 
        "s_obs_6553","s_obs_8028", "s_obs_8192","s_obs_8355", "s_obs_9666", "s_obs_9830",
        "s_obs_9994", "s_obs_11468", "s_obs_13107"
    ]
    
    # Configurações do gráfico
    PLOT_PARAMS = {
        "cmap_cenarios": "flag",  # Para diferentes cenários
        "cmap_obs": "viridis",     # Para diferentes densidades φ
        "figsize": (12, 8),
        "linewidth": 2.5,
        "alpha": 0.8,
        "grid_alpha": 0.3
    }
    
    # Configuração de fonte
    plt.rcParams.update({
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "font.family": "serif"
    })

config = Config()

# ---------------------- Funções auxiliares ----------------------
def parse_filename(filename):
    """
    Extrai informações do nome do arquivo.
    Exemplo: NC_409_NE_819_O_13107_TCC_1.000000_SR_2_TCT_1.000000_SR_2.dat_run_0_prey_per_step.csv
    """
    pattern = r"NC_(\d+)_NE_(\d+)_O_(\d+)_TCC_([\d\.]+)_SR_(\d+)_TCT_([\d\.]+)_SR_(\d+)\.dat_run_(\d+)_prey_per_step\.csv"
    match = re.match(pattern, filename)
    
    if match:
        return {
            "NC": int(match.group(1)),
            "NE": int(match.group(2)),
            "OBS": int(match.group(3)),
            "TCC": float(match.group(4)),
            "SR_TCC": int(match.group(5)),
            "TCT": float(match.group(6)),
            "SR_TCT": int(match.group(7)),
            "RUN": int(match.group(8))
        }
    return None

def calcular_densidade(obs_num):
    """Calcula a densidade φ a partir do número de obstáculos"""
    return obs_num / (128 * 128)

def identificar_coluna_prey(df):
    """
    Identifica automaticamente a coluna que contém a contagem de presas
    """
    # Primeiro, procurar por padrões conhecidos
    prey_patterns = ['prey', 'ne', 'presa', 'count', 'numero', 'n_', 'npresas']
    
    for col in df.columns:
        col_lower = str(col).lower()
        for pattern in prey_patterns:
            if pattern in col_lower:
                # Excluir colunas de metadados
                if col_lower not in ['neo', 'obs', 'run', 'step', 'time', 't']:
                    return col
    
    # Se não encontrar, usar a primeira coluna numérica que não seja de metadados
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    meta_cols = ['step', 'time', 't', 'iteration', 'run', 'obs_num', 'phi', 'NEO']
    
    for col in numeric_cols:
        if col not in meta_cols:
            return col
    
    # Último recurso: usar a segunda coluna (assumindo que a primeira é tempo)
    if len(df.columns) > 1:
        return df.columns[1]
    
    return None

def calcular_custo_dinamico_por_arquivo(df, file_info, cenario):
    """
    Calcula o custo dinâmico para um arquivo individual
    c(t) = f * (N_EO / N_E(t)) * t
    
    Onde f é determinado pelo cenário:
    - Nc=Np*0.5 → f = 0.5
    - Nc=Np*0.8 → f = 0.8  
    - Nc=Np → f = 1.0
    """
    results = []
    
    # Identificar coluna de presas
    prey_col = identificar_coluna_prey(df)
    if prey_col is None:
        logger.warning(f"Não encontrei coluna de presas")
        return pd.DataFrame()
    
    # Obter N_EO (valor inicial de NE do nome do arquivo)
    N_EO = file_info["NE"]
    
    # Determinar f baseado no cenário
    if cenario == "Nc=Np*0.5":
        f = 0.5
    elif cenario == "Nc=Np*0.8":
        f = 0.8
    elif cenario == "Nc=Np":
        f = 1.0
    else:
        f = 1.0  # default
    
    # Obter step/time
    if 'step' in df.columns:
        time_col = 'step'
    elif 'time' in df.columns:
        time_col = 'time'
    elif 't' in df.columns:
        time_col = 't'
    else:
        # Usar o índice como tempo
        df = df.reset_index()
        time_col = 'index'
    
    # Calcular custo para cada ponto no tempo
    for idx, row in df.iterrows():
        time = row[time_col]
        N_E = row[prey_col]
        
        # Evitar divisão por zero
        if N_E == 0 or pd.isna(N_E):
            cost = np.nan
        else:
            cost = f * (N_EO / N_E) 
        
        results.append({
            'cenario': cenario,
            'f_value': f,
            'obs_num': file_info["OBS"],
            'phi': calcular_densidade(file_info["OBS"]),
            'time': time,
            'N_E': N_E,
            'N_EO': N_EO,
            'cost': cost,
            'run': file_info["RUN"],
            'prey_col': prey_col
        })
    
    return pd.DataFrame(results)

def processar_todos_dados():
    """
    Processa todos os arquivos em todas as pastas e cenários
    """
    all_results = []
    
    for cenario in config.CENARIOS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processando cenário: {cenario}")
        logger.info(f"{'='*50}")
        
        for obs_folder in config.OBS_FOLDERS:
            # Caminho completo: BASE_ROOT / cenário / obs_folder
            folder_path = config.BASE_ROOT / cenario / obs_folder
            
            if not folder_path.exists():
                logger.warning(f"Pasta não encontrada: {folder_path}")
                continue
            
            # Procurar todos os arquivos CSV
            csv_files = list(folder_path.glob("*_prey_per_step.csv"))
            
            if csv_files:
                logger.info(f"  {obs_folder}: {len(csv_files)} arquivos")
            
            for csv_file in csv_files:
                file_info = parse_filename(csv_file.name)
                
                if file_info is None:
                    logger.warning(f"    Não consegui parsear: {csv_file.name}")
                    continue
                
                try:
                    # Ler o arquivo CSV
                    df = pd.read_csv(csv_file)
                    
                    if df.empty:
                        continue
                    
                    # Calcular custo dinâmico
                    df_cost = calcular_custo_dinamico_por_arquivo(df, file_info, cenario)
                    
                    if not df_cost.empty:
                        df_cost['obs_folder'] = obs_folder
                        df_cost['filename'] = csv_file.name
                        all_results.append(df_cost)
                        logger.debug(f"    ✓ {csv_file.name} processado")
                    
                except Exception as e:
                    logger.error(f"    Erro ao processar {csv_file.name}: {e}")
    
    if not all_results:
        logger.error("Nenhum dado foi processado!")
        return pd.DataFrame()
    
    # Concatenar todos os resultados
    final_df = pd.concat(all_results, ignore_index=True)
    
    logger.info(f"\n{'='*50}")
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info(f"{'='*50}")
    logger.info(f"Total de registros: {len(final_df)}")
    logger.info(f"Cenários processados: {final_df['cenario'].unique()}")
    logger.info(f"Valores de f únicos: {final_df['f_value'].unique()}")
    logger.info(f"Observações únicas: {sorted(final_df['obs_num'].unique())}")
    logger.info(f"Coluna de presas usada: {final_df['prey_col'].iloc[0] if len(final_df) > 0 else 'N/A'}")
    
    return final_df

def agregar_dados_para_plot(df):
    """
    Agrega os dados calculando média e desvio padrão
    """
    if df.empty:
        return pd.DataFrame()
    
    aggregated = []
    
    # Agrupar por cenário, observação e tempo
    group_cols = ['cenario', 'f_value', 'obs_folder', 'obs_num', 'phi', 'time']
    
    for group_key, group in df.groupby(group_cols):
        if group['cost'].notna().sum() > 0:
            cenario, f_val, obs_folder, obs_num, phi, time = group_key
            
            aggregated.append({
                'cenario': cenario,
                'f_value': f_val,
                'obs_folder': obs_folder,
                'obs_num': obs_num,
                'phi': phi,
                'time': time,
                'cost_mean': group['cost'].mean(),
                'cost_std': group['cost'].std(),
                'cost_median': group['cost'].median(),
                'cost_min': group['cost'].min(),
                'cost_max': group['cost'].max(),
                'n_runs': len(group['run'].unique()),
                'N_E_mean': group['N_E'].mean()
            })
    
    return pd.DataFrame(aggregated)
def plot_custo_por_phi(df_agg):
    """
    Cria gráficos separados INDIVIDUAIS para cada densidade φ
    Um gráfico por vez, não todos juntos
    """
    if df_agg.empty:
        logger.error("Dados agregados vazios!")
        return
    
    # Obter lista única de φ
    unique_phis = sorted(df_agg['phi'].unique())
    
    logger.info(f"Gerando {len(unique_phis)} gráficos individuais...")
    
    # Para cada densidade φ, criar um gráfico separado
    for phi in unique_phis:
        logger.info(f"  Processando φ = {phi:.3f}")
        
        # Criar figura individual
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filtrar dados para este φ
        phi_data = df_agg[df_agg['phi'] == phi]
        
        cmap = colormaps.get_cmap(config.PLOT_PARAMS["cmap_cenarios"])
        
        # Para cada cenário
        for cenario_idx, cenario in enumerate(config.CENARIOS):
            cenario_phi_data = phi_data[phi_data['cenario'] == cenario]
            
            if cenario_phi_data.empty:
                logger.warning(f"    Sem dados para {cenario} em φ={phi:.3f}")
                continue
            
            # Ordenar por tempo
            cenario_phi_data = cenario_phi_data.sort_values('time')
            
            color = cmap(cenario_idx)
            
            # Determinar legenda com notação LaTeX
            if cenario == "Nc=Np*0.5":
                f_val = 0.5
                label = r'$N^{C} = 0.5N_{0}^{E}$'
            elif cenario == "Nc=Np*0.8":
                f_val = 0.8
                label = r'$N^{C} = 0.8N_{0}^{E}$'
            else:
                f_val = 1.0
                label = r'$N^{C} = N_{0}^{E}$'
            
            # Plotar linha da média
            ax.plot(cenario_phi_data['time'], cenario_phi_data['cost_mean'],
                   color=color, linewidth=2.5,
                   label=label, alpha=0.9)
            
            # Plotar desvio padrão (se tiver dados)
            if len(cenario_phi_data) > 1 and cenario_phi_data['cost_std'].notna().any():
                # Faixa sombreada (desvio padrão)
                ax.fill_between(cenario_phi_data['time'],
                              cenario_phi_data['cost_mean'] - cenario_phi_data['cost_std'],
                              cenario_phi_data['cost_mean'] + cenario_phi_data['cost_std'],
                              color=color, alpha=0.2)
                
                # Linhas tracejadas nos limites (opcional)
                ax.plot(cenario_phi_data['time'], 
                       cenario_phi_data['cost_mean'] + cenario_phi_data['cost_std'],
                       color=color, linestyle='--', linewidth=0.8, alpha=0.4)
                
                ax.plot(cenario_phi_data['time'], 
                       cenario_phi_data['cost_mean'] - cenario_phi_data['cost_std'],
                       color=color, linestyle='--', linewidth=0.8, alpha=0.4)
        
        # Configurar o gráfico individual
        ax.set_xlabel('steps', fontsize=14)
        ax.set_ylabel(r'$c(t)$', fontsize=14)
        ax.set_title(fr'$\phi = {phi:.2f}$', fontsize=16)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=12, loc='best')
        ax.set_xlim(left=0)
        
        # Remover spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Salvar figura individual
        # Substituir pontos por underlines no nome do arquivo
        phi_str = f"{phi:.2f}".replace('.', '_')
        out_file = config.OUT_DIR / f"custo_phi_{phi_str}.pdf"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        logger.info(f"    Gráfico salvo: {out_file}")
        
        # Mostrar este gráfico individual
        plt.show()
        plt.close(fig)  # Fechar figura para liberar memória
    
    logger.info(f"✓ Todos os {len(unique_phis)} gráficos individuais foram gerados!")

# ---------------------- Main ----------------------
def main():
    logger.info("Iniciando processamento do custo dinâmico...")
    logger.info(f"Diretório base: {config.BASE_ROOT}")
    logger.info(f"Cenários: {config.CENARIOS}")
    logger.info(f"Observações: {config.OBS_FOLDERS}")
    
    # Processar todos os dados
    df = processar_todos_dados()
    
    if df.empty:
        logger.error("Nenhum dado processado. Encerrando.")
        return
    
    # Salvar dados brutos processados
    raw_output = config.OUT_DIR / "custo_dinamico_bruto.csv"
    df.to_csv(raw_output, index=False)
    logger.info(f"Dados brutos salvos: {raw_output}")
    
    # Agregar dados
    df_agg = agregar_dados_para_plot(df)
    
    if df_agg.empty:
        logger.error("Falha na agregação dos dados.")
        return
    
    # Salvar dados agregados
    agg_output = config.OUT_DIR / "custo_dinamico_agregado.csv"
    df_agg.to_csv(agg_output, index=False)
    logger.info(f"Dados agregados salvos: {agg_output}")
    
    # Gerar gráficos
    #plot_custo_por_cenario(df_agg)
    plot_custo_por_phi(df_agg)
    
    logger.info("Processamento concluído!")

if __name__ == "__main__":
    main()