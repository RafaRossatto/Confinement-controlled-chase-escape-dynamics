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

# ---------------------- Config (ADAPTADO) ----------------------
class Config:
    # BASE_ROOT adaptada para a pasta raiz dos resultados_modelos
    BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128" / "resultados_modelos" 
    
    # PASTA DOS DADOS DE SOBREVIVÊNCIA (NOVO)
    DATA_DIR = BASE_ROOT / "survival_analysis"
    
    OUT_DIR = BASE_ROOT / "custo_survival_analysis" # Nova pasta de saída
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Os nomes dos arquivos contêm os cenários (Nc=Np*0.5, etc.) e a obs.
    # Vamos escanear DATA_DIR diretamente e usar o parse_filename adaptado.
    
    # Cenários e seus valores 'f'
    CENARIO_MAP = {
        "Nc=Np*0.5": 0.5,
        "Nc=Np*0.8": 0.8,
        "Nc=Np": 1.0
    }
    
    # Colunas esperadas no arquivo CSV de sobrevivência
    COLUNAS_DADOS = {
        'step_col': 'step',
        's_t_col': 'S(t)',
        's_t_lower_col': 'S(t)_lower',
        's_t_upper_col': 'S(t)_upper'
    }

    # Configurações do gráfico
    PLOT_PARAMS = {
        "cmap_cenarios": "flag",
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

def parse_filename_survival(filename):
    """
    Extrai informações do nome do arquivo de sobrevivência.
    Exemplo: Nc=Np*0.5_ob00_survival_data.csv
    
    Tenta mapear 'obXX' para um número de observação.
    Se 'ob00' -> obs_num = 0
    Se 'ob1638' -> obs_num = 1638
    """
    
    # Padrão para cenários (Nc=Np*X) e obs (obYYY)
    pattern = r"^(Nc=Np\*(?:0\.5|0\.8)|Nc=Np)_ob(\d+)_survival_data\.csv$"
    match = re.match(pattern, filename)
    
    if match:
        cenario_str = match.group(1)
        obs_num_str = match.group(2)
        
        # Mapear o string do cenário para o nome padrão (Nc=Np*0.5 etc)
        if cenario_str == "$N^{C}$=N8*0.5" and cenario_str in config.CENARIO_MAP:
            f_value = config.CENARIO_MAP[cenario_str]
        elif cenario_str == "$N^{C}$=Np*0.8" and cenario_str in config.CENARIO_MAP:
            f_value = config.CENARIO_MAP[cenario_str]
        elif cenario_str == "$N^{C}$=Np" and cenario_str in config.CENARIO_MAP:
            f_value = config.CENARIO_MAP[cenario_str]
        else:
            f_value = 1.0
        
        return {
            "cenario": cenario_str,
            "f_value": f_value,
            "obs_num": int(obs_num_str),
        }
    return None

def calcular_densidade(obs_num):
    """Calcula a densidade φ a partir do número de obstáculos"""
    return obs_num / (128 * 128)

def calcular_custo_survival(df, file_info):
    """
    Calcula o custo dinâmico para o arquivo de sobrevivência
    c(t) = f / S(t)
    Onde os erros são propagados:
    c_lower(t) = f / S(t)_upper
    c_upper(t) = f / S(t)_lower
    """
    
    f = file_info["f_value"]
    
    # Verificar se as colunas estão presentes
    cols = config.COLUNAS_DADOS
    missing_cols = [c for c in cols.values() if c not in df.columns]
    if missing_cols:
        logger.error(f"  Colunas faltando: {missing_cols}")
        return pd.DataFrame()

    try:
        # Calcular o custo
        df['cost'] = f / df[cols['s_t_col']]
        
        # Calcular os limites do custo (propagação do erro)
        # Note: Inverso porque S(t) está no denominador.
        df['cost_lower'] = f / df[cols['s_t_upper_col']]
        df['cost_upper'] = f / df[cols['s_t_lower_col']]
        
        # Adicionar metadados
        df['cenario'] = file_info["cenario"]
        df['f_value'] = f
        df['obs_num'] = file_info["obs_num"]
        df['phi'] = calcular_densidade(file_info["obs_num"])
        
        # Renomear coluna de tempo
        df = df.rename(columns={cols['step_col']: 'time'})
        
        # Selecionar colunas de interesse para o resultado final
        return df[['cenario', 'f_value', 'obs_num', 'phi', 'time', 
                   'S(t)', 'S(t)_lower', 'S(t)_upper', 
                   'cost', 'cost_lower', 'cost_upper']].copy()
        
    except Exception as e:
        logger.error(f"  Erro no cálculo do custo: {e}")
        return pd.DataFrame()

def processar_todos_dados():
    """
    Processa todos os arquivos na pasta DATA_DIR
    """
    all_results = []
    
    logger.info(f"Escanenado diretório: {config.DATA_DIR}")
    if not config.DATA_DIR.exists():
        logger.error(f"Diretório não encontrado: {config.DATA_DIR}")
        return pd.DataFrame()
    
    # Procurar todos os arquivos CSV com o padrão "_survival_data.csv"
    csv_files = list(config.DATA_DIR.glob("*_survival_data.csv"))
    
    if not csv_files:
        logger.error("Nenhum arquivo '_survival_data.csv' encontrado!")
        return pd.DataFrame()
    
    logger.info(f"Encontrados {len(csv_files)} arquivos para processar.")
            
    for csv_file in csv_files:
        file_info = parse_filename_survival(csv_file.name)
        
        if file_info is None:
            logger.warning(f"  Não consegui parsear o nome: {csv_file.name}")
            continue
        
        logger.info(f"  Processando: {csv_file.name} (f={file_info['f_value']}, obs={file_info['obs_num']})")
        
        try:
            # Ler o arquivo CSV
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"  Arquivo vazio: {csv_file.name}")
                continue
            
            # Calcular custo dinâmico
            df_cost = calcular_custo_survival(df, file_info)
            
            if not df_cost.empty:
                df_cost['filename'] = csv_file.name
                all_results.append(df_cost)
                logger.debug(f"  ✓ {csv_file.name} processado")
            
        except Exception as e:
            logger.error(f"  Erro ao processar {csv_file.name}: {e}")
            
    if not all_results:
        logger.error("Nenhum dado foi processado com sucesso!")
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
    
    return final_df

def plot_custo_por_phi(df):
    """
    Cria gráficos separados INDIVIDUAIS para cada densidade φ
    Plotando a curva de custo (c(t)) com o intervalo de erro.
    """
    if df.empty:
        logger.error("Dados vazios para plotagem!")
        return
    
    # Obter lista única de φ
    unique_phis = sorted(df['phi'].unique())
    
    logger.info(f"Gerando {len(unique_phis)} gráficos individuais...")
    
    cmap = colormaps.get_cmap(config.PLOT_PARAMS["cmap_cenarios"])
    
    # Para cada densidade φ, criar um gráfico separado
    for phi in unique_phis:
        logger.info(f"  Processando φ = {phi:.3f}")
        
        # Criar figura individual
        fig, ax = plt.subplots(figsize=config.PLOT_PARAMS["figsize"])
        
        # Filtrar dados para este φ
        phi_data = df[df['phi'] == phi]
        
        # Para cada cenário
        for cenario_idx, cenario in enumerate(config.CENARIO_MAP.keys()):
            cenario_phi_data = phi_data[phi_data['cenario'] == cenario].sort_values('time')
            
            if cenario_phi_data.empty:
                continue
            
            color = cmap(cenario_idx ) if len(config.CENARIO_MAP) > 1 else 0.5
            
            # Determinar legenda com notação LaTeX
            if cenario == "Nc=Np*0.5":
                label = r'$N^{C} = 0.5N_{0}^{E}$'
            elif cenario == "Nc=Np*0.8":
                label = r'$N^{C} = 0.8N_{0}^{E}$'
            else:
                label = r'$N^{C} = N_{0}^{E}$'
            
            # Plotar linha do custo médio (c(t))
            ax.plot(cenario_phi_data['time'], cenario_phi_data['cost'],
                   color=color, linewidth=config.PLOT_PARAMS['linewidth'],
                   label=label, alpha=config.PLOT_PARAMS['alpha'])
            
            # Plotar a faixa de erro (fill_between)
            ax.fill_between(cenario_phi_data['time'],
                          cenario_phi_data['cost_lower'],
                          cenario_phi_data['cost_upper'],
                          color=color, alpha=0.2)
        
        # Configurar o gráfico individual
        obs_num = phi_data['obs_num'].iloc[0] if not phi_data.empty else 'N/A'
        ax.set_xlabel('steps', fontsize=14)
        ax.set_ylabel(r'$c(t)$', fontsize=14)
        ax.set_title(fr'$\phi = {phi:.2f}$', fontsize=16)
        
        ax.grid(True, alpha=config.PLOT_PARAMS['grid_alpha'], linestyle='--')
        ax.legend(fontsize=12, loc='best')
        ax.set_xlim(left=0)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Salvar figura individual
        phi_str = f"{phi:.2f}".replace('.', '_')
        out_file = config.OUT_DIR / f"custo_survival_phi_{phi_str}.pdf"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        logger.info(f"    Gráfico salvo: {out_file}")
        
        # Mostrar este gráfico individual
        plt.show()
        plt.close(fig)
    
    logger.info(f"✓ Todos os {len(unique_phis)} gráficos individuais foram gerados!")


# ---------------------- Main ----------------------
def main():
    logger.info("Iniciando processamento do custo dinâmico (Survival Analysis)...")
    logger.info(f"Diretório dos dados: {config.DATA_DIR}")
    logger.info(f"Diretório de saída: {config.OUT_DIR}")
    
    # Processar todos os dados
    df = processar_todos_dados()
    
    if df.empty:
        logger.error("Nenhum dado processado. Encerrando.")
        return
    
    # Salvar dados brutos processados (agora é o resultado final, pois não há runs para agregar)
    output_file = config.OUT_DIR / "custo_survival_final.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Dados de custo salvos: {output_file}")
    
    # Gerar gráficos (a função de plotagem agora usa o DataFrame processado diretamente)
    plot_custo_por_phi(df)
    
    logger.info("Processamento concluído!")

if __name__ == "__main__":
    main()