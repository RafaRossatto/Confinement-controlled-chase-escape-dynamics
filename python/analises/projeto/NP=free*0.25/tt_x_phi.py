#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logging
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import seaborn as sns

# ---------------------- Configuração de Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- Parâmetros principais ----------------------
class Config:
    L = 128
    AREA = L**2
    BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"
    
    BASES = [
        (r"$N^{C}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
        (r"$N^{C}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
        (r"$N^{C}=N^{E}_{0}$", "Nc=Np"),
    ]
    
    PLOT_PARAMS = {
        'cmap_name': "flag",
        'phi_separators_start': 0.05,
        'phi_separators_step': 0.10,
        'phi_separators_max': 0.90,
        'phi_line_special': 0.60,
        'boxplot_width': 0.22,
        'offset_delta': 0.25
    }
    
    FILTRO_STEPS = "todos"  # "todos" ou "apenas_extintos"
    
    # Configuração da legenda
    LEGEND_FRAME = True  # Com quadro na legenda
    LEGEND_LOCATION = 'upper left'  # Posição da legenda
    
    # Escala logarítmica apenas no gráfico principal
    LOG_SCALE = True
    Y_LIM_LOG = (7, 200)  # Começando em 1 na escala log
    X_LIM = (-0.5, 12.5)  # Começando em 1 na escala log
    # Configuração do inset
    INSET_Y_RANGE = (1000, 1200)  # Faixa Y para o inset
    INSET_PHI_RANGE = (0.65, 0.8)  # Faixa φ para o inset

        # --- Configuração global de fonte nos eixos e legenda ---
    plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14})


config = Config()
OUT_DIR = config.BASE_ROOT / "resultados_modelos" / "zeros_escapers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Utilitários Melhorados ----------------------
@lru_cache(maxsize=32)
def ler_csv(fp: Path) -> pd.DataFrame:
    """Lê simulation_results.csv tolerante a delimitadores e cabeçalhos."""
    try:
        # Tentar diferentes separadores para CSV
        for sep in [',', ';', '\t', ' ']:
            try:
                df = pd.read_csv(fp, sep=sep, engine='python', comment='#')
                if {'run', 'steps', 'remaining_prey', 'seed'}.issubset(df.columns):
                    logger.info(f"Arquivo {fp.name} lido com separador: {repr(sep)}")
                    break
            except Exception as e:
                continue
        else:
            # Fallback para leitura padrão do pandas
            logger.warning(f"Usando fallback para {fp.name}")
            df = pd.read_csv(fp)
        
        # Verificar se as colunas necessárias existem
        required_cols = ['run', 'steps', 'remaining_prey']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colunas ausentes em {fp.name}: {missing_cols}")
            return pd.DataFrame()
        
        # Conversão robusta de tipos
        numeric_cols = ['run', 'steps', 'remaining_prey']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)
        
        if df.empty:
            logger.warning(f"Arquivo {fp.name} vazio após limpeza")
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao ler {fp}: {e}")
        return pd.DataFrame()

def parse_directory_parameters(dir_path: Path) -> Dict[str, int]:
    """Extrai parâmetros do nome do diretório de forma robusta"""
    dir_name = dir_path.name
    
    patterns = {
        'O': r"s_obs_(\d+)$",
        'NC': r"NC_(\d+)",
        'NE': r"NE_(\d+)",
        'TCC': r"TCC_(\d+)",
        'SR': r"SR_(\d+)",
        'TCT': r"TCT_(\d+)"
    }
    
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, dir_name)
        if match:
            try:
                params[key] = int(match.group(1))
            except ValueError:
                logger.warning(f"Valor inválido para {key} em {dir_name}")
    
    return params

def symmetric_offsets(n_series: int, delta: float = 0.25) -> np.ndarray:
    """Retorna offsets simétricos para boxplots."""
    idx = np.arange(n_series)
    if n_series % 2 == 1:
        center = n_series // 2
        return (idx - center) * delta
    else:
        return (idx - (n_series - 1)/2) * delta

# ---------------------- Funções de Processamento ----------------------
def coletar_steps_por_run(base_root: Path, label_tex: str) -> List[Tuple[float, int, np.ndarray]]:
    """Coleta dados de steps por run."""
    resultados = []
    
    if not base_root.exists():
        logger.warning(f"Raiz não encontrada: {base_root}")
        return resultados

    subdirs = [d for d in base_root.iterdir() if d.is_dir() and d.name.startswith("s_obs_")]
    
    if not subdirs:
        logger.warning(f"Sem subpastas s_obs_* em {base_root}")
        return resultados

    for sdir in sorted(subdirs):
        # Buscar especificamente por simulation_results.csv
        csv_file = sdir / "simulation_results.csv"
        
        if not csv_file.exists():
            logger.warning(f"Arquivo simulation_results.csv não encontrado em {sdir}")
            continue

        try:
            df = ler_csv(csv_file)
            if df.empty:
                continue
                
            # Aplicar filtro
            if config.FILTRO_STEPS == "apenas_extintos":
                df = df[df["remaining_prey"] == 0]
            elif config.FILTRO_STEPS != "todos":
                raise ValueError("FILTRO_STEPS deve ser 'todos' ou 'apenas_extintos'")

            if df.empty:
                continue

            # Extrair parâmetros do nome do diretório
            params = parse_directory_parameters(sdir)
            O = params.get('O')
            
            if O is None:
                logger.warning(f"Não consegui obter O do diretório {sdir.name}")
                continue
                    
            n_obs = int(O)
            phi = n_obs / float(config.AREA)

            steps_vals = df["steps"].dropna().to_numpy()
            
            if len(steps_vals) > 0:
                resultados.append((phi, n_obs, steps_vals))
                logger.info(f"Processado: phi={phi:.3f}, n_obs={n_obs}, runs={len(steps_vals)}")

        except Exception as e:
            logger.error(f"Erro processando {csv_file}: {e}")
            continue

    return resultados

def plot_kde_distribuicoes(dados_por_cenario, out_dir: Path):
    """Gera gráficos KDE das distribuições de steps para cada configuração."""
    for base_idx, label_tex, triplets in dados_por_cenario:
        for phi, n_obs, arr in triplets:
            if len(arr) < 2:
                continue  # precisa de pelo menos 2 pontos para KDE
            
            plt.figure(figsize=(7, 4))
            sns.kdeplot(arr, fill=True, color="blue", alpha=0.4, linewidth=2)
            
            plt.title(f"Distribuição dos steps\n{label_tex}, φ={phi:.2f}")
            plt.xlabel("Steps até a captura")
            plt.ylabel("Densidade KDE")
            plt.grid(alpha=0.3)
            
            safe_label = re.sub(r'[^a-zA-Z0-9_]+', '_', label_tex)  # mantém só letras/números/underscore
            out_file = out_dir / f"kde_steps_{safe_label}_phi{phi:.2f}.pdf"
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()
            
            logger.info(f"KDE salvo: {out_file}")


def add_phi_separators_and_phi60(ax, phi_sorted, tick_positions):
    """Adiciona linhas verticais ENTRE cada tick e uma faixa em φ=0.60."""
    phi_max = max(phi_sorted) if phi_sorted else 1.0
    
    # LINHAS ENTRE CADA PAR DE TICKS
    for i in range(len(tick_positions) - 1):
        # Posição exatamente no meio entre dois ticks consecutivos
        x_sep = (tick_positions[i] + tick_positions[i + 1]) / 2
        ax.axvline(x=x_sep, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    
    # faixa em φ=0.60 - encontra o tick mais próximo
    if config.PLOT_PARAMS['phi_line_special'] <= phi_max:
        # Encontra o índice do valor de phi mais próximo de 0.60
        idx_60 = min(range(len(phi_sorted)), key=lambda i: abs(phi_sorted[i] - 0.60))
        x_phi60 = tick_positions[idx_60]
        
        ax.axvspan(x_phi60 - 0.5, x_phi60 + 0.5,
                   color="red", alpha=0.2,
                   label=rf"$\phi = {config.PLOT_PARAMS['phi_line_special']:.2f}$")

def put_phi60_first_in_legend(ax):
    """Reorganiza a legenda para colocar phi=0.60 primeiro com quadro."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    
    linha60, outros = [], []
    special_label = rf"$\phi = {config.PLOT_PARAMS['phi_line_special']:.1f}$"
    
    for h, l in zip(handles, labels):
        if special_label in l:
            linha60.append((h, l))
        else:
            outros.append((h, l))
    
    new = linha60 + outros if linha60 else outros
    if new:
        new_handles, new_labels = zip(*new)
        # LEGENDA COM QUADRO - configurações melhoradas
        ax.legend(new_handles, new_labels, 
                 frameon=config.LEGEND_FRAME,
                 loc=config.LEGEND_LOCATION,
                 fancybox=True,          # Bordas arredondadas
                 shadow=False,            # Sombra
                 framealpha=0.7,         # Transparência do fundo
                 edgecolor='grey',      # Cor da borda
                 facecolor='white',      # Cor de fundo
                 fontsize=10)            # Tamanho da fonte

def main():
    """Função principal executável."""
    logger.info("Iniciando análise de dados...")
    
    # Coletar dados por cenário
    dados_por_cenario = []
    rows_all = []
    
    for base_idx, (label_tex, base_dirname) in enumerate(config.BASES):
        root_path = config.BASE_ROOT / base_dirname
        logger.info(f"Processando cenário: {label_tex}")
        
        triplets = coletar_steps_por_run(root_path, label_tex)
        if not triplets:
            logger.warning(f"Nenhum dado encontrado para {label_tex}")
            continue
            
        # Salvar runs individuais
        for phi, n_obs, arr in triplets:
            for v in arr:
                rows_all.append({
                    "cenario_label": label_tex,
                    "cenario_tag": base_dirname,
                    "phi": phi,
                    "num_obs": n_obs,
                    "steps_run": float(v),
                })
        
        dados_por_cenario.append((base_idx, label_tex, triplets))
    
    if not dados_por_cenario:
        logger.error("Nenhum dado encontrado para análise. Verifique os caminhos.")
        return
    
    # Salvar CSV combinado
    if rows_all:
        filtro_tag = "ALL" if config.FILTRO_STEPS == "todos" else "EXTINTOS"
        out_runs = OUT_DIR / f"steps_runs_{filtro_tag}__ALL.csv"
        pd.DataFrame(rows_all).sort_values(["cenario_tag","phi"]).to_csv(out_runs, index=False)
        logger.info(f"CSV salvo: {out_runs}")
    
    # Preparar eixo categórico
    phi_labels = sorted({phi for _, _, trips in dados_por_cenario for (phi, _n, _arr) in trips})
    tick_positions = np.arange(len(phi_labels))
    
    # Configurar plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    cmap = colormaps.get_cmap(config.PLOT_PARAMS['cmap_name'])
    offsets = symmetric_offsets(len(config.BASES), config.PLOT_PARAMS['offset_delta'])
    
    # Plotar boxplots
    for base_idx, label_tex, triplets in dados_por_cenario:
        groups = []
        pos_idx = []
        
        for phi, _n_obs, arr in triplets:
            idx = phi_labels.index(phi)
            pos_idx.append(idx)
            groups.append(arr)
        
        if not groups:
            continue
            
        pos_idx, groups = zip(*sorted(zip(pos_idx, groups)))
        positions = np.array(pos_idx, dtype=float) + offsets[base_idx]
        
        ax.boxplot(
            groups,
            positions=positions,
            widths=config.PLOT_PARAMS['boxplot_width'],
            patch_artist=True,
            boxprops=dict(facecolor=cmap(base_idx), alpha=0.5),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=cmap(base_idx)),
            capprops=dict(color=cmap(base_idx)),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor=cmap(base_idx), markeredgecolor="none")
        )
        
        # Handle para legenda
        ax.plot([], [], color=cmap(base_idx), label=label_tex, linewidth=3)
    
    # Eixos
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{phi:.2f}" for phi in phi_labels], rotation=0)
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xlabel(r"$\phi$",fontsize=18)
    
    if config.LOG_SCALE:
        ax.set_yscale('log')
        ax.set_ylabel(r"$TT$",fontsize=18)
        ax.set_ylim(config.Y_LIM_LOG[0], config.Y_LIM_LOG[1])
        ax.set_xlim(config.X_LIM[0], config.X_LIM[1])
    else:
        ax.set_ylabel("Passos por run até a captura (distribuição)")
    
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    
    # Adicionar separadores + faixa em φ=0.60
    add_phi_separators_and_phi60(ax, phi_labels, tick_positions)
    
    # Legenda
    put_phi60_first_in_legend(ax)
    
    # Layout e salvar
    plt.tight_layout()
    out_fig = OUT_DIR / ("TT_vs_phi_boxplot.pdf" if config.LOG_SCALE else "TT_vs_phi.pdf")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    logger.info(f"Figura salva: {out_fig}")
    
    # Salvar os KDEs também
    #plot_kde_distribuicoes(dados_por_cenario, OUT_DIR)
    plt.show()
    logger.info("Análise concluída com sucesso!")

if __name__ == "__main__":
    main()