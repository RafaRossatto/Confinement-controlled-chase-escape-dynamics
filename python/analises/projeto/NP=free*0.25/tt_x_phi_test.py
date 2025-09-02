#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib import colormaps
import logging
from typing import List, Tuple, Dict, Optional
from functools import lru_cache

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
    BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
    
    BASES = [
        (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
        (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
        (r"$N^{C}_{0}=N^{E}_{0}$", "Nc=Np"),
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
    Y_LIM_LOG = (1, 1250)  # Começando em 1 na escala log

config = Config()
OUT_DIR = config.BASE_ROOT / "resultados_modelos" / "zeros_escapers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Utilitários Melhorados ----------------------
@lru_cache(maxsize=32)
def ler_dat(fp: Path) -> pd.DataFrame:
    """Lê .dat tolerante a delimitadores e cabeçalhos."""
    try:
        # Tentar diferentes separadores
        for sep in [None, '\s+', ',', '\t']:
            try:
                df = pd.read_csv(fp, sep=sep, engine='python', comment='#')
                if {'run', 'steps', 'escapers', 'seed'}.issubset(df.columns):
                    logger.info(f"Arquivo {fp.name} lido com separador: {repr(sep)}")
                    break
            except Exception as e:
                continue
        else:
            # Fallback para formato fixo
            logger.warning(f"Usando fallback para {fp.name}")
            df = pd.read_csv(
                fp, delim_whitespace=True, header=None,
                names=["run","steps","escapers","seed"], comment="#"
            )
        
        # Conversão robusta de tipos
        numeric_cols = ['run', 'steps', 'escapers']
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

def parse_filename_parameters(filename: str) -> Dict[str, int]:
    """Extrai parâmetros do nome do arquivo de forma robusta"""
    patterns = {
        'O': r"_O_(\d+)_",
        'NC': r"NC_(\d+)_",
        'NE': r"NE_(\d+)_",
        'TCC': r"TCC_(\d+)_",
        'SR': r"SR_(\d+)_",
        'TCT': r"TCT_(\d+)_"
    }
    
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            try:
                params[key] = int(match.group(1))
            except ValueError:
                logger.warning(f"Valor inválido para {key} em {filename}")
    
    # Fallback: tentar extrair do nome do diretório
    if 'O' not in params:
        match = re.search(r"s_obs_(\d+)$", filename)
        if match:
            try:
                params['O'] = int(match.group(1))
            except ValueError:
                pass
    
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
        # Buscar arquivos .dat
        dat_files = list(sdir.glob("*.dat"))
        if not dat_files:
            logger.warning(f"Sem arquivos .dat em {sdir}")
            continue
        
        # Preferir arquivos com padrão específico
        preferred_files = [f for f in dat_files if "NC_" in f.name and "NE_" in f.name]
        fp = preferred_files[0] if preferred_files else dat_files[0]

        try:
            df = ler_dat(fp)
            if df.empty:
                continue
                
            # Aplicar filtro
            if config.FILTRO_STEPS == "apenas_extintos":
                df = df[df["escapers"] == 0]
            elif config.FILTRO_STEPS != "todos":
                raise ValueError("FILTRO_STEPS deve ser 'todos' ou 'apenas_extintos'")

            if df.empty:
                continue

            # Extrair parâmetros do nome do arquivo
            params = parse_filename_parameters(fp.name)
            O = params.get('O')
            
            if O is None:
                logger.warning(f"Não consegui obter O em {fp.name}")
                continue
                
            n_obs = int(O)
            phi = n_obs / float(config.AREA)

            steps_vals = df["steps"].dropna().to_numpy()
            
            if len(steps_vals) > 0:
                resultados.append((phi, n_obs, steps_vals))
                logger.info(f"Processado: phi={phi:.3f}, n_obs={n_obs}, runs={len(steps_vals)}")

        except Exception as e:
            logger.error(f"Erro processando {fp}: {e}")
            continue

    return resultados

def add_phi_separators_and_phi60(ax, phi_sorted, tick_positions):
    """Adiciona linhas verticais nos separadores de phi."""
    phi_max = max(phi_sorted) if phi_sorted else 1.0
    
    for phi_sep in np.arange(config.PLOT_PARAMS['phi_separators_start'], 
                           min(config.PLOT_PARAMS['phi_separators_max'], phi_max) + 1e-12, 
                           config.PLOT_PARAMS['phi_separators_step']):
        if np.isclose(phi_sep, config.PLOT_PARAMS['phi_separators_max']):
            continue
        x_sep = np.interp(phi_sep, phi_sorted, tick_positions)
        ax.axvline(x=x_sep, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    
    # Linha especial
    if config.PLOT_PARAMS['phi_line_special'] <= phi_max:
        x_phi60 = np.interp(config.PLOT_PARAMS['phi_line_special'], phi_sorted, tick_positions)
        ax.axvline(x=x_phi60, color="black", linestyle="--", linewidth=1.5, 
                  label=rf"$\phi = {config.PLOT_PARAMS['phi_line_special']:.2f}$")

def put_phi60_first_in_legend(ax):
    """Reorganiza a legenda para colocar phi=0.60 primeiro com quadro."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    
    linha60, outros = [], []
    special_label = rf"$\phi = {config.PLOT_PARAMS['phi_line_special']:.2f}$"
    
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

# ---------------------- Função Principal ----------------------
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
    
    # Verificar se há dados
    if not dados_por_cenario:
        logger.error("Nenhum dado encontrado para boxplot. Verifique os caminhos.")
        return
    
    # Salvar CSV combinado
    if rows_all:
        filtro_tag = "ALL" if config.FILTRO_STEPS == "todos" else "EXTINTOS"
        out_runs = OUT_DIR / f"steps_boxplot_runs_{filtro_tag}__ALL.csv"
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
        
        # Boxplot
        bp = ax.boxplot(
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
    
    # Configurar eixo com escala logarítmica COMEÇANDO EM 1
    ax.set_xticks(tick_positions)
    # MUDANÇA AQUI: uma casa decimal e sem rotação
    ax.set_xticklabels([f"{phi:.1f}" for phi in phi_labels], rotation=0)  # rotation=0 para reto
    ax.set_xlabel(r"$\phi$")
    
    if config.LOG_SCALE:
        ax.set_yscale('log')
        ax.set_ylabel("Passos por run até a captura (escala logarítmica)")
        # COMEÇANDO EM 1 na escala log
        ax.set_ylim(config.Y_LIM_LOG[0], config.Y_LIM_LOG[1])
    else:
        ax.set_ylabel("Passos por run até a captura (distribuição)")
    
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    
    # Adicionar separadores
    add_phi_separators_and_phi60(ax, phi_labels, tick_positions)
    
    # Legenda com quadro - INSET ELIMINADO
    put_phi60_first_in_legend(ax)
    
    # Adicionar título informativo
    if config.LOG_SCALE:
        ax.set_title("Distribuição de Passos até a Captura - Escala Logarítmica")
    
    # Layout e salvamento
    plt.tight_layout()
    
    if config.LOG_SCALE:
        out_fig = OUT_DIR / "steps_vs_phi_boxplot_log_no_inset.pdf"
    else:
        out_fig = OUT_DIR / "steps_vs_phi_boxplot_no_inset.pdf"
        
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    logger.info(f"Figura salva: {out_fig}")
    
    plt.show()
    logger.info("Análise concluída com sucesso!")

if __name__ == "__main__":
    main()