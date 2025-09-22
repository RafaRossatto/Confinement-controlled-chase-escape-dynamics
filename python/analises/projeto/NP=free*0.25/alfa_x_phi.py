from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------- Configuração de Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuração global de fonte nos eixos e legenda ---
plt.rcParams.update({
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

# Saídas
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
OUT_DIR = BASE_ROOT / "resultados_modelos" / "alpha_x_phi"  # Mudar para alpha
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- parâmetros ----------------------
class Config:
    OBS_LIST = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
                "obs_8192", "obs_9830", "obs_11468", "obs_13107"]
    
    BASES = [
        (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np_0.5"),
        (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np_0.8"),
        (r"$N^{C}_{0}=N^{E}_{0}$", "Nc=Np"),
    ]
    
    L = 128
    AREA = L**2
    BASE_RES = Path.home() / "Dados_Doc" / "Np=free*0.25" / "resultados_modelos" / "msd"
    MODEL_TAG = "msd_ensemble_hunter_s_"  # Mudar para refletir arquivos de MSD
    
    # Configuração do plot
    PLOT_PARAMS = {
        'cmap_name': "flag",
        'phi_separators_start': 0.05,
        'phi_separators_step': 0.10,
        'phi_separators_max': 0.90,
        'phi_line_special': 0.60,
        'boxplot_width': 0.22,
        'offset_delta': 0.25
    }
    
    # Configuração da legenda
    LEGEND_FRAME = True
    LEGEND_LOCATION = 'lower left'

config = Config()

# ---------------------- funções auxiliares ----------------------
def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / config.AREA

def symmetric_offsets(n_series: int, delta: float = 0.25):
    """Offsets simétricos: n=3 -> [-d, 0, +d]; n=2 -> [-d/2, +d/2]; n=4 -> [-1.5d,-0.5d,+0.5d,+1.5d]."""
    idx = np.arange(n_series)
    if n_series % 2 == 1:
        center = n_series // 2
        return (idx - center) * delta
    else:
        return (idx - (n_series - 1)/2) * delta

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
        ax.legend(new_handles, new_labels, 
                 frameon=config.LEGEND_FRAME,
                 loc=config.LEGEND_LOCATION,
                 fancybox=True,
                 shadow=False,
                 framealpha=0.5,
                 edgecolor='grey',
                 facecolor='white',
                 fontsize=14)

@lru_cache(maxsize=32)
def ler_csv_com_fallback(fp: Path) -> pd.DataFrame:
    """Lê CSV com fallback para diferentes formatos."""
    try:
        for sep in [',', ';', '\t', None]:
            try:
                df = pd.read_csv(fp, sep=sep, engine='python')
                if not df.empty and 'alpha' in df.columns:  # Mudar para alpha
                    logger.info(f"Arquivo {fp.name} lido com separador: {repr(sep)}")
                    return df
            except Exception as e:
                continue
        
        logger.warning(f"Usando fallback para {fp.name}")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Erro ao ler {fp}: {e}")
        return pd.DataFrame()

# ---------------------- MAIN ----------------------
def main():
    logger.info("Iniciando análise de dados alpha...")
    
    # Paleta
    cmap_flag = plt.cm.get_cmap(config.PLOT_PARAMS['cmap_name'])
    
    plt.figure(figsize=(12, 6), dpi=150)
    ax = plt.gca()
    
    rows_all = []          # CSV combinado
    phi_labels = []        # valores únicos de phi
    phi_to_index = {}      # mapeia phi -> índice categórico
    all_alpha_groups = []   # Para armazenar todos os grupos de alpha
    
    OFFSETS = symmetric_offsets(len(config.BASES), delta=config.PLOT_PARAMS['offset_delta'])
    
    for base_idx, (label_tex, nc_tag) in enumerate(config.BASES):
        phi_vals, alpha_groups = [], []
    
        for obs_folder in config.OBS_LIST:
            n_obs = extrai_num_obs(obs_folder)
            phi = densidade_obs(n_obs)
    
            # arquivo esperado - MUDAR O PADRÃO DO NOME
            fname = f"msd_ensemble_hunter_s_obs_{n_obs:02d}_{nc_tag}.csv"  # Padrão dos seus arquivos
            
            fpath = config.BASE_RES / fname
            if not fpath.exists():
                # Tentar padrão alternativo
                fname_alt = f"msd_ensemble_hunter_obs_{n_obs:02d}_{nc_tag}.csv"
                fpath = config.BASE_RES / fname_alt
                if not fpath.exists():
                    logger.warning(f"Arquivo não encontrado: {fname} ou {fname_alt}")
                    continue
    
            df = ler_csv_com_fallback(fpath)
            if df.empty or "alpha" not in df.columns:  # Mudar para alpha
                logger.warning(f"Sem coluna 'alpha' ou vazio: {fpath.name}")
                continue
    
            alpha_values = df["alpha"].dropna().values
            # Filtrar valores extremos (opcional)
            alpha_values = alpha_values[(alpha_values >= 0) & (alpha_values <= 2)]
            
            if len(alpha_values) == 0:
                continue
    
            if phi not in phi_to_index:
                phi_to_index[phi] = len(phi_labels)
                phi_labels.append(phi)
    
            phi_vals.append(phi_to_index[phi])
            alpha_groups.append(alpha_values)
            all_alpha_groups.append(alpha_values)
    
            # CSV combinado com valores individuais
            for v in alpha_values:
                rows_all.append({
                    "scenario": nc_tag,
                    "phi": phi,
                    "num_obs": n_obs,
                    "alpha_run": float(v)  # Mudar para alpha
                })
    
        if alpha_groups:
            # ordenar por índice de phi e posicionar com offsets simétricos
            phi_vals, alpha_groups = zip(*sorted(zip(phi_vals, alpha_groups)))
            positions = np.array(phi_vals, dtype=float) + OFFSETS[base_idx]
    
            plt.boxplot(
                alpha_groups,
                positions=positions,
                widths=config.PLOT_PARAMS['boxplot_width'],
                patch_artist=True,
                boxprops=dict(facecolor=cmap_flag(base_idx), alpha=0.5),
                medianprops=dict(color="grey"),
                whiskerprops=dict(color=cmap_flag(base_idx)),
                capprops=dict(color=cmap_flag(base_idx)),
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor=cmap_flag(base_idx), markeredgecolor="none")
            )
            plt.plot([], [], color=cmap_flag(base_idx), label=label_tex, linewidth=3)
    
    # ---------------------- salvar CSV combinado ----------------------
    if rows_all:
        df_all = pd.DataFrame(rows_all).sort_values(["scenario", "phi"])
        df_all.to_csv("alpha_vs_phi__todas_bases__runs.csv", index=False)
        logger.info("CSV salvo: alpha_vs_phi__todas_bases__runs.csv")
    
    # ---------------------- decoração ----------------------
    phi_sorted = sorted(phi_labels)
    tick_positions = np.arange(len(phi_sorted))

    tick_labels = [f"{phi:.1f}" for phi in phi_sorted]
    plt.xticks(tick_positions, tick_labels, rotation=0)
    # REMOVER OS TICKS DO EIXO X
    ax.tick_params(axis='x', which='both', length=0)  # length=0 remove os tracinhos

    plt.xlabel(r"$\phi$", fontsize=20)
    plt.ylabel(r"$\alpha$", fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.xlim(-0.5, len(phi_sorted) - 0.5)

    # ADICIONAR: Fill entre phi=0.5 e 0.65 (PRIMEIRO para ficar no TOPO da legenda)
    phi_min_fill = 0.55
    phi_max_fill = 0.65

    fill_handle = None
    if phi_sorted:  # Verificar se há valores de phi
        phi_max = max(phi_sorted)
        if phi_min_fill <= phi_max:
            x_min_fill = np.interp(phi_min_fill, phi_sorted, tick_positions)
            x_max_fill = np.interp(phi_max_fill, phi_sorted, tick_positions)
            
            # Adicionar área preenchida - SALVAR O HANDLE
            fill_patch = ax.axvspan(x_min_fill, x_max_fill, color='red', alpha=0.2, 
                                label=r'$\phi = 0.60$')
            fill_handle = fill_patch

    # separadores
    for phi_sep in np.arange(config.PLOT_PARAMS['phi_separators_start'], 
                        min(config.PLOT_PARAMS['phi_separators_max'], phi_max) + 1e-12, 
                        config.PLOT_PARAMS['phi_separators_step']):
        if np.isclose(phi_sep, config.PLOT_PARAMS['phi_separators_max']):
            continue
        x_sep = np.interp(phi_sep, phi_sorted, tick_positions)
        ax.axvline(x=x_sep, color="gray", linestyle="-", alpha=0.5, linewidth=1)

    # Função para colocar a área preenchida no TOPO da legenda
    def put_legend_with_fill_on_top(ax, fill_handle=None):
        """Legenda com a área preenchida no topo."""
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        
        # Se temos um handle para a área preenchida, colocá-lo no início
        if fill_handle is not None:
            # Encontrar o índice do handle da área preenchida
            fill_indices = [i for i, h in enumerate(handles) if h == fill_handle]
            if fill_indices:
                fill_idx = fill_indices[0]
                # Mover para o início da lista
                handles = [handles[fill_idx]] + [h for i, h in enumerate(handles) if i != fill_idx]
                labels = [labels[fill_idx]] + [l for i, l in enumerate(labels) if i != fill_idx]
        
        ax.legend(handles, labels, 
                frameon=config.LEGEND_FRAME,
                loc=config.LEGEND_LOCATION,
                fancybox=True,
                shadow=False,
                framealpha=0.5,
                edgecolor='grey',
                facecolor='white',
                fontsize=14)

    put_legend_with_fill_on_top(ax, fill_handle)

    plt.tight_layout()

    fig_pdf = OUT_DIR / "alpha_vs_phi_boxplot.pdf"
    plt.savefig(fig_pdf, bbox_inches="tight")

    plt.show()
    logger.info("Análise de alpha concluída com sucesso!")

if __name__ == "__main__":
    main()