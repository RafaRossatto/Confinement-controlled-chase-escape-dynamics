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

# ---------------------- parâmetros ----------------------
class Config:
    OBS_LIST = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
                "obs_8192", "obs_9830", "obs_11468", "obs_13107", "obs_14745"]
    
    BASES = [
        (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
        (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
        (r"$N^{C}_{0}=N^{E}_{0}$", "Nc=Np"),
    ]
    
    L = 128
    AREA = L**2
    BASE_RES = Path.home() / "Dados_Doc" / "Np=free*0.25" / "resultados_modelos"
    MODEL_TAG = "EXPbeta"
    
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
        # LEGENDA COM QUADRO - configurações melhoradas
        ax.legend(new_handles, new_labels, 
                 frameon=config.LEGEND_FRAME,
                 loc=config.LEGEND_LOCATION,
                 fancybox=True,          # Bordas arredondadas
                 shadow=False,            # Sombra
                 framealpha=0.8,         # Transparência do fundo
                 edgecolor='black',      # Cor da borda
                 facecolor='white',      # Cor de fundo
                 fontsize=10)            # Tamanho da fonte

@lru_cache(maxsize=32)
def ler_csv_com_fallback(fp: Path) -> pd.DataFrame:
    """Lê CSV com fallback para diferentes formatos."""
    try:
        # Tentar diferentes separadores
        for sep in [',', ';', '\t', None]:
            try:
                df = pd.read_csv(fp, sep=sep, engine='python')
                if not df.empty and 'beta' in df.columns:
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
    logger.info("Iniciando análise de dados beta...")
    
    # Paleta
    cmap_flag = plt.cm.get_cmap(config.PLOT_PARAMS['cmap_name'])
    
    plt.figure(figsize=(12, 6), dpi=150)
    ax = plt.gca()
    
    rows_all = []          # CSV combinado (todas as bases, todos os runs)
    phi_labels = []        # valores únicos de phi
    phi_to_index = {}      # mapeia phi -> índice categórico
    all_beta_groups = []   # Para armazenar todos os grupos de beta
    
    OFFSETS = symmetric_offsets(len(config.BASES), delta=config.PLOT_PARAMS['offset_delta'])
    
    for base_idx, (label_tex, nc_tag) in enumerate(config.BASES):
        phi_vals, beta_groups = [], []
    
        for obs_folder in config.OBS_LIST:
            n_obs = extrai_num_obs(obs_folder)
            phi = densidade_obs(n_obs)
    
            # arquivo esperado
            fname = f"params_por_run_{config.MODEL_TAG}_{nc_tag}_s_obs_{n_obs:02d}.csv"
            fpath = config.BASE_RES / fname
            if not fpath.exists():
                logger.warning(f"Arquivo não encontrado: {fpath.name}")
                continue
    
            df = ler_csv_com_fallback(fpath)
            if df.empty or "beta" not in df.columns:
                logger.warning(f"Sem coluna 'beta' ou vazio: {fpath.name}")
                continue
    
            beta_values = df["beta"].dropna().values
            if len(beta_values) == 0:
                continue
    
            if phi not in phi_to_index:
                phi_to_index[phi] = len(phi_labels)
                phi_labels.append(phi)
    
            phi_vals.append(phi_to_index[phi])
            beta_groups.append(beta_values)
            all_beta_groups.append(beta_values)  # Adicionar à lista geral
    
            # CSV combinado com valores individuais
            for v in beta_values:
                rows_all.append({
                    "scenario": nc_tag,
                    "phi": phi,
                    "num_obs": n_obs,
                    "beta_run": float(v)
                })
    
        if beta_groups:
            # ordenar por índice de phi e posicionar com offsets simétricos
            phi_vals, beta_groups = zip(*sorted(zip(phi_vals, beta_groups)))
            positions = np.array(phi_vals, dtype=float) + OFFSETS[base_idx]
    
            plt.boxplot(
                beta_groups,
                positions=positions,
                widths=config.PLOT_PARAMS['boxplot_width'],
                patch_artist=True,
                boxprops=dict(facecolor=cmap_flag(base_idx), alpha=0.5),
                medianprops=dict(color="black"),
                whiskerprops=dict(color=cmap_flag(base_idx)),
                capprops=dict(color=cmap_flag(base_idx)),
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor=cmap_flag(base_idx), markeredgecolor="none")
            )
            # "handle" para legenda
            plt.plot([], [], color=cmap_flag(base_idx), label=label_tex, linewidth=3)
    
    # ---------------------- salvar CSV combinado ----------------------
    if rows_all:
        df_all = pd.DataFrame(rows_all).sort_values(["scenario", "phi"])
        df_all.to_csv("beta_vs_phi__todas_bases__runs.csv", index=False)
        logger.info("CSV salvo: beta_vs_phi__todas_bases__runs.csv")
    
    # ---------------------- decoração ----------------------
    phi_sorted = sorted(phi_labels)
    tick_positions = np.arange(len(phi_sorted))

    # RÓTULOS RETOS E UMA CASA DECIMAL
    tick_labels = [f"{phi:.1f}" for phi in phi_sorted]
    plt.xticks(tick_positions, tick_labels, rotation=0)  # rotation=0 para reto

    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\beta $")
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # LIMITAR EIXO Y ATÉ 1.5
    plt.ylim(0.4, 1.0)
    plt.xlim(-0.5, 8.5)

    # separadores e linha 0.60
    add_phi_separators_and_phi60(ax, phi_sorted, tick_positions)
    put_phi60_first_in_legend(ax)


    """
    # CRIAR INSET APENAS COM OS PONTOS (β > 1.5)
    if all_beta_groups:  # Verificar se há dados
        max_beta = max([np.max(group) for group in all_beta_groups if len(group) > 0])
        if max_beta > 1.5:
            ax_inset = inset_axes(ax, width="25%", height="25%", loc="upper center")
            
            # Valores de φ que queremos mostrar no inset
            target_phis = [0.85, 0.9, 1.0]  # 0.8, 0.9, 1.0
            actual_phis = []  # Para armazenar os valores reais de φ encontrados
            all_inset_points = []  # Para todos os pontos do inset
            
            # Coletar pontos para os φ alvo
            for pos, target_phi in enumerate(target_phis):
                # Encontrar o φ mais próximo disponível nos dados
                closest_phi = None
                min_diff = float('inf')
                
                for available_phi in phi_labels:
                    diff = abs(available_phi - target_phi)
                    if diff < min_diff:
                        min_diff = diff
                        closest_phi = available_phi
                
                if closest_phi is not None:
                    # Coletar todos os pontos β > 1.5 para este φ
                    phi_points = []
                    
                    for base_idx, (label_tex, nc_tag) in enumerate(config.BASES):
                        for obs_folder in config.OBS_LIST:
                            n_obs = extrai_num_obs(obs_folder)
                            phi_val = densidade_obs(n_obs)
                            
                            if abs(phi_val - closest_phi) < 0.001:  # Tolerância
                                fname = f"params_por_run_{config.MODEL_TAG}_{nc_tag}_s_obs_{n_obs:02d}.csv"
                                fpath = config.BASE_RES / fname
                                
                                if fpath.exists():
                                    df = ler_csv_com_fallback(fpath)
                                    if not df.empty and "beta" in df.columns:
                                        beta_values = df["beta"].dropna().values
                                        beta_values_above = beta_values[beta_values > 1.5]
                                        if len(beta_values_above) > 0:
                                            # Adicionar cada ponto individualmente
                                            for beta_val in beta_values_above:
                                                phi_points.append(beta_val)
                                                all_inset_points.append((pos, beta_val))
                                    break  # Sair do loop de obs_folder
                    
                    if phi_points:
                        # Plotar todos os pontos para este φ
                        ax_inset.scatter(
                            [pos] * len(phi_points),  # X position
                            phi_points,               # Y values
                            color=cmap_flag(base_idx % len(config.BASES)),  # Cor por cenário
                            alpha=0.6,
                            s=25,                     # Tamanho dos pontos
                            edgecolor='white',
                            linewidth=0.5,
                            zorder=10                 # Para ficar acima de outros elementos
                        )
                        actual_phis.append(closest_phi)
            
            if all_inset_points:
                # CONFIGURAÇÃO DOS TICKS - três colunas: 0.8, 0.9, 1.0
                ax_inset.set_xticks([0, 1, 2])
                
                # Usar os valores reais encontrados ou os target se não encontrou
                tick_labels = []
                for i in range(3):
                    if i < len(actual_phis):
                        tick_labels.append(f"{actual_phis[i]:.1f}")
                    else:
                        tick_labels.append(f"{target_phis[i]:.1f}")
                
                ax_inset.set_xticklabels(tick_labels, rotation=0, fontsize=9)
                
                # Configurar limites e aparência
                ax_inset.set_xlim(-0.5, 2.5)  # Espaço para 3 colunas
                
                # Ajustar limites Y para focar nos pontos
                y_values = [point[1] for point in all_inset_points]
                y_min = min(y_values) if y_values else 1.4
                y_max = max(y_values) if y_values else 3.0
                #ax_inset.set_ylim(max(1.4, y_min * 0.95), y_max * 1.05)
                ax_inset.set_ylim(1.4, 4)
                
                
                #ax_inset.set_title('Pontos com β > 1.5', fontsize=10, pad=8)
                ax_inset.set_ylabel(r'$\langle \beta \rangle$', fontsize=9)
                ax_inset.set_xlabel(r'$\phi$', fontsize=9)
                ax_inset.grid(True, alpha=0.3, axis='y')
                ax_inset.tick_params(labelsize=8)
                
                # Adicionar legenda apenas para a linha de referência
                #ax_inset.legend(fontsize=8, framealpha=0.8, loc='upper left')
                
                # Adicionar borda ao inset para melhor destaque
                for spine in ax_inset.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(1)
                
                # Adicionar fundo levemente colorido para melhor visualização
                ax_inset.set_facecolor('#f8f9fa')
    """
    plt.tight_layout()
    plt.savefig("beta_vs_phi_boxplot_sep.pdf", dpi=200, bbox_inches='tight')
    logger.info("Figura salva: beta_vs_phi.pdf")
    
    plt.show()
    logger.info("Análise concluída com sucesso!")

if __name__ == "__main__":
    main()