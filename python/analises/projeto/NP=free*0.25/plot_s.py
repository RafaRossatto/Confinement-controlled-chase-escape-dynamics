import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path


# -------------------------------
# 0. Configurações Principais
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128"

# Frações de caçadores para comparar
FRACOES_C = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]
NOMES_LEGENDA = {
    "Nc=Np*0.5": r"$N^C = 0.5 N^E_0$",
    "Nc=Np*0.8": r"$N^C = 0.8 N^E_0$", 
    "Nc=Np": r"$N^C = N^E_0$"
}

# Lista de obstáculos para análise (pode adicionar quantos quiser)
OBS_LISTA = [
    "s_obs_00",
    "s_obs_1638",
    "s_obs_3276",
    "s_obs_4915",
    "s_obs_6553",
    "s_obs_8028",
    "s_obs_8192",
    "s_obs_8355",
    "s_obs_9666",
    "s_obs_9830",
    "s_obs_9994",
    "s_obs_11468",
    "s_obs_13107"
]  # ← ADICIONE AQUI OS QUE QUISER
# Exemplo: OBS_LISTA = ["s_obs_00", "s_obs_4915", "s_obs_9830", "s_obs_13107"]

# Diretórios
OUT_DIR = BASE_ROOT / "resultados_modelos" /"paper_response" /"plot_s"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuração de plotagem
plt.rcParams.update({
    "font.size": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.labelsize": 16
})

# -------------------------------
# 1. Função para calcular densidade do título
# -------------------------------
def calcular_titulo_com_densidade(obs_nome):
    """Calcula o título com a densidade de obstáculos"""
    try:
        # Extrair número do nome do obstáculo
        numero_str = obs_nome.split('_')[-1]
        numero = int(numero_str)
        
        # Calcular densidade
        L = 128
        densidade = numero / (L * L)
        
        return f'Curvas de Sobrevivência Normalizadas\nDensidade: {densidade:.4f}'
    
    except (ValueError, IndexError):
        return f'Curvas de Sobrevivência Normalizadas\nObstáculo {obs_nome}'

# -------------------------------
# 2. Carregar dados brutos do Kaplan-Meier
# -------------------------------
def carregar_dados_survival_brutos(frac_c, obs_name):
    """Carrega dados BRUTOS de S(t) com comportamento em escada"""
    survival_dir = BASE_ROOT / "resultados_modelos" / "survival_analysis"
    
    # Construir nome do arquivo
    tag = f"{frac_c}_{obs_name.replace('s_', '')}"
    arquivo_survival = survival_dir / f"{tag}_survival_data.csv"
    
    if arquivo_survival.exists():
        dados = pd.read_csv(arquivo_survival)
        return dados
    else:
        print(f" Arquivo não encontrado: {arquivo_survival}")
        return None


def plotar_survival_para_todos_obs():
    """Plota gráficos de sobrevivência para cada obstáculo na lista"""
    
    print("=" * 60)
    print(f"PLOTANDO S(t)/S(0) E S(t) ORIGINAL PARA {len(OBS_LISTA)} OBSTÁCULOS")
    print(f"Obstáculos: {OBS_LISTA}")
    print("=" * 60)
    
    # Para cada obstáculo na lista
    for obs_alvo in OBS_LISTA:
        print(f"\n📊 Processando obstáculo: {obs_alvo}")
        
        # Criar duas figuras separadas
        fig1, ax1 = plt.subplots(figsize=(10, 6))  # Figura para S(t)/S(0)
        fig2, ax2 = plt.subplots(figsize=(10, 6))  # Figura para S(t) original
        
        # OBTER PALETA DE CORES "flag" (mesma para ambos os gráficos)
        cmap = colormaps.get_cmap("flag")
        
        for i, frac_c in enumerate(FRACOES_C):
            dados = carregar_dados_survival_brutos(frac_c, obs_alvo)
            
            if dados is not None and len(dados) > 0:
                # Usar cores da paleta "flag"
                cor = cmap(i)
                
                steps = dados['step'].values
                survival_norm = dados['S(t)_norm'].values
                
                # Obter S(t) original (se existir na coluna, caso contrário calcular)
                if 'S(t)' in dados.columns:
                    survival_original = dados['S(t)'].values
                else:
                    # Se não existir, calcular a partir do normalizado e S(0)
                    survival_original = dados['S(t)_norm'].values
                    # Aqui você precisaria ter S(0) disponível
                    # Supondo que S(0) seja o primeiro valor não normalizado
                    # Isso é um placeholder - ajuste conforme seus dados
                    S_0 = 1.0  # Ajuste este valor conforme necessário
                    survival_original = survival_norm * S_0
                
                # GRÁFICO 1: S(t)/S(0) normalizado
                # --------------------------------------------
                # Plotar como degraus (step) - comportamento real do Kaplan-Meier
                ax1.step(steps, survival_norm, 
                       color=cor, linewidth=2.0, 
                       where='post',  # 'post' para degraus à direita
                       label=NOMES_LEGENDA[frac_c])
                
                # Adicionar pontos nos degraus para maior clareza
                ax1.plot(steps, survival_norm, 
                       color=cor, linewidth=0, marker='o', 
                       markersize=2, alpha=0.6)
                
                # GRÁFICO 2: S(t) original
                # --------------------------------------------
                # Plotar como degraus (step)
                ax2.step(steps, survival_original, 
                       color=cor, linewidth=2.0, 
                       where='post',
                       label=NOMES_LEGENDA[frac_c])
                
                # Adicionar pontos nos degraus para maior clareza
                ax2.plot(steps, survival_original, 
                       color=cor, linewidth=0, marker='o', 
                       markersize=2, alpha=0.6)
                
                print(f"  {frac_c}: {len(dados)} degraus")
                print(f"    Normalizado: S(0)={survival_norm[0]:.3f}, S(final)={survival_norm[-1]:.3f}")
                print(f"    Original: S(0)={survival_original[0]:.3f}, S(final)={survival_original[-1]:.3f}")
        
        # Configuração do título com densidade
        try:
            numero_str = obs_alvo.split('_')[-1]
            numero = int(numero_str)
            densidade = numero / (128 * 128)
            titulo_base = fr'$\phi=${densidade:.2f}'
        except:
            titulo_base = f'Obstáculo {obs_alvo}'
        
        # CONFIGURAR GRÁFICO 1: S(t)/S(0) normalizado
        ax1.set_xlabel('steps', fontsize=16)
        ax1.set_ylabel(r'$S(t)/S(0)$', fontsize=16)
        #ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 casas para S(t)/S(0)

        ax1.set_title(f'{titulo_base}', fontsize=16)
        
        # Aplicar configurações visuais
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(alpha=0.3)
        #ax1.legend(fontsize=14, loc='upper right')
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # CONFIGURAR GRÁFICO 2: S(t) original
        ax2.set_xlabel('steps', fontsize=16)
        ax2.set_ylabel(r'$S(t)$', fontsize=16)
        ax2.set_title(f'{titulo_base}', fontsize=16)
        
        # Aplicar configurações visuais
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(alpha=0.3)
        #ax2.legend(fontsize=14, loc='upper right')
        # Não fixar limites de y para permitir visualização da escala original
        ax2.set_xlim(0)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Salvar figuras
        nome_base = obs_alvo.replace('s_obs_', '')
        
        # Figura normalizada
        nome_arquivo_norm = OUT_DIR / f"survival_escada_norm_obs_{nome_base}.pdf"
        fig1.savefig(nome_arquivo_norm, bbox_inches='tight', dpi=300)
        print(f" Gráfico normalizado salvo: {nome_arquivo_norm}")
        
        # Figura original
        nome_arquivo_orig = OUT_DIR / f"survival_escada_original_obs_{nome_base}.pdf"
        fig2.savefig(nome_arquivo_orig, bbox_inches='tight', dpi=300)
        print(f" Gráfico original salvo: {nome_arquivo_orig}")
        
        # Mostrar ambos os gráficos
        plt.show()
        plt.close(fig1)
        plt.close(fig2)

def main():
    print("INICIANDO ANÁLISE COMPARATIVA - COMPORTAMENTO EM ESCADA")
    print(f"Número de obstáculos: {len(OBS_LISTA)}")
    print(f"Diretório de saída: {OUT_DIR}")
    print("-" * 60)
    
    # 1. Plotar gráfico individual para cada obstáculo
    plotar_survival_para_todos_obs()
    
    # 2. Plotar zoom (opcional)
    # plotar_survival_zoom()
    
    # 3. Plotar todos juntos em um gráfico (opcional)
    # if len(OBS_LISTA) > 1:
    #     plotar_todos_juntos()
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 60)
    print(f"Total de gráficos gerados: {len(OBS_LISTA)}")
    print("Configurações aplicadas:")
    print("✓ Paleta de cores: 'flag'")
    print("✓ Títulos com densidade calculada")
    print("✓ Fontes tamanho 14 para ticks")
    print("✓ Grid com alpha 0.3")
    print("✓ Legendas tamanho 14")

if __name__ == "__main__":
    main()