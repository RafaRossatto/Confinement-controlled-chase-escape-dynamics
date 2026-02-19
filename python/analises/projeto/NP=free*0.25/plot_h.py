import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colormaps
from matplotlib.ticker import FormatStrFormatter

# -------------------------------
# 0. Configurações Principais
# -------------------------------
# Diretório com os arquivos H(t) cumulativos
H_CUMULATIVO_DIR = Path("/home/rafael/Dados_Doc/Np=free*0.25/L_128/resultados_modelos/survival_analysis")

# Frações de caçadores para comparar
FRACOES_C = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]

NOMES_LEGENDA = {
    "Nc=Np*0.5": r"$N^C = 0.5 N^E_0$",
    "Nc=Np*0.8": r"$N^C = 0.8 N^E_0$", 
    "Nc=Np": r"$N^C = N^E_0$"
}

# Lista de obstáculos
OBS_LISTA = [
    "ob00", "ob1638", "ob3276", "ob4915", "ob6553","ob8028",
    "ob8192","ob8355", "ob9666", "ob9830", "ob9994", "ob11468", "ob13107"]

#OBS_LISTA = ["ob8028","ob8355"]

# Diretório de saída para os plots
OUT_DIR = H_CUMULATIVO_DIR.parent / "plots_H_cumulativo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuração de plotagem
plt.rcParams.update({
    "font.size": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.labelsize": 16,
    "figure.figsize": (10, 6)
})


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)

# -------------------------------
# 1. Função para carregar dados H(t) cumulativo
# -------------------------------
def carregar_dados_H_cumulativo(frac_c, obs_name):
    """Carrega dados de H(t) cumulativo"""
    
    # Ajustar nome do arquivo
    nome_arquivo = f"{frac_c}_{obs_name}_hazard_data.csv"  # MODIFICADO AQUI
    caminho_arquivo = H_CUMULATIVO_DIR / nome_arquivo
    
    if not caminho_arquivo.exists():
        print(f"  Arquivo não encontrado: {nome_arquivo}")
        print(f"Caminho: {caminho_arquivo}")
        input()
        return None
    
    try:
        dados = pd.read_csv(caminho_arquivo)
        
        print(f" Arquivo carregado: {len(dados)} linhas")
        
        # VERIFICAÇÃO DAS COLUNAS
        colunas_necessarias = ['step', 'H(t)']
        colunas_faltando = [c for c in colunas_necessarias if c not in dados.columns]
        if colunas_faltando:
            print(f"  Colunas faltando: {colunas_faltando}")
            return None
        
        # RENOMEAR COLUNAS DE IC SE EXISTIREM
        # Se tiver H(t)_lower_0.95 e H(t)_upper_0.95, renomear para formato simplificado
        if 'H(t)_lower_0.95' in dados.columns and 'H(t)_upper_0.95' in dados.columns:
            dados = dados.rename(columns={
                'H(t)_lower_0.95': 'H(t)_lower',
                'H(t)_upper_0.95': 'H(t)_upper'
            })
            print(f" Colunas de IC renomeadas")
        elif 'H(t)_lower_0.95' not in dados.columns and 'H(t)_upper_0.95' not in dados.columns:
            print(f" Colunas de IC não encontradas")
        
        # Ordenar por step
        dados = dados.sort_values('step').dropna()
        
        return dados
        
    except Exception as e:
        print(f" Erro ao carregar {nome_arquivo}: {e}")
        return None

def plotar_H_por_fracoes():
    """Cria um gráfico para cada φ, com todas as frações - VERSÃO H(t)/H(0)"""
    
    for obs_alvo in OBS_LISTA:
        print("\n" + "=" * 60)
        print(f"PLOTANDO H(t)/H(0) CUMULATIVO E H(t) ORIGINAL: {obs_alvo}")
        print("=" * 60)
        
        # Criar duas figuras separadas
        fig1, ax1 = plt.subplots(1,1,figsize=(10, 6))  # Figura para H(t)/H(0)
        fig2, ax2 = plt.subplots(1,1,figsize=(10, 6))  # Figura para H(t) original
        
        # Obter paleta de cores (mesma para ambos os gráficos)
        cmap = colormaps.get_cmap("flag")
        
        # Calcular densidade φ
        try:
            numero_str = obs_alvo.replace('ob', '')
            numero = int(numero_str)
            phi = numero / (128**2)
            titulo_phi = fr'$\phi = {phi:.2f}$'
        except:
            phi = np.nan
            titulo_phi = f'Obstáculo {obs_alvo}'
        
        # Listas para estatísticas
        estatisticas_norm = []
        estatisticas_orig = []
        
        # Plotar cada fração em ambos os gráficos
        for i, frac_c in enumerate(FRACOES_C):
            dados = carregar_dados_H_cumulativo(frac_c, obs_alvo)
            
            if dados is not None and len(dados) > 0:
                cor = cmap(i)
                
                # Obter H(0) para normalização
                H_0 = dados['H(t)'].iloc[0] if dados['H(t)'].iloc[0] != 0 else dados['H(t)'].iloc[1]
                
                # Calcular H(t)/H(0) para gráfico normalizado
                H_normalizado = dados['H(t)'] / H_0
                
                # GRÁFICO 1: H(t)/H(0) cumulativo (normalizado)
                # --------------------------------------------
                ax1.step(dados['step'], H_normalizado, 
                        where='post', color=cor, linewidth=2.5,
                        label=NOMES_LEGENDA[frac_c], alpha=0.8)
                
                # Intervalo de confiança para gráfico normalizado
                if 'H(t)_lower_0.95' in dados.columns and 'H(t)_upper_0.95' in dados.columns:
                    H_lower_normalizado = dados['H(t)_lower_0.95'] / H_0
                    H_upper_normalizado = dados['H(t)_upper_0.95'] / H_0
                    
                    ax1.fill_between(dados['step'], 
                                   H_lower_normalizado, 
                                   H_upper_normalizado,
                                   color=cor, alpha=0.2, step='post')
                
                # Calcular estatísticas NORMALIZADAS
                if not dados.empty:
                    H_final_normalizado = H_normalizado.iloc[-1]
                    H_max_normalizado = H_normalizado.max()
                    estatisticas_norm.append((frac_c, H_final_normalizado, H_max_normalizado, H_0))
                
                # GRÁFICO 2: H(t) original
                # --------------------------------------------
                ax2.step(dados['step'], dados['H(t)'], 
                        where='post', color=cor, linewidth=2.5,
                        label=NOMES_LEGENDA[frac_c], alpha=0.8)
                
                # Intervalo de confiança para gráfico original
                if 'H(t)_lower_0.95' in dados.columns and 'H(t)_upper_0.95' in dados.columns:
                    ax2.fill_between(dados['step'], 
                                   dados['H(t)_lower_0.95'], 
                                   dados['H(t)_upper_0.95'],
                                   color=cor, alpha=0.2, step='post')
                
                # Calcular estatísticas ORIGINAIS
                if not dados.empty:
                    H_final_original = dados['H(t)'].iloc[-1]
                    H_max_original = dados['H(t)'].max()
                    estatisticas_orig.append((frac_c, H_final_original, H_max_original))
        
        # Configurar GRÁFICO 1: H(t)/H(0) normalizado
        ax1.set_xlabel('steps', fontsize=16)
        ax1.set_ylabel('$H(t)/H(0)$', fontsize=16)
        ax1.set_title(f'{titulo_phi}', fontsize=16)
        ax1.legend(fontsize=14)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0.95)  # Ajuste o limite conforme necessário
        
        # Configurar GRÁFICO 2: H(t) original
        ax2.set_xlabel('steps', fontsize=14)
        ax2.set_ylabel('$H(t)$', fontsize=14)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 2 casas para S(t)/S(0)
        ax2.set_title(f'{titulo_phi}', fontsize=16)
        ax2.legend(fontsize=14)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(left=0)
        # ax2.set_ylim()  # Remova ou ajuste conforme necessário
        
        # Ajustar layout e salvar ambas as figuras
        plt.tight_layout()
        
        # Salvar figura normalizada
        nome_arquivo_norm = OUT_DIR / f"H_cumulativo_normalizado_{obs_alvo}.pdf"
        fig1.savefig(nome_arquivo_norm, dpi=300, bbox_inches='tight')
        print(f" Gráfico normalizado salvo: {nome_arquivo_norm}")
        
        # Salvar figura original
        nome_arquivo_orig = OUT_DIR / f"H_cumulativo_original_{obs_alvo}.pdf"
        fig2.savefig(nome_arquivo_orig, dpi=300, bbox_inches='tight')
        print(f" Gráfico original salvo: {nome_arquivo_orig}")
        
        # Mostrar estatísticas NORMALIZADAS
        if estatisticas_norm:
            print(f"\n ESTATÍSTICAS H(t)/H(0) final:")
            print(f"    {'Frações':<20} | H(0)     | H(final)/H(0) | H(max)/H(0)")
            print(f"    {'-'*20} | {'-'*8} | {'-'*13} | {'-'*12}")
            for frac_c, H_final_norm, H_max_norm, H_0 in estatisticas_norm:
                print(f"    {NOMES_LEGENDA[frac_c]:<20} | {H_0:.4f}   | {H_final_norm:.3f}         | {H_max_norm:.3f}")
        
        # Mostrar estatísticas ORIGINAIS
        if estatisticas_orig:
            print(f"\n  📊 ESTATÍSTICAS H(t) original final:")
            print(f"    {'Frações':<20} | H(final)  | H(max)")
            print(f"    {'-'*20} | {'-'*9} | {'-'*6}")
            for frac_c, H_final_orig, H_max_orig in estatisticas_orig:
                print(f"    {NOMES_LEGENDA[frac_c]:<20} | {H_final_orig:.4f}  | {H_max_orig:.4f}")
        
        # Mostrar ambos os gráficos
        plt.show()
        plt.close(fig1)
        plt.close(fig2)

# -------------------------------
# 6. Função para plotar hazard instantâneo
# -------------------------------
def plotar_hazard_instantaneo_por_fracoes():
    """Cria um gráfico do hazard instantâneo para cada φ, com todas as frações"""
    
    for obs_alvo in OBS_LISTA:
        print("\n" + "=" * 60)
        print(f"PLOTANDO HAZARD INSTANTÂNEO: {obs_alvo}")
        print("=" * 60)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Obter paleta de cores
        cmap = colormaps.get_cmap("flag")
        
        # Calcular densidade φ
        try:
            numero_str = obs_alvo.replace('ob', '')
            numero = int(numero_str)
            phi = numero / (128**2)
            titulo_phi = fr'$\phi = {phi:.2f}$'
        except:
            phi = np.nan
            titulo_phi = f'Obstáculo {obs_alvo}'
        
        # Plotar cada fração
        for i, frac_c in enumerate(FRACOES_C):
            dados = carregar_dados_H_cumulativo(frac_c, obs_alvo)
            
            if dados is not None and len(dados) > 0 and 'hazard_instantaneo' in dados.columns:
                cor = cmap(i)
                
                # Filtrar valores não nulos e finitos
                mask = dados['hazard_instantaneo'].notna() & np.isfinite(dados['hazard_instantaneo'])
                dados_filtrados = dados[mask]
                
                if len(dados_filtrados) > 0:
                    # Encontrar h(0) - primeiro valor não nulo
                    h0 = dados_filtrados['hazard_instantaneo'].iloc[0] if len(dados_filtrados) > 0 else 1.0
                    
                    # Normalizar por h(0)
                    if h0 != 0:
                        hazard_normalizado = dados_filtrados['hazard_instantaneo'] / h0
                    else:
                        # Se h(0) for zero, não normalizar (ou usar 1 como fallback)
                        hazard_normalizado = dados_filtrados['hazard_instantaneo']
                        print(f"  Atenção: h(0) = 0 para {NOMES_LEGENDA[frac_c]}")
                    
                    # Plotar hazard instantâneo normalizado
                    ax.scatter(dados_filtrados['step'], hazard_normalizado,
                              color=cor, s=30, alpha=0.7,
                              label=NOMES_LEGENDA[frac_c])
                    
                    # Conectar pontos com linha
                    ax.plot(dados_filtrados['step'], hazard_normalizado,
                           color=cor, linewidth=1.5, alpha=0.5, linestyle='-')
                    
                    # Calcular estatísticas do hazard normalizado
                    h_mean = hazard_normalizado.mean()
                    h_std = hazard_normalizado.std()
                    h_max = hazard_normalizado.max()
                    
                    print(f"  {NOMES_LEGENDA[frac_c]:<20}: h(0)={h0:.4f}, média_norm={h_mean:.4f}, std_norm={h_std:.4f}, max_norm={h_max:.4f}")
        
        # Configurar gráfico
        ax.set_xlabel('steps', fontsize=14)
        ax.set_ylabel('$h(t)/h(0)$ (hazard normalizado)', fontsize=14)
        ax.set_title(f'Hazard Instantâneo Normalizado por h(0): {titulo_phi}', fontsize=16)
        ax.legend(fontsize=14, loc='upper right')
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # Salvar figura
        nome_arquivo = OUT_DIR / f"hazard_instantaneo_normalizado_{obs_alvo}.pdf"
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        print(f" Gráfico salvo: {nome_arquivo}")
        plt.show()
        plt.close(fig)


# -------------------------------
# 5. Função principal
# -------------------------------
def main():
    print("PLOTAGEM H(t) CUMULATIVO")
    print(f"Diretório dados: {H_CUMULATIVO_DIR}")
    print(f"Diretório saída: {OUT_DIR}")
    
    # Verificar se existem arquivos
    arquivos = list(H_CUMULATIVO_DIR.glob("*_hazard_data.csv"))
    print(f"\nArquivos encontrados: {len(arquivos)}")
    
    if len(arquivos) == 0:
        print(" Nenhum arquivo encontrado. Execute primeiro o programa de conversão S→H.")
        return
    
    # Mostrar alguns exemplos
    print("\n Exemplos de arquivos:")
    for arquivo in arquivos[:5]:
        print(f"  {arquivo.name}")
    
    # Menu
    print("\n" + "="*60)
    print("OPÇÕES DE PLOTAGEM:")
    print("="*60)
    print("1. Plotar H(t) e S(t) lado a lado (um gráfico por φ)")
    print("2. Plotar hazard instantâneo h(t) (um gráfico por φ)")
    print("3. Todas as opções")
    opcao = input("\nEscolha (1-3): ").strip()
    
    if opcao in ["1", "3"]:
        plotar_H_por_fracoes()
    
    if opcao in ["2", "3"]:
        plotar_hazard_instantaneo_por_fracoes()


    print("\n" + "="*60)
    print(" PLOTAGEM CONCLUÍDA!")

if __name__ == "__main__":
    main()