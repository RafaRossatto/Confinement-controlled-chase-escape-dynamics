import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# 0. Configurações Principais
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128"

# Frações de caçadores para comparar
FRACOES_C = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]
CORES = ['blue', 'red', 'green']
NOMES_LEGENDA = {
    "Nc=Np*0.5": r"$N^C = 0.5 N^E_0$",
    "Nc=Np*0.8": r"$N^C = 0.8 N^E_0$", 
    "Nc=Np": r"$N^C = N^E_0$"
}

# Obstáculo para análise
OBS_ALVO = "s_obs_00"

# Diretórios
OUT_DIR = BASE_ROOT / "resultados_modelos" / "plots_comparativos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuração de plotagem
plt.rcParams.update({
    "font.size": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.labelsize": 16
})

# -------------------------------
# 1. Carregar dados brutos do Kaplan-Meier
# -------------------------------
def carregar_dados_survival_brutos(frac_c, obs_name):
    """Carrega dados BRUTOS de S(t) com comportamento em escada"""
    survival_dir = BASE_ROOT / "resultados_modelos" / "survival_analysis"
    
    # Construir nome do arquivo
    tag = f"{frac_c}_{obs_name.replace('s_', '')}"
    arquivo_survival = survival_dir / f"{tag}_survival_data.csv"
    
    print(f"Tentando carregar: {arquivo_survival}")
    
    if arquivo_survival.exists():
        dados = pd.read_csv(arquivo_survival)
        print(f"✅ Carregado: {len(dados)} pontos para {frac_c}")
        return dados
    else:
        print(f"❌ Arquivo não encontrado: {arquivo_survival}")
        return None

# -------------------------------
# 2. Plotar S(t)/S(0) com comportamento em ESCADA
# -------------------------------
def plotar_survival_escada():
    """Plota S(t)/S(0) mostrando o comportamento em escada do Kaplan-Meier"""
    
    print("=" * 60)
    print("PLOTANDO S(t)/S(0) - COMPORTAMENTO EM ESCADA")
    print(f"Obstáculo: {OBS_ALVO}")
    print("=" * 60)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, frac_c in enumerate(FRACOES_C):
        dados = carregar_dados_survival_brutos(frac_c, OBS_ALVO)
        
        if dados is not None and len(dados) > 0:
            # Para mostrar o comportamento em escada, precisamos do plot step
            steps = dados['step'].values
            survival_norm = dados['S(t)_norm'].values
            
            # Plotar como degraus (step) - isso mostra o comportamento real do Kaplan-Meier
            ax.step(steps, survival_norm, 
                   color=CORES[i], linewidth=2.0, 
                   where='post',  # 'post' para degraus à direita
                   label=NOMES_LEGENDA[frac_c])
            
            # Opcional: adicionar pontos nos degraus para maior clareza
            ax.plot(steps, survival_norm, 
                   color=CORES[i], linewidth=0, marker='o', 
                   markersize=2, alpha=0.6)
            
            print(f"📊 {frac_c}: {len(dados)} degraus, S(0)={survival_norm[0]:.3f}, S(final)={survival_norm[-1]:.3f}")
    
    # Configurações do gráfico
    ax.set_xlabel('Tempo (steps)', fontsize=16)
    ax.set_ylabel('S(t)/S(0)', fontsize=16)
    ax.set_title('Curvas de Sobrevivência Normalizadas - Comportamento em Escada\nKaplan-Meier', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Salvar figura
    nome_arquivo = OUT_DIR / f"comparacao_S_t_S0_escada_{OBS_ALVO}.pdf"
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
    plt.savefig(OUT_DIR / f"comparacao_S_t_S0_escada_{OBS_ALVO}.png", bbox_inches='tight', dpi=300)
    
    print(f"\n💾 Gráfico em escada salvo como:")
    print(f"   {nome_arquivo}")
    
    plt.show()

# -------------------------------
# 3. Plotar versão detalhada com zoom nos primeiros degraus
# -------------------------------

# 4. Plotar comparação lado a lado: suave vs escada
# -------------------------------
def plotar_survival_escada():
    """Plota S(t)/S(0) mostrando o comportamento em escada do Kaplan-Meier"""
    
    print("=" * 60)
    print("PLOTANDO S(t)/S(0) - COMPORTAMENTO EM ESCADA")
    print(f"Obstáculo: {OBS_ALVO}")
    print("=" * 60)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, frac_c in enumerate(FRACOES_C):
        dados = carregar_dados_survival_brutos(frac_c, OBS_ALVO)
        
        if dados is not None and len(dados) > 0:
            # Para mostrar o comportamento em escada, precisamos do plot step
            steps = dados['step'].values
            survival_norm = dados['S(t)_norm'].values
            
            # Plotar como degraus (step) - isso mostra o comportamento real do Kaplan-Meier
            ax.step(steps, survival_norm, 
                   color=CORES[i], linewidth=2.0, 
                   where='post',  # 'post' para degraus à direita
                   label=NOMES_LEGENDA[frac_c])
            
            # Opcional: adicionar pontos nos degraus para maior clareza
            ax.plot(steps, survival_norm, 
                   color=CORES[i], linewidth=0, marker='o', 
                   markersize=2, alpha=0.6)
            
            print(f"📊 {frac_c}: {len(dados)} degraus, S(0)={survival_norm[0]:.3f}, S(final)={survival_norm[-1]:.3f}")
    
    # Configurações do gráfico
    ax.set_xlabel('Tempo (steps)', fontsize=16)
    ax.set_ylabel('S(t)/S(0)', fontsize=16)
    ax.set_title('Curvas de Sobrevivência Normalizadas - Comportamento em Escada\nKaplan-Meier', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Salvar figura
    nome_arquivo = OUT_DIR / f"comparacao_S_t_S0_escada_{OBS_ALVO}.pdf"
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
    plt.savefig(OUT_DIR / f"comparacao_S_t_S0_escada_{OBS_ALVO}.png", bbox_inches='tight', dpi=300)
    
    print(f"\n💾 Gráfico em escada salvo como:")
    print(f"   {nome_arquivo}")
    
    plt.show()
# -------------------------------
# 5. Função principal
# -------------------------------
def main():
    print("INICIANDO ANÁLISE COMPARATIVA - COMPORTAMENTO EM ESCADA")
    print(f"Diretório de saída: {OUT_DIR}")
    
    # Plotar versão em escada
    plotar_survival_escada()
    
    # Plotar versão detalhada com zoom
    plotar_survival_escada()
    
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 60)
    print("Gráficos gerados:")
    print("1. comparacao_S_t_S0_escada_s_obs_00.pdf - Comportamento em escada")
    print("2. comparacao_S_t_S0_escada_detalhe_s_obs_00.pdf - Zoom nos primeiros degraus")
    print("3. comparacao_suave_vs_escada_s_obs_00.pdf - Comparação entre versões")

if __name__ == "__main__":
    main()