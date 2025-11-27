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
    "Nc=Np*0.5": r"$N^C_0 = 0.5 N^E_0$",
    "Nc=Np*0.8": r"$N^C_0 = 0.8 N^E_0$", 
    "Nc=Np": r"$N^C_0 = N^E_0$"
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
# 1. Carregar dados de h(t) para uma fração e obstáculo
# -------------------------------
def carregar_dados_hazard(frac_c, obs_name):
    """Carrega dados de h(t) para uma fração e obstáculo específicos"""
    hazard_dir = BASE_ROOT / "resultados_modelos" / "hazard_analysis"
    
    # Construir nome do arquivo
    tag = f"{frac_c}_{obs_name.replace('s_', '')}"
    arquivo_hazard = hazard_dir / f"{tag}_hazard_data.csv"
    
    print(f"Tentando carregar: {arquivo_hazard}")
    
    if arquivo_hazard.exists():
        dados = pd.read_csv(arquivo_hazard)
        # Remover possíveis NaNs
        dados = dados.dropna()
        print(f"✅ Carregado: {len(dados)} pontos para {frac_c}")
        return dados
    else:
        print(f"❌ Arquivo não encontrado: {arquivo_hazard}")
        return None

# -------------------------------
# 2. Plotar h(t)/h(0) com bandas de confiança
# -------------------------------
def plotar_comparacao_hazard_normalizado():
    """Plota h(t)/h(0) para as três frações de caçadores"""
    
    print("=" * 60)
    print("PLOTANDO COMPARAÇÃO h(t)/h(0)")
    print(f"Obstáculo: {OBS_ALVO}")
    print(f"Frações: {FRACOES_C}")
    print("=" * 60)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Carregar e plotar dados para cada fração
    dados_por_fracao = {}
    
    for i, frac_c in enumerate(FRACOES_C):
        dados = carregar_dados_hazard(frac_c, OBS_ALVO)
        
        if dados is not None and len(dados) > 0:
            # Plotar h(t)/h(0) com banda de confiança
            linha = ax.plot(dados['step'], dados['h(t)_norm'], 
                          color=CORES[i], linewidth=2.5, 
                          label=NOMES_LEGENDA[frac_c])
            
            # Adicionar banda de confiança (área sombreada)
            ax.fill_between(dados['step'], 
                          dados['h(t)_lower_norm'], 
                          dados['h(t)_upper_norm'],
                          color=CORES[i], alpha=0.3)
            
            dados_por_fracao[frac_c] = dados
            
            print(f"📊 {frac_c}: h(0)≈{dados['h(t)_norm'].iloc[0]:.3f}, h(final)≈{dados['h(t)_norm'].iloc[-1]:.3f}")
    
    # Configurações do gráfico
    ax.set_xlabel('Tempo (steps)', fontsize=16)
    ax.set_ylabel('h(t)/h(0)', fontsize=16)
    ax.set_title(f'Funções de Hazard Normalizadas\nObstáculo {OBS_ALVO}', fontsize=14)
    
    # Grid e limites
    ax.grid(True, alpha=0.3)
    #ax.set_ylim(0, 2.0)  # Ajustar conforme os dados
    
    # Legenda
    ax.legend(loc='upper right')
    
    # Melhorar aparência
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Salvar figura
    nome_arquivo = OUT_DIR / f"comparacao_h_t_h0_{OBS_ALVO}.pdf"
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
    plt.savefig(OUT_DIR / f"comparacao_h_t_h0_{OBS_ALVO}.png", bbox_inches='tight', dpi=300)
    
    print(f"\n💾 Gráfico salvo como:")
    print(f"   {nome_arquivo}")
    print(f"   {OUT_DIR / f'comparacao_h_t_h0_{OBS_ALVO}.png'}")
    
    plt.show()
    
    return dados_por_fracao

# -------------------------------
# 3. Plotar h(t) original (não normalizado) também
# -------------------------------
def plotar_hazard_original():
    """Plota h(t) original (não normalizado) para comparação"""
    
    print("\n" + "=" * 60)
    print("PLOTANDO h(t) ORIGINAL (NÃO NORMALIZADO)")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, frac_c in enumerate(FRACOES_C):
        dados = carregar_dados_hazard(frac_c, OBS_ALVO)
        
        if dados is not None and len(dados) > 0:
            # Plotar h(t) original
            ax.plot(dados['step'], dados['h(t)'], 
                   color=CORES[i], linewidth=2.5, 
                   label=NOMES_LEGENDA[frac_c])
            
            # Banda de confiança para h(t) original
            ax.fill_between(dados['step'], 
                          dados['h(t)_lower'], 
                          dados['h(t)_upper'],
                          color=CORES[i], alpha=0.3)
            
            print(f"📊 {frac_c}: h(0) original≈{dados['h(t)'].iloc[0]:.6f}, h(final) original≈{dados['h(t)'].iloc[-1]:.6f}")
    
    # Configurações do gráfico
    ax.set_xlabel('Tempo (steps)', fontsize=16)
    ax.set_ylabel('h(t)', fontsize=16)
    ax.set_title(f'Funções de Hazard Originais\nObstáculo {OBS_ALVO}', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    #ax.set_ylim(0, 0.1)  # Ajustar conforme os dados
    ax.legend(loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Salvar figura
    nome_arquivo = OUT_DIR / f"comparacao_h_t_original_{OBS_ALVO}.pdf"
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=300)
    plt.savefig(OUT_DIR / f"comparacao_h_t_original_{OBS_ALVO}.png", bbox_inches='tight', dpi=300)
    
    print(f"\n💾 Gráfico original salvo como:")
    print(f"   {nome_arquivo}")
    
    plt.show()


# -------------------------------
# 5. Função principal para hazard
# -------------------------------
def main_hazard():
    print("INICIANDO ANÁLISE COMPARATIVA DE HAZARD")
    print(f"Diretório de saída: {OUT_DIR}")
    
    # Plotar comparação h(t)/h(0)
    dados_comparacao = plotar_comparacao_hazard_normalizado()
    
    # Plotar h(t) original também
    plotar_hazard_original()
    
    # Mostrar estatísticas resumidas
    print("\n" + "=" * 60)
    print("RESUMO ESTATÍSTICO - HAZARD")
    print("=" * 60)
    
    for frac_c in FRACOES_C:
        if frac_c in dados_comparacao:
            dados = dados_comparacao[frac_c]
            h0_norm = dados['h(t)_norm'].iloc[0] if len(dados) > 0 else np.nan
            h_final_norm = dados['h(t)_norm'].iloc[-1] if len(dados) > 0 else np.nan
            h0_original = dados['h(t)'].iloc[0] if len(dados) > 0 else np.nan
            h_final_original = dados['h(t)'].iloc[-1] if len(dados) > 0 else np.nan
            
            print(f"{NOMES_LEGENDA[frac_c]:<20} | h(0) norm={h0_norm:.3f} | h(final) norm={h_final_norm:.3f}")
            print(f"{'':<20} | h(0) orig={h0_original:.6f} | h(final) orig={h_final_original:.6f}")
            print("-" * 70)

if __name__ == "__main__":
    main_hazard()