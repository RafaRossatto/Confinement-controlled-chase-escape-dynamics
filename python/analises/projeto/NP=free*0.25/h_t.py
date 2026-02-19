import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------
# 0. Configurações Principais
# -------------------------------
SURVIVAL_DIR = Path("/home/rafael/Dados_Doc/Np=free*0.25/resultados_modelos/survival_analysis")

# Frações de caçadores para processar
FRACOES_C = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]

# Diretório de saída para H(t)
OUT_DIR = SURVIVAL_DIR.parent / "hazard_cumulativo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuração de plotagem
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12
})

# -------------------------------
# 1. Função para calcular H(t) = -log(S(t))
# -------------------------------
def calcular_H_de_S(arquivo_S, frac_c, obs_name):
    """Calcula H(t) = -log(S(t)) a partir de arquivo de sobrevivência"""
    
    print(f"\n📊 Processando: {arquivo_S.name}")
    
    try:
        # Carregar dados de S(t)
        dados_S = pd.read_csv(arquivo_S)
        
        # Verificar colunas necessárias
        colunas_necessarias = ["step", "S(t)"]
        for col in colunas_necessarias:
            if col not in dados_S.columns:
                print(f"   Coluna '{col}' não encontrada em {arquivo_S.name}")
                print(f"  Colunas disponíveis: {list(dados_S.columns)}")
                return None
        
        print(f"  ✅ {len(dados_S)} pontos | S(0) = {dados_S['S(t)'].iloc[0]:.6f}")
        
        # Criar DataFrame para H(t)
        dados_H = pd.DataFrame()
        dados_H["step"] = dados_S["step"]
        
        # -------------------------------
        # CALCULAR H(t) = -log(S(t))
        # -------------------------------
        # Garantir que S(t) > 0 para evitar log(0)
        epsilon = 1e-10
        S_t = np.maximum(dados_S["S(t)"], epsilon)
        
        # Calcular H(t) principal
        dados_H["H(t)"] = -np.log(S_t)
        
        # -------------------------------
        # Calcular intervalos de confiança se existirem
        # -------------------------------
        if "S(t)_lower" in dados_S.columns and "S(t)_upper" in dados_S.columns:
            S_lower = np.maximum(dados_S["S(t)_lower"], epsilon)
            S_upper = np.maximum(dados_S["S(t)_upper"], epsilon)
            
            # IMPORTANTE: Inverter porque -log é função decrescente
            # H_lower = -log(S_upper)  (maior S → menor H)
            # H_upper = -log(S_lower)  (menor S → maior H)
            dados_H["H(t)_lower"] = -np.log(S_upper)
            dados_H["H(t)_upper"] = -np.log(S_lower)
            
            print(f"  ✅ Intervalos de confiança calculados")
        
        # -------------------------------
        # Adicionar metadados
        # -------------------------------
        dados_H["frac_c"] = frac_c
        dados_H["obs"] = obs_name
        
        # Calcular densidade φ do nome do arquivo
        obs_numero = obs_name.replace("ob", "")
        try:
            phi = int(obs_numero) / (128**2)
        except:
            phi = 0.0
        dados_H["phi"] = phi
        
        # -------------------------------
        # Estatísticas
        # -------------------------------
        H0 = dados_H["H(t)"].iloc[0]
        H_final = dados_H["H(t)"].iloc[-1]
        
        print(f"  📈 H(0) = {H0:.6f}")
        print(f"  📈 H(final) = {H_final:.6f}")
        print(f"  📈 ΔH = {H_final - H0:.6f}")
        
        if "S(t)_norm" in dados_S.columns:
            print(f"  ℹ️  S(t) já normalizado encontrado")
        
        return dados_H
        
    except Exception as e:
        print(f"  ❌ Erro ao processar {arquivo_S.name}: {e}")
        return None

# -------------------------------
# 2. Processar todos os arquivos
# -------------------------------
def processar_todos_arquivos():
    """Processa todos os arquivos S(t) e calcula H(t)"""
    
    print("="*60)
    print("CALCULANDO H(t) = -log(S(t))")
    print(f"Diretório de entrada: {SURVIVAL_DIR}")
    print(f"Diretório de saída: {OUT_DIR}")
    print("="*60)
    
    # Encontrar todos os arquivos de sobrevivência
    arquivos_S = list(SURVIVAL_DIR.glob("*_survival_data.csv"))
    print(f"Encontrados {len(arquivos_S)} arquivos S(t)")
    
    if len(arquivos_S) == 0:
        print("❌ Nenhum arquivo encontrado!")
        return []
    
    # Lista para armazenar todos os H(t)
    todos_H = []
    
    for arquivo in arquivos_S:
        # Extrair informações do nome do arquivo
        nome = arquivo.stem.replace("_survival_data", "")
        
        # Extrair fração e obstáculo do nome
        # Formato esperado: "Nc=Np*0.5_ob00"
        partes = nome.split("_")
        
        if len(partes) >= 2:
            frac_c = partes[0]  # "Nc=Np*0.5"
            obs_name = partes[1]  # "ob00"
            
            # Verificar se a fração está na lista de interesse
            if frac_c in FRACOES_C:
                # Calcular H(t)
                dados_H = calcular_H_de_S(arquivo, frac_c, obs_name)
                
                if dados_H is not None:
                    todos_H.append(dados_H)
        
        else:
            print(f"  ⚠️  Nome do arquivo incomum: {nome}")
    
    return todos_H

# -------------------------------
# 3. Salvar e plotar resultados
# -------------------------------
def salvar_e_plotar_resultados(todos_H):
    """Salva os dados e gera gráficos"""
    
    if not todos_H:
        print("\n❌ Nenhum dado H(t) foi gerado!")
        return
    
    print(f"\n" + "="*60)
    print(f"SALVANDO RESULTADOS ({len(todos_H)} datasets)")
    print("="*60)
    
    # -------------------------------
    # 3.1 Salvar cada H(t) individualmente
    # -------------------------------
    for dados_H in todos_H:
        frac_c = dados_H["frac_c"].iloc[0]
        obs_name = dados_H["obs"].iloc[0]
        
        # Nome do arquivo de saída
        nome_arquivo = f"{frac_c}_{obs_name}_H_cumulativo.csv"
        caminho_saida = OUT_DIR / nome_arquivo
        
        # Salvar
        dados_H.to_csv(caminho_saida, index=False)
        print(f"  💾 {nome_arquivo}")
    
    # -------------------------------
    # 3.2 Salvar dados combinados
    # -------------------------------
    df_combinado = pd.concat(todos_H, ignore_index=True)
    
    # Salvar todos juntos
    saida_combinado = OUT_DIR / "todos_H_cumulativo.csv"
    df_combinado.to_csv(saida_combinado, index=False)
    print(f"\n  💾 Arquivo combinado: todos_H_cumulativo.csv")
    
    # -------------------------------
    # 3.3 Criar resumo estatístico
    # -------------------------------
    resumo = []
    for dados_H in todos_H:
        frac_c = dados_H["frac_c"].iloc[0]
        obs_name = dados_H["obs"].iloc[0]
        phi = dados_H["phi"].iloc[0]
        
        H0 = dados_H["H(t)"].iloc[0]
        H_final = dados_H["H(t)"].iloc[-1]
        H_max = dados_H["H(t)"].max()
        
        resumo.append({
            "frac_c": frac_c,
            "obs": obs_name,
            "phi": phi,
            "H(0)": H0,
            "H(final)": H_final,
            "H(max)": H_max,
            "ΔH": H_final - H0,
            "n_pontos": len(dados_H)
        })
    
    df_resumo = pd.DataFrame(resumo)
    saida_resumo = OUT_DIR / "resumo_H_cumulativo.csv"
    df_resumo.to_csv(saida_resumo, index=False)
    
    print(f"  💾 Resumo estatístico: resumo_H_cumulativo.csv")
    
    # Mostrar tabela resumo
    print(f"\n" + "="*60)
    print("RESUMO ESTATÍSTICO")
    print("="*60)
    print(df_resumo[["frac_c", "obs", "phi", "H(0)", "H(final)", "ΔH"]].to_string(index=False))
    
    # -------------------------------
    # 3.4 Plotar gráfico rápido de verificação
    # -------------------------------
    plotar_verificacao(todos_H)
    
    return df_resumo

# -------------------------------
# 4. Plotar gráfico de verificação
# -------------------------------
def plotar_verificacao(todos_H):
    """Plota um gráfico rápido para verificar os resultados"""
    
    print(f"\n" + "="*60)
    print("GERANDO GRÁFICO DE VERIFICAÇÃO")
    print("="*60)
    
    # Criar figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cores para diferentes frações
    cores = {"Nc=Np*0.5": "blue", "Nc=Np*0.8": "green", "Nc=Np": "red"}
    
    # Agrupar por fração
    for frac_c in FRACOES_C:
        # Filtrar dados desta fração
        dados_frac = [d for d in todos_H if d["frac_c"].iloc[0] == frac_c]
        
        if not dados_frac:
            continue
        
        cor = cores.get(frac_c, "gray")
        
        for dados_H in dados_frac:
            obs_name = dados_H["obs"].iloc[0]
            phi = dados_H["phi"].iloc[0]
            
            # Plot H(t) vs tempo
            axes[0].plot(dados_H["step"], dados_H["H(t)"], 
                        color=cor, alpha=0.7, linewidth=1,
                        label=f"{frac_c}, φ={phi:.3f}")
            
            # Plot H(final) vs φ
            H_final = dados_H["H(t)"].iloc[-1]
            axes[1].scatter(phi, H_final, color=cor, s=50, alpha=0.7)
    
    # Configurar gráfico 1: H(t) vs tempo
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel("H(t) = -log(S(t))")
    axes[0].set_title("Hazard Cumulativo")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best", ncol=2)
    
    # Configurar gráfico 2: H(final) vs φ
    axes[1].set_xlabel("Densidade φ")
    axes[1].set_ylabel("H(final)")
    axes[1].set_title("Hazard Final vs Densidade")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    saida_grafico = OUT_DIR / "verificacao_H_cumulativo.png"
    plt.savefig(saida_grafico, dpi=150, bbox_inches='tight')
    print(f"  💾 Gráfico salvo: {saida_grafico}")
    
    plt.show()
    plt.close(fig)

# -------------------------------
# 5. Função principal
# -------------------------------
def main():
    print("🔧 PROGRAMA: CÁLCULO DE H(t) = -log(S(t))")
    print("="*60)
    
    # 1. Processar arquivos
    todos_H = processar_todos_arquivos()
    
    # 2. Salvar e plotar
    if todos_H:
        salvar_e_plotar_resultados(todos_H)
        
        print(f"\n" + "="*60)
        print("✅ PROCESSAMENTO CONCLUÍDO!")
        print("="*60)
        print(f"📁 Arquivos H(t) salvos em: {OUT_DIR}")
        print(f"📊 Número de datasets processados: {len(todos_H)}")
    else:
        print("\n❌ Nenhum dado processado.")

if __name__ == "__main__":
    main()