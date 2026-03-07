import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pathlib import Path
import re

# -------------------------------
# Configurações
# -------------------------------
DATA_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"
OUT_DIR = DATA_ROOT /"resultados_modelos" /"paper_response"/ "cost_x_time"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1. Função para extrair parâmetros
# -------------------------------
def extract_parameters(filename):
    pattern = r'NC_(\d+)_NE_(\d+)_O_(\d+)'
    match = re.search(pattern, str(filename))
    if match:
        return {
            'NC': int(match.group(1)),
            'NE': int(match.group(2)), 
            'O': int(match.group(3))
        }
    return None

# -------------------------------
# 2. Carregar dados de sobrevivência
# -------------------------------
def load_survival_data(data_dir):
    files = sorted(Path(data_dir).glob("*_prey_trajectories.csv"))
    all_data = []
    for sim_id, f in enumerate(files):
        df = pd.read_csv(f)
        max_time = df["timestep"].max()
        last_times = df.groupby("cell_id")["timestep"].max().reset_index()
        last_times["event"] = (last_times["timestep"] < max_time).astype(int)
        last_times.rename(columns={"timestep": "time"}, inplace=True)
        last_times["sim_id"] = sim_id
        all_data.append(last_times)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

# -------------------------------
# 3. Calcular custo normalizado SIMPLIFICADO
# -------------------------------
def calculate_normalized_cost(survival_data, Nc, cost_per_hunter=1.0):
    """
    Calcula custo normalizado pelo máximo SEM incerteza
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_data["time"], event_observed=survival_data["event"])
    
    # Obter curva de sobrevivência
    survival_func = kmf.survival_function_
    times = survival_func.index.values
    S_t = survival_func['KM_estimate'].values
    
    # Número total inicial de presas
    Ne0 = len(survival_data)
    
    # Custo instantâneo
    instantaneous_cost = (Nc * cost_per_hunter) / np.maximum(Ne0 * S_t, 1)
    
    # Normalizar pelo custo máximo
    max_cost = instantaneous_cost.max()
    normalized_cost = instantaneous_cost / max_cost
    
    return times, normalized_cost, S_t, max_cost

# -------------------------------
# 4. Coletar dados para TODOS os gráficos
# -------------------------------
def collect_all_data():
    """
    Coleta todos os dados organizados de duas formas:
    1. Por configuração de agentes (original)
    2. Por densidade de obstáculos (novo)
    """
    configs = {
        "Nc=Np": DATA_ROOT / "Nc=Np",
        "Nc=Np*0.8": DATA_ROOT / "Nc=Np*0.8", 
        "Nc=Np*0.5": DATA_ROOT / "Nc=Np*0.5"
    }
    
    obstaculos = [
        "s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", 
        "s_obs_6553", "s_obs_8192", "s_obs_9830", "s_obs_11468", "s_obs_13107"
    ]
    
    # Dicionário para armazenar dados por configuração de AGENTES (original)
    config_data = {
        "Nc=Np": [],
        "Nc=Np*0.8": [],
        "Nc=Np*0.5": []
    }
    
    # Dicionário para armazenar dados por OBSTÁCULOS (NOVO)
    obstacle_data = {
        0: [],        # O=0
        1638: [],     # O=1638
        3276: [],     # O=3276
        4915: [],     # O=4915
        6553: [],     # O=6553
        8192: [],     # O=8192
        9830: [],     # O=9830
        11468: [],    # O=11468
        13107: []     # O=13107
    }
    
    for config_name, config_base in configs.items():
        print(f"\nColetando dados: {config_name}")
        print(f"Procurando em: {config_base}")
        
        for obs in obstaculos:
            obs_dir = config_base / obs
            print(f"  Verificando: {obs_dir}")
            
            if not obs_dir.exists():
                print(f" Diretório não existe: {obs_dir}")
                continue
            
            # Carregar dados
            survival_data = load_survival_data(obs_dir)
            if survival_data is None:
                print(f"Nenhum dado encontrado em: {obs_dir}")
                continue
            
            # Extrair parâmetros
            prey_files = list(obs_dir.glob("*prey_per_step.csv"))
            if prey_files:
                params = extract_parameters(prey_files[0].name)
                Nc = params['NC'] if params else len(survival_data)
                Ne = params['NE'] if params else len(survival_data)
            else:
                Nc = len(survival_data)
                Ne = len(survival_data)
            
            # Extrair valor dos obstáculos
            obs_match = re.search(r'obs_(\d+)', obs)
            obstacles = int(obs_match.group(1)) if obs_match else 0
            
            # Calcular custo normalizado SEM INCERTEZA
            times, normalized_cost, S_t, max_cost = calculate_normalized_cost(survival_data, Nc)
            
            # Dados para configuração de AGENTES (original)
            config_data[config_name].append({
                'obstacles': obstacles,
                'times': times,
                'normalized_cost': normalized_cost,
                'survival': S_t,
                'max_cost': max_cost,
                'Nc': Nc,
                'Ne': Ne
            })
            
            # Dados para configuração de OBSTÁCULOS (NOVO)
            obstacle_data[obstacles].append({
                'config': config_name,
                'times': times,
                'normalized_cost': normalized_cost,
                'survival': S_t,
                'max_cost': max_cost,
                'Nc': Nc,
                'Ne': Ne
            })
            
            print(f"O={obstacles}: {len(times)} pontos, {len(survival_data)} presas, Nc={Nc}")
    
    return config_data, obstacle_data

# -------------------------------
# 5. Criar gráficos por CONFIGURAÇÃO DE AGENTES (original)
# -------------------------------
def create_agent_config_plots(config_data):
    """
    Cria 3 gráficos individuais por configuração de agentes
    """
    # Cores para as diferentes densidades de obstáculos
    colors = plt.cm.flag(np.linspace(0, 1, 9))
    
    # Nomes dos obstáculos para legenda
    obstacle_names = {
        0: "O=0",
        1638: "O=1638", 
        3276: "O=3276",
        4915: "O=4915",
        6553: "O=6553",
        8192: "O=8192",
        9830: "O=9830",
        11468: "O=11468",
        13107: "O=13107"
    }
    
    for config_name, data_list in config_data.items():
        if not data_list:
            print(f" Nenhum dado para {config_name}")
            continue
            
        plt.figure(figsize=(12, 7))
        
        # Ordenar por número de obstáculos
        data_list.sort(key=lambda x: x['obstacles'])
        
        # Plotar TODAS as densidades para esta configuração
        for i, data in enumerate(data_list):
            obstacles = data['obstacles']
            color = colors[i]
            
            # Plotar curva principal
            plt.plot(data['times'], data['normalized_cost'], 
                    color=color, linewidth=2.5, alpha=0.8,
                    label=obstacle_names[obstacles])
        
        # Linhas de referência
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Máximo = 1.0')
        
        plt.xlabel('Tempo (steps)', fontsize=12)
        plt.ylabel('Custo Normalizado (c(t)/cₘₐₓ)', fontsize=12)
        plt.title(f'Custo Normalizado - {config_name}\n(Variação com Densidade de Obstáculos)', fontsize=14)
        
        # LEGENDA NO CANTO INFERIOR DIREITO
        plt.legend(fontsize=9, loc='lower right')
        
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.2)
        
        plt.tight_layout()
        
        # Salvar o gráfico
        output_path = OUT_DIR / f"agentes_{config_name}.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f" Gráfico por AGENTES salvo: {output_path}")

# -------------------------------
# 6. Criar gráficos por DENSIDADE DE OBSTÁCULOS (NOVO)
# -------------------------------
def create_obstacle_density_plots(obstacle_data):
    """
    Cria gráficos para mesma densidade de obstáculos variando agentes (NOVO)
    """
    # Cores para as diferentes configurações de agentes
    colors = {
        "Nc=Np": "red",
        "Nc=Np*0.8": "blue", 
        "Nc=Np*0.5": "green"
    }
    colors = plt.cm.flag(np.linspace(0, 1, 3))  # 3 cores para as 3 configurações
    
    # Nomes das configurações para legenda - ORDEM ALTERADA
    config_names = {
        "Nc=Np": "Nc = Np",
        "Nc=Np*0.8": "Nc = 0.8*Np", 
        "Nc=Np*0.5": "Nc = 0.5*Np"
    }
    
    # ORDEM DESEJADA para plotagem (do maior para o menor Nc)
    plot_order = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]
    
    for obstacle_density, data_list in obstacle_data.items():
        if not data_list:
            print(f" Nenhum dado para obstáculos O={obstacle_density}")
            continue
            
        plt.figure(figsize=(12, 7))
        
        # ORDENAR os dados pela ordem desejada antes de plotar
        # Criar um dicionário temporário para acesso rápido
        data_dict = {data['config']: data for data in data_list}
        
        # Plotar na ORDEM ESPECIFICADA com cores do padrão flag
        for i, config_name in enumerate(plot_order):
            if config_name in data_dict:
                data = data_dict[config_name]
                color = colors[i]  # Usar a i-ésima cor do colormap
                
                # Plotar curva principal
                plt.plot(data['times'], data['normalized_cost'], 
                        color=color, linewidth=2.5, alpha=0.8,
                        label=config_names[config_name])
        
        # Linhas de referência
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Máximo = 1.0')
        
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('c(t)', fontsize=12)
        normalized_phi = obstacle_density / (128**2)
        plt.title(fr'$\phi$ = {normalized_phi:.2f}', fontsize=14)
        
        # LEGENDA
        plt.legend(fontsize=10, loc='lower right')
        
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        
        plt.tight_layout()
        
        # Salvar o gráfico
        output_path = OUT_DIR / f"obstaculos_O_{obstacle_density}.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f" Gráfico por OBSTÁCULOS salvo: {output_path}")

# -------------------------------
# 7. Salvar dados completos
# -------------------------------
def save_complete_data(config_data, obstacle_data):
    """
    Salva dados completos SEM incertezas
    """
    all_data = []
    
    # Dados por configuração de agentes
    for config_name, data_list in config_data.items():
        for data in data_list:
            df = pd.DataFrame({
                'time': data['times'],
                'normalized_cost': data['normalized_cost'],
                'survival': data['survival']
            })
            df['agent_config'] = config_name
            df['obstacles'] = data['obstacles']
            df['Nc'] = data['Nc']
            df['Ne'] = data['Ne']
            all_data.append(df)
    
    if all_data:
        complete_df = pd.concat(all_data, ignore_index=True)
        output_path = OUT_DIR / "dados_completos.csv"
        complete_df.to_csv(output_path, index=False)
        print(f"Dados completos salvos: {output_path}")
    else:
        print("Nenhum dado para salvar")
    
    # Salvar resumo estatístico
    summary_data = []
    for config_name, data_list in config_data.items():
        for data in data_list:
            summary_data.append({
                'agent_config': config_name,
                'obstacles': data['obstacles'],
                'Nc': data['Nc'],
                'Ne': data['Ne'],
                'max_absolute_cost': data['max_cost'],
                'final_normalized_cost': data['normalized_cost'][-1],
                'final_survival': data['survival'][-1]
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        output_path = OUT_DIR / "resumo_estatistico.csv"
        df_summary.to_csv(output_path, index=False)
        print(f" Resumo estatístico salvo: {output_path}")

# -------------------------------
# 8. Loop principal
# -------------------------------
def main():
    print("=" * 60)
    print("INICIANDO ANÁLISE COMPLETA DE CUSTO NORMALIZADO")
    print("=" * 60)
    print(f" Diretório de dados: {DATA_ROOT}")
    print(f" Diretório de saída: {OUT_DIR}")
    
    # Coletar todos os dados (ambas as organizações)
    config_data, obstacle_data = collect_all_data()
    
    # Verificar se temos dados
    total_configs = sum(len(data) for data in config_data.values())
    total_obstacles = sum(len(data) for data in obstacle_data.values())
    
    if total_configs == 0:
        print(" NENHUM DADO ENCONTRADO!")
        print("Verifique se os diretórios existem:")
        print(f"  - {DATA_ROOT / 'Nc=Np'}")
        print(f"  - {DATA_ROOT / 'Nc=Np*0.8'}")
        print(f"  - {DATA_ROOT / 'Nc=Np*0.5'}")
        return
    
    print(f"\n Dados coletados:")
    print(f"   - Configurações de agentes: {total_configs} conjuntos")
    print(f"   - Densidades de obstáculos: {total_obstacles} conjuntos")
    
    # Criar os gráficos por CONFIGURAÇÃO DE AGENTES
    print(f"\n Criando gráficos por configuração de AGENTES...")
    create_agent_config_plots(config_data)
    
    # Criar os gráficos por DENSIDADE DE OBSTÁCULOS (NOVO)
    print(f"\n Criando gráficos por densidade de OBSTÁCULOS...")
    create_obstacle_density_plots(obstacle_data)
    
    # Salvar dados completos
    save_complete_data(config_data, obstacle_data)
    
    print(f"\n{'='*60}")
    print("ANÁLISE COMPLETA CONCLUÍDA!")
    print("\n GRÁFICOS GERADOS:")
    print("Por Configuração de Agentes:")
    print("  - agentes_Nc=Np.pdf")
    print("  - agentes_Nc=Np*0.8.pdf") 
    print("  - agentes_Nc=Np*0.5.pdf")
    print("\nPor Densidade de Obstáculos:")
    print("  - obstaculos_O_0.pdf")
    print("  - obstaculos_O_1638.pdf")
    print("  - obstaculos_O_3276.pdf")
    print("  - ... (um para cada densidade)")
    print("\n ARQUIVOS DE DADOS:")
    print("  - dados_completos.csv")
    print("  - resumo_estatistico.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()