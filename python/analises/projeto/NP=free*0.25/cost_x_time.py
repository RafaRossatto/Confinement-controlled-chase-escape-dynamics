import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pathlib import Path
import re

# -------------------------------
# Configurações
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
OUT_DIR = BASE_ROOT / "resultados_com_incerteza"
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
# 3. Calcular custo normalizado COM INCERTEZA
# -------------------------------
def calculate_normalized_cost_with_uncertainty(survival_data, Nc, cost_per_hunter=1.0):
    """
    Calcula custo normalizado pelo máximo COM propagação de incerteza do Kaplan-Meier
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_data["time"], event_observed=survival_data["event"])
    
    # Obter curva de sobrevivência com intervalos de confiança
    survival_func = kmf.survival_function_
    times = survival_func.index.values
    S_t = survival_func['KM_estimate'].values
    
    # Obter intervalos de confiança 95%
    ci_lower = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
    ci_upper = kmf.confidence_interval_['KM_estimate_upper_0.95'].values
    
    # Calcular desvio padrão aproximado (para 95% CI: ±1.96σ)
    std_S = (ci_upper - ci_lower) / 3.92
    
    # Número total inicial de presas
    Ne0 = len(survival_data)
    
    # Custo instantâneo (estimativa pontual)
    instantaneous_cost = (Nc * cost_per_hunter) / np.maximum(Ne0 * S_t, 1)
    
    # Propagação de incerteza para o custo
    # custo = K / (Ne0 * S) onde K = Nc * cost_per_hunter
    # ∂custo/∂S = -K/(Ne0 * S²)
    # std_custo = |∂custo/∂S| * std_S = (K/(Ne0 * S²)) * std_S
    K = Nc * cost_per_hunter
    std_instantaneous_cost = (K / (Ne0 * np.maximum(S_t, 0.001)**2)) * std_S
    
    # Normalizar pelo custo máximo
    max_cost = instantaneous_cost.max()
    normalized_cost = instantaneous_cost / max_cost
    
    # Propagação de incerteza para o custo normalizado
    std_normalized_cost = std_instantaneous_cost / max_cost
    
    return times, normalized_cost, std_normalized_cost, S_t, std_S, max_cost

# -------------------------------
# 4. Coletar dados para os 3 gráficos COM INCERTEZA
# -------------------------------
def collect_data_for_plots_with_uncertainty():
    """
    Coleta todos os dados organizados por configuração COM INCERTEZA
    """
    configs = {
        "Nc=Np": BASE_ROOT / "Nc=Np",
        "Nc=Np*0.8": BASE_ROOT / "Nc=Np*0.8", 
        "Nc=Np*0.5": BASE_ROOT / "Nc=Np*0.5"
    }
    
    obstaculos = [
        "s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", 
        "s_obs_6553", "s_obs_8192", "s_obs_9830", "s_obs_11468", "s_obs_13107"
    ]
    
    # Dicionário para armazenar dados por configuração
    config_data = {
        "Nc=Np": [],
        "Nc=Np*0.8": [],
        "Nc=Np*0.5": []
    }
    
    for config_name, config_base in configs.items():
        print(f"\nColetando dados COM INCERTEZA: {config_name}")
        
        for obs in obstaculos:
            obs_dir = config_base / obs
            if not obs_dir.exists():
                continue
            
            # Carregar dados
            survival_data = load_survival_data(obs_dir)
            if survival_data is None:
                continue
            
            # Extrair parâmetros
            prey_files = list(obs_dir.glob("*prey_per_step.csv"))
            if prey_files:
                params = extract_parameters(prey_files[0].name)
                Nc = params['NC'] if params else len(survival_data)
            else:
                Nc = len(survival_data)
            
            # Extrair valor dos obstáculos
            obs_match = re.search(r'obs_(\d+)', obs)
            obstacles = int(obs_match.group(1)) if obs_match else 0
            
            # Calcular custo normalizado COM INCERTEZA
            times, normalized_cost, std_normalized_cost, S_t, std_S, max_cost = calculate_normalized_cost_with_uncertainty(survival_data, Nc)
            
            # Armazenar dados
            config_data[config_name].append({
                'obstacles': obstacles,
                'times': times,
                'normalized_cost': normalized_cost,
                'std_normalized_cost': std_normalized_cost,
                'survival': S_t,
                'std_survival': std_S,
                'max_cost': max_cost
            })
            
            print(f"  → O={obstacles}: {len(times)} pontos, incerteza média: {np.mean(std_normalized_cost):.4f}")
    
    return config_data

# -------------------------------
# 5. Criar 3 gráficos individuais COM BARRAS DE ERRO
# -------------------------------
def create_individual_plots_with_uncertainty(config_data):
    """
    Cria 3 gráficos individuais com barras de erro da incerteza
    """
    # Cores para as diferentes densidades
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
            continue
            
        plt.figure(figsize=(12, 7))
        
        # Ordenar por número de obstáculos
        data_list.sort(key=lambda x: x['obstacles'])
        
        # Plotar TODAS as densidades para esta configuração COM INCERTEZA
        for i, data in enumerate(data_list):
            obstacles = data['obstacles']
            color = colors[i]
            
            # Plotar curva principal
            plt.plot(data['times'], data['normalized_cost'], 
                    color=color, linewidth=2.5, alpha=0.8,
                    label=obstacle_names[obstacles])
            
            # Adicionar banda de incerteza (a cada 5 pontos para não poluir)
            if len(data['times']) > 10:
                # Amostrar pontos para a banda de incerteza
                step = max(1, len(data['times']) // 20)
                indices = range(0, len(data['times']), step)
                
                plt.fill_between(data['times'][indices],
                               data['normalized_cost'][indices] - data['std_normalized_cost'][indices],
                               data['normalized_cost'][indices] + data['std_normalized_cost'][indices],
                               color=color, alpha=0.2)
        
        # Linhas de referência
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Máximo = 1.0')
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='50% do máximo')
        
        plt.xlabel('Tempo (steps)', fontsize=12)
        plt.ylabel('Custo Normalizado (c(t)/cₘₐₓ)', fontsize=12)
        plt.title(f'Custo Normalizado - {config_name}\n(com incerteza do Kaplan-Meier)', fontsize=14)
        
        # LEGENDA NO CANTO INFERIOR DIREITO
        plt.legend(fontsize=9, loc='lower right')
        
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.2)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{config_name}_com_incerteza.pdf", bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"[OK] Gráfico com incerteza salvo: {config_name}_com_incerteza.pdf")

# -------------------------------
# 6. Salvar dados completos com incerteza
# -------------------------------
def save_complete_data_with_uncertainty(config_data):
    """
    Salva dados completos incluindo as incertezas
    """
    all_data = []
    
    for config_name, data_list in config_data.items():
        for data in data_list:
            df = pd.DataFrame({
                'time': data['times'],
                'normalized_cost': data['normalized_cost'],
                'std_normalized_cost': data['std_normalized_cost'],
                'survival': data['survival'],
                'std_survival': data['std_survival']
            })
            df['config'] = config_name
            df['obstacles'] = data['obstacles']
            all_data.append(df)
    
    if all_data:
        complete_df = pd.concat(all_data, ignore_index=True)
        complete_df.to_csv(OUT_DIR / "dados_completos_com_incerteza.csv", index=False)
        print("[OK] Dados completos com incerteza salvos!")
    
    # Salvar resumo estatístico
    summary_data = []
    for config_name, data_list in config_data.items():
        for data in data_list:
            summary_data.append({
                'config': config_name,
                'obstacles': data['obstacles'],
                'max_absolute_cost': data['max_cost'],
                'final_normalized_cost': data['normalized_cost'][-1],
                'final_std_normalized_cost': data['std_normalized_cost'][-1],
                'final_survival': data['survival'][-1],
                'final_std_survival': data['std_survival'][-1],
                'mean_uncertainty': np.mean(data['std_normalized_cost'])
            })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUT_DIR / "resumo_estatistico_com_incerteza.csv", index=False)
    print("[OK] Resumo estatístico com incerteza salvo!")

# -------------------------------
# 7. Loop principal COM INCERTEZA
# -------------------------------
def main():
    print("Coletando dados de todas as configurações COM INCERTEZA...")
    
    # Coletar todos os dados COM INCERTEZA
    config_data = collect_data_for_plots_with_uncertainty()
    
    # Criar os 3 gráficos individuais COM INCERTEZA
    create_individual_plots_with_uncertainty(config_data)
    
    # Salvar dados completos
    save_complete_data_with_uncertainty(config_data)
    
    print(f"\n{'='*60}")
    print("ANÁLISE COM INCERTEZA CONCLUÍDA!")
    print("Gerados 3 gráficos com barras de erro:")
    print("  - Nc=Np_com_incerteza.pdf")
    print("  - Nc=Np*0.8_com_incerteza.pdf") 
    print("  - Nc=Np*0.5_com_incerteza.pdf")
    print("  - dados_completos_com_incerteza.csv")
    print("  - resumo_estatistico_com_incerteza.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()