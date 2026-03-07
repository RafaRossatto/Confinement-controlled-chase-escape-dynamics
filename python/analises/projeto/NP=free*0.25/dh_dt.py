import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 0. Configurações Principais
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128"

# Fração de caçadores (escolha uma)
FRAC_C = "Nc=Np*0.8"

# -------- Lista de Obstáculos para Processar --------
OBSTACULOS = [
    "s_obs_00", "s_obs_1638", "s_obs_3276",
    "s_obs_4915", "s_obs_6553","s_obs_8028", "s_obs_8192", "s_obs_8355",
    "s_obs_9666","s_obs_9994","s_obs_9830", 
    "s_obs_11468", "s_obs_13107"
]

# Diretórios
FRAC_DIR = BASE_ROOT / FRAC_C
OUT_DIR = BASE_ROOT / "resultados_modelos" / "survival_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1. Funções do Modelo
# -------------------------------
def stretched_exp_model(t, tau, beta, A, C):
    return A * np.exp(-(t / tau) ** beta) + C

# -------------------------------
# 2. Carregar e Preparar Dados
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
    return pd.concat(all_data, ignore_index=True)

# -------------------------------
# 3. Função para calcular H(t) e variações
# -------------------------------
def calcular_hazard_e_variacoes(data, tag):
    """
    Calcula o hazard cumulativo H(t) e suas variações entre passos
    """
    # Ajustar modelo Nelson-Aalen
    naf = NelsonAalenFitter(alpha=0.95)  # 95% intervalo de confiança
    naf.fit(durations=data["time"], event_observed=data["event"])
    
    # Obter H(t) - Hazard Cumulativo
    hazard_df = naf.cumulative_hazard_.reset_index()
    hazard_df.columns = ['step', 'H(t)']
    
    # Obter intervalos de confiança
    ci_df = naf.confidence_interval_.reset_index()
    ci_df.columns = ['step', 'H(t)_lower_0.95', 'H(t)_upper_0.95']
    
    # Combinar dados
    hazard_data = pd.merge(hazard_df, ci_df, on='step', how='inner')
    
    # Calcular variações entre passos
    hazard_data['delta_H'] = hazard_data['H(t)'].diff()
    hazard_data['delta_H_percent'] = hazard_data['delta_H'] / hazard_data['H(t)'].shift() * 100
    
    # Calcular taxa de hazard instantâneo aproximada
    hazard_data['hazard_instantaneo'] = hazard_data['delta_H'] / hazard_data['step'].diff()
    
    # Obter tabela de eventos para contexto
    event_table = naf.event_table.reset_index()
    event_table.columns = ['step', 'removed', 'observed', 'censored', 'entrance', 'at_risk']
    
    # Combinar com tabela de eventos
    full_data = pd.merge(hazard_data, event_table, on='step', how='inner')
    
    # Calcular estatísticas de variação
    stats_variacao = {
        'media_delta_H': full_data['delta_H'].mean(),
        'std_delta_H': full_data['delta_H'].std(),
        'max_delta_H': full_data['delta_H'].max(),
        'min_delta_H': full_data['delta_H'].min(),
        'tempo_max_delta_H': full_data.loc[full_data['delta_H'].idxmax(), 'step'] if not full_data.empty else None,
        'tempo_min_delta_H': full_data.loc[full_data['delta_H'].idxmin(), 'step'] if not full_data.empty else None,
        'total_eventos': full_data['observed'].sum(),
        'total_censuras': full_data['censored'].sum(),
        'tempo_maximo': full_data['step'].max(),
        'hazard_final': full_data['H(t)'].iloc[-1] if not full_data.empty else 0
    }
    
    return naf, full_data, stats_variacao

# -------------------------------
# 5. Análise de Sobrevivência com Hazard
# -------------------------------
def analyze_survival_one_obs(data_dir, tag, obs_name):
    data = load_survival_data(data_dir)
    
    print(f"\n=== {tag} ===")
    print(f"Número total de presas: {len(data)}")
    capturadas_t0 = data[(data["time"] == 0) & (data["event"] == 1)]
    print(f"Presas capturadas em t=0: {len(capturadas_t0)}")
    
    # 1. Ajuste do modelo Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data["time"], event_observed=data["event"])
    
    # Fração de obstáculos
    obs_val = int(''.join(filter(str.isdigit, obs_name)))
    phi = obs_val / (128**2)
    
    # 2. Calcular H(t) usando Nelson-Aalen
    print("\n[INFO] Calculando Hazard Cumulativo H(t) com Nelson-Aalen...")
    naf, hazard_full_data, stats_variacao = calcular_hazard_e_variacoes(data, tag)
    
   
    # -------------------------------
    # 4. Salvar dados de H(t) e variações
    # -------------------------------
    # Salvar dados completos de hazard
    hazard_out = OUT_DIR / f"{tag}_hazard_data.csv"
    hazard_full_data.to_csv(hazard_out, index=False)
    print(f"[OK] Dados de H(t) e variações salvos em {hazard_out}")
    print(f"[OK] Número de pontos H(t): {len(hazard_full_data)}")
       
    # -------------------------------
    # 5. Salvar dados de S(t) brutos (Kaplan-Meier)
    # -------------------------------
    survival_df = kmf.survival_function_.reset_index()
    survival_ci = kmf.confidence_interval_.reset_index()
    
    survival_df_renamed = survival_df.rename(columns={
        survival_df.columns[0]: "step",
        "KM_estimate": "S(t)"
    })
    
    survival_ci_renamed = survival_ci.rename(columns={
        survival_ci.columns[0]: "step",
        "KM_estimate_lower_0.95": "S(t)_lower",
        "KM_estimate_upper_0.95": "S(t)_upper"
    })
    
    survival_data = pd.merge(
        survival_df_renamed[["step", "S(t)"]], 
        survival_ci_renamed[["step", "S(t)_lower", "S(t)_upper"]], 
        on="step", 
        how="inner"
    )
    
    # Normalização S(t)/S(0)
    S0 = survival_data["S(t)"].iloc[0] if len(survival_data) > 0 else 1.0
    print(f"S(0) = {S0:.6f}")
    survival_data["S(t)_norm"] = survival_data["S(t)"] / S0
    
    survival_out = OUT_DIR / f"{tag}_survival_data.csv"
    survival_data.to_csv(survival_out, index=False)
    print(f"[OK] Dados de S(t) salvos em {survival_out}")
    
    # -------------------------------
    # 6. Ajuste do Modelo Stretched Exponential (opcional)
    # -------------------------------
    print("[INFO] Realizando ajuste do modelo stretched exponential...")
    km_time = kmf.survival_function_.index.values
    km_survival = kmf.survival_function_['KM_estimate'].values
    
    ci_lower = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
    ci_upper = kmf.confidence_interval_['KM_estimate_upper_0.95'].values
    km_std = (ci_upper - ci_lower) / 2
    
    C = km_survival[-1] if len(km_survival) > 0 else 0.0
    total_presas = len(data)
    sobrevivencia_inicial = 1 - (len(capturadas_t0) / total_presas)
    A = sobrevivencia_inicial - C
    
    print(f"Parâmetros iniciais para ajuste: A={A:.4f}, C={C:.4f}")
    
    def model_fixed_ac(t, tau, beta):
        return stretched_exp_model(t, tau, beta, A, C)
    
    try:
        popt, pcov = curve_fit(
            model_fixed_ac, km_time, km_survival,
            p0=[10.0, 0.5], bounds=([0.1, 0.1], [100, 2.0]),
            sigma=km_std, absolute_sigma=True
        )
        tau_fit, beta_fit = popt
        tau_err, beta_err = np.sqrt(np.diag(pcov))
        
        print(f"[SUCESSO] Ajuste do modelo:")
        print(fr"  τ = {tau_fit:.2f} ± {tau_err:.2f}")
        print(fr"  β = {beta_fit:.2f} ± {beta_err:.2f}")
        
        # Salvar parâmetros do ajuste
        params_data = {
            'tag': [tag],
            'frac_c': [FRAC_C],
            'obs': [obs_name],
            'phi': [phi],
            'tau': [tau_fit],
            'tau_err': [tau_err],
            'beta': [beta_fit],
            'beta_err': [beta_err],
            'A': [A],
            'C': [C],
            'S0': [S0],
            'sobrevivencia_inicial': [sobrevivencia_inicial],
            'total_presas': [total_presas],
            'capturadas_t0': [len(capturadas_t0)],
            'hazard_final': [stats_variacao['hazard_final']],
            'media_delta_H': [stats_variacao['media_delta_H']],
            'std_delta_H': [stats_variacao['std_delta_H']]
        }
        
        params_df = pd.DataFrame(params_data)
        params_out = OUT_DIR / f"{tag}_fit_parameters.csv"
        params_df.to_csv(params_out, index=False)
        print(f"[OK] Parâmetros do ajuste salvos em {params_out}")
        
        return {
            'hazard_stats': stats_variacao,
            'fit_params': {
                'tau': tau_fit, 'tau_err': tau_err,
                'beta': beta_fit, 'beta_err': beta_err,
                'A': A, 'C': C
            },
            'hazard_data': hazard_full_data,
            'survival_data': survival_data
        }
        
    except Exception as e:
        print(f"[AVISO] Falha no ajuste do modelo stretched exponential: {e}")
        return {
            'hazard_stats': stats_variacao,
            'fit_params': None,
            'hazard_data': hazard_full_data,
            'survival_data': survival_data
        }

# -------------------------------
# 7. Loop principal
# -------------------------------
def main():
    print("=" * 80)
    print("ANÁLISE DE SOBREVIVÊNCIA COM HAZARD CUMULATIVO H(t)")
    print("=" * 80)
    print(f"Fração de caçadores: {FRAC_C}")
    print(f"Diretório de saída: {OUT_DIR}")
    print("=" * 80)

   
    resultados_dict = {}
    
    for obs_name in OBSTACULOS:
        obs_dir = FRAC_DIR / obs_name
        if not obs_dir.is_dir():
            print(f"[AVISO] Pasta não encontrada: {obs_dir}")
            continue
            
        tag = f"{FRAC_C}_{obs_name.replace('s_', '')}"
        print(f"\n" + "=" * 60)
        print(f"PROCESSANDO: {tag}")
        print("=" * 60)
        
        try:
            # CORREÇÃO: Primeiro rodar a análise completa
            result = analyze_survival_one_obs(obs_dir, tag, obs_name)
            
            # CORREÇÃO: Extrair variáveis do resultado
            # O analyze_survival_one_obs agora retorna um dicionário
            hazard_stats = result['hazard_stats']
            hazard_full_data = result['hazard_data']
            
            # Calcular phi e obs_val
            obs_val = int(''.join(filter(str.isdigit, obs_name))) if any(char.isdigit() for char in obs_name) else 0
            phi = obs_val / (128**2)
            
            # CORREÇÃO: Salvar o resultado no dicionário
            resultados_dict[tag] = result
            print(f"[SUCESSO] {tag} processado completamente")
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar {tag}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Criar gráfico comparativo se houver resultados
    if resultados_dict:
        print("\n" + "=" * 60)
        print("CRIANDO ANÁLISE COMPARATIVA")
        print("=" * 60)

        # Salvar resumo geral
        print("\n[INFO] Salvando resumo geral dos resultados...")
        resumo_geral = []
        
        for tag, results in resultados_dict.items():
            if 'hazard_stats' in results:
                obs_val = int(''.join(filter(str.isdigit, tag.split('_')[-1]))) if any(char.isdigit() for char in tag) else 0
                phi = obs_val / (128**2)
                
                linha = {
                    'tag': tag,
                    'frac_c': FRAC_C,
                    'obs': tag.split('_')[-1],
                    'phi': phi,
                    'total_eventos': results['hazard_stats']['total_eventos'],
                    'total_censuras': results['hazard_stats']['total_censuras'],
                    'tempo_maximo': results['hazard_stats']['tempo_maximo']
                }
                
                # Adicionar parâmetros do ajuste se disponíveis
                if 'fit_params' in results and results['fit_params'] is not None:
                    linha.update({
                        'tau': results['fit_params']['tau'],
                        'tau_err': results['fit_params']['tau_err'],
                        'beta': results['fit_params']['beta'],
                        'beta_err': results['fit_params']['beta_err'],
                        'A': results['fit_params']['A'],
                        'C': results['fit_params']['C']
                    })
                
                resumo_geral.append(linha)
        
        if resumo_geral:
            resumo_df = pd.DataFrame(resumo_geral)
            resumo_out = OUT_DIR / f"{FRAC_C}_hazard_analysis_summary.csv"
            resumo_df.to_csv(resumo_out, index=False)
            print(f"[OK] Resumo geral salvo em: {resumo_out}")
        
        print("\n" + "=" * 80)
        print(f"ANÁLISE CONCLUÍDA COM SUCESSO!")
        print(f"Obstáculos processados: {len(resultados_dict)}/{len(OBSTACULOS)}")
        print(f"Arquivos gerados em: {OUT_DIR}")
        print("=" * 80)
        
    else:
        print("\n[ERRO] Nenhum obstáculo foi processado com sucesso!")
        print("Verifique os diretórios de entrada e os dados.")

if __name__ == "__main__":
    main()