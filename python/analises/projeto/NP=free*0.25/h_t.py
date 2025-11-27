import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from lifelines import NelsonAalenFitter
from pathlib import Path

# -------------------------------
# 0. Configurações Principais
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128"

# Fração de caçadores (escolha uma)
FRAC_C = "Nc=Np*0.5"

# -------- Lista de Obstáculos para Processar --------
OBSTACULOS = [
    "s_obs_00",
    "s_obs_1638",
    "s_obs_3276",
    "s_obs_4915",
    "s_obs_6553",
    "s_obs_8192",
    "s_obs_9666",
    "s_obs_9830",
    "s_obs_9994",
    "s_obs_11468",
    "s_obs_13107"
]

# Diretórios
FRAC_DIR = BASE_ROOT / FRAC_C
OUT_DIR = BASE_ROOT / "resultados_modelos" / "hazard_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1. Funções do Modelo de Hazard
# -------------------------------
def weibull_hazard_model(t, tau, beta):
    """Modelo de hazard de Weibull"""
    return (beta / tau) * (t / tau) ** (beta - 1)

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
# 3. Análise de Hazard para um obstáculo
# -------------------------------
def analyze_hazard_one_obs(data_dir, tag, obs_name):
    data = load_survival_data(data_dir)
    
    print(f"\n=== {tag} ===")
    print(f"Número total de presas: {len(data)}")
    capturadas_t0 = data[(data["time"] == 0) & (data["event"] == 1)]
    print(f"Presas capturadas em t=0: {len(capturadas_t0)}")
    
    # Ajuste do modelo Nelson-Aalen para hazard
    naf = NelsonAalenFitter()
    naf.fit(durations=data["time"], event_observed=data["event"])
    
    # Fração de obstáculos
    obs_val = int(''.join(filter(str.isdigit, obs_name)))
    phi = obs_val / (128**2)
    
    # -------------------------------
    # Obter dados de h(t) do Nelson-Aalen
    # -------------------------------
    # Hazard suavizado
    hazard_df = naf.smoothed_hazard_(bandwidth=5)
    ci_df = naf.smoothed_hazard_confidence_intervals_(bandwidth=5)
    
    # Criar DataFrame manualmente para evitar problemas de merge
    hazard_times = hazard_df.index.values
    hazard_values = hazard_df.values.flatten()
    
    # Obter intervalos de confiança
    ci_lower = ci_df['NA_estimate_lower_0.95'].values
    ci_upper = ci_df['NA_estimate_upper_0.95'].values
    
    # Criar DataFrame com dados de hazard
    hazard_data = pd.DataFrame({
        'step': hazard_times,
        'h(t)': hazard_values,
        'h(t)_lower': ci_lower,
        'h(t)_upper': ci_upper
    })
    
    # Remover possíveis NaNs
    hazard_data = hazard_data.dropna()
    
    # -------------------------------
    # Normalização h(t)/h(0)
    # -------------------------------
    # Encontrar h(0) - primeiro valor não-zero e não-NaN
    if len(hazard_data) > 0:
        # Pegar o primeiro valor válido (pode não ser exatamente em t=0)
        valid_h0 = hazard_data[hazard_data['h(t)'] > 0]
        if len(valid_h0) > 0:
            h0 = valid_h0['h(t)'].iloc[0]
            h0_step = valid_h0['step'].iloc[0]
        else:
            h0 = hazard_data['h(t)'].iloc[0] if hazard_data['h(t)'].iloc[0] > 0 else 1.0
            h0_step = hazard_data['step'].iloc[0]
    else:
        h0 = 1.0
        h0_step = 0
    
    print(f"h(0) ≈ {h0:.6f} (em step {h0_step})")
    
    # Adicionar colunas normalizadas
    hazard_data["h(t)_norm"] = hazard_data["h(t)"] / h0
    hazard_data["h(t)_lower_norm"] = hazard_data["h(t)_lower"] / h0
    hazard_data["h(t)_upper_norm"] = hazard_data["h(t)_upper"] / h0
    
    # Salvar dados completos de hazard
    hazard_out = OUT_DIR / f"{tag}_hazard_data.csv"
    hazard_data.to_csv(hazard_out, index=False)
    print(f"[OK] Dados COMPLETOS de h(t) salvos em {hazard_out}")
    print(f"[OK] Número de pontos: {len(hazard_data)}")
    print(f"[OK] h(t) normalizado: {hazard_data['h(t)_norm'].iloc[0]:.3f} -> {hazard_data['h(t)_norm'].iloc[-1]:.3f}")
    
    # -------------------------------
    # Ajuste do Modelo de Weibull ao hazard
    # -------------------------------
    # Preparar dados para o fit (usar apenas valores válidos)
    valid_mask = (hazard_data['h(t)'] > 0) & (hazard_data['step'] > 0)
    fit_times = hazard_data[valid_mask]['step'].values
    fit_hazard = hazard_data[valid_mask]['h(t)'].values
    
    if len(fit_times) < 10:
        print(f"[AVISO] Poucos pontos válidos para ajuste ({len(fit_times)})")
        fit_success = False
        tau_fit, beta_fit, tau_err, beta_err = np.nan, np.nan, np.nan, np.nan
    else:
        try:
            # Calcular erros para o fit
            fit_errors = (hazard_data[valid_mask]['h(t)_upper'] - hazard_data[valid_mask]['h(t)_lower']) / 2
            
            popt, pcov = curve_fit(
                weibull_hazard_model, fit_times, fit_hazard,
                p0=[50.0, 0.8], 
                bounds=([1.0, 0.1], [1000, 3.0]),
                sigma=fit_errors, 
                absolute_sigma=True, 
                maxfev=5000
            )
            
            tau_fit, beta_fit = popt
            tau_err, beta_err = np.sqrt(np.diag(pcov))
            
            print(f"[OK] Ajuste do hazard bem-sucedido!")
            print(fr"τ = {tau_fit:.2f} ± {tau_err:.2f}")
            print(fr"β = {beta_fit:.4f} ± {beta_err:.4f}")
            
            fit_success = True
            
            # -------------------------------
            # Gerar curva do modelo ajustado para plotagem
            # -------------------------------
            time_range = np.linspace(fit_times.min(), fit_times.max(), 1000)
            fitted_hazard = weibull_hazard_model(time_range, tau_fit, beta_fit)
            
            # Criar DataFrame com a curva ajustada
            fit_data = pd.DataFrame({
                'step': time_range,
                'fitted_h(t)': fitted_hazard,
                'fitted_h(t)_norm': fitted_hazard / h0  # Versão normalizada também
            })
            
            fit_out = OUT_DIR / f"{tag}_fitted_hazard.csv"
            fit_data.to_csv(fit_out, index=False)
            print(f"[OK] Curva do modelo ajustado salva em {fit_out}")
            
        except Exception as e:
            print(f"[AVISO] Ajuste do modelo de hazard falhou: {e}")
            fit_success = False
            tau_fit, beta_fit, tau_err, beta_err = np.nan, np.nan, np.nan, np.nan
    
    # -------------------------------
    # Salvar parâmetros do análise
    # -------------------------------
    params_data = {
        'tag': [tag],
        'frac_c': [FRAC_C],
        'obs': [obs_name],
        'phi': [phi],
        'tau_hazard': [tau_fit],
        'tau_err_hazard': [tau_err],
        'beta_hazard': [beta_fit],
        'beta_err_hazard': [beta_err],
        'h0': [h0],
        'h0_step': [h0_step],
        'total_presas': [len(data)],
        'capturadas_t0': [len(capturadas_t0)],
        'n_points_hazard': [len(hazard_data)],
        'n_points_fit': [len(fit_times) if 'fit_times' in locals() else 0],
        'min_time': [hazard_data['step'].min()],
        'max_time': [hazard_data['step'].max()],
        'fit_success': [fit_success]
    }
    
    params_df = pd.DataFrame(params_data)
    params_out = OUT_DIR / f"{tag}_hazard_parameters.csv"
    params_df.to_csv(params_out, index=False)
    print(f"[OK] Parâmetros da análise de hazard salvos em {params_out}")
    
    return {
        'tau': tau_fit, 'tau_err': tau_err,
        'beta': beta_fit, 'beta_err': beta_err,
        'h0': h0, 'h0_step': h0_step,
        'hazard_data': hazard_data,
        'fit_success': fit_success,
        'n_points': len(hazard_data)
    }

# -------------------------------
# 4. Loop para todos obstáculos
# -------------------------------
def main():
    print("INICIANDO ANÁLISE DE HAZARD h(t)")
    print(f"Diretório de saída: {OUT_DIR}")
    print(f"Objetivo: Gerar dados completos de h(t) e h(t)/h(0) para plotagem")
    
    resultados = []
    for obs_name in OBSTACULOS:
        obs_dir = FRAC_DIR / obs_name
        if not obs_dir.is_dir():
            print(f"[Aviso] Pasta não encontrada: {obs_dir}")
            continue
            
        tag = f"{FRAC_C}_{obs_name.replace('s_', '')}"
        print(f"\n▶ Processando: {tag}")
        
        result = analyze_hazard_one_obs(obs_dir, tag, obs_name)
        
        if result is not None:
            # Verificar qualidade dos dados
            hazard_data = result['hazard_data']
            if len(hazard_data) > 0:
                h0 = result['h0']
                h_final = hazard_data['h(t)_norm'].iloc[-1] if len(hazard_data) > 0 else np.nan
                print(f"   ✅ Dados: h(0)={h0:.3f}, h(final)={h_final:.3f}, {len(hazard_data)} pontos")
                print(f"   📊 Ajuste: {'SUCESSO' if result['fit_success'] else 'FALHA'}")
                
                resultados.append((
                    FRAC_C, obs_name, result['tau'], result['tau_err'],
                    result['beta'], result['beta_err'], result['h0'], result['h0_step'],
                    len(hazard_data), result['fit_success']
                ))
            else:
                print(f"   ❌ Dados vazios para {tag}")
        else:
            print(f"   ❌ Falha completa no processamento de {tag}")
    
    # Salvar resumo geral
    if resultados:
        df_res = pd.DataFrame(resultados, columns=[
            "frac_c", "obs", "tau_hazard", "tau_err_hazard", 
            "beta_hazard", "beta_err_hazard", "h0", "h0_step",
            "n_points", "fit_success"
        ])
        
        summary_out = OUT_DIR / f"{FRAC_C}_hazard_analysis_summary.csv"
        df_res.to_csv(summary_out, index=False)
        
        # Estatísticas
        success_count = sum(result[-1] for result in resultados)
        
        print(f"\n" + "="*60)
        print("✅ ANÁLISE DE HAZARD CONCLUÍDA COM SUCESSO!")
        print(f"📊 Obstáculos processados: {len(resultados)}/{len(OBSTACULOS)}")
        print(f"🎯 Ajustes bem-sucedidos: {success_count}/{len(resultados)}")
        print(f"💾 Dados salvos em: {OUT_DIR}")
        print(f"📈 Arquivos gerados por obstáculo:")
        print(f"   - {FRAC_C}_*_hazard_data.csv → Dados completos h(t) e h(t)/h(0)")
        print(f"   - {FRAC_C}_*_fitted_hazard.csv → Curva do modelo ajustado")
        print(f"   - {FRAC_C}_*_hazard_parameters.csv → Parâmetros da análise")
        print(f"   - {FRAC_C}_hazard_analysis_summary.csv → Resumo geral")
        print("="*60)
    else:
        print("\n❌ Nenhum obstáculo foi processado com sucesso!")

if __name__ == "__main__":
    main()