import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter
from pathlib import Path

# -------------------------------
# 0. Configurações Principais
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25" / "L_128"

# Fração de caçadores (escolha uma)
FRAC_C = "Nc=Np*0.8"

# -------- Lista de Obstáculos para Processar --------
OBSTACULOS = [
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
# 3. Análise de Sobrevivência para um obstáculo
# -------------------------------
def analyze_survival_one_obs(data_dir, tag, obs_name):
    data = load_survival_data(data_dir)
    
    print(f"\n=== {tag} ===")
    print(f"Número total de presas: {len(data)}")
    capturadas_t0 = data[(data["time"] == 0) & (data["event"] == 1)]
    print(f"Presas capturadas em t=0: {len(capturadas_t0)}")
    
    # Ajuste do modelo Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data["time"], event_observed=data["event"])
    
    # Fração de obstáculos
    obs_val = int(''.join(filter(str.isdigit, obs_name)))
    phi = obs_val / (128**2)
    
    # -------------------------------
    # Salvar dados de S(t) brutos - CORRIGIDO
    # -------------------------------
    # Obter dados de sobrevivência do KaplanMeier
    survival_df = kmf.survival_function_.reset_index()
    survival_ci = kmf.confidence_interval_.reset_index()
    
    print(f"DEBUG - Colunas survival_df: {list(survival_df.columns)}")
    print(f"DEBUG - Colunas survival_ci: {list(survival_ci.columns)}")
    
    # Verificar qual é a coluna de tempo
    time_col_df = survival_df.columns[0]
    time_col_ci = survival_ci.columns[0]
    
    print(f"DEBUG - Coluna tempo survival_df: '{time_col_df}'")
    print(f"DEBUG - Coluna tempo survival_ci: '{time_col_ci}'")
    
    # Renomear colunas primeiro para garantir consistência
    survival_df_renamed = survival_df.rename(columns={
        time_col_df: "step",
        "KM_estimate": "S(t)"
    })
    
    survival_ci_renamed = survival_ci.rename(columns={
        time_col_ci: "step",
        "KM_estimate_lower_0.95": "S(t)_lower",
        "KM_estimate_upper_0.95": "S(t)_upper"
    })
    
    # Combinar dados - método mais seguro
    survival_data = pd.merge(
        survival_df_renamed[["step", "S(t)"]], 
        survival_ci_renamed[["step", "S(t)_lower", "S(t)_upper"]], 
        on="step", 
        how="inner"
    )
    
    print(f"DEBUG - Primeiros steps: {survival_data['step'].head().tolist()}")
    print(f"DEBUG - Primeiros S(t): {survival_data['S(t)'].head().tolist()}")
    
    # -------------------------------
    # Normalização S(t)/S(0)
    # -------------------------------
    S0 = survival_data["S(t)"].iloc[0] if len(survival_data) > 0 else 1.0
    print(f"S(0) = {S0:.6f}")
    
    survival_data["S(t)_norm"] = survival_data["S(t)"] / S0
    survival_data["S(t)_lower_norm"] = survival_data["S(t)_lower"] / S0
    survival_data["S(t)_upper_norm"] = survival_data["S(t)_upper"] / S0
    
    # Salvar dados de sobrevivência
    survival_out = OUT_DIR / f"{tag}_survival_data.csv"
    survival_data.to_csv(survival_out, index=False)
    print(f"[OK] Dados de S(t) salvos em {survival_out}")
    print(f"[OK] Tamanho do dataset: {len(survival_data)} pontos")
    
    # -------------------------------
    # Ajuste do Modelo
    # -------------------------------
    km_time = kmf.survival_function_.index.values
    km_survival = kmf.survival_function_['KM_estimate'].values
    
    # Extraindo survival + erro (CI do Kaplan–Meier)
    ci_lower = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
    ci_upper = kmf.confidence_interval_['KM_estimate_upper_0.95'].values
    km_std = (ci_upper - ci_lower) / 2
    
    C = km_survival[-1] if len(km_survival) > 0 else 0.0
    total_presas = len(data)
    sobrevivencia_inicial = 1 - (len(capturadas_t0) / total_presas)
    A = sobrevivencia_inicial - C
    
    print(f"Parâmetros iniciais: A={A:.4f}, C={C:.4f}")
    print(f"Tamanho dados para fit: {len(km_time)} pontos")
    
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
        
        print(fr"τ = {tau_fit:.2f} ± {tau_err:.2f}, β = {beta_fit:.2f} ± {beta_err:.2f}")
        
        # -------------------------------
        # Salvar dados do modelo ajustado
        # -------------------------------
        time_range = np.linspace(0, km_time.max(), 1000)
        fitted_survival = model_fixed_ac(time_range, tau_fit, beta_fit)
        
        fit_data = pd.DataFrame({
            'step': time_range,
            'fitted_S(t)': fitted_survival,
            'fitted_S(t)_norm': fitted_survival / S0
        })
        
        fit_out = OUT_DIR / f"{tag}_fitted_model.csv"
        fit_data.to_csv(fit_out, index=False)
        print(f"[OK] Dados do modelo ajustado salvos em {fit_out}")
        
        # -------------------------------
        # Salvar parâmetros do ajuste
        # -------------------------------
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
            'cov_tau_tau': [pcov[0, 0]],
            'cov_beta_beta': [pcov[1, 1]],
            'cov_tau_beta': [pcov[0, 1]]
        }
        
        params_df = pd.DataFrame(params_data)
        params_out = OUT_DIR / f"{tag}_fit_parameters.csv"
        params_df.to_csv(params_out, index=False)
        print(f"[OK] Parâmetros do ajuste salvos em {params_out}")
        
        return tau_fit, beta_fit, tau_err, beta_err, A, C, S0, pcov
        
    except Exception as e:
        print(f"[ERRO] Falha no ajuste do modelo: {e}")
        return None

# -------------------------------
# 4. Loop para todos obstáculos
# -------------------------------
def main():
    print("INICIANDO ANÁLISE DE SOBREVIVÊNCIA S(t)")
    print(f"Diretório de saída: {OUT_DIR}")
    
    resultados = []
    for obs_name in OBSTACULOS:
        obs_dir = FRAC_DIR / obs_name
        if not obs_dir.is_dir():
            print(f"[Aviso] Pasta não encontrada: {obs_dir}")
            continue
            
        tag = f"{FRAC_C}_{obs_name.replace('s_', '')}"
        print(f"\nProcessando: {tag}")
        
        result = analyze_survival_one_obs(obs_dir, tag, obs_name)
        
        if result is not None:
            tau, beta, tau_err, beta_err, A, C, S0, pcov = result
            resultados.append((
                FRAC_C, obs_name, tau, tau_err, beta, beta_err, A, C, S0,
                pcov[0, 0], pcov[1, 1], pcov[0, 1], pcov[1, 0]
            ))
            print(f"[SUCESSO] {tag} processado")
        else:
            print(f"[FALHA] {tag} não pôde ser processado")
    
    # Salvar resumo geral apenas se houver resultados
    if resultados:
        df_res = pd.DataFrame(resultados, columns=[
            "frac_c", "obs",
            "tau", "tau_err",
            "beta", "beta_err",
            "A", "C", "S0",
            "cov_tau_tau", "cov_beta_beta",
            "cov_tau_beta", "cov_beta_tau"
        ])
        
        summary_out = OUT_DIR / f"{FRAC_C}_survival_analysis_summary.csv"
        df_res.to_csv(summary_out, index=False)
        print(f"\n[OK] Resumo da análise salvo em {summary_out}")
        print(f"Obstáculos processados com sucesso: {len(resultados)}/{len(OBSTACULOS)}")
    else:
        print("\n[ERRO] Nenhum obstáculo foi processado com sucesso!")

if __name__ == "__main__":
    main()