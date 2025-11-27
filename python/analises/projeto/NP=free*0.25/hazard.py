import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter, NelsonAalenFitter
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
    "s_obs_8192",
    "s_obs_9666",
    "s_obs_9830",
    "s_obs_9994",
    "s_obs_11468",
    "s_obs_13107"
]

FRAC_C_LABELS = {
    "Nc=Np":      r"$N^{C}_{0} = N^{E}_{0}$",
    "Nc=Np*0.8":  r"$N^{C}_{0} = 0.8\,N^{E}_{0}$",
    "Nc=Np*0.5":  r"$N^{C}_{0} = 0.5\,N^{E}_{0}$"
}

# Diretórios
FRAC_DIR = BASE_ROOT / FRAC_C
OUT_DIR = BASE_ROOT / "resultados_modelos" / "hazard"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# 1. Funções do Modelo
# -------------------------------
def stretched_exp_model(t, tau, beta, A, C):
    return A * np.exp(-(t / tau) ** beta) + C

def calculate_hazard(t, tau, beta):
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
# 3. Análise para um obstáculo
# -------------------------------
def analyze_one_obs(data_dir, tag, obs_name):
    data = load_survival_data(data_dir)
    
    print(f"\n=== {tag} ===")
    print(f"Número total de presas: {len(data)}")
    capturadas_t0 = data[(data["time"] == 0) & (data["event"] == 1)]
    print(f"Presas capturadas em t=0: {len(capturadas_t0)}")
    
    kmf = KaplanMeierFitter()
    naf = NelsonAalenFitter()
    kmf.fit(durations=data["time"], event_observed=data["event"])
    naf.fit(durations=data["time"], event_observed=data["event"])

    # Fração de obstáculos
    obs_val = int(''.join(filter(str.isdigit, obs_name)))
    phi = obs_val / (128**2)

    # Texto da legenda
    frac_label = FRAC_C_LABELS.get(FRAC_C, FRAC_C)
    legenda = fr"{frac_label}, $\phi={phi:.2f}$"

    # -------------------------------
    # Dados de Sobrevivência (S(t))
    # -------------------------------
    survival_df = kmf.survival_function_.reset_index()
    survival_ci = kmf.confidence_interval_.reset_index()
    
    # Combinar dados de sobrevivência
    survival_data = pd.merge(survival_df, survival_ci, on="timeline")
    survival_data = survival_data.rename(columns={
        "timeline": "step",
        "KM_estimate": "S(t)",
        "KM_estimate_lower_0.95": "S(t)-",
        "KM_estimate_upper_0.95": "S(t)+"
    })
    
    # Normalização S(t)/S(0)
    S0 = survival_data["S(t)"].iloc[0] if len(survival_data) > 0 else 1.0
    survival_data["S(t)_norm"] = survival_data["S(t)"] / S0
    survival_data["S(t)-_norm"] = survival_data["S(t)-"] / S0
    survival_data["S(t)+_norm"] = survival_data["S(t)+"] / S0
    
    # Salvar dados de sobrevivência
    survival_out = OUT_DIR / f"{tag}_survival_data.csv"
    survival_data.to_csv(survival_out, index=False)
    print(f"[OK] Dados de sobrevivência salvos em {survival_out}")

    # -------------------------------
    # Dados de Hazard (h(t))
    # -------------------------------
    hazard_df = naf.smoothed_hazard_(bandwidth=5)
    ci_df = naf.smoothed_hazard_confidence_intervals_(bandwidth=5)

    # Combinar hazard + intervalos
    hazard_all = pd.concat([hazard_df, ci_df], axis=1).reset_index()
    hazard_all = hazard_all.rename(columns={
        "index": "step",
        "differenced-NA_estimate": "h(t)",
        "NA_estimate_lower_0.95": "h(t)-",
        "NA_estimate_upper_0.95": "h(t)+"
    })
    
    # Normalização h(t)/h(0)
    # Pegar o primeiro valor não-NaN para h(0)
    h0_valid = hazard_all["h(t)"].dropna()
    h0 = h0_valid.iloc[0] if len(h0_valid) > 0 else 1.0
    
    hazard_all["h(t)_norm"] = hazard_all["h(t)"] / h0
    hazard_all["h(t)-_norm"] = hazard_all["h(t)-"] / h0
    hazard_all["h(t)+_norm"] = hazard_all["h(t)+"] / h0
    
    # Salvar dados de hazard
    hazard_out = OUT_DIR / f"{tag}_hazard_data.csv"
    hazard_all.to_csv(hazard_out, index=False)
    print(f"[OK] Dados de hazard salvos em {hazard_out}")

    # -------------------------------
    # Fit do Modelo
    # -------------------------------
    km_time = kmf.survival_function_.index.values
    km_survival = kmf.survival_function_['KM_estimate'].values
    
    # Extraindo survival + erro (CI do Kaplan–Meier)
    ci_lower = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
    ci_upper = kmf.confidence_interval_['KM_estimate_upper_0.95'].values
    km_std = (ci_upper - ci_lower) / 2
    
    C = km_survival[-1]
    total_presas = len(data)
    sobrevivencia_inicial = 1 - (len(capturadas_t0) / total_presas)
    A = sobrevivencia_inicial - C
    
    def model_fixed_ac(t, tau, beta):
        return stretched_exp_model(t, tau, beta, A, C)
    
    popt, pcov = curve_fit(
        model_fixed_ac, km_time, km_survival,
        p0=[10.0, 0.5], bounds=([0.1, 0.1], [100, 2.0]),
        sigma=km_std, absolute_sigma=True
    )
    tau_fit, beta_fit = popt
    tau_err, beta_err = np.sqrt(np.diag(pcov))
    
    print(fr"\tau = {tau_fit:.2f} ± {tau_err:.2f}, \beta = {beta_fit:.2f} ± {beta_err:.2f}")

    # -------------------------------
    # Salvar dados do Fit
    # -------------------------------
    time_range = np.linspace(0, km_time.max(), 1000)
    fitted_survival = model_fixed_ac(time_range, tau_fit, beta_fit)
    
    fit_data = pd.DataFrame({
        'step': time_range,
        'fitted_S(t)': fitted_survival,
        'fitted_S(t)_norm': fitted_survival / S0  # Fit normalizado também
    })
    
    fit_out = OUT_DIR / f"{tag}_fit_data.csv"
    fit_data.to_csv(fit_out, index=False)
    print(f"[OK] Dados do fit salvos em {fit_out}")

    # -------------------------------
    # Salvar parâmetros do modelo
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
        'h0': [h0],
        'sobrevivencia_inicial': [sobrevivencia_inicial],
        'total_presas': [total_presas],
        'capturadas_t0': [len(capturadas_t0)]
    }
    
    params_df = pd.DataFrame(params_data)
    params_out = OUT_DIR / f"{tag}_model_params.csv"
    params_df.to_csv(params_out, index=False)
    print(f"[OK] Parâmetros do modelo salvos em {params_out}")
    
    return tau_fit, beta_fit, tau_err, beta_err, A, C, pcov, S0, h0

# -------------------------------
# 4. Loop para todos obstáculos
# -------------------------------
def main():
    resultados = []
    for obs_name in OBSTACULOS:
        obs_dir = FRAC_DIR / obs_name
        if not obs_dir.is_dir():
            print(f"[Aviso] Pasta não encontrada: {obs_dir}")
            continue
        tag = f"{FRAC_C}_{obs_name.replace('s_', '')}"
        tau, beta, tau_err, beta_err, A, C, pcov, S0, h0 = analyze_one_obs(obs_dir, tag, obs_name)
        
        resultados.append((
            FRAC_C, obs_name, tau, tau_err, beta, beta_err, A, C, S0, h0,
            pcov[0, 0], pcov[1, 1], pcov[0, 1], pcov[1, 0]
        ))
    
    df_res = pd.DataFrame(resultados, columns=[
        "frac_c", "obs",
        "tau", "tau_err",
        "beta", "beta_err",
        "A", "C", "S0", "h0",
        "cov_tau_tau", "cov_beta_beta",
        "cov_tau_beta", "cov_beta_tau"
    ])
    
    df_res.to_csv(OUT_DIR / f"{FRAC_C}_fit_results.csv", index=False)
    print("\n[OK] Resultados salvos em CSV!")

if __name__ == "__main__":
    main()