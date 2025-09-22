import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from pathlib import Path

# -------------------------------
# 0. Configurações Principais
# -------------------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"

# Fração de caçadores (escolha uma)
FRAC_C = "Nc=Np"

# -------- Lista de Obstáculos para Processar --------
OBSTACULOS = [
    "s_obs_00",
    "s_obs_1638",
    "s_obs_3276",
    "s_obs_4915",
    "s_obs_6553",
    "s_obs_8192",
    "s_obs_9830",
    "s_obs_11468",
    "s_obs_13107"
]

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
def analyze_one_obs(data_dir, tag):
    data = load_survival_data(data_dir)
    
    print(f"\n=== {tag} ===")
    print(f"Número total de presas: {len(data)}")
    capturadas_t0 = data[(data["time"] == 0) & (data["event"] == 1)]
    print(f"Presas capturadas em t=0: {len(capturadas_t0)}")
    
    kmf = KaplanMeierFitter()
    naf = NelsonAalenFitter()
    kmf.fit(durations=data["time"], event_observed=data["event"])
    naf.fit(durations=data["time"], event_observed=data["event"])
    
    # Hazard plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kmf.plot_survival_function(ax=axes[0], ci_show=True)
    axes[0].set_title("Função de Sobrevivência (Kaplan–Meier)")
    axes[0].set_ylabel("S(t)"); axes[0].grid(True, alpha=0.3)
    naf.plot_hazard(ax=axes[1], bandwidth=5, ci_show=True)
    axes[1].set_title("Taxa de Risco Instantânea (Nelson–Aalen)")
    axes[1].set_ylabel("h(t)"); axes[1].grid(True, alpha=0.3)
    naf.plot_cumulative_hazard(ax=axes[2], ci_show=True)
    axes[2].set_title("Hazard Acumulado (Nelson–Aalen)")
    axes[2].set_ylabel("H(t)"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_hazard_x_t.pdf", bbox_inches="tight")
    plt.close()
    
    # Fit
    km_time = kmf.survival_function_.index.values
    km_survival = kmf.survival_function_['KM_estimate'].values
    C = km_survival[-1]
    total_presas = len(data)
    sobrevivencia_inicial = 1 - (len(capturadas_t0) / total_presas)
    A = sobrevivencia_inicial - C
    
    def model_fixed_ac(t, tau, beta):
        return stretched_exp_model(t, tau, beta, A, C)
    
    popt, pcov = curve_fit(model_fixed_ac, km_time, km_survival, 
                          p0=[10.0, 0.5], bounds=([0.1, 0.1], [100, 2.0]))
    tau_fit, beta_fit = popt
    tau_err, beta_err = np.sqrt(np.diag(pcov))
    
    print(f"τ = {tau_fit:.3f} ± {tau_err:.3f}, β = {beta_fit:.3f} ± {beta_err:.3f}")
    
    # Fit plot
    time_range = np.linspace(0, km_time.max(), 1000)
    fitted_survival = model_fixed_ac(time_range, tau_fit, beta_fit)
    
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function(ci_show=False, label='Dados Empíricos (KM)')
    plt.plot(time_range, fitted_survival, 'r-', linewidth=2, 
             label=f'Modelo: A·exp(-(t/τ)^β) + C\nτ={tau_fit:.2f}, β={beta_fit:.2f}')
    plt.axhline(y=sobrevivencia_inicial, color='g', linestyle='--', alpha=0.7, 
                label=f'S(0) = {sobrevivencia_inicial:.3f}')
    plt.axhline(y=C, color='purple', linestyle='--', alpha=0.7, 
                label=f'S(∞) = {C:.3f}')
    plt.xlabel('Tempo (steps)')
    plt.ylabel('Probabilidade de Sobrevivência S(t)')
    plt.title(f'Ajuste do Modelo ({tag})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{tag}_fit_hazard_x_t.pdf", bbox_inches="tight")
    plt.close()
    
    return tau_fit, beta_fit, A, C


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
        tau, beta, A, C = analyze_one_obs(obs_dir, tag)
        resultados.append((FRAC_C, obs_name, tau, beta, A, C))
    
    # salvar tabela com resultados
    df_res = pd.DataFrame(resultados, columns=["frac_c", "obs", "tau", "beta", "A", "C"])
    df_res.to_csv(OUT_DIR / f"{FRAC_C}_fit_results.csv", index=False)
    print("\n[OK] Resultados salvos em CSV!")


if __name__ == "__main__":
    main()
