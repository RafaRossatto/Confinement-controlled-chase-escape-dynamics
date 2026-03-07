import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from pathlib import Path

# -------------------------------
# 0. CONFIGURAÇÕES PRINCIPAIS - ESCOLHA AQUI OS CAMINHOS
# -------------------------------

# Diferentes tamanhos de rede - ⬅️ ALTERE OS CAMINHOS AQUI
REDE_CONFIGS = [
    (64,  "L_64",  Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE AQUI
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"),   # ⬅️ ALTERE AQUI  
    (256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")    # ⬅️ ALTERE AQUI
]

# Apenas a proporção 0.5
FRAC_C = "Nc=Np*0.5"

# Lista de obstáculos para cada tamanho de rede (ajuste conforme necessário)
OBS_TEMPLATES = {
    64: ["s_obs_00", "s_obs_409", "s_obs_819", "s_obs_1228", "s_obs_1638", "s_obs_2048",
         "s_obs_2416", "s_obs_2457","s_obs_2498","s_obs_2867", "s_obs_3276"],
    128: ["s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", "s_obs_6553","s_obs_8028", "s_obs_8192","s_obs_8355", "s_obs_9666",
    "s_obs_9830", "s_obs_9994", "s_obs_11468", "s_obs_13107"],
    256: ["s_obs_00", "s_obs_6553", "s_obs_13107", "s_obs_19660", "s_obs_26214","s_obs_32112", "s_obs_32768","s_obs_33423", "s_obs_38666",
    "s_obs_39321", "s_obs_39976", "s_obs_45875", "s_obs_52428"]  # exemplo
}

# Diretório de saída
OUT_BASE = Path.home() / "Dados_Doc" / "resultados_modelos" / "hazard_multiple_L"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# --- Configuração global de fonte ---
plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# -------------------------------
# 1. FUNÇÕES DO MODELO
# -------------------------------
def stretched_exp_model(t, tau, beta, A, C):
    return A * np.exp(-(t / tau) ** beta) + C

# -------------------------------
# 2. CARREGAR DADOS
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
# 3. ANÁLISE PARA UM OBSTÁCULO
# -------------------------------
def analyze_one_obs(data_dir, tag, obs_name, L, rede_nome, out_dir):
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
    area = L**2
    phi = obs_val / area

    # Legenda
    legenda = fr"{rede_nome}, $N^C_0 = 0.5N^E_0$, $\phi={phi:.3f}$"

    # Hazard plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Kaplan–Meier
    kmf.plot_survival_function(ax=axes[0], ci_show=True, label=legenda)
    axes[0].set_ylabel("S(t)", fontsize=18)
    axes[0].set_xlabel("steps", fontsize=18)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Hazard
    naf.plot_hazard(ax=axes[1], bandwidth=5, ci_show=True, label=legenda)
    axes[1].set_ylabel("h(t)", fontsize=18)
    axes[1].set_xlabel("steps", fontsize=18)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_hazard_x_t.pdf", bbox_inches="tight")
    plt.close()
    
    # Salvar dados do hazard
    hazard_df = naf.smoothed_hazard_(bandwidth=5)
    ci_df = naf.smoothed_hazard_confidence_intervals_(bandwidth=5)

    hazard_all = pd.concat([hazard_df, ci_df], axis=1).reset_index()
    hazard_all = hazard_all.rename(columns={
        "index": "step",
        "differenced-NA_estimate": "h(t)",
        "NA_estimate_lower_0.95": "h(t)-",
        "NA_estimate_upper_0.95": "h(t)+"
    })

    hazard_out = out_dir / f"{tag}_hazard_data.csv"
    hazard_all.to_csv(hazard_out, index=False)
    print(f"[OK] Hazard + CI salvo em {hazard_out}")
    
    # Fit do modelo
    km_time = kmf.survival_function_.index.values
    km_survival = kmf.survival_function_['KM_estimate'].values
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
    
    print(fr"τ = {tau_fit:.2f} ± {tau_err:.2f}, β = {beta_fit:.2f} ± {beta_err:.2f}")

    # Plot do ajuste
    time_range = np.linspace(0, km_time.max(), 1000)
    fitted_survival = model_fixed_ac(time_range, tau_fit, beta_fit)
    
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function(ci_show=True, label=legenda)
    plt.plot(time_range, fitted_survival, 'r-', linewidth=2,
            label=fr'Fit: $exp(-(t/\tau)^\beta) + C$')
    plt.axhline(y=sobrevivencia_inicial, color='g', linestyle='--', alpha=0.7, 
                label=f'A = {sobrevivencia_inicial:.2f}')
    plt.axhline(y=C, color='purple', linestyle='--', alpha=0.7, 
                label=f'C = {C:.2f}')
    plt.xlabel('steps', fontsize=18)
    plt.ylabel('S(t)', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_fit_hazard_x_t.pdf", bbox_inches="tight")
    plt.close()
    
    return tau_fit, beta_fit, tau_err, beta_err, A, C, pcov

# -------------------------------
# 4. LOOP PRINCIPAL
# -------------------------------
def main():
    todos_resultados = []
    
    for L, rede_nome, base_path in REDE_CONFIGS:
        # Verifica se a pasta existe
        if not base_path.exists():
            print(f"❌ ERRO: Pasta não encontrada: {base_path}")
            print(f"   Por favor, ajuste o caminho em REDE_CONFIGS")
            continue
            
        print(f"\n🎯 Processando rede: {rede_nome} (L={L})")
        print(f"📁 Caminho: {base_path}")
        
        # Diretório de saída para esta rede
        out_dir_rede = OUT_BASE / rede_nome
        out_dir_rede.mkdir(parents=True, exist_ok=True)
        
        # Lista de obstáculos para este L
        obs_list = OBS_TEMPLATES.get(L, [f"s_obs_{i}" for i in range(0, L**2, L**2//8)])
        
        resultados_rede = []
        
        for obs_name in obs_list:
            # Caminho: base_path / Nc=Np*0.5 / obs_name
            obs_dir = base_path / FRAC_C / obs_name
            if not obs_dir.is_dir():
                print(f"[Aviso] Pasta não encontrada: {obs_dir}")
                continue
                
            tag = f"{rede_nome}_{FRAC_C}_{obs_name.replace('s_', '')}"
            
            try:
                tau, beta, tau_err, beta_err, A, C, pcov = analyze_one_obs(
                    obs_dir, tag, obs_name, L, rede_nome, out_dir_rede
                )
                
                resultados_rede.append((
                    rede_nome, L, FRAC_C, obs_name, tau, tau_err, beta, beta_err, 
                    A, C, pcov[0, 0], pcov[1, 1], pcov[0, 1], pcov[1, 0]
                ))
                
            except Exception as e:
                print(f"❌ Erro ao processar {obs_name}: {e}")
                continue
        
        # Salvar resultados desta rede
        if resultados_rede:
            df_res = pd.DataFrame(resultados_rede, columns=[
                "rede", "L", "frac_c", "obs",
                "tau", "tau_err", "beta", "beta_err",
                "A", "C", "cov_tau_tau", "cov_beta_beta",
                "cov_tau_beta", "cov_beta_tau"
            ])
            
            csv_rede = out_dir_rede / f"{rede_nome}_{FRAC_C}_fit_results.csv"
            df_res.to_csv(csv_rede, index=False)
            print(f"\n[OK] Resultados de {rede_nome} salvos em: {csv_rede}")
            
            todos_resultados.extend(resultados_rede)
    
    # Salvar todos os resultados combinados
    if todos_resultados:
        df_todos = pd.DataFrame(todos_resultados, columns=[
            "rede", "L", "frac_c", "obs",
            "tau", "tau_err", "beta", "beta_err",
            "A", "C", "cov_tau_tau", "cov_beta_beta",
            "cov_tau_beta", "cov_beta_tau"
        ])
        
        csv_todos = OUT_BASE / f"TODOS_{FRAC_C}_fit_results.csv"
        df_todos.to_csv(csv_todos, index=False)
        print(f"\n[OK] Todos os resultados salvos em: {csv_todos}")

if __name__ == "__main__":
    main()