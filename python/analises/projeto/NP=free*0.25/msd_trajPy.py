#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD Ensemble-Averaged usando TrajPy
- Ideal para trajetórias curtas
- Usa ensemble_MSD() do TrajPy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import trajpy.trajpy as tj

# -------- Configurações --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
base = Path("/home/rrossatto/Dados_Doc/Np=free*0.25/Nc=Np*0.5/s_obs_00")
pattern = "*_hunter_trajectories.csv"
Lx = Ly = 128
min_traj_length = 5
out_prefix = "msd_ensemble_hunter_obs_00"

# Saídas
OUT_DIR = BASE_ROOT / "resultados_modelos" / "msd"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- Funções de Unwrapping --------
def unwrap_1d(x, L):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    dx -= np.round(dx / L) * L
    xu = np.empty_like(x, dtype=float)
    xu[0] = x[0]
    xu[1:] = xu[0] + np.cumsum(dx)
    return xu

def unwrap_xy(x, y, Lx, Ly):
    return unwrap_1d(x, Lx), unwrap_1d(y, Ly)

# -------- Função Principal --------
def main():
    # Coleta arquivos
    arquivos = sorted(base.glob(pattern))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {base}")
    
    print(f"Encontrados {len(arquivos)} arquivos")
    
    # Lista para armazenar todas as trajetórias processadas
    todas_trajetorias = []
    comprimentos = []
    
    for arq in arquivos:
        print(f"Processando: {arq.name}")
        
        try:
            df = pd.read_csv(arq, usecols=["timestep", "cell_id", "x", "y"])
            df = df.sort_values(["cell_id", "timestep"])
            
            for cid, g in df.groupby("cell_id"):
                t = g["timestep"].to_numpy(float)
                x = g["x"].to_numpy(float)
                y = g["y"].to_numpy(float)
                
                # Unwrapping
                x, y = unwrap_xy(x, y, Lx, Ly)
                
                # Cria array no formato [x, y] para TrajPy
                traj = np.column_stack([x, y])
                
                if len(traj) >= min_traj_length:
                    todas_trajetorias.append(traj)
                    comprimentos.append(len(traj))
                    
        except Exception as e:
            print(f"Erro processando {arq.name}: {e}")
            continue
    
    if not todas_trajetorias:
        raise RuntimeError("Nenhuma trajetória válida encontrada")
    
    print(f"\nTotal de trajetórias válidas: {len(todas_trajetorias)}")
    print(f"Comprimento médio: {np.mean(comprimentos):.1f} frames")
    print(f"Comprimento mínimo: {np.min(comprimentos)} frames")
    print(f"Comprimento máximo: {np.max(comprimentos)} frames")
    
    # -------- MSD ENSEMBLE-AVERAGED com TrajPy --------
    print("\n📊 Calculando MSD Ensemble-Averaged...")
    
    # Encontra o comprimento máximo possível para análise
    max_tau = min(100, np.min(comprimentos) - 1)  # Não pode exceder trajetória mais curta
    
    # Calcula MSD Ensemble-Averaged usando TrajPy
    try:
        # O TrajPy tem uma função específica para ensemble MSD
        taus = np.arange(1, max_tau + 1)
        
        # Inicializa array para armazenar MSD ensemble
        msd_ensemble = np.zeros(max_tau)
        n_contrib = np.zeros(max_tau, dtype=int)
        
        # Para cada time lag, calcula a média sobre todas as trajetórias
        for tau in taus:
            msd_values = []
            for traj in todas_trajetorias:
                if len(traj) > tau:
                    # MSD para esta trajetória neste tau
                    dr = traj[tau] - traj[0]  # Deslocamento desde o início
                    msd_val = np.sum(dr**2)
                    msd_values.append(msd_val)
            
            if msd_values:
                msd_ensemble[tau-1] = np.mean(msd_values)
                n_contrib[tau-1] = len(msd_values)
            else:
                msd_ensemble[tau-1] = np.nan
                n_contrib[tau-1] = 0
        
        print("✅ Cálculo do Ensemble MSD concluído")
        
    except Exception as e:
        print(f"❌ Erro no cálculo do Ensemble MSD: {e}")
        # Fallback: método manual
        print("Usando método manual...")
        msd_ensemble = np.zeros(max_tau)
        n_contrib = np.zeros(max_tau, dtype=int)
        
        for tau in range(1, max_tau + 1):
            msd_values = []
            for traj in todas_trajetorias:
                if len(traj) > tau:
                    dr = traj[tau] - traj[0]
                    msd_values.append(np.sum(dr**2))
            
            if msd_values:
                msd_ensemble[tau-1] = np.mean(msd_values)
                n_contrib[tau-1] = len(msd_values)
    
    # -------- Análise e Fitting --------
    # Filtra valores com contribuição suficiente
    mask = (n_contrib >= 10) & np.isfinite(msd_ensemble)
    taus_valid = taus[mask]
    msd_valid = msd_ensemble[mask]
    n_contrib_valid = n_contrib[mask]

        # -------- Ajuste log-log para expoente α --------
    # Usamos apenas pontos com msd > 0 (evita log(0))
    mask_fit = (msd_valid > 0)
    x = np.log(taus_valid[mask_fit])
    y = np.log(msd_valid[mask_fit])

    if len(x) >= 3:
        a1, a0 = np.polyfit(x, y, 1)  # y = a1*x + a0  => MSD ≈ exp(a0) * τ^{a1}
        alpha_global = a1
        K_global = np.exp(a0)
    else:
        alpha_global = np.nan
        K_global = np.nan

        # -------- Expoente local α(τ) por diferença finita em log-log --------
    alpha_local = np.full_like(msd_valid, np.nan, dtype=float)
    log_tau = np.log(taus_valid[mask_fit])
    log_msd = np.log(msd_valid[mask_fit])
    # derivada central em log-log
    for i in range(1, len(log_tau)-1):
        num = log_msd[i+1] - log_msd[i-1]
        den = log_tau[i+1] - log_tau[i-1]
        alpha_local[i] = num / den
    # alinhar de volta (guardando só nos índices válidos)
    alpha_series = pd.Series(np.nan, index=taus_valid)
    alpha_series.iloc[mask_fit.nonzero()[0][1:-1]] = alpha_local[1:-1]

        # -------- Curva de ratio R(τ, τ2fixo) --------
    tau2 = max_tau  # ou escolha outro grande com boa estatística
    R_curve = []
    T_curve = []
    for t1 in taus_valid:
        if 1 <= t1 < tau2 <= max_tau:
            i1, i2 = t1-1, tau2-1
            if (np.isfinite(msd_ensemble[i1]) and np.isfinite(msd_ensemble[i2])
                and n_contrib[i1] >= 10 and n_contrib[i2] >= 10):
                try:
                    Rval = tj.Trajectory.msd_ratio_(msd_ensemble, int(t1), int(tau2))
                    if not np.isfinite(Rval):
                        raise ValueError
                except Exception:
                    Rval = (msd_ensemble[i1] / msd_ensemble[i2]) - (t1 / tau2)
                R_curve.append(Rval)
                T_curve.append(t1)
    R_curve = np.array(R_curve, dtype=float)
    T_curve = np.array(T_curve, dtype=int)

       # -------- MSD RATIO (TrajPy) --------
    # Pares de lags que você quer comparar (em frames)
    pares = [(1, 5), (1, 10), (2, 20), (5, int(max_tau/2)), (10, max_tau)]
    result_ratio = []

    def msd_ratio_fallback(msd_vec, t1, t2, eps=1e-12):
        # msd_vec[tau-1] = MSD(tau)
        return (msd_vec[t1-1] / max(msd_vec[t2-1], eps)) - (t1 / t2)

    for t1, t2 in pares:
        # checagens básicas
        if not (1 <= t1 < t2 <= max_tau):
            continue
        # checar se ambos os lags são válidos (não-NaN e com contribuições)
        i1, i2 = t1 - 1, t2 - 1
        if not (np.isfinite(msd_ensemble[i1]) and np.isfinite(msd_ensemble[i2])):
            continue
        if not (n_contrib[i1] >= 10 and n_contrib[i2] >= 10):
            continue

        # tentar TrajPy; se não, cair para manual
        try:
            R = tj.Trajectory.msd_ratio_(msd_ensemble, t1, t2)
            if not np.isfinite(R):
                raise ValueError("msd_ratio_ retornou não finito")
        except Exception:
            R = msd_ratio_fallback(msd_ensemble, int(t1), int(t2))

        result_ratio.append({"tau1": t1, "tau2": t2, "msd_ratio": float(R),
                             "msd_tau1": float(msd_ensemble[i1]),
                             "msd_tau2": float(msd_ensemble[i2]),
                             "n1": int(n_contrib[i1]), "n2": int(n_contrib[i2])})

    ratio_path   = OUT_DIR / f"{out_prefix}_msd_ratio.csv"
    ratio_df = pd.DataFrame(result_ratio)
    ratio_df.to_csv(ratio_path, index=False)
    print("\n📗 MSD ratio (tau1, tau2, R):")
    print(ratio_df if not ratio_df.empty else "sem pares válidos")
    
    if len(taus_valid) < 3:
        raise RuntimeError("Poucos pontos válidos para análise")
    
    # Ajuste linear para estimar D
    try:
        # MSD = 4D·t para difusão 2D normal
        coeffs = np.polyfit(taus_valid, msd_valid, 1)
        slope = coeffs[0]
        D = slope / 4.0  # MSD = 4Dt
        
        # Calcula R²
        y_pred = np.polyval(coeffs, taus_valid)
        ss_res = np.sum((msd_valid - y_pred) ** 2)
        ss_tot = np.sum((msd_valid - np.mean(msd_valid)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"\n📈 Resultados do Ensemble MSD:")
        print(f"Coeficiente de difusão D = {D:.6f}")
        print(f"R² do ajuste linear = {r2:.4f}")
        
    except Exception as e:
        print(f"Erro no ajuste linear: {e}")
        D = np.nan
        r2 = np.nan
    

    # -------- Visualização --------
    plt.figure(figsize=(14, 10))

    # (1) MSD linear
    plt.subplot(2, 2, 1)
    plt.plot(taus_valid, msd_valid, 'o-', label='MSD Ensemble', markersize=4)
    if not np.isnan(D):
        plt.plot(taus_valid, 4*D*taus_valid, '--', label=f'Fit linear: MSD=4·({D:.3f})·τ')
    plt.xlabel('Time Lag τ (frames)')
    plt.ylabel('MSD(τ)')
    plt.title('Ensemble-Averaged MSD (Linear)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (2) MSD log-log + fit global α
    plt.subplot(2, 2, 2)
    plt.loglog(taus_valid, msd_valid, 'o-', label='MSD Ensemble', markersize=4)
    if not np.isnan(alpha_global):
        # reta do fit em log-log
        tau_fit = np.array([taus_valid.min(), taus_valid.max()], dtype=float)
        msd_fit = K_global * tau_fit**alpha_global
        plt.loglog(tau_fit, msd_fit, '--',
                   label=rf'Fit log-log: $\alpha={alpha_global:.3f}$')
    plt.xlabel('Time Lag τ (frames)')
    plt.ylabel('MSD(τ)')
    plt.title('MSD (Log-Log)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)

    # (3) Expoente local α(τ)
    plt.subplot(2, 2, 3)
    plt.plot(alpha_series.index, alpha_series.values, '-', marker='o', ms=3)
    plt.axhline(1.0, linestyle='--', alpha=0.5, label='Difusão normal (α=1)')
    plt.xlabel('τ (frames)')
    plt.ylabel(r'$\alpha(\tau)$')
    plt.title('Expoente local em log-log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (4) Curva de ratio R(τ, τ2 fixo)
    plt.subplot(2, 2, 4)
    if len(T_curve) > 0:
        plt.plot(T_curve, R_curve, '-', marker='o', ms=3,
                 label=rf'$R(\tau,{tau2}) = \frac{{\mathrm{{MSD}}(\tau)}}{{\mathrm{{MSD}}({tau2})}} - \frac{{\tau}}{{{tau2}}}$')
        plt.axhline(0.0, linestyle='--', alpha=0.6, label='Difusão normal (R=0)')
        plt.xlabel('τ (frames)')
        plt.ylabel(r'$R(\tau,\tau_2)$')
        plt.title(rf'Ratio vs τ (com $\tau_2={tau2}$)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Sem pares válidos para R(τ, τ2)', ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    fig_pdf = OUT_DIR / f"{out_prefix}_plot.pdf"
    plt.savefig(fig_pdf, dpi=150, bbox_inches='tight')
    
    # -------- Salva Resultados --------
    resultados = pd.DataFrame({
        'tau': taus,
        'msd_ensemble': msd_ensemble,
        'n_contrib': n_contrib
    })
    csv_path   = OUT_DIR / f"{out_prefix}_results.csv"
    resultados.to_csv(csv_path, index=False)
    
    """
    with open(f"{out_prefix}_summary.txt", "w") as f:
        f.write("=== ENSEMBLE-AVERAGED MSD ANALYSIS ===\n\n")
        f.write(f"Total trajectories: {len(todas_trajetorias)}\n")
        f.write(f"Mean length: {np.mean(comprimentos):.1f} frames\n")
        f.write(f"Min length: {np.min(comprimentos)} frames\n")
        f.write(f"Max length: {np.max(comprimentos)} frames\n")
        f.write(f"Max tau analyzed: {max_tau} frames\n\n")
        
        f.write("\nMSD ratios (tau1, tau2, R):\n")
        if result_ratio:
            for r in result_ratio:
                f.write(f"  ({r['tau1']:>3},{r['tau2']:>3})  R={r['msd_ratio']:.6f}  "
                        f"MSD1={r['msd_tau1']:.6g} (n={r['n1']}), "
                        f"MSD2={r['msd_tau2']:.6g} (n={r['n2']})\n")
        else:
            f.write("  [nenhum par válido]\n")


        f.write("LOG-LOG fit:\n")
        if not np.isnan(alpha_global):
            f.write(f"Alpha (global) = {alpha_global:.6f}\n")
            f.write(f"K (prefactor)  = {K_global:.6g}  [MSD ≈ K·τ^alpha]\n\n")
        else:
            f.write("Alpha (global): n/d\n\n")

        if not np.isnan(D):
            f.write(f"Diffusion coefficient D = {D:.6f}\n")
            f.write(f"R² of linear fit = {r2:.4f}\n")
            f.write(f"Equation: MSD(τ) = {4*D:.4f}·τ\n")
        else:
            f.write("Linear fit failed\n")
    """
    print(f"\n💾 Resultados salvos:")
    print(f"- Gráfico: {out_prefix}_plot.png")
    print(f"- Dados: {out_prefix}_results.csv")
    print(f"- Resumo: {out_prefix}_summary.txt")
    print("\n✅ Análise concluída!")

if __name__ == "__main__":
    main()