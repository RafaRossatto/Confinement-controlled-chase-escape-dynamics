#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

from typing import Callable, Dict, Tuple, Optional, List

from scipy.optimize import curve_fit
from scipy.stats import shapiro, chi2
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# --- Configuração global de fonte nos eixos e legenda ---
plt.rcParams.update({
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14
})

# ============================
# Modelos e utilidades gerais
# ============================

class ModelSpec:
    def __init__(self, name: str, func: Callable, p0=None, param_names=None):
        self.name = name
        self.func = func
        self.p0 = p0
        self.param_names = param_names

def _ensure_1d(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.ravel()
    return x

def linear_f(x, a, b):
    x = _ensure_1d(x)
    return a + b*x

def exp_f(x, A, tau):
    x = _ensure_1d(x)
    return A * np.exp(-x / tau)

def power_f(t, k, a, eps=1e-9):
    t = np.asarray(t, dtype=float)
    return k * np.power(t + eps, a)

def logistic_f(x, L, k, x0):
    x = _ensure_1d(x)
    return L / (1 + np.exp(-k*(x - x0)))

BUILTINS: Dict[str, ModelSpec] = {
    "linear":   ModelSpec("linear",   linear_f,   p0=(0.0, 1.0),         param_names=("a","b")),
    "exp":      ModelSpec("exp",      exp_f,      p0=(1.0, 1.0),         param_names=("A","tau")),
    "power":    ModelSpec("power",    power_f,    p0=(1.0, 1.0),         param_names=("k","a")),
    "logistic": ModelSpec("logistic", logistic_f, p0=(1.0, 1.0, 0.0),    param_names=("L","k","x0")),
}

def fit_model(x, y, spec: ModelSpec, bounds=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ajusta f(x,*theta) com curve_fit e retorna (params, yhat).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = y > 0
    x = x[mask]; y = y[mask]

    if spec.name == "exp":
        ln_C = np.log(y)
        slope, intercept = np.polyfit(x, ln_C, 1)
        A0 = float(np.exp(intercept))
        tau0 = float(-1.0 / slope if slope != 0 else (x.max() - x.min()) / 5)
        if bounds is None:
            bounds = ([0.0, 0.0], [np.inf, np.inf])
        p0 = (A0, max(tau0, 1e-9))

    elif spec.name == "power":
        def power_safe(t, k, a, eps=1e-9):
            return k * np.power(t + eps, a)
        if bounds is None:
            bounds = ([0.0, -np.inf], [np.inf, np.inf])
        p0 = (y[0], -1.0)
        spec = ModelSpec(spec.name, power_safe, p0=p0, param_names=spec.param_names)

    else:
        p0 = spec.p0
        if bounds is None:
            bounds = (-np.inf, np.inf)

    params, _ = curve_fit(spec.func, x, y, p0=p0, bounds=bounds, maxfev=10000)
    yhat = spec.func(x, *params)
    return params, yhat

def metrics(y, yhat) -> Dict[str, float]:
    y = np.asarray(y); yhat = np.asarray(yhat)
    rss = float(np.sum((y - yhat)**2))
    tss = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    rmse = float(np.sqrt(rss / max(len(y),1)))
    return {"R2": r2, "RMSE": rmse, "RSS": rss}

def durbin_watson(res) -> float:
    res = np.asarray(res)
    num = np.sum(np.diff(res)**2)
    den = np.sum(res**2)
    return float(num/den) if den > 0 else np.nan

def residual_tests(y, yhat, exog="x", x=None) -> Dict[str, float]:
    """DW, BP (p-valor), SW (p-valor). exog='x' usa x no BP; 'yhat' usa yhat."""
    res = np.asarray(y) - np.asarray(yhat)
    dw = durbin_watson(res)
    if exog == "yhat":
        X = sm.add_constant(np.asarray(yhat))
    else:
        if x is None:
            X = sm.add_constant(np.asarray(yhat))
        else:
            X = np.asarray(x)
            if X.ndim == 1: X = X[:, None]
            X = sm.add_constant(X)
    _, bp_p, _, _ = het_breuschpagan(res, X)
    W, sw_p = shapiro(res)
    return {"DW": float(dw), "BP_p": float(bp_p), "SW_p": float(sw_p)}

def aic_bic_from_rss(rss: float, n: int, k: int) -> Tuple[float, float]:
    if n <= 0 or rss <= 0: return (np.nan, np.nan)
    aic = n*np.log(rss/n) + 2*k
    bic = n*np.log(rss/n) + k*np.log(n)
    return float(aic), float(bic)

def dummy_constant(y):
    """Prevê sempre a média de y."""
    return np.full_like(y, np.mean(y), dtype=float)

def compare_to_dummy(
    x, y, yhat_model, k_model: int,
    bp_exog="x",
    dummy_func=None, dummy_params=None, k_dummy=None
) -> Dict[str, float]:
    n = len(y)

    if dummy_func is None:
        yhat_dummy = dummy_constant(y)
        k_dummy = 1 if k_dummy is None else k_dummy
    else:
        if dummy_params is None:
            raise ValueError("Se dummy_func for fornecido, dummy_params também deve ser.")
        yhat_dummy = dummy_func(x, *dummy_params)
        if k_dummy is None:
            k_dummy = len(dummy_params)

    met_model = metrics(y, yhat_model)
    met_dummy = metrics(y, yhat_dummy)

    aic_m, bic_m = aic_bic_from_rss(met_model["RSS"], n, k_model)
    aic_d, bic_d = aic_bic_from_rss(met_dummy["RSS"], n, k_dummy)
    dAIC = aic_m - aic_d
    dBIC = bic_m - bic_d

    tests_model = residual_tests(y, yhat_model, exog=bp_exog, x=x)
    tests_dummy = residual_tests(y, yhat_dummy, exog=bp_exog, x=x)

    out = {
        "R2_model": met_model["R2"], "RMSE_model": met_model["RMSE"],
        "R2_dummy": met_dummy["R2"], "RMSE_dummy": met_dummy["RMSE"],
        "dAIC": dAIC, "dBIC": dBIC,
        "DW_model": tests_model["DW"], "BP_p_model": tests_model["BP_p"], "SW_p_model": tests_model["SW_p"],
        "DW_dummy": tests_dummy["DW"], "BP_p_dummy": tests_dummy["BP_p"], "SW_p_dummy": tests_dummy["SW_p"],
    }
    return {k: (float(v) if v is not None else v) for k, v in out.items()}

def evaluate(
    x, y,
    model: str or ModelSpec = "exp",
    bp_exog: str = "x",
    dummy: str or ModelSpec = None,
    dummy_params=None,
    k_dummy: int = None):
    if isinstance(model, str):
        spec = BUILTINS[model]
    else:
        spec = model

    params, yhat = fit_model(x, y, spec)
    k_model = len(params)
    mets = metrics(y, yhat)
    tests = residual_tests(y, yhat, exog=bp_exog, x=x)

    if dummy is not None:
        if isinstance(dummy, str):
            if dummy == "power":
                spec_dummy = ModelSpec("power", power_f, p0=(y[0], -1.0), param_names=("k","a"))
            elif dummy == "exp":
                spec_dummy = BUILTINS["exp"]
            else:
                spec_dummy = BUILTINS[dummy]
        else:
            spec_dummy = dummy

        if dummy_params is None:
            dummy_params, yhat_dummy = fit_model(x, y, spec_dummy)
        else:
            yhat_dummy = spec_dummy.func(x, *dummy_params)

        if k_dummy is None:
            k_dummy = len(dummy_params)

        comp = compare_to_dummy(
            x, y, yhat_model=yhat, k_model=k_model,
            bp_exog=bp_exog,
            dummy_func=spec_dummy.func, dummy_params=dummy_params, k_dummy=k_dummy
        )
    else:
        comp = compare_to_dummy(x, y, yhat, k_model=k_model, bp_exog=bp_exog)

    return {
        "model_name": spec.name,
        "params": dict(zip(spec.param_names or tuple(f"p{i}" for i in range(len(params))), params)),
        "metrics": mets,
        "tests": tests,
        "vs_dummy": comp,
        "yhat": yhat,
    }

# ============================
# I/O e varredura de arquivos
# ============================

RE_RUN = re.compile(r"_run_(\d+)_presas_por_passo\.csv$")

def collect_runs(scen_dir: Path) -> List[Path]:
    return sorted([p for p in scen_dir.rglob("*_run_*_presas_por_passo.csv") if p.is_file()])

def load_xy_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).sort_values("passo")
    if "passo" not in df.columns or "presas_vivas" not in df.columns:
        raise ValueError(f"CSV {path.name} não tem colunas 'passo' e 'presas_vivas'.")
    c = df["presas_vivas"].to_numpy(float)
    t = df["passo"].to_numpy(float)
    mask = np.isfinite(c) & np.isfinite(t) & (c > 0)
    x = t[mask]; y = c[mask]
    if x.size < 3:
        raise ValueError(f"Pontos insuficientes após filtro (>0) em {path.name}.")
    return x, y

# ============================
# Combinação de p-values
# ============================

def fisher_combined_p(pvals: np.ndarray) -> float:
    """Combina p-values (BP ou SW) de múltiplos runs via método de Fisher."""
    ps = np.asarray(pvals, dtype=float)
    ps = ps[np.isfinite(ps)]
    ps = ps[(ps > 0.0) & (ps <= 1.0)]
    if ps.size == 0:
        return float('nan')
    T = -2.0 * np.sum(np.log(np.clip(ps, 1e-300, 1.0)))
    df = 2 * ps.size
    return float(chi2.sf(T, df))

# ============================
# Pipeline multi-runs + plot + VALORES FINAIS por modelo
# ============================

def analisar_cenario(
    base: Path,
    SCENARIO: str,
    n_obs: int,
    salvar_csv: bool = True,
    salvar_fig: bool = True,
    out_dir: Optional[Path] = None,
    alpha: float = 0.05,
):
    """
    Lê todos os runs do cenário, ajusta exp e potência, agrega estatísticas,
    plota a curva média e imprime VALORES FINAIS agregados (por Fisher) para
    os dois modelos (exponencial e potência).
    """
    if out_dir is None:
        out_dir = base / "resultados_comb"
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_dir = base / SCENARIO / (f"s_obs_{n_obs}" if n_obs != 0 else "s_obs_00")
    if not scen_dir.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {scen_dir}")

    arquivos = collect_runs(scen_dir)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV *_run_*_presas_por_passo.csv em {scen_dir}")

    series_list = []
    rows = []

    # coletores de parâmetros (opcional)
    exp_params = []
    pow_params = []

    for path in arquivos:
        try:
            x, y = load_xy_from_csv(path)
        except Exception:
            continue

        # Ajustes
        res_exp = evaluate(x, y, model="exp",   dummy=None)
        res_pow = evaluate(x, y, model="power", dummy=None)

        # parâmetros (opcional, para plot de médias)
        A  = res_exp["params"].get("A", np.nan)
        tau= res_exp["params"].get("tau", np.nan)
        k  = res_pow["params"].get("k", np.nan)
        a  = res_pow["params"].get("a", np.nan)
        exp_params.append((A, tau))
        pow_params.append((k, a))

        # armazena métricas/testes de ambos
        linha = {
            "arquivo": path.name,

            "A": A, "tau": tau, "R2_exp": res_exp["metrics"]["R2"],
            "DW_exp": res_exp["tests"]["DW"],
            "BP_p_exp": res_exp["tests"]["BP_p"],
            "SW_p_exp": res_exp["tests"]["SW_p"],

            "k": k, "a": a, "R2_pow": res_pow["metrics"]["R2"],
            "DW_pow": res_pow["tests"]["DW"],
            "BP_p_pow": res_pow["tests"]["BP_p"],
            "SW_p_pow": res_pow["tests"]["SW_p"],
        }
        rows.append(linha)

        # série para curva média
        df_tmp = pd.read_csv(path)[["passo", "presas_vivas"]]
        df_tmp = df_tmp[df_tmp["presas_vivas"] > 0].set_index("passo").sort_index()
        series_list.append(df_tmp["presas_vivas"])

    if not rows or not series_list:
        raise RuntimeError("Nenhum run válido após filtros.")

    df_runs = pd.DataFrame(rows).sort_values("arquivo")

    # Médias de parâmetros e R² (para título/plot)
    A_med, tau_med = np.nanmean(exp_params, axis=0)
    k_med, a_med   = np.nanmean(pow_params, axis=0)
    r2_exp_med = float(np.nanmean(df_runs["R2_exp"]))
    r2_pow_med = float(np.nanmean(df_runs["R2_pow"]))

    # Curva média real e std
    df_all = pd.concat(series_list, axis=1)
    y_mean = df_all.mean(axis=1)
    y_std  = df_all.std(axis=1)
    x_vals = y_mean.index.values.astype(float)

    # Plot (médias + ajustes médios)
    plt.figure(figsize=(8,6))
    plt.errorbar(x_vals, y_mean.values, yerr=y_std.values,
                 fmt='o', capsize=3, markersize=3, alpha=0.7,
                 label="Dados médios ±1σ")

    x_fit = np.linspace(x_vals.min(), x_vals.max(), 300)
    plt.plot(x_fit, exp_f(x_fit, A_med, tau_med), '-', label=f"Exp médio: A={A_med:.2g}, τ={tau_med:.2g}, R²≈{r2_exp_med:.3f}")
    plt.plot(x_fit, power_f(x_fit, k_med, a_med), '-', label=f"Potência média: k={k_med:.2g}, a={a_med:.2g}, R²≈{r2_pow_med:.3f}")
    plt.xlabel("Passo"); plt.ylabel("Presas vivas")
    plt.title(f"{SCENARIO} • s_obs_{n_obs:02d}" if n_obs==0 else f"{SCENARIO} • s_obs_{n_obs}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

    if salvar_fig:
        fig_base = f"media_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}"
        plt.savefig(out_dir / f"{fig_base}.pdf", bbox_inches="tight")
        plt.savefig(out_dir / f"{fig_base}.png", bbox_inches="tight", dpi=300)
    plt.show()

    if salvar_csv:
        csv_name = f"resultados_por_run_{SCENARIO.replace('/','-')}_s_obs_{n_obs if n_obs!=0 else '00'}.csv"
        df_runs.to_csv(out_dir / csv_name, index=False)

    # ============== VALORES FINAIS (agregados por Fisher) ==============
    # EXPONENCIAL
    p_bp_global_exp = fisher_combined_p(df_runs["BP_p_exp"].values)
    p_sw_global_exp = fisher_combined_p(df_runs["SW_p_exp"].values)
    dw_vals_exp = df_runs["DW_exp"].values
    dw_med_exp = float(np.nanmean(dw_vals_exp))
    dw_frac_ok_exp = float(np.mean((dw_vals_exp >= 1.8) & (dw_vals_exp <= 2.2)))

    pass_norm_exp = (p_sw_global_exp > alpha)
    pass_homo_exp = (p_bp_global_exp > alpha)
    pass_dw_exp   = (1.8 <= dw_med_exp <= 2.2)
    decisao_exp = "APROVADO" if (pass_norm_exp and pass_homo_exp and pass_dw_exp) else "REPROVADO"

    # POTÊNCIA
    p_bp_global_pow = fisher_combined_p(df_runs["BP_p_pow"].values)
    p_sw_global_pow = fisher_combined_p(df_runs["SW_p_pow"].values)
    dw_vals_pow = df_runs["DW_pow"].values
    dw_med_pow = float(np.nanmean(dw_vals_pow))
    dw_frac_ok_pow = float(np.mean((dw_vals_pow >= 1.8) & (dw_vals_pow <= 2.2)))

    pass_norm_pow = (p_sw_global_pow > alpha)
    pass_homo_pow = (p_bp_global_pow > alpha)
    pass_dw_pow   = (1.8 <= dw_med_pow <= 2.2)
    decisao_pow = "APROVADO" if (pass_norm_pow and pass_homo_pow and pass_dw_pow) else "REPROVADO"

    # Prints finais (um bloco por modelo)
    print("\n=== VALOR FINAL da análise de hipóteses — EXPONENCIAL (α = {:.3f}) ===".format(alpha))
    print("Normalidade (Shapiro):             p_global = {:.3g}  → {}".format(
        p_sw_global_exp, "aceita H0 (normais)" if pass_norm_exp else "rejeita H0 (não normais)"))
    print("Homoscedasticidade (Breusch–Pagan): p_global = {:.3g}  → {}".format(
        p_bp_global_exp, "aceita H0 (homoscedásticos)" if pass_homo_exp else "rejeita H0 (heteroscedasticidade)"))
    print("Autocorrelação (Durbin–Watson):     DW_médio = {:.3f} (fração em [1.8,2.2]: {:>.1%}) → {}".format(
        dw_med_exp, dw_frac_ok_exp, "sem autocorr. significativa (médio≈2)" if pass_dw_exp else "possível autocorr."))
    print("Conclusão geral (EXP): {}".format(decisao_exp))

    print("\n=== VALOR FINAL da análise de hipóteses — POTÊNCIA (α = {:.3f}) ===".format(alpha))
    print("Normalidade (Shapiro):             p_global = {:.3g}  → {}".format(
        p_sw_global_pow, "aceita H0 (normais)" if pass_norm_pow else "rejeita H0 (não normais)"))
    print("Homoscedasticidade (Breusch–Pagan): p_global = {:.3g}  → {}".format(
        p_bp_global_pow, "aceita H0 (homoscedásticos)" if pass_homo_pow else "rejeita H0 (heteroscedasticidade)"))
    print("Autocorrelação (Durbin–Watson):     DW_médio = {:.3f} (fração em [1.8,2.2]: {:>.1%}) → {}".format(
        dw_med_pow, dw_frac_ok_pow, "sem autocorr. significativa (médio≈2)" if pass_dw_pow else "possível autocorr."))
    print("Conclusão geral (POT): {}".format(decisao_pow))

    return {
        "df_runs": df_runs,
        "x_mean": x_vals,
        "y_mean": y_mean.values,
        "y_std": y_std.values,
        "params_mean": {"A": A_med, "tau": tau_med, "k": k_med, "a": a_med},
        "r2_mean": {"exp": r2_exp_med, "power": r2_pow_med},
        "final_exp": {
            "alpha": alpha,
            "p_global_shapiro": p_sw_global_exp,
            "p_global_breusch_pagan": p_bp_global_exp,
            "dw_medio": dw_med_exp,
            "dw_frac_intervalo_1p8_2p2": dw_frac_ok_exp,
            "decisao": decisao_exp,
        },
        "final_power": {
            "alpha": alpha,
            "p_global_shapiro": p_sw_global_pow,
            "p_global_breusch_pagan": p_bp_global_pow,
            "dw_medio": dw_med_pow,
            "dw_frac_intervalo_1p8_2p2": dw_frac_ok_pow,
            "decisao": decisao_pow,
        }
    }

# ============================
# Execução
# ============================

if __name__ == "__main__":
    base = Path.home() / "Dados_Doc/Np=free*0.25"

    # --- ajuste aqui ---
    SCENARIO = "Nc=Np"        # ex.: "Nc=Np" ou "Nc=Np*0.5"
    n_obs = 0                 # 0 -> usa s_obs_00; caso contrário s_obs_{n_obs}
    ALPHA = 0.05              # nível de significância
    # --------------------

    _ = analisar_cenario(
        base=base,
        SCENARIO=SCENARIO,
        n_obs=n_obs,
        salvar_csv=True,
        salvar_fig=True,
        out_dir=base / "resultados_comb",
        alpha=ALPHA,
    )
