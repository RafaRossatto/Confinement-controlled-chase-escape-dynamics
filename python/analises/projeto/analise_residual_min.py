# === Avaliação Genérica de Modelos + Dummy (sem plots) ===
# Requisitos: numpy, scipy, statsmodels

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

# --------------------- Modelos prontos ---------------------

@dataclass
class ModelSpec:
    name: str
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]  # f(x, *theta) -> y
    p0: Optional[Tuple[float, ...]] = None                # chute inicial
    param_names: Optional[Tuple[str, ...]] = None         # nomes dos parâmetros

# Exemplos de funções de modelo (x pode ser 1D ou matriz com shape (n, d))
def _ensure_1d(x):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1: x = x.ravel()
    return x

def linear_f(x, a, b):
    x = _ensure_1d(x); return a + b*x

def exp_f(x, A, tau):
    x = _ensure_1d(x); return A * np.exp(-x / tau)

def power_f(x, k, a):
    x = _ensure_1d(x); return k * np.power(x, a)

def logistic_f(x, L, k, x0):
    x = _ensure_1d(x); return L / (1 + np.exp(-k*(x - x0)))

BUILTINS: Dict[str, ModelSpec] = {
    "linear":   ModelSpec("linear",   linear_f,   p0=(0.0, 1.0),         param_names=("a","b")),
    "exp":      ModelSpec("exp",      exp_f,      p0=(1.0, 1.0),         param_names=("A","tau")),
    "power":    ModelSpec("power",    power_f,    p0=(1.0, 1.0),         param_names=("k","a")),
    "logistic": ModelSpec("logistic", logistic_f, p0=(1.0, 1.0, 0.0),    param_names=("L","k","x0")),
}

# --------------------- Ajuste + Métricas ---------------------

def fit_model(x, y, spec: ModelSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Ajusta f(x,*theta) com curve_fit e retorna (params, yhat)."""
    x = np.asarray(x); y = np.asarray(y)
    params, _ = curve_fit(spec.func, x, y, p0=spec.p0, maxfev=10000)
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
            # fallback: usa yhat
            X = sm.add_constant(np.asarray(yhat))
        else:
            X = np.asarray(x)
            if X.ndim == 1: X = X[:, None]
            X = sm.add_constant(X)
    _, bp_p, _, _ = het_breuschpagan(res, X)
    W, sw_p = shapiro(res)
    return {"DW": float(dw), "BP_p": float(bp_p), "SW_p": float(sw_p)}

def aic_bic_from_rss(rss: float, n: int, k: int) -> Tuple[float, float]:
    """AIC/BIC de um modelo com k parâmetros (incluindo intercepto, se houver)."""
    if n <= 0 or rss <= 0: return (np.nan, np.nan)
    aic = n*np.log(rss/n) + 2*k
    bic = n*np.log(rss/n) + k*np.log(n)
    return float(aic), float(bic)

# --------------------- Dummy constante + Comparação ---------------------

def dummy_constant(y):
    return np.full_like(np.asarray(y, dtype=float), float(np.mean(y)))

def compare_to_dummy(x, y, yhat_model, k_model: int, bp_exog="x") -> Dict[str, float]:
    """Compara modelo ajustado vs dummy constante: R2, RMSE, ΔAIC, ΔBIC, p-tests do modelo."""
    n = len(y)
    yhat_dummy = dummy_constant(y)
    met_model = metrics(y, yhat_model)
    met_dummy = metrics(y, yhat_dummy)

    # AIC/BIC (k_model = nº parâmetros livres do modelo; dummy tem k=1)
    aic_m, bic_m = aic_bic_from_rss(met_model["RSS"], n, k_model)
    aic_d, bic_d = aic_bic_from_rss(met_dummy["RSS"], n, 1)
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

# --------------------- Pipeline principal ---------------------

def evaluate(x, y,
             model: str | ModelSpec = "exp",
             bp_exog: str = "x"):
    """
    x, y: dados
    model: nome em BUILTINS ('linear','exp','power','logistic') OU uma ModelSpec custom
    bp_exog: 'x' usa X no Breusch–Pagan; 'yhat' usa yhat como regressora da variância
    """
    if isinstance(model, str):
        spec = BUILTINS[model]
    else:
        spec = model

    # Ajuste
    params, yhat = fit_model(x, y, spec)
    # nº de parâmetros (p/ AIC/BIC): usa tamanho do vetor de parâmetros do modelo
    k_model = len(params)

    # Métricas + testes do modelo
    mets = metrics(y, yhat)
    tests = residual_tests(y, yhat, exog=bp_exog, x=x)

    # Comparação com dummy
    comp = compare_to_dummy(x, y, yhat, k_model=k_model, bp_exog=bp_exog)

    result = {
        "model_name": spec.name,
        "params": dict(zip(spec.param_names or tuple(f"p{i}" for i in range(len(params))), params)),
        "metrics": mets,
        "tests": tests,
        "vs_dummy": comp,
    }
    return result


import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

# mapeia chaves de cenário -> possíveis subpastas dentro de obs/s_obs
MODEL_CATALOG: Dict[str, List[str]] = {
    "Nc=Np":      ["Nc=Np"],
    "Nc=0.5Np":   ["Nc=Np*0.5", "Nc= Np*0.5"],
    # adicione mais se precisar
}

# --- já existentes ---
RE_RUN = re.compile(r"_run_(\d+)_presas_por_passo\.csv$")

def find_obs_dir(base: Path, n_obs: int) -> Optional[Path]:
    """Procura por .../obs_{n_obs} ou .../s_obs_{n_obs} dentro de base."""
    cand = [base / f"obs_{n_obs}", base / f"s_obs_{n_obs}"]
    for d in cand:
        if d.is_dir():
            return d
    return None

# --- NOVA: listar subpastas candidatas (cenários) ---
def list_scenario_dirs(obs_dir: Path) -> list[Path]:
    """Lista apenas as subpastas diretas (cenários) dentro de obs_dir."""
    return [d for d in obs_dir.iterdir() if d.is_dir()]

# --- NOVA: escolher cenário automaticamente a partir de um run específico ---
def find_scenario_dir_auto(obs_dir: Path, run: int | str = "latest") -> Path:
    """
    Procura, entre as subpastas de obs_dir, aquela que contém:
      - se run == 'latest': algum CSV *_run_*_presas_por_passo.csv (pega a mais recente)
      - se run é int: pelo menos um CSV com _run_{run}_... (escolhe a mais recente entre as que têm esse run)
    Retorna o diretório do cenário escolhido.
    """
    candidates = list_scenario_dirs(obs_dir)
    best_dir: Path | None = None
    best_time = -1.0

    for d in candidates:
        csvs = list(d.rglob("*_run_*_presas_por_passo.csv"))
        if not csvs:
            continue

        if isinstance(run, str) and run.lower() == "latest":
            # pega o arquivo mais recente de cada subpasta e escolhe a subpasta com o arquivo mais novo
            p = max(csvs, key=lambda p: p.stat().st_mtime)
            t = p.stat().st_mtime
            if t > best_time:
                best_time = t
                best_dir = d
        else:
            # filtra pelos arquivos que têm exatamente esse run
            wanted = []
            for p in csvs:
                m = RE_RUN.search(p.name)
                if m and int(m.group(1)) == int(run):
                    wanted.append(p)
            if not wanted:
                continue
            p = max(wanted, key=lambda p: p.stat().st_mtime)
            t = p.stat().st_mtime
            if t > best_time:
                best_time = t
                best_dir = d

    if best_dir is None:
        if isinstance(run, str) and run.lower() == "latest":
            raise FileNotFoundError("Nenhuma subpasta com arquivos *_run_*_presas_por_passo.csv encontrada.")
        else:
            raise FileNotFoundError(f"Nenhuma subpasta contém arquivos do run {run}.")
    return best_dir

def load_xy_from_run(scen_dir: Path, run: int | str = "latest") -> tuple[np.ndarray, np.ndarray]:
    """Carrega x=passo, y=presas_vivas (>0) do run escolhido (exato ou 'latest') dentro de scen_dir."""
    csvs = list(scen_dir.rglob("*_run_*_presas_por_passo.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV de run encontrado em {scen_dir}")

    if isinstance(run, str) and run.lower() == "latest":
        path = max(csvs, key=lambda p: p.stat().st_mtime)
    else:
        candidatos = []
        for p in csvs:
            m = RE_RUN.search(p.name)
            if m and int(m.group(1)) == int(run):
                candidatos.append(p)
        if not candidatos:
            raise FileNotFoundError(f"Run {run} não encontrado em {scen_dir}")
        path = max(candidatos, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(path).sort_values("passo")
    if "passo" not in df.columns or "presas_vivas" not in df.columns:
        raise ValueError(f"CSV {path.name} não tem colunas 'passo' e 'presas_vivas'.")

    c = df["presas_vivas"].to_numpy(float)
    t = df["passo"].to_numpy(float)
    mask = np.isfinite(c) & np.isfinite(t) & (c > 0)
    x = t[mask]
    y = c[mask]
    if x.size < 3:
        raise ValueError(f"Pontos insuficientes após filtro (>0) no arquivo {path.name}.")
    return x, y



def interpretar_resultados(res):
    interpretacoes = []

    # Modelo principal
    interpretacoes.append("📊 Modelo principal:")
    r2 = res["metrics"].get("R2", None)
    if r2 is not None:
        if r2 > 0.9:
            interpretacoes.append(f"  - R² = {r2:.3f} → Excelente ajuste.")
        elif r2 > 0.7:
            interpretacoes.append(f"  - R² = {r2:.3f} → Bom ajuste.")
        elif r2 > 0.5:
            interpretacoes.append(f"  - R² = {r2:.3f} → Ajuste moderado.")
        else:
            interpretacoes.append(f"  - R² = {r2:.3f} → Ajuste fraco.")

    dw = res["tests"].get("DW", None)
    if dw is not None:
        if 1.8 <= dw <= 2.2:
            interpretacoes.append(f"  - DW = {dw:.3f} → Sem autocorrelação significativa.")
        elif dw < 1.8:
            interpretacoes.append(f"  - DW = {dw:.3f} → Autocorrelação positiva.")
        else:
            interpretacoes.append(f"  - DW = {dw:.3f} → Autocorrelação negativa.")

    bp = res["tests"].get("BP_p", None)
    if bp is not None:
        if bp > 0.05:
            interpretacoes.append(f"  - BP_p = {bp:.3g} → Sem evidência de heteroscedasticidade.")
        else:
            interpretacoes.append(f"  - BP_p = {bp:.3g} → Evidência de heteroscedasticidade.")

    sw = res["tests"].get("SW_p", None)
    if sw is not None:
        if sw > 0.05:
            interpretacoes.append(f"  - SW_p = {sw:.3g} → Resíduos normais.")
        else:
            interpretacoes.append(f"  - SW_p = {sw:.3g} → Resíduos não normais.")

    # Dummy
    interpretacoes.append("\n📊 Dummy constante:")
    r2_d = res["vs_dummy"].get("R2_dummy", None)
    if r2_d is not None:
        if r2_d > 0.9:
            interpretacoes.append(f"  - R² = {r2_d:.3f} → Excelente ajuste.")
        elif r2_d > 0.7:
            interpretacoes.append(f"  - R² = {r2_d:.3f} → Bom ajuste.")
        elif r2_d > 0.5:
            interpretacoes.append(f"  - R² = {r2_d:.3f} → Ajuste moderado.")
        else:
            interpretacoes.append(f"  - R² = {r2_d:.3f} → Ajuste fraco.")

    dw_d = res["vs_dummy"].get("DW_dummy", None)
    if dw_d is not None:
        if 1.8 <= dw_d <= 2.2:
            interpretacoes.append(f"  - DW = {dw_d:.3f} → Sem autocorrelação significativa.")
        elif dw_d < 1.8:
            interpretacoes.append(f"  - DW = {dw_d:.3f} → Autocorrelação positiva.")
        else:
            interpretacoes.append(f"  - DW = {dw_d:.3f} → Autocorrelação negativa.")

    bp_d = res["vs_dummy"].get("BP_p_dummy", None)
    if bp_d is not None:
        if bp_d > 0.05:
            interpretacoes.append(f"  - BP_p = {bp_d:.3g} → Sem evidência de heteroscedasticidade.")
        else:
            interpretacoes.append(f"  - BP_p = {bp_d:.3g} → Evidência de heteroscedasticidade.")

    sw_d = res["vs_dummy"].get("SW_p_dummy", None)
    if sw_d is not None:
        if sw_d > 0.05:
            interpretacoes.append(f"  - SW_p = {sw_d:.3g} → Resíduos normais.")
        else:
            interpretacoes.append(f"  - SW_p = {sw_d:.3g} → Resíduos não normais.")

    return interpretacoes

if __name__ == "__main__":

    base = Path.home() / "Dados_Doc"
    n_obs = 0  # pode ser 0, 1638, 3276, ...
    SCENARIO = "Nc=Np*0.5"
    RUN = 30  # número inteiro

    # Ajuste do caminho do cenário
    if SCENARIO == "Nc=Np*0.5":
        scen_dir = base / "Nc=Np*0.5"
    else:
        scen_dir = base / SCENARIO

    if not scen_dir.exists():
        raise FileNotFoundError(f"Cenário '{SCENARIO}' não encontrado em {base}")

    # Nome da pasta s_obs
    if n_obs == 0:
        obs_dir = scen_dir / "s_obs_00"
    else:
        obs_dir = scen_dir / f"s_obs_{n_obs}"

    if not obs_dir.exists():
        raise FileNotFoundError(f"{obs_dir.name} não encontrado em {scen_dir}")

    # Busca pelo CSV do run
    arquivo = next(obs_dir.glob(f"*run_{RUN}_presas_por_passo.csv"), None)
    if arquivo is None:
        raise FileNotFoundError(f"Nenhum arquivo para run_{RUN} encontrado em {obs_dir}")

    
    # Leitura e debug das colunas
    print(f"Lendo {arquivo}...")
    df = pd.read_csv(arquivo)
    print("Colunas encontradas:", df.columns.tolist())
    print(df)
    
    df = pd.read_csv(arquivo)
    x = df["passo"].to_numpy()
    y = df["presas_vivas"].to_numpy()

    # 🔹 Remove todos os pontos com presas_vivas = 0
    mask = y > 0
    x = x[mask]
    y = y[mask]

    """
    print("x:", x)
    print("y:", y)
    print("n pontos:", len(x))
"""
    # ---------- escolha do FIT ----------
    # pode ser: "exp" | "linear" | "power" | "logistic" | uma ModelSpec custom
    MODEL_NAME = "exp"

    # opcional: para Breusch–Pagan, usar 'x' (recom.) ou 'yhat' como regressora da variância
    BP_EXOG = "x"

    # avalia
    res = evaluate(x, y, model=MODEL_NAME, bp_exog=BP_EXOG)
"""
    # ---------- saída resumida ----------
    print(f"\n=== Modelo: {res['model_name']} | cenário '{SCENARIO}' | n_obs={n_obs} | run={RUN} ===")
    print("Parâmetros:", {k: f"{v:.6g}" for k, v in res["params"].items()})
    print("Métricas:",   {k: f"{v:.6g}" for k, v in res["metrics"].items()})
    print("Testes:",     {k: f"{v:.6g}" for k, v in res["tests"].items()})
    print("\n--- Comparação vs Dummy Constante ---")
    for k, v in res["vs_dummy"].items():
        print(f"{k}: {v:.6g}")
        """
print("\n--- Interpretação automática ---")
for linha in interpretar_resultados(res):
    print(linha)
print("Parâmetros ajustados:", res["params"])