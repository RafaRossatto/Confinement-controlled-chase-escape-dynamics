# === KIT A + ROTA 2 (com diagnósticos numéricos pós-R²) ===
# Requisitos: numpy, pandas, matplotlib, scipy, statsmodels

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from scipy.stats import linregress, probplot, t as tdist, mannwhitneyu, shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# ===== Parâmetros =====
base_root = Path.home() / "Dados_Doc"
n_obs = 0  # ajuste (ex.: 0, 3276, 11468...)

# Catálogo e seleção dos modelos (cenários) a comparar
MODEL_CATALOG: dict[str, tuple[str, list[str]]] = {
    "Nc=Np":      (r"$N_C = N_P$",        ["Nc=Np"]),
    "Nc=0.5Np":   (r"$N_C = 0.5\,N_P$",   ["Nc=Np*0.5", "Nc= Np*0.5"]),
    # Adicione aqui se quiser mais:
    # "Nc=2Np":   (r"$N_C = 2\,N_P$",     ["Nc=2Np"]),
}
SELECT_MODELS: list[str] = ["Nc=Np", "Nc=0.5Np"]

SEED = 12345        # reprodutibilidade do bootstrap
B = 5000            # nº de réplicas bootstrap (reduza p/ 2000 se ficar pesado)
MIN_PONTOS_RUN = 3  # mínimo de pontos (C>0) por run para ajustar lnC vs t
SALVAR = False      # True para salvar PDFs e CSVs
PREFIXO_SAIDA = f"analise_phi_{round(n_obs/(128**2),1)}"

# Controle do relatório numérico pós-R²
RELATAR_NUMERICO = True   # imprime tabela com R², DW, BP p, SW p
CSV_DIAGNOSTICOS = f"{PREFIXO_SAIDA}_diagnosticos.csv"

rng = np.random.default_rng(SEED)

# ===== Helpers de navegação =====
def find_obs_dir(base_root: Path, subpasta: str, n_obs: int) -> Optional[Path]:
    base = base_root / subpasta
    if not base.exists():
        return None
    for d in base.glob("s_obs_*"):
        if d.is_dir():
            m = re.fullmatch(r"s_obs_(\d+)", d.name)
            if m and int(m.group(1)) == n_obs:
                return d
    for name in (f"s_obs_{n_obs}", f"s_obs_{n_obs:02d}", f"s_obs_{n_obs:03d}"):
        d = base / name
        if d.exists():
            return d
    return None

RE_RUN = re.compile(r"_run_(\d+)_presas_por_passo\.csv$")

def listar_por_run(dir_obs: Path) -> Dict[int, List[Path]]:
    por_run: Dict[int, List[Path]] = {}
    for p in dir_obs.rglob("**/NC_*_O_*_run_*_presas_por_passo.csv"):
        m = RE_RUN.search(p.name)
        if not m:
            continue
        r = int(m.group(1))
        por_run.setdefault(r, []).append(p)
    return por_run

def escolher_mais_recente(paths: List[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)

# ===== Kit A: pontos médios de lnC por passo + diagnósticos (plots) =====
def media_std_por_passo_lnC(
    por_run: Dict[int, List[Path]],
    limitar_runs: Tuple[int,int] | None = None  # ex.: (0,99); None usa todos os presentes
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs_presentes = sorted(por_run.keys())
    if limitar_runs is not None:
        a, b = limitar_runs
        runs_presentes = [r for r in runs_presentes if a <= r <= b]

    series_lnC: list[np.ndarray] = []
    max_t = 0
    for r in runs_presentes:
        arq = escolher_mais_recente(por_run[r])
        df = pd.read_csv(arq)
        if "passo" not in df.columns or "presas_vivas" not in df.columns:
            continue
        df = df.sort_values("passo")
        c = df["presas_vivas"].to_numpy(dtype=float)
        lnC = np.full_like(c, np.nan, dtype=float)
        pos = c > 0
        lnC[pos] = np.log(c[pos])
        max_t = max(max_t, lnC.size)
        series_lnC.append(lnC)

    if not series_lnC:
        return np.array([]), np.array([]), np.array([])

    arr = np.full((len(series_lnC), max_t), np.nan, dtype=float)
    for i, lnC in enumerate(series_lnC):
        arr[i, :lnC.size] = lnC

    col_ok = ~np.all(np.isnan(arr), axis=0)
    if not np.any(col_ok):
        return np.array([]), np.array([]), np.array([])

    arr_ok = arr[:, col_ok]
    t_axis = np.arange(max_t)[col_ok]

    lnC_media = np.nanmean(arr_ok, axis=0)
    lnC_std   = np.nanstd(arr_ok, axis=0)

    mask_fin = np.isfinite(lnC_media)
    return t_axis[mask_fin], lnC_media[mask_fin], lnC_std[mask_fin]

def durbin_watson(residuos: np.ndarray) -> float:
    dif = np.diff(residuos)
    num = np.sum(dif**2)
    den = np.sum(residuos**2)
    return np.nan if den == 0 else num/den

def sanitize_label_for_file(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)

def kitA_diagnosticos(t: np.ndarray, y_lnC: np.ndarray, yerr: np.ndarray | None,
                      label: str, salvar: bool, prefixo_saida: str):
    if t.size < 3:
        print(f"[{label}] Pontos insuficientes para OLS (n={t.size}).")
        return

    res = linregress(t, y_lnC)
    a, b = res.intercept, res.slope
    R2 = res.rvalue**2
    p_b = res.pvalue
    n = t.size
    df_ = max(n - 2, 1)
    tcrit = tdist.ppf(0.975, df_)
    b_ci = (b - tcrit*res.stderr, b + tcrit*res.stderr)

    tau = -1.0/b if b != 0 else np.inf
    if b_ci[0] < 0 and b_ci[1] < 0:
        tau_ci = (-1.0/b_ci[1], -1.0/b_ci[0])
    else:
        tau_ci = (np.nan, np.nan)

    y_hat = a + b*t
    resids = y_lnC - y_hat

    DW = durbin_watson(resids)

    # Breusch–Pagan (heteroscedasticidade)
    X = sm.add_constant(t)
    bp_stat, bp_p, f_stat, f_p = het_breuschpagan(resids, X)

    # --- Relatório curto ---
    print(f"\n=== KIT A :: {label} ===")
    print(f"n = {n}")
    print(f"b (declive) = {b:.6g}  [IC95%: {b_ci[0]:.6g}, {b_ci[1]:.6g}]   p(b)= {p_b:.3e}")
    print(f"tau = {(-1/b):.6g}  [IC95%: {tau_ci[0]:.6g}, {tau_ci[1]:.6g}]" if np.isfinite(tau) else "tau = inf")
    print(f"R^2 = {R2:.4f}")
    print(f"Durbin–Watson = {DW:.3f}   (≈2 indica pouca autocorrelação)")
    print(f"Breusch–Pagan: LM={bp_stat:.3f}  p={bp_p:.3g}  (p<0.05 sugere heteroscedasticidade)")

    # --- Gráfico 1: pontos médios (±desvio) + reta ---
    plt.figure(figsize=(9,6))
    if yerr is not None and np.any(np.isfinite(yerr)):
        plt.errorbar(t, y_lnC, yerr=yerr, fmt='o', markersize=4, capsize=3, label=f"Média {label}")
    else:
        plt.scatter(t, y_lnC, s=18, label=f"Média {label}")
    t_plot = np.linspace(t.min(), t.max(), 200)
    plt.plot(t_plot, a + b*t_plot, linewidth=2,
             label=f"Ajuste {label}: $\\tau$={tau:.2f}, $R^2$={R2:.3f}")
    plt.xlabel("Passos (t)")
    plt.ylabel("ln(C médio) ± desvio padrão")
    plt.title(f"Kit A — Ajuste e pontos médios | $\\phi$ = {round(n_obs/(128**2),1)}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if salvar:
        fname = f"{prefixo_saida}_ajuste_{sanitize_label_for_file(label)}.pdf"
        plt.savefig(fname, format="pdf"); print(f"[salvo] {fname}")
    plt.show()

    # --- Gráfico 2: resíduos vs t ---
    resids = y_lnC - (a + b*t)  # reforça
    plt.figure(figsize=(9,5))
    plt.axhline(0, lw=1, ls='--')
    plt.scatter(t, resids, s=16)
    plt.xlabel("Passos (t)")
    plt.ylabel("Resíduos (lnC - ajuste)")
    plt.title(f"Kit A — Resíduos vs t ({label})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        fname = f"{prefixo_saida}_residuos_{sanitize_label_for_file(label)}.pdf"
        plt.savefig(fname, format="pdf"); print(f"[salvo] {fname}")
    plt.show()

    # --- Gráfico 3: Q–Q plot dos resíduos ---
    plt.figure(figsize=(6,6))
    (osm, osr), (slope, intercept, r) = probplot(resids, dist="norm")
    plt.scatter(osm, osr, s=16)
    plt.plot(osm, slope*osm + intercept, lw=2)
    plt.xlabel("Quantis teóricos N(0,1)")
    plt.ylabel("Quantis dos resíduos")
    plt.title(f"Kit A — Q–Q plot ({label})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if salvar:
        fname = f"{prefixo_saida}_qqplot_{sanitize_label_for_file(label)}.pdf"
        plt.savefig(fname, format="pdf"); print(f"[salvo] {fname}")
    plt.show()

# ===== Diagnósticos numéricos pós-R² (SEM plots) =====
def diagnosticos_numericos_pos_R2(t: np.ndarray, y_lnC: np.ndarray) -> dict:
    """
    Calcula R², Durbin–Watson, Breusch–Pagan (p-valor) e Shapiro–Wilk (p-valor)
    sem gerar plots. Retorna dict com os valores numéricos.
    """
    if t.size < 3:
        return {"R2": np.nan, "DW": np.nan, "BP_p": np.nan, "SW_p": np.nan}

    res = linregress(t, y_lnC)
    a, b = res.intercept, res.slope
    R2 = res.rvalue**2
    y_hat = a + b*t
    resids = y_lnC - y_hat

    # Durbin–Watson
    DW = durbin_watson(resids)

    # Breusch–Pagan (usa regressão dos resíduos^2 nas covariáveis)
    X = sm.add_constant(t)
    _, bp_p, _, _ = het_breuschpagan(resids, X)

    # Shapiro–Wilk
    # Para n muito grande, o p pode ser sempre baixo; use em conjunto com QQ-plot
    W_stat, p_shapiro = shapiro(resids)

    return {"R2": float(R2), "DW": float(DW), "BP_p": float(bp_p), "SW_p": float(p_shapiro)}

# ===== Rota 2: τ por run + comparações =====
def tau_por_run(por_run: Dict[int, List[Path]],
                min_pontos: int = MIN_PONTOS_RUN) -> Tuple[np.ndarray, List[int]]:
    taus, runs_ok = [], []
    for r, paths in sorted(por_run.items()):
        arq = escolher_mais_recente(paths)
        df = pd.read_csv(arq)
        if "passo" not in df.columns or "presas_vivas" not in df.columns:
            continue
        df = df.sort_values("passo")
        c = df["presas_vivas"].to_numpy(float)
        mask = c > 0
        if np.count_nonzero(mask) < min_pontos:
            continue
        t = df.loc[mask, "passo"].to_numpy(float)
        lnC = np.log(c[mask])
        if t.size < 2:
            continue
        res = linregress(t, lnC)
        b = res.slope
        if np.isfinite(b) and b < 0:
            tau = -1.0 / b
            if np.isfinite(tau):
                taus.append(tau)
                runs_ok.append(r)
    return np.array(taus, float), runs_ok

def bootstrap_ci_mediana(x: np.ndarray, B: int = B, alpha: float = 0.05) -> Tuple[float, Tuple[float,float]]:
    x = np.asarray(x, float)
    if x.size == 0:
        return (np.nan, (np.nan, np.nan))
    meds = []
    n = x.size
    for _ in range(B):
        sample = rng.choice(x, size=n, replace=True)
        meds.append(np.median(sample))
    m = float(np.median(x))
    lo, hi = np.percentile(meds, [100*alpha/2, 100*(1-alpha/2)])
    return m, (float(lo), float(hi))

def bootstrap_ci_diff_ratio_medianas(x: np.ndarray, y: np.ndarray,
                                     B: int = B, alpha: float = 0.05) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return ((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan))
    n1, n2 = x.size, y.size
    diffs, ratios = [], []
    for _ in range(B):
        xb = rng.choice(x, size=n1, replace=True)
        yb = rng.choice(y, size=n2, replace=True)
        mx, my = np.median(xb), np.median(yb)
        diffs.append(mx - my)
        ratios.append(mx / my if my != 0 else np.nan)
    diff_hat = np.median(x) - np.median(y)
    ratio_hat = np.median(x) / np.median(y) if np.median(y) != 0 else np.nan
    dlo, dhi = np.nanpercentile(diffs, [2.5, 97.5])
    rlo, rhi = np.nanpercentile(ratios, [2.5, 97.5])
    return ((float(diff_hat), float(dlo), float(dhi)),
            (float(ratio_hat), float(rlo), float(rhi)))

def rank_biserial_from_mw(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = x.size, y.size
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0*U/(n1*n2))

# ===== Main =====
# Carregar info de todos os modelos selecionados
packs = []
for key in SELECT_MODELS:
    if key not in MODEL_CATALOG:
        print(f"[aviso] Modelo '{key}' não está no catálogo; ignorando.")
        packs.append({"key": key, "label": key, "dir_obs": None, "por_run": {}})
        continue

    label, subs = MODEL_CATALOG[key]
    dir_obs = None
    for sub in subs:
        tentativa = find_obs_dir(base_root, sub, n_obs)
        if tentativa is not None:
            dir_obs = tentativa
            break

    if dir_obs is None:
        packs.append({"key": key, "label": label, "dir_obs": None, "por_run": {}})
    else:
        por_run = listar_por_run(dir_obs)
        packs.append({"key": key, "label": label, "dir_obs": dir_obs, "por_run": por_run})

print("\n[debug] Runs encontrados:")
for p in packs:
    print(f"  {p['label']}: dir={p['dir_obs']}")
    if p["por_run"]:
        rkeys = sorted(p["por_run"].keys())
        print(f"    runs: {rkeys[:40]}{' ...' if len(p['por_run'])>40 else ''}")
    else:
        print("    (nenhum arquivo encontrado)")

# ---- KIT A (por modelo) + Diagnósticos numéricos pós-R² ----
tem_algum = False
diagnosticos_rows = []

for p in packs:
    if not p["por_run"]:
        continue

    t_med, lnC_med, lnC_std = media_std_por_passo_lnC(p["por_run"], limitar_runs=None)  # ou (0,99)
    if t_med.size == 0:
        print(f"[aviso] Sem pontos válidos para {p['label']}.")
        continue

    # 1) Parte visual/exploratória (plots)
    kitA_diagnosticos(t_med, lnC_med, lnC_std, p["label"], SALVAR, PREFIXO_SAIDA)

    # 2) Parte numérica pós-R² (sem plots)
    if RELATAR_NUMERICO:
        diag = diagnosticos_numericos_pos_R2(t_med, lnC_med)
        print(f"\n[Diagnósticos numéricos — {p['label']}]")
        print(f"R^2={diag['R2']:.4f} | DW={diag['DW']:.3f} | BP p={diag['BP_p']:.3g} | SW p={diag['SW_p']:.3g}")
        diagnosticos_rows.append({
            "modelo": p["label"],
            "R2": diag["R2"],
            "DW": diag["DW"],
            "BP_p": diag["BP_p"],
            "SW_p": diag["SW_p"],
            "phi": round(n_obs/(128**2), 6),
        })

    tem_algum = True

if not tem_algum:
    raise FileNotFoundError("Nenhum modelo com dados válidos para o Kit A.")

# (Opcional) salvar CSV dos diagnósticos numéricos
if RELATAR_NUMERICO and SALVAR and len(diagnosticos_rows) > 0:
    df_diag = pd.DataFrame(diagnosticos_rows)
    df_diag.to_csv(CSV_DIAGNOSTICOS, index=False)
    print(f"[salvo] {CSV_DIAGNOSTICOS}")

# ---- ROTA 2: τ por run + comparação ----
print("\n[ROTA 2] Estimando τ por run e comparando modelos...")

for p in packs:
    if not p["por_run"]:
        p["taus"] = np.array([])
        p["runs_ok"] = []
        continue
    taus, runs_ok = tau_por_run(p["por_run"], MIN_PONTOS_RUN)
    p["taus"], p["runs_ok"] = taus, runs_ok
    print(f"[info] {p['label']}: {len(runs_ok)} runs válidos (τ_i)")

validos = [p for p in packs if p.get("taus", np.array([])).size > 0]
if len(validos) < 2:
    raise RuntimeError("Preciso de τ em pelo menos dois modelos com dados válidos (Rota 2).")

# --- Resumos por modelo ---
for p in validos:
    x = p["taus"]
    med, (lo, hi) = bootstrap_ci_mediana(x, B=B)
    print(f"\n=== Resumo {p['label']} ===")
    print(f"n_runs = {x.size}")
    print(f"Mediana τ = {med:.4g}  [IC95% (bootstrap): {lo:.4g}, {hi:.4g}]")
    if x.size > 1:
        print(f"Média = {np.mean(x):.4g}   DP = {np.std(x, ddof=1):.4g}")

# --- Comparações par-a-par entre TODOS os modelos válidos ---
from itertools import combinations

print("\n=== Comparações entre modelos (Rota 2) ===")
for p1, p2 in combinations(validos, 2):
    x, y = p1["taus"], p2["taus"]
    U, p_mw = mannwhitneyu(x, y, alternative="two-sided")
    r_rb = rank_biserial_from_mw(x, y)
    (diff_hat, dlo, dhi), (ratio_hat, rlo, rhi) = bootstrap_ci_diff_ratio_medianas(x, y, B=B)
    print(f"\n[{p1['label']}] vs [{p2['label']}]")
    print(f"Mann–Whitney U: U = {U:.3f},  p = {p_mw:.3e}  (two-sided)")
    print(f"Tamanho de efeito (rank-biserial): r_rb = {r_rb:.3f}  (~0.1 pequeno, 0.3 médio, 0.5 grande)")
    print(f"Δ mediana (1 - 2) = {diff_hat:.4g}  [IC95%: {dlo:.4g}, {dhi:.4g}]")
    print(f"Razão medianas (1 / 2) = {ratio_hat:.4g}  [IC95%: {rlo:.4g}, {rhi:.4g}]")

# ---- Gráficos Rota 2 (hist, boxplot, forest) ----
for p in validos:
    taus = p["taus"]
    if taus.size == 0:
        continue
    plt.figure(figsize=(8,5))
    plt.hist(taus, bins="auto", alpha=0.85)
    plt.xlabel(r"$\tau_i$ por run")
    plt.ylabel("Frequência")
    plt.title(f"Distribuição de $\\tau$ por run — {p['label']}")
    plt.tight_layout()
    if SALVAR:
        fname = f"{PREFIXO_SAIDA}_hist_{sanitize_label_for_file(p['label'])}.pdf"
        plt.savefig(fname, format="pdf"); print(f"[salvo] {fname}")
    plt.show()

labels = [p["label"] for p in validos]
data = [p["taus"] for p in validos]
plt.figure(figsize=(8,5))
plt.boxplot(data, labels=labels, showfliers=False)
plt.ylabel(r"$\tau$ por run")
plt.title(r"$\tau$ por run — comparação entre modelos")
plt.tight_layout()
if SALVAR:
    fname = f"{PREFIXO_SAIDA}_boxplots.pdf"
    plt.savefig(fname, format="pdf"); print(f"[salvo] {fname}")
plt.show()

xs, y_med, y_lo, y_hi = [], [], [], []
for i, p in enumerate(validos, start=1):
    med, (lo, hi) = bootstrap_ci_mediana(p["taus"], B=B)
    xs.append(i); y_med.append(med); y_lo.append(lo); y_hi.append(hi)

plt.figure(figsize=(7,5))
plt.errorbar(xs, y_med,
             yerr=[np.array(y_med)-np.array(y_lo), np.array(y_hi)-np.array(y_med)],
             fmt='o', capsize=4)
plt.xticks(xs, labels, rotation=0)
plt.ylabel("Mediana de $\\tau$ (IC95% bootstrap)")
plt.title("Medianas de $\\tau$ por modelo")
plt.tight_layout()
if SALVAR:
    fname = f"{PREFIXO_SAIDA}_forest_medianas.pdf"
    plt.savefig(fname, format="pdf"); print(f"[salvo] {fname}")
plt.show()
