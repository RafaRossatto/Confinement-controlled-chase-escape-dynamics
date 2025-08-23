import pandas as pd
from pathlib import Path
from scipy.stats import shapiro
import statsmodels.api as sm
import numpy as np

# === Configurações ===
base = Path.home() / "Dados_Doc" / "resultados_modelos" / "curvas_por_run"
caso = "Nc_eq_Np"   # escolha: "Nc_eq_Np" ou "Nc_lt_Np"
n_obs_alvo = 14745  # número de obstáculos que você quer analisar

# Lista de arquivos candidatos
todos = sorted(base.glob("*_ajuste.csv"))

def extrair_nc_ne(nome: str):
    partes = nome.split("_")
    nc = int(partes[1])
    ne = int(partes[3])
    return nc, ne

# Filtrar arquivos
arquivos = []
for arq in todos:
    if f"_O_{n_obs_alvo}_" not in arq.name:
        continue  # ignora se não bate o número de obstáculos

    nc, ne = extrair_nc_ne(arq.name)
    if caso == "Nc_eq_Np" and nc == ne:
        arquivos.append(arq)
    elif caso == "Nc_lt_Np" and nc < ne:
        arquivos.append(arq)

print(f"Total de arquivos selecionados: {len(arquivos)}")
for a in arquivos[:5]:
    print(" ->", a.name)

# === Modelos a avaliar ===
modelos = {
    "Exp": "y_exp",
    "Exp^β": "y_expbeta",
    "PowerShift": "y_powershift",
}

rows = []

for arq in arquivos:
    df = pd.read_csv(arq)
    y_true = df["dado"].to_numpy()

    for nome, col in modelos.items():
        if col not in df:
            continue
        y_pred = df[col].to_numpy()

        # --- Resíduos ---
        residuos = y_true - y_pred

        # R²
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

        # AIC e BIC
        n = len(y_true)
        k = 2 if nome == "Exp" else 3  # nº de parâmetros (ajustar se precisar)
        sigma2 = ss_res/n
        loglik = -0.5 * n * (np.log(2*np.pi*sigma2) + 1) if sigma2 > 0 else np.nan
        aic = -2*loglik + 2*k if not np.isnan(loglik) else np.nan
        bic = -2*loglik + k*np.log(n) if not np.isnan(loglik) else np.nan

        # Teste Shapiro
        try:
            stat, p_shapiro = shapiro(residuos)
        except Exception:
            p_shapiro = float("nan")

        # Durbin-Watson
        try:
            dw = sm.stats.stattools.durbin_watson(residuos)
        except Exception:
            dw = float("nan")

        rows.append({
            "arquivo": arq.name,
            "modelo": nome,
            "R2": r2,
            "AIC": aic,
            "BIC": bic,
            "shapiro_p": p_shapiro,
            "durbin_watson": dw,
        })

# Monta dataframe final
df_comp = pd.DataFrame(rows)

print("\n=== Resumo comparativo ===")
print(df_comp.groupby("modelo")[["R2","AIC","BIC","shapiro_p","durbin_watson"]].median())

print("\nNúmero de runs avaliados por modelo:")
print(df_comp.groupby("modelo")["arquivo"].count())
