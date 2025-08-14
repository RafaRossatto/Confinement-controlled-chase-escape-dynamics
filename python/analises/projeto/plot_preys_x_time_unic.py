import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ===== Parâmetros =====
base_root = Path.home() / "Dados_Doc"
subpasta = "Nc=Np"     # escolha o cenário
n_obs = 11468                 # se for 0, vira s_obs_00
run_desejado = 30          # run que você quer ver

# ===== Funções auxiliares =====
def get_obs_dir(base_root: Path, subpasta: str, n_obs: int) -> Path:
    """Retorna o caminho da pasta s_obs_* de acordo com n_obs."""
    nome_obs = f"s_obs_{n_obs:02d}" if n_obs == 0 else f"s_obs_{n_obs}"
    caminho = base_root / subpasta / nome_obs
    if not caminho.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {caminho}")
    return caminho

# Regex para identificar arquivos
RE_RUN = re.compile(r"_run_(\d+)_presas_por_passo\.csv$")

def encontrar_arquivo_run(dir_obs: Path, run_alvo: int) -> Path:
    for p in dir_obs.rglob("*.csv"):
        m = RE_RUN.search(p.name)
        if m and int(m.group(1)) == run_alvo:
            return p
    raise FileNotFoundError(f"Run {run_alvo} não encontrado em {dir_obs}")

# ===== Função exponencial =====
def expo(t, A, tau):
    return A * np.exp(-t / tau)

# ===== Execução =====
dir_obs = get_obs_dir(base_root, subpasta, n_obs)
arquivo = encontrar_arquivo_run(dir_obs, run_desejado)
print(f"[✓] Lendo {arquivo}")

df = pd.read_csv(arquivo)

# Remove pontos com presas_vivas = 0 para evitar log(0)
df_nonzero = df[df["presas_vivas"] > 0]

# ===== Ajuste exponencial direto =====
popt_exp, _ = curve_fit(expo, df_nonzero["passo"], df_nonzero["presas_vivas"], maxfev=5000)
y_pred_exp = expo(df_nonzero["passo"], *popt_exp)
r2_exp = r2_score(df_nonzero["presas_vivas"], y_pred_exp)

# ===== Ajuste linear no log =====
ln_C = np.log(df_nonzero["presas_vivas"])
coef = np.polyfit(df_nonzero["passo"], ln_C, 1)  # slope, intercept
y_pred_ln = np.polyval(coef, df_nonzero["passo"])
r2_ln = r2_score(ln_C, y_pred_ln)

# ===== Plot C(t) =====
plt.figure(figsize=(8,5))
plt.scatter(df["passo"], df["presas_vivas"], s=15, label="Dados")
plt.plot(df_nonzero["passo"], y_pred_exp, 'r-', label=f"Ajuste exp: A={popt_exp[0]:.2f}, tau={popt_exp[1]:.2f}, R²={r2_exp:.3f}")
plt.xlabel("Passos")
plt.ylabel("Presas vivas C(t)")
plt.title(f"C(t) | φ = {round(n_obs / (128**2), 3)}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== Plot ln C(t) =====
plt.figure(figsize=(8,5))
plt.scatter(df_nonzero["passo"], ln_C, s=15, label="ln(C) dados")
plt.plot(df_nonzero["passo"], y_pred_ln, 'g-', label=f"Ajuste linear: slope={coef[0]:.4f}, intercept={coef[1]:.2f}, R²={r2_ln:.3f}")
plt.xlabel("Passos")
plt.ylabel("ln(C(t))")
plt.title(f"ln(C(t)) | $\phi$ = {round(n_obs / (128**2), 3)}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
