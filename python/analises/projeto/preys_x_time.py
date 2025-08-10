
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # type: ignore
import numpy as np

# Caminho do arquivo
arquivo = "/home/rafarossatto/Dados_Doc/s_obs_11468/NC_1500_NE_2458_O_11468_TCC_1.00_SR_2_TCT_1.00_SR_2.dat_run_0_presas_por_passo.csv"

# Ler o CSV
df = pd.read_csv(arquivo)

# Verificar as primeiras linhas para garantir que os nomes das colunas estão corretos
print(df.head())

# Plotar
plt.figure(figsize=(8, 5))
plt.plot(df["passo"], df["presas_vivas"], marker='o', markersize=2, linewidth=1)
plt.xlabel("Passos")
plt.ylabel("Presas vivas")
plt.title("Presas vivas por passo")
plt.grid(True)
plt.tight_layout()
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Caminho do arquivo
arquivo = "/home/rafarossatto/Dados_Doc/s_obs_11468/NC_1500_NE_2458_O_11468_TCC_1.00_SR_2_TCT_1.00_SR_2.dat_run_0_presas_por_passo.csv"

# Ler CSV
df = pd.read_csv(arquivo)

# Filtrar apenas valores positivos para evitar log de zero/negativo
mask = df["presas_vivas"] > 0
t = df["passo"][mask]
lnC = np.log(df["presas_vivas"][mask])

# Ajuste linear: ln(C) = ln(C0) - t/tau
slope, intercept, r_value, p_value, std_err = linregress(t, lnC)
tau = -1 / slope

print(f"ln(C0) = {intercept:.4f}")
print(f"C0 = {np.exp(intercept):.2f}")
print(f"tau = {tau:.4f}")
print(f"R² = {r_value**2:.4f}")

# Plot do ajuste
plt.figure(figsize=(8,5))
plt.scatter(t, lnC, s=10, label="Dados (ln C)")
plt.plot(t, intercept + slope*t, 'r-', label=f"Ajuste: tau={tau:.2f}, R²={r_value**2:.3f}")
plt.xlabel("Passos (t)")
plt.ylabel("ln(C)")
plt.legend()
plt.grid(True)
plt.show()
