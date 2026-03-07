import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 1. Ler o arquivo CSV (ajuste o nome do arquivo e delimitador)
dados = pd.read_csv('msd_hunters_all_runs.csv')

# 2. Assumindo que as colunas se chamam 'tau' e 'MSD'
tau = dados['tau'].values
msd = dados['msd_mean'].values

# 3. Aplicar a transformação logarítmica (usando log10 para facilitar a interpretação)
log_tau = np.log10(tau)
log_msd = np.log10(msd)

# 4. Realizar a regressão linear nos dados transformados
# A função linregress retorna: slope, intercept, r_value, p_value, std_err
resultado = linregress(log_tau, log_msd)

alpha = resultado.slope
intercept = resultado.intercept
r_quadrado = resultado.rvalue**2 # Coeficiente de determinação (qualidade do ajuste)

print(f"Expoente α (alfa) = {alpha:.4f}")
print(f"Coeficiente linear = {intercept:.4f}")
print(f"Coeficiente de determinação (R²) = {r_quadrado:.4f}")

# (Opcional) Calcular D a partir do coeficiente linear
# Lembre-se: MSD = C * τ^α --> log10(MSD) = log10(C) + α*log10(τ)
# Portanto, C = 10^(intercept)
C = 10**intercept
print(f"Constante C (aproximadamente D) = {C:.4f}")

# Gera os valores da reta de ajuste para plotar
y_ajustado = intercept + alpha * log_tau

# ... (código anterior para ler dados e calcular logs) ...

# Definir o ponto de transição (tau_critico) com base na sua inspeção visual
tau_critico = 20 # Altere este valor conforme necessário
log_tau_critico = np.log10(tau_critico)

# Encontrar o índice do array onde log_tau é aproximadamente igual ao log do crítico
# (Isso divide os dados nos dois regimes)
indice_critico = np.argmax(log_tau > log_tau_critico)

# Dados do Regime 1 (Curto Prazo - Balístico)
log_tau_regime1 = log_tau[:indice_critico]
log_msd_regime1 = log_msd[:indice_critico]

# Dados do Regime 2 (Longo Prazo - Difusivo/Superdifusivo)
log_tau_regime2 = log_tau[indice_critico:]
log_msd_regime2 = log_msd[indice_critico:]

# Ajustar o Regime 1
resultado_regime1 = linregress(log_tau_regime1, log_msd_regime1)
alpha_1 = resultado_regime1.slope
r2_1 = resultado_regime1.rvalue**2

# Ajustar o Regime 2
resultado_regime2 = linregress(log_tau_regime2, log_msd_regime2)
alpha_2 = resultado_regime2.slope
r2_2 = resultado_regime2.rvalue**2

print("--- Regime de Curto Prazo (Balístico) ---")
print(f"Expoente α₁ = {alpha_1:.4f}")
print(f"R² = {r2_1:.4f}")
print(f"Janela: tau < {tau_critico}")

print("\n--- Regime de Longo Prazo ---")
print(f"Expoente α₂ = {alpha_2:.4f}")
print(f"R² = {r2_2:.4f}")
print(f"Janela: tau > {tau_critico}")

# Plotar tudo para verificar
plt.figure(figsize=(10, 6))
plt.scatter(log_tau, log_msd, label='Dados (log-log)', color='grey', alpha=0.5)
plt.plot(log_tau_regime1, resultado_regime1.intercept + alpha_1 * log_tau_regime1, label=f'Ajuste Regime 1: α₁ = {alpha_1:.3f}', color='red', linewidth=2)
plt.plot(log_tau_regime2, resultado_regime2.intercept + alpha_2 * log_tau_regime2, label=f'Ajuste Regime 2: α₂ = {alpha_2:.3f}', color='green', linewidth=2)
plt.axvline(x=log_tau_critico, color='k', linestyle='--', label=f'τ = {tau_critico}')
plt.xlabel('log10(tau)')
plt.ylabel('log10(MSD)')
plt.title('Identificação de Dois Regimes Dinâmicos')
plt.legend()
plt.grid(True)
plt.show()


# ... depois de ler os dados ...

# Plotar dados BRUTOS em escala LINEAR
plt.figure(figsize=(12, 5))

# Gráfico 1: MSD vs tau (Linear)
plt.subplot(1, 2, 1)
plt.plot(tau, msd, 'b-')
plt.xlabel('tau')
plt.ylabel('MSD')
plt.title('MSD vs tau (Escala Linear)')
plt.grid(True)

# Gráfico 2: MSD vs tau (Log-Log)
plt.subplot(1, 2, 2)
plt.loglog(tau, msd, 'b-')
plt.xlabel('tau')
plt.ylabel('MSD')
plt.title('MSD vs tau (Escala Log-Log)')
plt.grid(True)

plt.tight_layout()
plt.show()