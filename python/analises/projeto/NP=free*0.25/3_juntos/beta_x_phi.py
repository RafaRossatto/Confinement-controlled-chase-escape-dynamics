from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

# ---------------------- CONFIGURAÇÕES PRINCIPAIS ----------------------
# Diferentes tamanhos de rede - ⬅️ ALTERE OS CAMINHOS AQUI
REDE_CONFIGS = [
    (64,  "L_64",  Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE AQUI
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"),   # ⬅️ ALTERE AQUI  
    (256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")    # ⬅️ ALTERE AQUI
]

# Apenas a proporção 0.5
FRAC_C = "Nc=Np*0.5"
FRAC_C_LABEL = r"$N^{C}_{0}=0.5\,N^{E}_{0}$"

# Diretórios
BASE_ROOT = Path.home() / "Dados_Doc" / "resultados_modelos"
HAZARD_DIR = BASE_ROOT / "hazard_multiple_L"  # Onde estão os CSVs do código anterior
OUT_DIR = BASE_ROOT / "beta_x_phi_multiple_L"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- CONFIGURAÇÃO DE CORES ----------------------
# Usando a mesma colormap "flag" do primeiro código para consistência
cmap = colormaps.get_cmap("flag")

# ---------------------- PLOT: β vs φ ----------------------
plt.figure(figsize=(10, 7))  # Tamanho padronizado
# plt.figure(figsize=(12, 8), dpi=150)  # Alternativa com maior resolução

for idx, (L, rede_nome, base_path) in enumerate(REDE_CONFIGS):
    # Caminho para o CSV de resultados desta rede
    fpath = HAZARD_DIR / rede_nome / f"{rede_nome}_{FRAC_C}_fit_results.csv"
    
    if not fpath.exists():
        print(f"[x] Arquivo não encontrado: {fpath}")
        continue

    df = pd.read_csv(fpath).copy()
    if not {"obs", "beta", "beta_err"}.issubset(df.columns):
        print(f"[x] Colunas esperadas não encontradas em {fpath}")
        continue

    # Calcular área e fração φ
    area = L**2
    df["n_obs"] = df["obs"].str.replace("s_obs_", "", regex=False).astype(int)
    df["phi"] = df["n_obs"] / area

    df_sorted = df.sort_values("phi")

    # Plota linha com pontos + barras de erro
    cor = cmap(idx)  # Cores distribuídas uniformemente
    plt.errorbar(
        df_sorted["phi"], 
        df_sorted["beta"],
        yerr=df_sorted["beta_err"],
        fmt="o-", 
        capsize=4, 
        markersize=6,
        linewidth=2,  # Adicionado para melhor visualização
        color=cor,
        label=f"L = {L}"  # Label simplificado e padronizado
    )

# ---------------------- DECORAÇÃO ----------------------
plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5,
            label=r"$\phi_c = 0.60$")

plt.xlabel(r"$\phi$", fontsize=18)
plt.ylabel(r"$\langle \beta \rangle$", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)  # Reduzido ligeiramente para melhor ajuste
plt.tight_layout()

# Salvar
out_file = OUT_DIR / "beta_vs_phi_multiple_L.pdf"
plt.savefig(out_file, bbox_inches="tight")
plt.show()

print(f"[OK] Gráfico β vs φ salvo em {out_file}")

# ---------------------- GRÁFICO ADICIONAL: τ vs φ ----------------------
plt.figure(figsize=(10, 7))  # Mesmo tamanho para consistência
# plt.figure(figsize=(12, 8), dpi=150)  # Alternativa com maior resolução

for idx, (L, rede_nome, base_path) in enumerate(REDE_CONFIGS):
    fpath = HAZARD_DIR / rede_nome / f"{rede_nome}_{FRAC_C}_fit_results.csv"
    
    if not fpath.exists():
        continue

    df = pd.read_csv(fpath).copy()
    if not {"obs", "tau", "tau_err"}.issubset(df.columns):
        continue

    area = L**2
    df["n_obs"] = df["obs"].str.replace("s_obs_", "", regex=False).astype(int)
    df["phi"] = df["n_obs"] / area

    df_sorted = df.sort_values("phi")

    cor = cmap(idx)  # Mesma cor para mesma rede
    plt.errorbar(
        df_sorted["phi"], 
        df_sorted["tau"],
        yerr=df_sorted["tau_err"],
        fmt="s-",  # Quadrados para diferenciar do gráfico anterior
        capsize=4, 
        markersize=6,
        linewidth=2,  # Adicionado para melhor visualização
        color=cor,
        label=f"L = {L}"  # Label consistente
    )

plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5,
            label=r"$\phi_c = 0.60$")

plt.xlabel(r"$\phi$", fontsize=18)
plt.ylabel(r"$\langle \tau \rangle$", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

out_file_tau = OUT_DIR / "tau_vs_phi_multiple_L.pdf"
plt.savefig(out_file_tau, bbox_inches="tight")
plt.show()

print(f"[OK] Gráfico τ vs φ salvo em {out_file_tau}")