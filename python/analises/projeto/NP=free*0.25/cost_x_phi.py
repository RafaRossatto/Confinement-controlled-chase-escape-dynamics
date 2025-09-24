import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib import colormaps

# -------------------------------
# Função auxiliar para extrair Nc e Ne
# -------------------------------
def parse_nc_ne_from_name(name: str):
    match_nc = re.search(r"NC_(\d+)", name)
    match_ne = re.search(r"NE_(\d+)", name)
    nc = int(match_nc.group(1)) if match_nc else None
    ne = int(match_ne.group(1)) if match_ne else None
    return nc, ne

# -------------------------------
# Calcular custo
# -------------------------------
def calcular_custos(base_root: Path, ne: int = 2048):
    df = pd.read_csv(base_root / "steps_boxplot_runs_ALL__ALL.csv")
    print("[DEBUG] Colunas:", df.columns)
    print("[DEBUG] Primeiras linhas:")
    print(df.head())

    resultados = []

    for (cenario, phi), grupo in df.groupby(["cenario_tag", "phi"]):
        steps = grupo["steps_run"].values
        if len(steps) == 0:
            continue

        # Determinar Nc/Ne a partir do cenario_tag
        
        if cenario == "Nc=Np*0.5":
            nc = int(0.5 * ne)
        elif cenario == "Nc=Np*0.8":
            nc = int(0.8 * ne)
        elif cenario == "Nc=Np":
            nc = int(ne)
        else:
            print(f"[Aviso] Cenário desconhecido: {cenario}")
            continue

        # Mediana e MAD
        tt_median = np.median(steps)
        mad = np.median(np.abs(steps - tt_median))

        # Cálculo do custo
        custo = (nc / ne) * tt_median

        resultados.append({
            "cenario": cenario,
            "phi": phi,
            "Nc": nc,
            "Ne": ne,
            "TT_median": tt_median,
            "MAD": mad,
            "custo": custo
        })

    if not resultados:
        print("[Aviso] Nenhum resultado encontrado!")
        return pd.DataFrame()

    df_res = pd.DataFrame(resultados).sort_values(["cenario", "phi"])
    df_res.to_csv(base_root / "custo_median.csv", index=False)
    print(f"[OK] Resultados salvos em {base_root/'custo_median.csv'}")
    return df_res



# -------------------------------
# Plotar custo vs φ
# -------------------------------
def plot_custo(df, out_dir: Path):
    plt.figure(figsize=(10,6))

    # Colormap flag
    cmap = colormaps.get_cmap("brg")
    
    # Ordem desejada dos cenários
    ordem = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]

    for idx, cenario in enumerate(ordem):
        grupo = df[df["cenario"] == cenario]
        if grupo.empty:
            continue

        plt.errorbar(
            grupo["phi"], grupo["custo"],
            yerr=grupo["MAD"],
            fmt="o-", capsize=4,
            color=cmap(idx / len(ordem)),
            label=cenario
        )

    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$c = \frac{N^C_0}{N^E_0} \tilde{TT}$")
    plt.title("Cost vs. $\phi$")
    plt.axvline(x=0.6, color="black", linestyle="--", linewidth=1.5,label=r"$\phi = 0.60$")
    plt.grid(alpha=0.3)
    plt.legend(title="Cenários", loc="best")
    plt.tight_layout()

    out_file = out_dir / "custo_vs_phi.pdf"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"[OK] Figura salva: {out_file}")

# -------------------------------
# Execução principal
# -------------------------------
def main():
    base_root = Path.home() / "Dados_Doc" / "Np=free*0.25" / "resultados_modelos" / "zeros_escapers"
    
    df_res = calcular_custos(base_root)
    plot_custo(df_res, base_root)

if __name__ == "__main__":
    main()
