import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib import colormaps
# ---------------------- parâmetros ----------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
IN_DIR = BASE_ROOT / "resultados_modelos" / "FTP_data"
OUT_DIR = BASE_ROOT / "resultados_modelos" / "FTP_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

bases = [
    "Nc=Np*0.5",
    "Nc=Np*0.8",
    "Nc=Np"
]

cmap = colormaps.get_cmap("flag")
L = 128
area_total = L * L

    # --- Configuração global de fonte nos eixos e legenda ---
plt.rcParams.update({
"xtick.labelsize": 16,
"ytick.labelsize": 16,
"legend.fontsize": 14})
# ---------------------- Agrupar por obstáculo ----------------------
csv_files = list(IN_DIR.glob("*.csv"))
obs_groups = {}
pat = re.compile(r"_(ob\d+)_FTP_data$")

for f in csv_files:
    m = pat.search(f.stem)
    if not m:
        print(f"[IGNORADO] Nome inesperado: {f.name}")
        continue
    obs_tag = m.group(1)
    obs_groups.setdefault(obs_tag, []).append(f)

from matplotlib import colormaps

# ---------------------- Loop por obstáculo ----------------------
for obs_tag, files in obs_groups.items():
    # calcula fração de obstáculos (phi)
    num_obs = int(obs_tag.replace("ob", ""))
    phi = round(num_obs / area_total, 2)

    # Cria índice base_tag -> Path
    idx = {}
    for f in files:
        base_part = f.stem[:f.stem.index(f"_{obs_tag}_FTP_data")]
        idx[base_part] = f

    # ---- 1) Comparativo ----
    plt.figure(figsize=(8, 5))
    found_any = False
    cmap = colormaps.get_cmap("flag")  # <<<<<< paleta de cores

    for i, base_tag in enumerate(bases):
        fp = idx.get(base_tag, None)
        if fp is None:
            continue
        df = pd.read_csv(fp)
        color = cmap(i)  # pega uma cor da paleta
        sns.kdeplot(
            x=df["ftp"], bw_adjust=1.0,
            fill=False, lw=2,
            label=base_tag, color=color
        )
        found_any = True

    if found_any:
        plt.title(fr"$\phi$={phi}",fontsize=20)
        plt.xlabel("timestep",fontsize=22)
        plt.ylabel("Density",fontsize=22)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_file = OUT_DIR / f"{obs_tag}_FTP_kde_comparativo.pdf"
        plt.savefig(out_file)
        plt.close()
        print(f"[OK] KDE comparativo salvo em: {out_file}")

    # ---- 2) Individuais ----
    for base_tag in bases:
        fp = idx.get(base_tag, None)
        if fp is None:
            continue
        df = pd.read_csv(fp)
        plt.figure(figsize=(8, 5))
        sns.kdeplot(x=df["ftp"], bw_adjust=1.0, fill=True, lw=2)
        plt.title(fr"FTP - {obs_tag}, {base_tag} ($\phi$={phi})")
        plt.xlabel("timestep",fontsize=22)
        plt.ylabel("Density",fontsize=22)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_file = OUT_DIR / f"{obs_tag}_{base_tag}_FTP_kde.pdf"
        plt.savefig(out_file)
        plt.close()
        print(f"[OK] KDE individual salvo em: {out_file}")

