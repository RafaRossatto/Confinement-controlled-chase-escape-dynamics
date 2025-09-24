import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re

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

L = 128
area_total = L * L

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
    for base_tag in bases:
        fp = idx.get(base_tag, None)
        if fp is None:
            continue
        df = pd.read_csv(fp)
        sns.kdeplot(x=df["ftp"], bw_adjust=1.0, fill=False, lw=2, label=base_tag)
        found_any = True

    if found_any:
        plt.title(fr"$\phi$={phi}")
        plt.xlabel("timestep")
        plt.ylabel("FTP - Density")
        #plt.xlim(left=-0.5)
        plt.legend()
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
        plt.xlabel("timestep")
        plt.ylabel("Density")
        #plt.xlim(left=-0.5)
        plt.tight_layout()
        out_file = OUT_DIR / f"{obs_tag}_{base_tag}_FTP_kde.pdf"
        plt.savefig(out_file)
        plt.close()
        print(f"[OK] KDE individual salvo em: {out_file}")
