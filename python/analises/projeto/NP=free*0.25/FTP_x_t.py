import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ---------------------- parâmetros ----------------------
obs_list = [
    "s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", "s_obs_6553",
    "s_obs_8192", "s_obs_9830", "s_obs_11468", "s_obs_13107"
]

bases = [
    ("0.5", "Nc=Np*0.5"),
    ("0.8", "Nc=Np*0.8"),
    ("1.0", "Nc=Np"),  # 1.0 não escreve "*1.0"
]

BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
BASE_RES = BASE_ROOT / "resultados_modelos"
OUT_DIR = BASE_RES / "FTP_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_obs_tag(obs: str) -> str:
    """Extrai o número do obstáculo de strings tipo 's_obs_1638' -> 'ob1638'."""
    m = re.search(r'(\d+)', obs)
    num = m.group(1) if m else obs
    return f"ob{num}"

# ---------------------- Loop principal ----------------------
for hunter_ratio, base_name in bases:
    for obs in obs_list:
        # Lista todos os arquivos de captura dessa pasta
        files = list((BASE_ROOT / base_name / obs).glob("*_captures.csv"))

        if not files:
            print(f"[Aviso] Nenhum arquivo encontrado em {base_name}/{obs}")
            continue

        resultados = []
        for f in files:
            try:
                run = int(f.stem.split("_run_")[1].split("_")[0])
            except Exception:
                run = -1

            tmp = pd.read_csv(f)
            if tmp.empty:
                continue

            # FTP = primeiro timestep de captura por hunter
            ftp = tmp.groupby("hunter_id")["timestep"].min().reset_index()
            ftp["run"] = run
            resultados.append(ftp)

        if not resultados:
            print(f"[Aviso] Nenhum dado válido em {base_name}/{obs}")
            continue

        df = pd.concat(resultados, ignore_index=True)
        df.rename(columns={"timestep": "ftp"}, inplace=True)

        # Nome base: Nc=Np ou Nc=Np*X
        base_tag = "Nc=Np" if hunter_ratio == "1.0" else f"Nc=Np*{hunter_ratio}"

        # Nome do obstáculo
        obs_tag = get_obs_tag(obs)

        # Nome final do arquivo
        out_file = OUT_DIR / f"{base_tag}_{obs_tag}_FTP_x_t.pdf"

        # ---------- Histograma normalizado (FTP global) ----------
        plt.figure(figsize=(8, 5))
        plt.hist(df["ftp"], bins=30, alpha=0.7, color="steelblue",
                 edgecolor="black", density=True)
        plt.title(f"Distribuição FTP - {base_tag}, {obs_tag}")
        plt.xlabel("FTP (timestep)")
        plt.ylabel("Densidade de probabilidade")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()

        print(f"[OK] Plot salvo em: {out_file}")
