import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
from scipy.optimize import curve_fit

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

# -------- Funções de ajuste --------
def exp_decay(x, lamb):
    """Exponencial simples normalizada: P(x) ~ lambda * exp(-lambda x)."""
    return lamb * np.exp(-lamb * x)

def stretched_exp(x, lamb, beta):
    """Exponencial esticada (KWW): ~ lambda * beta * (lambda*x)^(beta-1) * exp(-(lambda*x)^beta)."""
    return lamb * beta * (lamb * x) ** (beta - 1) * np.exp(-(lamb * x) ** beta)

def power_law(x, a, b):
    """Lei de potência: f(x) = a * x^(-b)."""
    return a * np.power(x, -b)

def sigmoid(x, L, k, x0):
    """Função sigmoide logística."""
    return L / (1 + np.exp(-k * (x - x0)))

# ---------------------- Primeiro passo: achar y_max global ----------------------
all_ftp = []
for hunter_ratio, base_name in bases:
    for obs in obs_list:
        files = list((BASE_ROOT / base_name / obs).glob("*_captures.csv"))
        for f in files:
            tmp = pd.read_csv(f)
            if not tmp.empty:
                ftp = tmp.groupby("hunter_id")["timestep"].min()
                all_ftp.extend(ftp.values)

# Calcula y_max global (densidade máxima em todos os histogramas)
hist_vals, hist_bins = np.histogram(all_ftp, bins=30, density=True)
ymax_global = hist_vals.max()

# ---------------------- Loop principal ----------------------
for hunter_ratio, base_name in bases:
    for obs in obs_list:
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

            ftp = tmp.groupby("hunter_id")["timestep"].min().reset_index()
            ftp["run"] = run
            resultados.append(ftp)

        if not resultados:
            print(f"[Aviso] Nenhum dado válido em {base_name}/{obs}")
            continue

        df = pd.concat(resultados, ignore_index=True)
        df.rename(columns={"timestep": "ftp"}, inplace=True)

        base_tag = "Nc=Np" if hunter_ratio == "1.0" else f"Nc=Np*{hunter_ratio}"
        obs_tag = get_obs_tag(obs)
        out_file = OUT_DIR / f"{base_tag}_{obs_tag}_FTP_x_t.pdf"

        # ---------- Histograma normalizado ----------
        plt.figure(figsize=(8, 5))
        counts, bins, _ = plt.hist(
            df["ftp"], bins=50, alpha=0.7, color="steelblue",
            edgecolor="black", density=True, label="Dados"
        )

        # ---------- Ajuste Exponencial ----------
        xdata = (bins[:-1] + bins[1:]) / 2
        ydata = counts

        try:
            popt, pcov = curve_fit(exp_decay, xdata, ydata, p0=[0.01])
            perr = np.sqrt(np.diag(pcov))
            plt.plot(xdata, exp_decay(xdata, *popt), "r-", lw=2,
                     label=rf"Exp: $\lambda$={popt[0]:.3f}$\pm${perr[0]:.3f}")
        except Exception as e:
            print(f"[Aviso] Fit exponencial falhou em {base_name}/{obs}: {e}")

        # ---------- Ajuste Exponencial Esticada (KWW) ----------
        mask = xdata > 0
        xdata_fit = xdata[mask]
        ydata_fit = ydata[mask]

        try:
            popt2, pcov2 = curve_fit(stretched_exp, xdata_fit, ydata_fit, p0=[0.01, 0.8])
            perr2 = np.sqrt(np.diag(pcov2))
            plt.plot(xdata_fit, stretched_exp(xdata_fit, *popt2), "g--", lw=2,
                     label=rf"KWW: $\lambda$={popt2[0]:.3f}$\pm${perr2[0]:.3f}, "
                           rf"$\beta$={popt2[1]:.2f}$\pm${perr2[1]:.2f}")
        except Exception as e:
            print(f"[Aviso] Fit KWW falhou em {base_name}/{obs}: {e}")

        # ---------- Ajuste Potencial ----------
        try:
            popt3, pcov3 = curve_fit(power_law, xdata_fit, ydata_fit, p0=[1.0, 1.0])
            perr3 = np.sqrt(np.diag(pcov3))
            plt.plot(xdata_fit, power_law(xdata_fit, *popt3), "m-.", lw=2,
                     label=rf"Power-law: a={popt3[0]:.3f}$\pm${perr3[0]:.3f}, "
                           rf"b={popt3[1]:.2f}$\pm${perr3[1]:.2f}")
        except Exception as e:
            print(f"[Aviso] Fit power-law falhou em {base_name}/{obs}: {e}")

        plt.title(f"Distribuição FTP - {base_tag}, {obs_tag}")
        plt.xlabel("FTP (timestep)")
        plt.ylabel("Densidade de probabilidade")
        #plt.ylim(0, 1.1)  # mesmo eixo Y para todos
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()

        print(f"[OK] Plot salvo em: {out_file}")
