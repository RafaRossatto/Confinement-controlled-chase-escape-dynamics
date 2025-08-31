#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
from matplotlib.ticker import PercentFormatter

# ---------------------- Parâmetros principais ----------------------
L = 128
AREA = L**2
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"

bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}_{0}=N^{E}_{0}$",      "Nc=Np"),
]

# saída
OUT_DIR = BASE_ROOT / "resultados_modelos" / "zeros_escapers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# plot
PHI_LINE = 0.59
CMAP_NAME = "flag"

# normalização do eixo-Y (fração de zeros)
# "por_cenario" -> cada curva/ cenário é dividida pelo seu máximo
# "global"      -> divide por max entre todos os pontos de todos os cenários
# "nenhuma"     -> plota a fração original em [0,1]
Y_NORMALIZACAO = "por_cenario"

# ---------------------- Utilitários ----------------------
def ler_dat(fp: Path) -> pd.DataFrame:
    """ Lê .dat tolerante a delimitadores e cabeçalhos. """
    try:
        df = pd.read_csv(fp, sep=None, engine="python", comment="#")
        ok = {"run", "steps", "escapers", "seed"}.issubset(df.columns)
        if not ok:
            raise ValueError("Cabeçalho inesperado")
    except Exception:
        df = pd.read_csv(
            fp, delim_whitespace=True, header=None,
            names=["run","steps","escapers","seed"], comment="#"
        )
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df["escapers"] = pd.to_numeric(df["escapers"], errors="coerce")
    return df.dropna(subset=["steps","escapers"])

def extrai(padrao: str, texto: str, default=None, cast=int):
    m = re.search(padrao, texto)
    return cast(m.group(1)) if m else default

def coletar(base_root: Path, label_tex: str) -> pd.DataFrame:
    """ Percorre s_obs_*; calcula fração de linhas com escapers==0. """
    linhas = []
    if not base_root.exists():
        print(f"[WARN] Raiz não encontrada: {base_root}")
        return pd.DataFrame()

    subdirs = [d for d in base_root.iterdir() if d.is_dir() and d.name.startswith("s_obs_")]
    if not subdirs:
        print(f"[WARN] Sem subpastas s_obs_* em {base_root}")
        return pd.DataFrame()

    for sdir in sorted(subdirs):
        # arquivo preferido; fallback p/ qualquer .dat
        cand = list(sdir.glob("NC_*_NE_*_O_*_TCC_*_SR_*_TCT_*_SR_*.dat"))
        if not cand:
            cand = list(sdir.glob("*.dat"))
        if not cand:
            print(f"[WARN] Sem .dat em {sdir}")
            continue
        fp = sorted(cand)[0]

        try:
            df = ler_dat(fp)
        except Exception as e:
            print(f"[WARN] Falha lendo {fp.name}: {e}")
            continue

        n_runs = len(df)
        if n_runs == 0:
            continue

        zeros_count = int((df["escapers"] == 0).sum())
        zeros_frac = zeros_count / float(n_runs)

        # tenta O no nome do arquivo; senão, pega da pasta s_obs_XXXX
        O = extrai(r"_O_(\d+)_", fp.name, default=extrai(r"s_obs_(\d+)$", sdir.name, default=None))
        if O is None:
            print(f"[WARN] Não consegui obter O em {fp.name}; pulando…")
            continue

        linhas.append({
            "cenario_label": label_tex,
            "cenario_tag": base_root.name,  # ex.: Nc=Np*0.5
            "pasta": sdir.name,             # ex.: s_obs_1638
            "arquivo": str(fp),
            "O": int(O),
            "phi": O / float(AREA),
            "n_runs": n_runs,
            "zeros_count": zeros_count,
            "zeros_frac": zeros_frac,  # [0,1]
        })

    if not linhas:
        return pd.DataFrame()
    return pd.DataFrame(linhas).sort_values("phi").reset_index(drop=True)

# ---------------------- Coleta + CSV ----------------------
tabelas = []
for label_tex, base_dirname in bases:
    root_path = BASE_ROOT / base_dirname
    df_one = coletar(root_path, label_tex)
    if df_one.empty:
        continue
    tabelas.append(df_one)
    # CSV por cenário (ordenado)
    df_one.sort_values("phi").to_csv(OUT_DIR / f"zeros_escapers_{base_dirname}.csv", index=False)
    print(f"[OK] CSV por cenário: {OUT_DIR / f'zeros_escapers_{base_dirname}.csv'}")

if not tabelas:
    raise SystemExit("[!] Nada encontrado em nenhuma raiz.")

df = pd.concat(tabelas, ignore_index=True)
# CSV combinado
out_all = OUT_DIR / "zeros_escapers__ALL.csv"
df.sort_values(["cenario_tag","phi"]).to_csv(out_all, index=False)
print(f"[OK] CSV combinado: {out_all}")

# --- após montar o df (mesma parte onde você salva o CSV combinado) ---
# Desvio padrão binomial da fração de zeros
df["zeros_std"] = np.sqrt(df["zeros_frac"] * (1.0 - df["zeros_frac"]) / df["n_runs"])

# Normalização (se estiver usando Y_NORMALIZACAO como no meu exemplo)
if Y_NORMALIZACAO == "por_cenario":
    df["norm_scale"] = df.groupby("cenario_label")["zeros_frac"].transform(lambda s: s.max().clip(min=1e-12))
elif Y_NORMALIZACAO == "global":
    gmax = df["zeros_frac"].max()
    df["norm_scale"] = gmax if gmax > 0 else 1.0
else:
    df["norm_scale"] = 1.0

df["zeros_norm"] = df["zeros_frac"] / df["norm_scale"]
df["zeros_std_norm"] = df["zeros_std"] / df["norm_scale"]

# ---------------------- Plot ----------------------
plt.figure(figsize=(10, 6), dpi=150)
ax = plt.gca()

from matplotlib import colormaps
cmap = colormaps.get_cmap("flag")

for base_idx, (label_tex, _dirname) in enumerate(bases):
    dfl = df[df["cenario_label"] == label_tex].sort_values("phi")
    if dfl.empty:
        continue
    ax.errorbar(
        dfl["phi"], dfl["zeros_norm"], 
        yerr=dfl["zeros_std_norm"],
        fmt="o-", capsize=5, linewidth=1.6, markersize=5,
        label=label_tex,
        color=cmap(base_idx)   # <<< usa índice fixo, igual ao modelo <d>
    )

# linha de referência em φ = 0.59
ax.axvline(0.59, color="black", linestyle="--", linewidth=1.5,
           label=r"$\phi_c \approx 0.59$")

ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$N^{E}_{t \to \infty}/N^{E}_{0}$")
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=False)
ax.set_ylim(-0.05,1.05)
plt.tight_layout()
plt.savefig("extinsion_vs_phi.pdf", dpi=200, bbox_inches="tight")
plt.show()