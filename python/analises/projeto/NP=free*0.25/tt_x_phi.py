#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import PercentFormatter
from matplotlib import colormaps

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

# O que entra na média de steps?
# "todos"         -> usa todos os runs
# "apenas_extintos" -> usa somente runs com escapers==0
FILTRO_STEPS = "todos"

# Barras de erro no gráfico: "std" (desvio-padrão) ou "sem" (erro-padrão da média)
BARRA_ERRO = "std"

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
    # coerções
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df["escapers"] = pd.to_numeric(df["escapers"], errors="coerce")
    return df.dropna(subset=["steps","escapers"])

def extrai(padrao: str, texto: str, default=None, cast=int):
    m = re.search(padrao, texto)
    return cast(m.group(1)) if m else default

def coletar_steps(base_root: Path, label_tex: str) -> pd.DataFrame:
    """
    Percorre s_obs_*; calcula média e desvio-padrão de steps por φ.
    Opcionalmente filtra por escapers==0 (apenas_extintos).
    """
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

        # Filtragem de acordo com a política escolhida
        if FILTRO_STEPS == "apenas_extintos":
            df = df[df["escapers"] == 0]
        elif FILTRO_STEPS == "todos":
            pass
        else:
            raise ValueError("FILTRO_STEPS deve ser 'todos' ou 'apenas_extintos'.")

        n_runs = len(df)
        if n_runs == 0:
            continue

        steps_mean = float(df["steps"].mean())
        steps_std  = float(df["steps"].std(ddof=1)) if n_runs > 1 else 0.0
        steps_sem  = steps_std / np.sqrt(n_runs) if n_runs > 0 else 0.0

        # tenta O no nome do arquivo; senão, pega da pasta s_obs_XXXX
        O = extrai(r"_O_(\d+)_", fp.name, default=extrai(r"s_obs_(\d+)$", sdir.name, default=None))
        if O is None:
            print(f"[WARN] Não consegui obter O em {fp.name}; pulando…")
            continue

        # também calcula a fração de extinção (útil p/ inspeção)
        zeros_frac = float((df["escapers"] == 0).sum()) / n_runs

        linhas.append({
            "cenario_label": label_tex,
            "cenario_tag": base_root.name,  # ex.: Nc=Np*0.5
            "pasta": sdir.name,             # ex.: s_obs_1638
            "arquivo": str(fp),
            "O": int(O),
            "phi": O / float(AREA),
            "n_runs": n_runs,
            "steps_mean": steps_mean,
            "steps_std": steps_std,
            "steps_sem": steps_sem,
            "zeros_frac_amostra": zeros_frac,
        })

    if not linhas:
        return pd.DataFrame()
    return pd.DataFrame(linhas).sort_values("phi").reset_index(drop=True)

# ---------------------- Coleta + CSV ----------------------
tabelas = []
for label_tex, base_dirname in bases:
    root_path = BASE_ROOT / base_dirname
    df_one = coletar_steps(root_path, label_tex)
    if df_one.empty:
        continue
    tabelas.append(df_one)
    # CSV por cenário (ordenado)
    filtro_tag = "ALL" if FILTRO_STEPS == "todos" else "EXTINTOS"
    df_one.sort_values("phi").to_csv(
        OUT_DIR / f"steps_vs_phi_{filtro_tag}_{base_dirname}.csv",
        index=False
    )
    print(f"[OK] CSV por cenário: {OUT_DIR / f'steps_vs_phi_{filtro_tag}_{base_dirname}.csv'}")

if not tabelas:
    raise SystemExit("[!] Nada encontrado em nenhuma raiz.")

df = pd.concat(tabelas, ignore_index=True)
# CSV combinado
filtro_tag = "ALL" if FILTRO_STEPS == "todos" else "EXTINTOS"
out_all = OUT_DIR / f"steps_vs_phi_{filtro_tag}__ALL.csv"
df.sort_values(["cenario_tag","phi"]).to_csv(out_all, index=False)
print(f"[OK] CSV combinado: {out_all}")

# ---------------------- Três gráficos: (1) mean±std, (2) mean, (3) dispersão ----------------------
from matplotlib import colormaps
cmap = colormaps.get_cmap(CMAP_NAME)

# Se True, corta a barra inferior para não “entrar” em valores negativos (passos >= 0)
CLIP_NONNEG_ERR = True

# Para o gráfico (3): escolha "std" (desvio-padrão) ou "sem" (erro-padrão da média)
DISP_METRIC = "std"   # ou "sem"

# --- FIGURA 1: média ± desvio-padrão (barras) ---
plt.figure(figsize=(10, 6), dpi=150)
ax = plt.gca()
ax.axvline(PHI_LINE, color="black", linestyle="--", linewidth=1.5,
           label=rf"$\phi_c \approx {PHI_LINE}$")
for base_idx, (label_tex, _dirname) in enumerate(bases):
    dfl = df[df["cenario_label"] == label_tex].sort_values("phi")
    if dfl.empty:
        continue

    x = dfl["phi"].to_numpy()
    y = dfl["steps_mean"].to_numpy()
    base = dfl["steps_std"].to_numpy()  # barras = DESVIO-PADRÃO

    if CLIP_NONNEG_ERR:
        yerr_lower = np.minimum(y, base)   # garante mean - lower >= 0
        yerr_upper = base
        yerr = np.vstack([yerr_lower, yerr_upper])  # formato (2, N) = assimétrico
    else:
        yerr = base  # simétrico

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-", capsize=5, linewidth=1.6, markersize=5,
        label=label_tex, color=cmap(base_idx)
    )


ax.set_xlabel(r"$\phi$")
ax.set_ylabel("Passos médios até capturar as presas disponíveis\n(barras = desvio-padrão)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=False)
ax.set_ylim(bottom=0)
plt.tight_layout()
fig1_path = OUT_DIR / "steps_mean_std_vs_phi.pdf"
plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
print(f"[OK] Figura (média±std): {fig1_path}")
plt.show()

# --- FIGURA 2: somente a média dos steps vs phi ---
plt.figure(figsize=(10, 6), dpi=150)

ax = plt.gca()
ax.axvline(PHI_LINE, color="black", linestyle="--", linewidth=1.5,
           label=rf"$\phi_c \approx {PHI_LINE}$")
for base_idx, (label_tex, _dirname) in enumerate(bases):
    dfl = df[df["cenario_label"] == label_tex].sort_values("phi")
    if dfl.empty:
        continue
    ax.plot(
        dfl["phi"], dfl["steps_mean"],
        "o-", linewidth=1.6, markersize=5,
        label=label_tex, color=cmap(base_idx)
    )


ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\langle TT \rangle$")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(frameon=False)
ax.set_ylim(bottom=0)
plt.tight_layout()
fig2_path = OUT_DIR / "steps_mean_vs_phi.pdf"
plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
print(f"[OK] Figura (média): {fig2_path}")
plt.show()

# --- FIGURA 3: dispersão (std/sem) vs phi ---
plt.figure(figsize=(10, 6), dpi=150)
ax = plt.gca()
ax.axvline(PHI_LINE, color="black", linestyle="--", linewidth=1.5,
           label=rf"$\phi_c \approx {PHI_LINE}$")
for base_idx, (label_tex, _dirname) in enumerate(bases):
    dfl = df[df["cenario_label"] == label_tex].sort_values("phi")
    if dfl.empty:
        continue
    if DISP_METRIC == "sem":
        y = dfl["steps_sem"].to_numpy()
        ylabel = "Erro-padrão da média dos passos"
    else:
        y = dfl["steps_std"].to_numpy()
        ylabel = "Desvio-padrão dos passos"

    ax.plot(
        dfl["phi"], y,
        "o-", linewidth=1.6, markersize=5,
        label=label_tex, color=cmap(base_idx)
    )


ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r"$\sigma$")
ax.grid(True, linestyle="--", alpha=0.6)
#ax.legend(frameon=False)
ax.set_ylim(bottom=0)
plt.tight_layout()
fig3_path = OUT_DIR / f"steps_disp_{DISP_METRIC}_vs_phi.pdf"
plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
print(f"[OK] Figura (dispersão): {fig3_path}")
plt.show()
