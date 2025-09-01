#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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
CMAP_NAME = "flag"     # paleta
PHI_SEPARATORS_START = 0.05
PHI_SEPARATORS_STEP  = 0.10
PHI_SEPARATORS_MAX   = 0.90   # NÃO desenhamos em 0.90
PHI_LINE_SPECIAL     = 0.60   # linha especial

# Filtro de runs: "todos" ou "apenas_extintos" (escapers==0)
FILTRO_STEPS = "todos"

# ---------------------- Utilitários ----------------------
def ler_dat(fp: Path) -> pd.DataFrame:
    """Lê .dat tolerante a delimitadores e cabeçalhos."""
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
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    return df.dropna(subset=["steps","escapers","run"])

def extrai(padrao: str, texto: str, default=None, cast=int):
    m = re.search(padrao, texto)
    return cast(m.group(1)) if m else default

def symmetric_offsets(n_series: int, delta: float = 0.25):
    """
    Retorna offsets simétricos:
      n=3 -> [-d, 0, +d]
      n=2 -> [-d/2, +d/2]
      n=4 -> [-1.5d,-0.5d,+0.5d,+1.5d]
    """
    idx = np.arange(n_series)
    if n_series % 2 == 1:
        center = n_series // 2
        return (idx - center) * delta
    else:
        return (idx - (n_series - 1)/2) * delta

def add_phi_separators_and_phi60(ax, phi_sorted, tick_positions):
    """Linhas verticais nos separadores (0.05, 0.15, ...) e linha especial em 0.60."""
    phi_max = max(phi_sorted) if len(phi_sorted) > 0 else 1.0
    for phi_sep in np.arange(PHI_SEPARATORS_START, min(PHI_SEPARATORS_MAX, phi_max)+1e-12, PHI_SEPARATORS_STEP):
        if np.isclose(phi_sep, PHI_SEPARATORS_MAX):
            continue
        x_sep = np.interp(phi_sep, phi_sorted, tick_positions)
        ax.axvline(x=x_sep, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    # linha especial em 0.60
    x_phi60 = np.interp(PHI_LINE_SPECIAL, phi_sorted, tick_positions)
    ax.axvline(x=x_phi60, color="black", linestyle="--", linewidth=1.5, label=rf"$\phi = {PHI_LINE_SPECIAL:.2f}$")

def put_phi60_first_in_legend(ax, frameon=False):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    linha60, outros = [], []
    for h, l in zip(handles, labels):
        if rf"$\phi = {PHI_LINE_SPECIAL:.2f}$" in l:
            linha60.append((h, l))
        else:
            outros.append((h, l))
    new = linha60 + outros if linha60 else outros
    if new:
        new_handles, new_labels = zip(*new)
        ax.legend(new_handles, new_labels, frameon=frameon)

# ---------------------- Coleta de dados por run (para boxplot) ----------------------
def coletar_steps_por_run(base_root: Path, label_tex: str):
    """
    Percorre s_obs_* em base_root e retorna uma lista de tuplas:
      (phi, n_obs, steps_values_array)
    onde steps_values_array contém os 'steps' dos runs (filtrados conforme FILTRO_STEPS).
    """
    resultados = []
    if not base_root.exists():
        print(f"[WARN] Raiz não encontrada: {base_root}")
        return resultados

    subdirs = [d for d in base_root.iterdir() if d.is_dir() and d.name.startswith("s_obs_")]
    if not subdirs:
        print(f"[WARN] Sem subpastas s_obs_* em {base_root}")
        return resultados

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

        # filtro
        if FILTRO_STEPS == "apenas_extintos":
            df = df[df["escapers"] == 0]
        elif FILTRO_STEPS == "todos":
            pass
        else:
            raise ValueError("FILTRO_STEPS deve ser 'todos' ou 'apenas_extintos'.")

        if df.empty:
            continue

        # obter O (número de obstáculos) e phi
        O = extrai(r"_O_(\d+)_", fp.name, default=extrai(r"s_obs_(\d+)$", sdir.name, default=None))
        if O is None:
            print(f"[WARN] Não consegui obter O em {fp.name}; pulando…")
            continue
        n_obs = int(O)
        phi = n_obs / float(AREA)

        steps_vals = df["steps"].dropna().to_numpy()
        if len(steps_vals) == 0:
            continue

        resultados.append((phi, n_obs, steps_vals))

    return resultados

# ---------------------- Montagem dos grupos para boxplot ----------------------
cmap = colormaps.get_cmap(CMAP_NAME)
OFFSETS = symmetric_offsets(len(bases), delta=0.25)

# Vamos coletar por cenário
dados_por_cenario = []
rows_all = []  # CSV combinado com todos os runs (útil depois)

for base_idx, (label_tex, base_dirname) in enumerate(bases):
    root_path = BASE_ROOT / base_dirname
    triplets = coletar_steps_por_run(root_path, label_tex)
    if not triplets:
        continue

    # salvar runs individuais no CSV combinado
    for phi, n_obs, arr in triplets:
        for v in arr:
            rows_all.append({
                "cenario_label": label_tex,
                "cenario_tag": base_dirname,
                "phi": phi,
                "num_obs": n_obs,
                "steps_run": float(v),
            })

    dados_por_cenario.append((base_idx, label_tex, triplets))

# salvar CSV combinado
if rows_all:
    filtro_tag = "ALL" if FILTRO_STEPS == "todos" else "EXTINTOS"
    out_runs = OUT_DIR / f"steps_boxplot_runs_{filtro_tag}__ALL.csv"
    pd.DataFrame(rows_all).sort_values(["cenario_tag","phi"]).to_csv(out_runs, index=False)
    print(f"[OK] CSV com runs: {out_runs}")
else:
    raise SystemExit("[!] Nenhum dado encontrado para boxplot.")

# ---------------------- Construir eixo categórico (φ -> índice) ----------------------
# reunir todos os φ usados para ter um eixo comum
phi_labels = sorted({phi for _, _, trips in dados_por_cenario for (phi, _n, _arr) in trips})
tick_positions = np.arange(len(phi_labels))

# ---------------------- Plot dos boxplots ----------------------
plt.figure(figsize=(12, 6), dpi=150)
ax = plt.gca()

for base_idx, label_tex, triplets in dados_por_cenario:
    # ordenar os grupos por índice categórico
    groups = []
    pos_idx = []
    for phi, _n_obs, arr in triplets:
        idx = phi_labels.index(phi)
        pos_idx.append(idx)
        groups.append(arr)

    # ordenar por idx
    pos_idx, groups = zip(*sorted(zip(pos_idx, groups)))
    positions = np.array(pos_idx, dtype=float) + OFFSETS[base_idx]

    # boxplot
    plt.boxplot(
        groups,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        boxprops=dict(facecolor=cmap(base_idx), alpha=0.5),
        medianprops=dict(color="black"),
        whiskerprops=dict(color=cmap(base_idx)),
        capprops=dict(color=cmap(base_idx)),
        flierprops=dict(marker="o", markersize=3, alpha=0.4,
                        markerfacecolor=cmap(base_idx), markeredgecolor="none")
    )
    # handle para legenda
    plt.plot([], [], color=cmap(base_idx), label=label_tex)

# ---------------------- Decoração ----------------------
# separadores (0.05, 0.15, ...) e linha 0.60
add_phi_separators_and_phi60(ax, phi_labels, tick_positions)

# eixo x com rótulos reais de φ
plt.xticks(tick_positions, [f"{phi:.3f}" for phi in phi_labels], rotation=45)
plt.xlabel(r"$\phi$")
plt.ylabel("Passos por run até a captura (distribuição)")
plt.grid(axis="y", linestyle="--", alpha=0.35)

# legenda (φ=0.60 primeiro)
put_phi60_first_in_legend(ax, frameon=False)

plt.tight_layout()
out_fig = OUT_DIR / "steps_vs_phi_boxplot_sep.pdf"
plt.savefig(out_fig, dpi=200)
print(f"[OK] Figura boxplot: {out_fig}")
plt.show()
