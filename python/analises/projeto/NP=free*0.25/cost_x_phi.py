#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import logging
import re

# ---------------------- Configuração de Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------- Config ----------------------
class Config:
    BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
    OUT_DIR = BASE_ROOT / "resultados_modelos" / "zeros_escapers"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    BASES = ["Nc=Np*0.5", "Nc=Np*0.8", "Nc=Np"]

    PLOT_PARAMS = {
        "cmap_name": "flag",
        "boxplot_width": 0.22,
        "offset_delta": 0.25,
        "phi_line_special": 0.60
    }

    LOG_SCALE = True
    Y_LIM_LOG = (3.9, 120)
    X_LIM = (-0.5, 8.5)
    LEGEND_FRAME = True
    LEGEND_LOCATION = "upper left"

    # --- Configuração global de fonte nos eixos e legenda ---
    plt.rcParams.update({
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14})

config = Config()

# ---------------------- Funções auxiliares ----------------------
def symmetric_offsets(n_series: int, delta: float = 0.25) -> np.ndarray:
    idx = np.arange(n_series)
    if n_series % 2 == 1:
        center = n_series // 2
        return (idx - center) * delta
    else:
        return (idx - (n_series - 1)/2) * delta

def put_phi60_first_in_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    linha60, outros = [], []
    special_label = rf"$\phi = {config.PLOT_PARAMS['phi_line_special']:.2f}$"
    for h, l in zip(handles, labels):
        if special_label in l:
            linha60.append((h, l))
        else:
            outros.append((h, l))
    new = linha60 + outros if linha60 else outros
    if new:
        new_handles, new_labels = zip(*new)
        ax.legend(new_handles, new_labels,
                  frameon=config.LEGEND_FRAME,
                  loc=config.LEGEND_LOCATION,
                  fancybox=True, framealpha=0.7,
                  edgecolor="grey", facecolor="white",
                  fontsize=10)

def add_phi_separators_and_span(ax, phi_sorted, tick_positions):
    """Adiciona linhas verticais regulares e a faixa em φ=0.60."""
    phi_max = max(phi_sorted) if phi_sorted else 1.0
    
    # linhas de separação em intervalos regulares
    for phi_sep in np.arange(0.05, min(0.90, phi_max) + 1e-12, 0.10):
        if np.isclose(phi_sep, 0.90):
            continue
        x_sep = np.interp(phi_sep, phi_sorted, tick_positions)
        ax.axvline(x=x_sep, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    
    # faixa em φ=0.60
    if config.PLOT_PARAMS["phi_line_special"] <= phi_max:
        x_phi60 = np.interp(config.PLOT_PARAMS["phi_line_special"],
                            phi_sorted, tick_positions)
        ax.axvspan(x_phi60 - 0.5, x_phi60 + 0.5,
                   color="red", alpha=0.2,
                   label=rf"$\phi = {config.PLOT_PARAMS['phi_line_special']:.2f}$")

# ---------------------- Calcular custos ----------------------
def calcular_custos(base_root: Path, ne: int = 2048):
    df = pd.read_csv(base_root / "steps_boxplot_runs_ALL__ALL.csv")
    resultados = []
    for (cenario, phi), grupo in df.groupby(["cenario_tag", "phi"]):
        steps = grupo["steps_run"].values
        if len(steps) == 0:
            continue
        if cenario == "Nc=Np*0.5":
            nc = int(0.5 * ne)
        elif cenario == "Nc=Np*0.8":
            nc = int(0.8 * ne)
        elif cenario == "Nc=Np":
            nc = int(ne)
        else:
            continue
        custos = (nc / ne) * steps
        for custo in custos:
            resultados.append({
                "cenario": cenario,
                "phi": phi,
                "Nc": nc,
                "Ne": ne,
                "custo": custo
            })
    df_res = pd.DataFrame(resultados).sort_values(["cenario", "phi"])
    df_res.to_csv(base_root / "custo_boxplot.csv", index=False)
    return df_res

# ---------------------- Plot ----------------------
def plot_custo(df, out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    cmap = colormaps.get_cmap(config.PLOT_PARAMS["cmap_name"])

    ordem = config.BASES
    phi_labels = sorted(df["phi"].unique())
    tick_positions = np.arange(len(phi_labels))
    offsets = symmetric_offsets(len(ordem), config.PLOT_PARAMS["offset_delta"])

    for idx, cenario in enumerate(ordem):
        grupos, pos_idx = [], []
        for phi in phi_labels:
            grupo = df[(df["cenario"] == cenario) & (df["phi"] == phi)]
            if grupo.empty:
                continue
            grupos.append(grupo["custo"].values)
            pos_idx.append(tick_positions[phi_labels.index(phi)])
        if not grupos:
            continue
        pos_idx, grupos = zip(*sorted(zip(pos_idx, grupos)))
        positions = np.array(pos_idx, dtype=float) + offsets[idx]
        ax.boxplot(
            grupos, positions=positions,
            widths=config.PLOT_PARAMS["boxplot_width"],
            patch_artist=True,
            boxprops=dict(facecolor=cmap(idx), alpha=0.5),
            medianprops=dict(color="black"),
            whiskerprops=dict(color=cmap(idx)),
            capprops=dict(color=cmap(idx)),
            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                            markerfacecolor=cmap(idx), markeredgecolor="none")
        )
        ax.plot([], [], color=cmap(idx), label=cenario, linewidth=3)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{phi:.2f}" for phi in phi_labels])
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xlabel(r"$\phi$",fontsize=22)
    ax.set_ylabel("c",fontsize=22)

    if config.LOG_SCALE:
        ax.set_yscale("log")
        ax.set_ylim(config.Y_LIM_LOG[0], config.Y_LIM_LOG[1])
        ax.set_xlim(config.X_LIM[0], config.X_LIM[1])

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    add_phi_separators_and_span(ax, phi_labels, tick_positions)
    put_phi60_first_in_legend(ax)

    plt.tight_layout()
    out_file = out_dir / "cost_vs_phi.pdf"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()
    logger.info(f"Figura salva: {out_file}")

# ---------------------- Main ----------------------
def main():
    base_root = config.OUT_DIR
    df_res = calcular_custos(base_root)
    plot_custo(df_res, base_root)

if __name__ == "__main__":
    main()
