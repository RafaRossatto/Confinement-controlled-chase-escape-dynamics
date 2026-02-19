#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from typing import List, Tuple, Optional, Dict

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import multi_source_dijkstra_path_length


# --- Configuração global de fonte nos eixos e legenda ---
plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ---------------------- parâmetros principais ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553","obs_8028",
            "obs_8192","obs_8355","obs_9666", "obs_9830","obs_9994", "obs_11468", "obs_13107"]

#obs_list = ["obs_00", "obs_409", "obs_819", "obs_1228", "obs_1638",
 #           "obs_2048", "obs_2457", "obs_2867", "obs_3276"]

bases = [
    (r"$N^{C}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}=N^{E}_{0}$",      "Nc=Np"),
]

L = 128
AREA = L**2
NUM_RUNS = 100
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"

# flags
USE_PBC = True                     # Condição periódica de contorno
RECALCULAR_SE_FALTAR = True        # Recalcula se arquivo não existir
FORCAR_RECALCULO = False           # Recalcula mesmo se arquivo existir

# nomes de arquivos dentro de cada run
OBST_FP_NAME = "obstacules.txt"
CHASER_CANDIDATES = ["chasers.txt", "cacs.txt", "ch.txt", "chaser.txt"]
ESCAPER_CANDIDATES = ["escapers.txt", "presas.txt", "es.txt", "escaper.txt"]

# figure style
CMAP = plt.get_cmap("flag")
PHI_LINE = 0.592746

# ---------------------- utilitários ----------------------
def ler_coords_txt(fp: Path) -> List[Tuple[int, int]]:
    pts = []
    if not fp.exists():
        return pts
    for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.sub(r"[;,]", " ", s)
        parts = s.split()
        if len(parts) >= 2:
            try:
                x = int(float(parts[0])); y = int(float(parts[1]))
                pts.append((x, y))
            except ValueError:
                pass
    return pts

def achar_primeiro_existente(dirpath: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        fp = dirpath / name
        if fp.exists():
            return fp
    return None

def in_bounds(p, L):
    x, y = p
    return (0 <= x < L) and (0 <= y < L)

def grafo_livre(L: int, obst_set: set, pbc: bool) -> nx.Graph:
    G = nx.Graph()
    for x in range(L):
        for y in range(L):
            if (x, y) not in obst_set:
                G.add_node((x, y))
    def viz(x, y):
        if pbc:
            return [((x-1) % L, y), ((x+1) % L, y), (x, (y-1) % L), (x, (y+1) % L)]
        out = []
        if x > 0:      out.append((x-1, y))
        if x < L-1:    out.append((x+1, y))
        if y > 0:      out.append((x, y-1))
        if y < L-1:    out.append((x, y+1))
        return out
    for (x, y) in list(G.nodes):
        for u in viz(x, y):
            if u in G:
                G.add_edge((x, y), u)
    return G

def NE0(num_obs: int) -> float:
    """N_E0 = metade dos sítios livres após obstáculos."""
    livres = AREA - num_obs
    return livres / 4.0  # CORRIGIDO (antes /4.0)

def densidade_obs(num_obs: int) -> float:
    return num_obs / AREA

def extrai_num_obs(nome_obs: str) -> int:
    # "obs_1638" -> 1638
    return int(nome_obs.split("_")[1])

def contar_inacessiveis_por_biblioteca(pasta_run: Path) -> Optional[int]:
    """
    Recalcula N_inacc no estado final do run via NetworkX:
    número de escapers que NÃO têm caminho para QUALQUER chaser (com CPD e obstáculos).
    """
    obst_fp = pasta_run / OBST_FP_NAME
    ch_fp = achar_primeiro_existente(pasta_run, CHASER_CANDIDATES)
    es_fp = achar_primeiro_existente(pasta_run, ESCAPER_CANDIDATES)
    if ch_fp is None or es_fp is None:
        return None

    obstacles = set([p for p in ler_coords_txt(obst_fp) if in_bounds(p, L)])
    chasers   = [p for p in ler_coords_txt(ch_fp) if in_bounds(p, L)]
    escapers  = [p for p in ler_coords_txt(es_fp) if in_bounds(p, L)]
    if not escapers:
        return 0

    G = grafo_livre(L=L, obst_set=obstacles, pbc=USE_PBC)
    sources = [s for s in chasers if s in G]
    # Dijkstra multi-fonte sem peso = BFS (distância em passos)
    distmap = dict(multi_source_dijkstra_path_length(G, sources, weight=None)) if sources else {}

    inac = 0
    for e in escapers:
        if e not in G:
            # escaper está em obstáculo (defensivo) → conte como inacessível
            inac += 1
        else:
            d = distmap.get(e, None)
            if d is None:
                inac += 1
    return inac

# ---------------------- pipeline ----------------------
def main():
    # figura
    plt.figure(figsize=(10, 6))

    for base_idx, (label_tex, base_dirname) in enumerate(bases):
        base_path = BASE_ROOT / base_dirname
        xs_phi, ys_mean, ys_std = [], [], []

        # CSV por cenário
        out_csv = base_path / f"inaccessible_preys_agregado_{base_dirname}.csv"
        rows_for_csv: List[Dict[str, float]] = []

        for obs_folder in obs_list:
            pasta_obs = base_path / obs_folder
            if not pasta_obs.exists():
                print(f"[WARN] Pasta não encontrada: {pasta_obs}")
                continue

            subpastas = sorted([p for p in pasta_obs.iterdir() if p.is_dir()])
            if not subpastas:
                print(f"[WARN] Nenhuma subpasta em {pasta_obs}")
                continue
            pasta_nC = subpastas[0]

            try:
                num_obs = extrai_num_obs(obs_folder)
            except Exception:
                print(f"[WARN] Não consegui extrair #obs de '{obs_folder}'")
                continue

            phi = densidade_obs(num_obs)
            ne0 = NE0(num_obs)
            if ne0 <= 0:
                print(f"[WARN] N_E0 inválido em {obs_folder}")
                continue

            valores_finais_runs = []

            for i in range(NUM_RUNS):
                pasta_run = pasta_nC / f"run_{i:02d}"
                if not pasta_run.exists():
                    alt = pasta_nC / f"run_{i}"
                    if alt.exists():
                        pasta_run = alt
                    else:
                        continue

                fp_inacc = pasta_run / "inaccessible_preys.txt"

                usar_arquivo = fp_inacc.exists() and not FORCAR_RECALCULO
                valor_final = None

                if usar_arquivo:
                    # lê o último valor do arquivo
                    conteudo = fp_inacc.read_text(encoding="utf-8", errors="ignore").split()
                    if conteudo:
                        try:
                            valor_final = float(conteudo[-1])
                        except ValueError:
                            valor_final = None

                if (not usar_arquivo and RECALCULAR_SE_FALTAR) or (usar_arquivo and valor_final is None):
                    # recalcula via NetworkX
                    rec = contar_inacessiveis_por_biblioteca(pasta_run)
                    if rec is not None:
                        valor_final = float(rec)

                if valor_final is not None:
                    valores_finais_runs.append(valor_final)

            if not valores_finais_runs:
                print(f"[INFO] Sem valores válidos em {pasta_nC}")
                continue

            media_bruta = float(np.mean(valores_finais_runs))
            std_bruto   = float(np.std(valores_finais_runs, ddof=1)) if len(valores_finais_runs) > 1 else 0.0

            media_norm = media_bruta / ne0
            std_norm   = std_bruto   / ne0

            xs_phi.append(phi)
            ys_mean.append(media_norm)
            ys_std.append(std_norm)

            rows_for_csv.append({
                "phi": phi,
                "num_obs": num_obs,
                "NE0": ne0,
                "inacc_mean": media_bruta,
                "inacc_std": std_bruto,
                "inacc_mean_norm": media_norm,
                "inacc_std_norm": std_norm,
                "n_runs_validos": len(valores_finais_runs),
            })

        # salvar CSV do cenário
        if rows_for_csv:
            rows_for_csv.sort(key=lambda r: r["phi"])
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
                w.writeheader()
                for r in rows_for_csv:
                    w.writerow(r)
            print(f"[OK] CSV salvo: {out_csv}")

        # plotar cenário
        if ys_mean:
            xs_phi, ys_mean, ys_std = zip(*sorted(zip(xs_phi, ys_mean, ys_std)))
            plt.errorbar(xs_phi, ys_mean, yerr=ys_std,
                         fmt='o-', capsize=5, markersize=5,
                         color=CMAP(base_idx), label=label_tex)

    # linha de referência (percolação de sítio ~0.592746; você marcou φ no eixo)
    plt.axvline(x=0.60, color='black', linestyle='--', linewidth=1.5,
                label=r'$\phi_{c} = 0.60$')

    plt.xlabel(r"$\phi$", fontsize=18)
    plt.ylabel(r"$\langle N^{E}_{\text{inacc}} \rangle /N^{E}_{0}$" , fontsize=18)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('inaccessible_preys.pdf', dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
