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

# ---------------------- NOVOS PARÂMETROS PRINCIPAIS ----------------------
# Diferentes tamanhos de rede
REDE_CONFIGS = [
    (64, "L_64"),
    (128, "L_128"), 
    (256, "L_256")
]

# Apenas a proporção 0.5 (como você mencionou)
bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
]

# Lista de observações (ajuste conforme necessário para cada L)
OBS_TEMPLATES = {
    64: ["obs_00", "obs_409", "obs_819", "obs_1228", "obs_1638", "obs_2048","obs_2416", "obs_2457","obs_2498","obs_2867",
    "obs_3276"],
    128: ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553", "obs_8192", "obs_9666",
    "obs_9830", "obs_9994", "obs_11468", "obs_13107"],
    256: ["obs_00", "obs_6553", "obs_13107", "obs_19660", "obs_26214", "obs_32768", "obs_38666",
    "obs_39321", "obs_39976", "obs_45875", "obs_52428"]  # exemplo
}

NUM_RUNS = 100
BASE_ROOT = Path.home() / "Dados_Doc/Np=free*0.25"  # pasta base contendo L64, L128, L256

# flags
USE_PBC = True
RECALCULAR_SE_FALTAR = True
FORCAR_RECALCULO = False

# nomes de arquivos dentro de cada run
OBST_FP_NAME = "obstacules.txt"
CHASER_CANDIDATES = ["chasers.txt", "cacs.txt", "ch.txt", "chaser.txt"]
ESCAPER_CANDIDATES = ["escapers.txt", "presas.txt", "es.txt", "escaper.txt"]

# ---------------------- CONFIGURAÇÃO DE CORES E ESTILO ----------------------
# Usando a mesma colormap "flag" para consistência com os outros códigos
CMAP = plt.get_cmap("flag")
PHI_LINE = 0.60

# ---------------------- FUNÇÕES UTILITÁRIAS (MANTIDAS) ----------------------
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

def NE0(num_obs: int, AREA: int) -> float:  # AGORA RECEBE AREA COMO PARÂMETRO
    """N_E0 = metade dos sítios livres após obstáculos."""
    livres = AREA - num_obs
    return livres / 4.0

def densidade_obs(num_obs: int, AREA: int) -> float:  # AGORA RECEBE AREA COMO PARÂMETRO
    return num_obs / AREA

def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def contar_inacessiveis_por_biblioteca(pasta_run: Path, L: int) -> Optional[int]:  # AGORA RECEBE L
    """
    Recalcula N_inacc no estado final do run via NetworkX.
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
    distmap = dict(multi_source_dijkstra_path_length(G, sources, weight=None)) if sources else {}

    inac = 0
    for e in escapers:
        if e not in G:
            inac += 1
        else:
            d = distmap.get(e, None)
            if d is None:
                inac += 1
    return inac

# ---------------------- PIPELINE PRINCIPAL MODIFICADA ----------------------
def main():
    # figura única com todas as curvas - tamanho padronizado
    plt.figure(figsize=(10, 7))

    for rede_idx, (L, rede_nome) in enumerate(REDE_CONFIGS):
        AREA = L**2
        obs_list = OBS_TEMPLATES.get(L, [f"obs_{i}" for i in range(0, AREA, AREA//8)])
        
        print(f"\n=== Processando {rede_nome} (L={L}) ===")

        for base_idx, (label_tex, base_dirname) in enumerate(bases):
            # Caminho base específico para este tamanho de rede
            base_path = BASE_ROOT / rede_nome / base_dirname
            
            if not base_path.exists():
                print(f"[WARN] Pasta não encontrada: {base_path}")
                continue

            xs_phi, ys_mean, ys_std = [], [], []
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

                phi = densidade_obs(num_obs, AREA)  # PASSA AREA
                ne0 = NE0(num_obs, AREA)  # PASSA AREA
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
                        conteudo = fp_inacc.read_text(encoding="utf-8", errors="ignore").split()
                        if conteudo:
                            try:
                                valor_final = float(conteudo[-1])
                            except ValueError:
                                valor_final = None

                    if (not usar_arquivo and RECALCULAR_SE_FALTAR) or (usar_arquivo and valor_final is None):
                        rec = contar_inacessiveis_por_biblioteca(pasta_run, L)  # PASSA L
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
                    "L": L,
                    "phi": phi,
                    "num_obs": num_obs,
                    "NE0": ne0,
                    "inacc_mean": media_bruta,
                    "inacc_std": std_bruto,
                    "inacc_mean_norm": media_norm,
                    "inacc_std_norm": std_norm,
                    "n_runs_validos": len(valores_finais_runs),
                })

            # salvar CSV para esta configuração
            if rows_for_csv:
                rows_for_csv.sort(key=lambda r: r["phi"])
                out_csv = base_path / f"inaccessible_preys_{rede_nome}_{base_dirname}.csv"
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
                    w.writeheader()
                    for r in rows_for_csv:
                        w.writerow(r)
                print(f"[OK] CSV salvo: {out_csv}")

            # plotar (uma curva por tamanho de rede)
            if ys_mean:
                xs_phi, ys_mean, ys_std = zip(*sorted(zip(xs_phi, ys_mean, ys_std)))
                cor = CMAP(rede_idx)  # Cores distribuídas uniformemente
                label_plot = f"L = {L}"  # Label simplificado e padronizado
                plt.errorbar(xs_phi, ys_mean, yerr=ys_std,
                             fmt='o-', capsize=4, markersize=6, linewidth=2,
                             color=cor, label=label_plot)

    # linha de referência
    plt.axvline(x=PHI_LINE, color='black', linestyle='--', linewidth=1.5,
                label=r'$\phi_{c} = 0.60$')

    plt.xlabel(r"$\phi$", fontsize=18)
    plt.ylabel(r"$N^{E}_{\text{inacc}}/N^{E}_{0}$", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.3)  # Grid mais suave
    plt.legend(fontsize=12)  # Tamanho consistente com outros códigos
    plt.tight_layout()
    
    # Salvar em PDF como nos outros códigos
    out_pdf = 'inaccessible_preys_multiple_L.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
    plt.show()
    
    print(f"[OK] Gráfico salvo em: {out_pdf}")

if __name__ == "__main__":
    main()