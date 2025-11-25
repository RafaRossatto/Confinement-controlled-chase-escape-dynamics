#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv
import re
from typing import List, Tuple, Dict, Optional
from networkx.algorithms.shortest_paths.weighted import multi_source_dijkstra_path_length

# --- Configuração global de fonte nos eixos e legenda ---
plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ---------------------- NOVOS PARÂMETROS PRINCIPAIS ----------------------
# Diferentes tamanhos de rede - VOCÊ VAI ESCOLHER OS CAMINHOS AQUI
REDE_CONFIGS = [
    (64,  "L_64",  Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE AQUI
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"),   # ⬅️ ALTERE AQUI  
    (256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")    # ⬅️ ALTERE AQUI
]

# Apenas a proporção 0.5 (como você mencionou)
bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
]

# Lista de observações para cada tamanho de rede (ajuste conforme necessário)
OBS_TEMPLATES = {
    64: ["obs_00", "obs_409", "obs_819", "obs_1228", "obs_1638", "obs_2048","obs_2416", "obs_2457","2498","obs_2867",
    "obs_3276"],
    128: ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553", "obs_8192", "obs_9666",
    "obs_9830", "obs_9994", "obs_11468", "obs_13107"],
    256: ["obs_00", "obs_6553", "obs_13107", "obs_19660", "obs_26214", "obs_32768", "obs_38666",
    "obs_39321", "obs_39976", "obs_45875", "obs_52428"]  # exemplo
}

NUM_RUNS = 100

# ---------------------- CONFIGURAÇÃO DE CORES E ESTILO ----------------------
# Usando a mesma colormap "flag" para consistência
CMAP = plt.get_cmap("flag")
PHI_LINE = 0.60

# Saída principal - vai criar em ~/Dados_Doc/resultados_modelos/dist_min
OUT_DIR = Path.home() / "Dados_Doc" / "resultados_modelos" / "dist_min_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotar ao final?
FAZER_PLOT = True

# Usar Condição Periódica de Contorno?
USE_PBC = True

# Candidatos de nomes dentro de cada run para chasers/escapers
CHASER_CANDIDATES = ["chasers.txt", "cacs.txt", "ch.txt", "chaser.txt"]
ESCAPER_CANDIDATES = ["escapers.txt", "presas.txt", "es.txt", "escaper.txt"]

# ---------------------- FUNÇÕES UTILITÁRIAS (MANTIDAS) ----------------------
def ler_coords_txt(fp: Path) -> List[Tuple[int, int]]:
    """Lê coordenadas (x y) de um txt; ignora comentários/linhas vazias."""
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

def in_bounds(p, L):
    x, y = p
    return (0 <= x < L) and (0 <= y < L)

def densidade_obs(num_obs: int, area: int) -> float:  # AGORA RECEBE area
    return num_obs / area

def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def grafo_livre(L: int, obst_set: set, pbc: bool) -> nx.Graph:
    G = nx.Graph()
    for x in range(L):
        for y in range(L):
            if (x, y) not in obst_set:
                G.add_node((x, y))

    def vizinhos(x, y):
        if pbc:
            return [((x-1) % L, y), ((x+1) % L, y), (x, (y-1) % L), (x, (y+1) % L)]
        else:
            cand = []
            if x > 0:      cand.append((x-1, y))
            if x < L-1:    cand.append((x+1, y))
            if y > 0:      cand.append((x, y-1))
            if y < L-1:    cand.append((x, y+1))
            return cand

    for (x, y) in list(G.nodes):
        for u in vizinhos(x, y):
            if u in G:
                G.add_edge((x, y), u)
    return G

def achar_primeiro_existente(dirpath: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        fp = dirpath / name
        if fp.exists():
            return fp
    return None

def multi_source_bfs_distmap(G: nx.Graph, sources: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    sources = [s for s in sources if s in G]
    if not sources:
        return {}
    return dict(multi_source_dijkstra_path_length(G, sources, weight=None))

# ---------------------- PIPELINE PRINCIPAL MODIFICADA ----------------------
def processar_cenario(label_tex: str, base_dirname: str, L: int, rede_nome: str, base_path: Path):
    area = L**2
    
    # Usa a lista de observações específica para este L
    obs_list = OBS_TEMPLATES.get(L, [f"obs_{i}" for i in range(0, area, area//8)])
    
    print(f"\n=== Processando {rede_nome} (L={L}) - {label_tex} ===")

    # agregação por-phi (para CSV agregado)
    agg_phi: Dict[float, Dict[str, List[float]]] = {}

    # CSV por-run (uma linha por run)
    out_por_run = OUT_DIR / f"dist_min_por_run_{rede_nome}_{base_dirname}.csv"
    with open(out_por_run, "w", newline="") as fout:
        w = csv.writer(fout)
        header = ["L", "rede_nome", "cenario_label", "cenario_tag", "obs_folder", "phi", "run",
                  "n_escapers", "n_reach", "frac_unreach",
                  "dist_mean", "dist_min", "dist_max"]
        w.writerow(header)

        for obs_folder in obs_list:
            pasta_obs = base_path / base_dirname / obs_folder  # ⚠️ estrutura: base_path / base_dirname / obs_folder
            if not pasta_obs.exists():
                print(f"[WARN] Pasta não encontrada: {pasta_obs}")
                continue

            # pega a 1ª subpasta (ex.: 'nC_XXX' ou similar)
            subpastas = sorted([p for p in pasta_obs.iterdir() if p.is_dir()])
            if not subpastas:
                print(f"[WARN] Nenhuma subpasta em {pasta_obs}")
                continue
            pasta_nC = subpastas[0]

            # φ calculado pelo número de obstáculos no nome da pasta
            num_obs = extrai_num_obs(obs_folder)
            phi = densidade_obs(num_obs, area)  # PASSA area

            # loop nos runs
            for i in range(NUM_RUNS):
                pasta_run = pasta_nC / f"run_{i:02d}"
                if not pasta_run.exists():
                    alt = pasta_nC / f"run_{i}"
                    if alt.exists():
                        pasta_run = alt
                    else:
                        continue

                # arquivos dentro do run
                obst_fp = pasta_run / "obstacules.txt"
                ch_fp = achar_primeiro_existente(pasta_run, CHASER_CANDIDATES)
                es_fp = achar_primeiro_existente(pasta_run, ESCAPER_CANDIDATES)

                if ch_fp is None or es_fp is None:
                    continue

                obstacles = set([p for p in ler_coords_txt(obst_fp) if in_bounds(p, L)])
                chasers   = [p for p in ler_coords_txt(ch_fp) if in_bounds(p, L)]
                escapers  = [p for p in ler_coords_txt(es_fp) if in_bounds(p, L)]

                if len(escapers) == 0:
                    continue

                # monta grafo e calcula distâncias (biblioteca)
                G = grafo_livre(L=L, obst_set=obstacles, pbc=USE_PBC)
                distmap = multi_source_bfs_distmap(G, chasers)

                dists = []
                unreachable = 0
                for e in escapers:
                    if e not in G:
                        unreachable += 1
                        continue
                    d = distmap.get(e, np.inf)
                    if np.isinf(d):
                        unreachable += 1
                    else:
                        dists.append(float(d))

                nE = len(escapers)
                n_reach = len(dists)
                frac_unreach = (unreachable / nE) if nE > 0 else np.nan

                dist_mean = float(np.mean(dists)) if dists else np.nan
                dist_min  = float(np.min(dists))  if dists else np.nan
                dist_max  = float(np.max(dists))  if dists else np.nan

                # escreve CSV por-run
                w.writerow([L, rede_nome, label_tex, base_dirname, obs_folder, phi, i,
                            nE, n_reach, frac_unreach,
                            dist_mean, dist_min, dist_max])

                # agrega para por-phi
                bucket = agg_phi.setdefault(phi, {"mean": [], "min": [], "max": [], "frac_unreach": []})
                if not np.isnan(dist_mean):
                    bucket["mean"].append(dist_mean)
                    bucket["min"].append(dist_min)
                    bucket["max"].append(dist_max)
                if not np.isnan(frac_unreach):
                    bucket["frac_unreach"].append(frac_unreach)

    print(f"[OK] CSV por-run salvo: {out_por_run}")

    # CSV agregado por-phi (média e desvio)
    out_agregado = OUT_DIR / f"dist_min_agregado_{rede_nome}_{base_dirname}.csv"
    with open(out_agregado, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["L", "rede_nome", "cenario_label", "cenario_tag", "phi",
                    "dist_mean_avg", "dist_mean_std",
                    "frac_unreach_avg", "frac_unreach_std",
                    "n_runs_validos"])
        for phi, d in sorted(agg_phi.items()):
            means = d["mean"]
            frus  = d["frac_unreach"]
            dist_mean_avg = float(np.mean(means)) if means else np.nan
            dist_mean_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
            fr_avg = float(np.mean(frus)) if frus else np.nan
            fr_std = float(np.std(frus, ddof=1)) if len(frus) > 1 else 0.0
            n_ok = len(means)
            w.writerow([L, rede_nome, label_tex, base_dirname, phi,
                        dist_mean_avg, dist_mean_std,
                        fr_avg, fr_std, n_ok])

    print(f"[OK] CSV agregado salvo: {out_agregado}")

    return out_por_run, out_agregado

# ---------------------- main ----------------------
def main():
    csvs_por_run = []
    csvs_agreg   = []
    
    # Verifica se os caminhos existem
    for L, rede_nome, base_path in REDE_CONFIGS:
        if not base_path.exists():
            print(f"❌ ERRO: Pasta não encontrada: {base_path}")
            print(f"   Por favor, ajuste o caminho em REDE_CONFIGS")
            continue
            
        print(f"\n🎯 Processando rede: {rede_nome} (L={L})")
        print(f"📁 Caminho: {base_path}")
        
        for label_tex, base_dirname in bases:
            por_run, agreg = processar_cenario(label_tex, base_dirname, L, rede_nome, base_path)
            csvs_por_run.append(por_run)
            csvs_agreg.append(agreg)

    # (opcional) plot rápido da média da distância vs phi por cenário
    if FAZER_PLOT and csvs_agreg:
        # Tamanho padronizado como nos outros códigos
        plt.figure(figsize=(10, 7))

        for k, (L, rede_nome, base_path) in enumerate(REDE_CONFIGS):
            for j, (label_tex, base_dirname) in enumerate(bases):
                # lê o CSV agregado de cada cenário
                csv_file = OUT_DIR / f"dist_min_agregado_{rede_nome}_{base_dirname}.csv"
                if not csv_file.exists():
                    continue
                    
                rows = []
                with open(csv_file, "r") as f:
                    rdr = csv.DictReader(f)
                    for r in rdr:
                        try:
                            phi = float(r["phi"])
                            m   = float(r["dist_mean_avg"])
                            s   = float(r["dist_mean_std"])
                            rows.append((phi, m, s))
                        except Exception:
                            pass
                if not rows:
                    continue
                    
                rows.sort(key=lambda t: t[0])
                xs = [r[0] for r in rows]
                ys = [r[1] for r in rows]
                es = [r[2] for r in rows]
                
                # Cores distribuídas uniformemente usando a mesma colormap
                cor = CMAP(k)
                # Label simplificado e padronizado
                label_plot = f"L = {L}"
                
                plt.errorbar(xs, ys, yerr=es, fmt='o-', capsize=4, markersize=6, linewidth=2,
                            color=cor, label=label_plot)

        plt.axvline(PHI_LINE, color="black", linestyle="--", linewidth=1.5, 
                   label=fr"$\phi_c = {PHI_LINE:.2f}$")
        plt.xlabel(r"$\phi$", fontsize=18)
        plt.ylabel(r"$\langle d \rangle$", fontsize=18)
        plt.grid(True, linestyle="--", alpha=0.3)  # Grid mais suave
        plt.legend(fontsize=12)  # Tamanho consistente
        plt.tight_layout()
        
        # Salvar em PDF como nos outros códigos
        out_pdf = OUT_DIR / "dist_min_vs_phi_multiple_L.pdf"
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.show()
        print(f"[OK] Figura salva: {out_pdf}")

if __name__ == "__main__":
    main()