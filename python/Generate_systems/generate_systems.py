#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de inacessibilidade de presas em uma grade com obstáculos,
usando um grafo (NetworkX) para determinar regiões alcançáveis a partir
dos caçadores. Nesta versão, impomos a razão fixa N_H / N_P = ratio (padrão 0.5),
ou seja, os caçadores são metade do número de presas.

Saídas:
- Gera pastas no formato: obs_{num_obstaculos}/nC_{n_chasers}/run_{sim}
  contendo arquivos:
    - chasers.txt         : posições (x y) dos caçadores
    - escapers.txt        : posições (x y) das presas
    - obstacules.txt      : posições (x y) dos obstáculos
    - inaccessible_preys.txt : número de presas inacessíveis
- Salva a figura "curva_ratio_0_5.pdf" com a fração de presas inacessíveis vs densidade de obstáculos.

Requisitos:
  numpy, matplotlib, tqdm, networkx
"""

from __future__ import annotations
import os
import csv
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

# ----- Constantes -----
OBSTACLE = -1
EMPTY    = 0
CHASER   = 1
PREY     = 2


# ----- Construção do grafo e BFS -----
def construir_grafo(grid: np.ndarray) -> nx.Graph:
    """Constroi um grafo 4-vizinhos (von Neumann) ignorando células com OBSTACLE."""
    L = grid.shape[0]
    G = nx.Graph()

    for x in range(L):
        for y in range(L):
            if grid[x, y] == OBSTACLE:
                continue  # Obstáculos não entram no grafo

            # Vizinhança 4
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < L and 0 <= ny_ < L:
                    if grid[nx_, ny_] != OBSTACLE:
                        G.add_edge((x, y), (nx_, ny_))

    return G


def bfs_usando_grafo(grid: np.ndarray) -> np.ndarray:
    """Marca (True/False) as células alcançáveis a partir de todos os caçadores."""
    L = grid.shape[0]
    G = construir_grafo(grid)

    chasers = [(x, y) for x in range(L) for y in range(L) if grid[x, y] == CHASER]

    acessiveis = set()
    for c in chasers:
        if c in G:  # segurança
            for node in nx.single_source_shortest_path_length(G, c).keys():
                acessiveis.add(node)

    visited = np.zeros_like(grid, dtype=bool)
    for x, y in acessiveis:
        visited[x, y] = True
    return visited


# ----- Geração da grade com razão fixa NH/NP -----
def criar_grid(L: int, p_obstaculos: float = 0.3, ratio_chasers_to_preys: float = 0.5,
               rng: np.random.Generator | None = None) -> Tuple[np.ndarray, int, int]:
    """
    Cria uma grade LxL com obstáculos, caçadores e presas.
    Mantém N_H = ratio * N_P. Distribui tudo aleatoriamente.

    Retorna: (grid, n_preys, n_chasers)
    """
    if rng is None:
        rng = np.random.default_rng()

    total_celulas = L * L
    max_obstaculos = int(p_obstaculos * total_celulas)
    livres = total_celulas - max_obstaculos

    if livres <= 0:
        raise ValueError("Sem sítios livres após colocar obstáculos.")

    r = float(ratio_chasers_to_preys)
    if r < 0:
        raise ValueError("A razão NH/NP deve ser não-negativa.")

    # N_H = r * N_P e (N_P + N_H) <= livres  =>  N_P*(1+r) <= livres
    n_preys  = int(livres / (4))
    n_chasers = int(r * n_preys)

    if n_preys + n_chasers > livres:
        raise ValueError("Soma de caçadores e presas excede os sítios livres.")

    grid = np.full((L, L), EMPTY, dtype=int)
    todas_posicoes = [(x, y) for x in range(L) for y in range(L)]
    rng.shuffle(todas_posicoes)
    idx = 0

    # Obstáculos
    for _ in range(max_obstaculos):
        x, y = todas_posicoes[idx]; idx += 1
        grid[x, y] = OBSTACLE

    # Caçadores
    for _ in range(n_chasers):
        x, y = todas_posicoes[idx]; idx += 1
        grid[x, y] = CHASER

    # Presas
    for _ in range(n_preys):
        x, y = todas_posicoes[idx]; idx += 1
        grid[x, y] = PREY

    return grid, n_preys, n_chasers


# ----- Métrica de inacessibilidade -----
def contar_presas_inacessiveis(grid: np.ndarray, acessiveis: np.ndarray) -> Tuple[int, int]:
    total_presas = int(np.sum(grid == PREY))
    inacessiveis = int(np.sum((grid == PREY) & (~acessiveis)))
    return total_presas, inacessiveis


# ----- Loop de simulações varrendo p com razão fixa -----
def gerar_curvas_metade(
    L: int = 128,
    p_min: float = 0.0,
    p_max: float = 0.9,
    passos_p: int = 20,
    num_simulacoes: int = 10,
    ratio: float = 0.5,
    seed: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Varrendo p (densidade de obstáculos), mantendo a razão NH/NP = ratio.
    Retorna um dicionário no formato esperado pelo plotador.
    """
    rng = np.random.default_rng(seed)
    valores_p = np.linspace(p_min, p_max, passos_p)

    medias, desvios, preys, chasers = [], [], [], []

    for p in tqdm(valores_p, desc=f"Razão NH/NP = {ratio}"):
        inacessiveis_por_sim: List[int] = []
        n_preys_amostrado = None
        n_chasers_amostrado = None

        for sim in range(num_simulacoes):
            try:
                grid, n_preys, n_chs = criar_grid(L, p_obstaculos=p,
                                                  ratio_chasers_to_preys=ratio,
                                                  rng=rng)
                n_preys_amostrado = n_preys
                n_chasers_amostrado = n_chs

                acessiveis = bfs_usando_grafo(grid)
                _, inacessiveis = contar_presas_inacessiveis(grid, acessiveis)
                inacessiveis_por_sim.append(inacessiveis)

                # salvar arquivos (mantendo padrão de pastas)
                num_obstaculos = int(p * L * L)
                pasta_p = f"obs_{num_obstaculos:02d}"
                pasta_n = f"nC_{n_chs:02d}"
                pasta_run = f"run_{sim:02d}"
                folder_name = os.path.join(pasta_p, pasta_n, pasta_run)
                os.makedirs(folder_name, exist_ok=True)

                # Caçadores
                with open(os.path.join(folder_name, "chasers.txt"), "w") as f:
                    for x in range(L):
                        for y in range(L):
                            if grid[x, y] == CHASER:
                                f.write(f"{x} {y}\n")

                # Presas
                with open(os.path.join(folder_name, "escapers.txt"), "w") as f:
                    for x in range(L):
                        for y in range(L):
                            if grid[x, y] == PREY:
                                f.write(f"{x} {y}\n")

                # Obstáculos
                with open(os.path.join(folder_name, "obstacules.txt"), "w") as f:
                    for x in range(L):
                        for y in range(L):
                            if grid[x, y] == OBSTACLE:
                                f.write(f"{x} {y}\n")

                # Presas inacessíveis
                with open(os.path.join(folder_name, "inaccessible_preys.txt"), "w") as f:
                    f.write(f"{inacessiveis}\n")

            except ValueError:
                # grade inviável para esse p
                inacessiveis_por_sim.append(0)

        if n_preys_amostrado and n_preys_amostrado > 0:
            media_norm = float(np.mean(inacessiveis_por_sim) / n_preys_amostrado)
            desvio_norm = float(np.std(inacessiveis_por_sim) / n_preys_amostrado)
        else:
            media_norm = 0.0
            desvio_norm = 0.0

        medias.append(media_norm)
        desvios.append(desvio_norm)
        preys.append(int(n_preys_amostrado or 0))
        chasers.append(int(n_chasers_amostrado or 0))

    return {
        "metade": {
            "p_vals": valores_p,
            "medias": medias,
            "desvios": desvios,
            "n_preys": preys,
            "n_chasers": chasers,
            "label": r"$\frac{N_H}{N_P}=0.5$"
        }
    }


# ----- Plot -----
def plotar_curvas(curvas: Dict[str, Dict[str, Any]], nome_pdf: str = "curva_ratio_0_5.pdf") -> None:
    plt.figure(figsize=(10, 6))

    chaves = list(curvas.keys())
    n_curvas = len(chaves)
    cmap = plt.cm.viridis

    for i, chave in enumerate(chaves):
        dados = curvas[chave]
        cor = cmap(i / max(1, (n_curvas - 1)))

        label = dados.get("label")
        if label is None:
            nh = dados.get("n_chasers")
            label = fr"$N_H = {nh[0] if nh else '?'}$"

        plt.errorbar(
            dados["p_vals"], dados["medias"], yerr=dados["desvios"],
            fmt='o-', capsize=3, label=label, color=cor
        )

    # Linhas de referência
    p_critico = 0.59
    plt.axvline(x=p_critico, linestyle='--', color='green', label=r"$p = 0.59$")
    plt.axhline(y=1/np.e, linestyle='--', color='red', label=r"$\frac{1}{e}$")
    plt.axhline(y=1/2, color='blue', label=r"$\frac{1}{2}$")

    plt.xlabel(r"$\rho$")
    plt.ylabel("Number of inaccessible prey (fraction)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_pdf)
    plt.close()
    # plt.show()  # deixe comentado para scripts batch


# # ----- Main -----
# def main() -> None:
#     curvas = gerar_curvas_metade(
#         L=128,
#         passos_p=10,
#         num_simulacoes=200,
#         ratio=0.5,
#         seed=1,  # defina um inteiro para reprodutibilidade
#     )
#     plotar_curvas(curvas)

# ----- Main -----
def main() -> None:
        curvas = gerar_curvas_metade(
            L=128,
            p_min = 0.61,
            p_max = 0.61,
            passos_p=1,
            num_simulacoes=200,
            ratio=0.5,
            seed=1,  # defina um inteiro para reprodutibilidade
        )
        plotar_curvas(curvas)


if __name__ == "__main__":
    main()
