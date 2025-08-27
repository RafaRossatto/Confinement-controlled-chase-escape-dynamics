from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# ---------------------- parâmetros ----------------------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
cenario_dir = "Nc=Np"    # ou "Nc=Np*0.5"
phi_alvo = "obs_9830"    # escolha o diretório do φ
NUM_RUNS = 100
L = 128
# --------------------------------------------------------

def ler_coords_txt(fp: Path):
    pts = []
    if not fp.exists():
        return pts
    for line in fp.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace(",", " ").replace(";", " ")
        parts = s.split()
        if len(parts) >= 2:
            try:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                pts.append((x, y))
            except ValueError:
                pass
    return pts

def in_bounds(p, L):
    x, y = p
    return (0 <= x < L) and (0 <= y < L)

def construir_grafo_grade(L, obst_set):
    G = nx.grid_2d_graph(L, L)
    if obst_set:
        G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
    return G

def analisa_run(pasta_run: Path, L: int):
    """
    Retorna: (num_ilhas, num_ilhas_ativas)
    """
    chasers_fp   = pasta_run / "chasers.txt"
    escapers_fp  = pasta_run / "escapers.txt"
    obstacles_fp = pasta_run / "obstacules.txt"

    chasers  = [p for p in ler_coords_txt(chasers_fp)  if in_bounds(p, L)]
    presas   = [p for p in ler_coords_txt(escapers_fp) if in_bounds(p, L)]
    obst_set = set([p for p in ler_coords_txt(obstacles_fp) if in_bounds(p, L)])

    if not chasers or not presas:
        return None

    G = construir_grafo_grade(L, obst_set)
    comps = list(nx.connected_components(G))
    comp_map = {node: cid for cid, comp in enumerate(comps) for node in comp}

    if not comp_map:
        return None

    presas_por_ilha = {}
    cacadores_por_ilha = {}

    for p in presas:
        if p in comp_map:
            cid = comp_map[p]
            presas_por_ilha[cid] = presas_por_ilha.get(cid, 0) + 1

    for c in chasers:
        if c in comp_map:
            cid = comp_map[c]
            cacadores_por_ilha[cid] = cacadores_por_ilha.get(cid, 0) + 1

    # conta ilhas com pelo menos 1 presa e 1 caçador
    ativas = sum(
        1 for cid, nP in presas_por_ilha.items()
        if nP > 0 and cacadores_por_ilha.get(cid, 0) > 0
    )

    return len(comps), ativas

# ---------------------- coleta de dados ----------------------
base_path = BASE_ROOT / cenario_dir / phi_alvo
subpastas = sorted([p for p in base_path.iterdir() if p.is_dir()])
if not subpastas:
    raise RuntimeError(f"Nenhuma subpasta em {base_path}")
pasta_nC = subpastas[0]

dados = []
for i in range(NUM_RUNS):
    pasta_run = pasta_nC / f"run_{i:02d}"
    if not pasta_run.exists():
        alt = pasta_nC / f"run_{i}"
        if alt.exists():
            pasta_run = alt
        else:
            continue
    res = analisa_run(pasta_run, L)
    if res is not None:
        num_ilhas, num_ativas = res
        dados.append({"num_ilhas": num_ilhas, "ativas": num_ativas})

df = pd.DataFrame(dados)

# ---------------------- gráficos ----------------------

fig, axs = plt.subplots(1, 2, figsize=(12,5))

# (1) Histograma do número de ilhas
axs[0].hist(df["num_ilhas"], bins=20, color="steelblue", alpha=0.7)
axs[0].set_xlabel("Número de ilhas (componentes)")
axs[0].set_ylabel("Frequência (runs)")
axs[0].set_title(f"Distribuição do nº de ilhas em φ={phi_alvo}")

# (2) Número médio de ilhas ativas em função do nº de ilhas
media_ativas = df.groupby("num_ilhas")["ativas"].mean()
axs[1].plot(media_ativas.index, media_ativas.values, "o-", color="darkred")
axs[1].set_xlabel("Número total de ilhas")
axs[1].set_ylabel("Média de ilhas ativas")
axs[1].set_title(f"Ilhas ativas vs nº de ilhas em φ={phi_alvo}")
axs[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# (3) Densidade de estados (heatmap 2D)
plt.figure(figsize=(8,6))
plt.hist2d(df["num_ilhas"], df["ativas"], bins=(30,30), cmap="viridis", density=True)
plt.colorbar(label="densidade de probabilidade")
plt.xlabel("Número total de ilhas (k)")
plt.ylabel("Número de ilhas ativas (m)")
plt.title(f"Densidade de estados em φ={phi_alvo}")
plt.show()
