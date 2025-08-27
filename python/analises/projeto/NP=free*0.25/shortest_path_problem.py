from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------- parâmetros principais ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
            "obs_8192", "obs_9830", "obs_11468", "obs_13107", "obs_14745"]

bases = [
    (r"$N_C=N_P$",      "Nc=Np"),
    (r"$N_C=0.5\,N_P$", "Nc=Np*0.5"),
]

L = 128
area = L**2
NUM_RUNS = 100
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"

# >>> novos controles <<<
MODO = "por_presa"   # "por_cacador" ou "por_presa"
PERIODIC = True       # True = contorno periódico (toro); False = aberto
# -------------------------------------------------------------------

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

def _construir_grafo_grade_periodico_fallback(L, obst_set):
    """Constrói grafo 2D LxL com 4-vizinhos e wrap-around manual (toro)."""
    G = nx.Graph()
    # adiciona apenas sites livres
    livres = {(x, y) for x in range(L) for y in range(L)} - set(obst_set)
    G.add_nodes_from(livres)
    for x, y in list(livres):
        vizinhos = [((x+1) % L, y), ((x-1) % L, y),
                    (x, (y+1) % L), (x, (y-1) % L)]
        for nb in vizinhos:
            if nb in livres:
                G.add_edge((x, y), nb)
    return G

def construir_grafo_grade(L, obst_set, periodic=False):
    """
    Constrói grafo 2D LxL (4-vizinhos).
    - periodic=False: borda aberta (sem wrap)
    - periodic=True : contorno periódico (toro)
    """
    if periodic:
        # tenta usar suporte nativo do NetworkX; se não houver, faz fallback manual
        try:
            G = nx.grid_2d_graph(L, L, periodic=True)
            if obst_set:
                G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
            return G
        except TypeError:
            # versões antigas não aceitam 'periodic'
            return _construir_grafo_grade_periodico_fallback(L, obst_set)
    else:
        G = nx.grid_2d_graph(L, L)
        if obst_set:
            G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
        return G

def d_media_run(pasta_run: Path, L: int, modo: str = "por_cacador", periodic: bool = False):
    """
    Calcula a distância média ⟨d⟩ para UM run.
    - modo="por_cacador": média da distância MÍNIMA de cada caçador até a presa mais próxima.
    - modo="por_presa"  : média da distância MÍNIMA de cada presa    até o caçador mais próximo.
    Retorna: (media, n_alcancados, n_total_no_modo)

    Contorno:
    - periodic=True  -> toro (wrap nas bordas)
    - periodic=False -> borda aberta (sem wrap)
    """
    chasers_fp   = pasta_run / "chasers.txt"
    escapers_fp  = pasta_run / "escapers.txt"
    obstacles_fp = pasta_run / "obstacules.txt"

    chasers  = [p for p in ler_coords_txt(chasers_fp)  if in_bounds(p, L)]
    presas   = [p for p in ler_coords_txt(escapers_fp) if in_bounds(p, L)]
    obst_set = set([p for p in ler_coords_txt(obstacles_fp) if in_bounds(p, L)])

    if not chasers or not presas:
        # n_total depende do modo
        n_tot = len(chasers) if modo == "por_cacador" else len(presas)
        return np.nan, 0, n_tot

    G = construir_grafo_grade(L, obst_set, periodic=periodic)

    # filtra agentes que realmente estão no grafo
    chasers = [c for c in chasers if c in G]
    presas  = [e for e in presas  if e in G]

    if modo not in ("por_cacador", "por_presa"):
        raise ValueError("modo deve ser 'por_cacador' ou 'por_presa'.")

    # define 'origens' cujas distâncias queremos (queries) e 'alvos' com peso 0
    if modo == "por_cacador":
        queries = chasers
        targets = presas
    else:
        queries = presas
        targets = chasers

    if not queries or not targets:
        return np.nan, 0, len(queries)

    # --- pesos unitários nas arestas (Manhattan em espaço livre) ---
    for u, v in G.edges():
        G.edges[u, v]['w'] = 1

    # --- super-source com peso 0 até todos os 'targets' ---
    SRC = ('SRC', -1)
    G.add_node(SRC)
    for t in targets:
        G.add_edge(SRC, t, w=0)

    # dijkstra a partir do SRC
    try:
        gen = nx.shortest_path_length(G, source=SRC, weight='w', method='dijkstra')
        dists = dict(gen) if not isinstance(gen, dict) else gen
    except TypeError:
        dists = dict(nx.shortest_path_length(G, source=SRC, weight='w'))

    vals = [dists[q] for q in queries if q in dists]

    # limpeza
    G.remove_node(SRC)

    if vals:
        return float(np.mean(vals)), len(vals), len(queries)
    else:
        return np.nan, 0, len(queries)

def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / area

# ---------------------- Agregação por densidade ----------------------
cmap = plt.cm.viridis
plt.figure(figsize=(10, 6))

for base_idx, (label_tex, base_dirname) in enumerate(bases):
    base_path = BASE_ROOT / base_dirname

    xs_phi = []
    ys_mean_d = []
    ys_std_d  = []
    frac_reach_list = []

    for obs_folder in obs_list:
        pasta_obs = base_path / obs_folder
        if not pasta_obs.exists():
            print(f"[WARN] Pasta não encontrada: {pasta_obs}")
            continue

        # estrutura típica: obs_xxxx / nC_YYYY / run_00..run_99
        subpastas = sorted([p for p in pasta_obs.iterdir() if p.is_dir()])
        if not subpastas:
            print(f"[WARN] Nenhuma subpasta em {pasta_obs}")
            continue
        pasta_nC = subpastas[0]

        ds = []
        reach_counts = 0
        total_counts = 0

        for i in range(NUM_RUNS):
            pasta_run = pasta_nC / f"run_{i:02d}"
            if not pasta_run.exists():
                alt = pasta_nC / f"run_{i}"
                if alt.exists():
                    pasta_run = alt
                else:
                    continue

            d_med, n_reach, n_tot = d_media_run(pasta_run, L, modo=MODO, periodic=PERIODIC)
            if not np.isnan(d_med):
                ds.append(d_med)
            reach_counts += n_reach
            total_counts += n_tot

        if not ds:
            print(f"[WARN] Sem valores válidos de <d> em {pasta_nC}")
            continue

        num_obs = extrai_num_obs(obs_folder)
        phi = densidade_obs(num_obs)

        xs_phi.append(phi)
        ys_mean_d.append(float(np.mean(ds)))
        ys_std_d.append(float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0)
        frac_reach_list.append((reach_counts / total_counts) if total_counts > 0 else np.nan)

    if ys_mean_d:
        xs_phi, ys_mean_d, ys_std_d, frac_reach_list = zip(
            *sorted(zip(xs_phi, ys_mean_d, ys_std_d, frac_reach_list))
        )
        plt.errorbar(xs_phi, ys_mean_d, yerr=ys_std_d,
                     fmt='o-', capsize=5, markersize=5,
                     label=label_tex,
                     color=cmap(base_idx / max(1, len(bases) - 1)))

# linha de referência (ex.: percolação de sítios livres ~0.407, obsts ~0.593; ajuste conforme seu caso)
plt.axvline(x=0.59, color='black', linestyle='--', linewidth=1.2, label='$\\phi\\,{\sim}\\,0.59$')

plt.xlabel("$\\phi$")
plt.ylabel("$\\langle d \\rangle$")
suffix = "caçador→presa" if MODO == "por_cacador" else "presa→caçador"
plt.title(f"$\\langle d \\rangle$ ({suffix}) vs $\\phi$ — $N_C=N_P$  vs  $N_C=0.5\\,N_P$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
# plt.savefig("mean_distance_vs_phi.pdf", dpi=200)  # se quiser salvar
plt.show()
