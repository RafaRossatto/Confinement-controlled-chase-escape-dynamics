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
# -------------------------------------------------------------------

def ler_coords_txt(fp: Path):
    """
    Lê um arquivo de coordenadas com formato flexível:
    - colunas separadas por espaço/comma/semicolon
    - ignora linhas vazias e que começam com '#'
    Retorna lista de tuplas (x, y) como inteiros.
    """
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
    """
    Constrói grafo 2D LxL (4-vizinhos) e remove nós que são obstáculos.
    """
    G = nx.grid_2d_graph(L, L)  # nós como (x,y)
    if obst_set:
        # só remove os que estão no grafo
        G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
    return G

def d_media_run(pasta_run: Path, L: int):
    """
    Calcula a distância média (⟨d⟩) caçador→presa mais próxima para UM run,
    usando 'super-source' + Dijkstra com nx.shortest_path_length (ou fallback
    para nx.shortest_path se a função de 'length' não existir).
    """
    chasers_fp   = pasta_run / "chasers.txt"
    escapers_fp  = pasta_run / "escapers.txt"
    obstacles_fp = pasta_run / "obstacules.txt"

    chasers  = [p for p in ler_coords_txt(chasers_fp)  if in_bounds(p, L)]
    presas   = [p for p in ler_coords_txt(escapers_fp) if in_bounds(p, L)]
    obst_set = set([p for p in ler_coords_txt(obstacles_fp) if in_bounds(p, L)])

    if not chasers or not presas:
        return np.nan, 0, len(chasers)

    # grafo da grade (4-vizinhos), removendo obstáculos
    G = construir_grafo_grade(L, obst_set)

    # filtra agentes que realmente estão no grafo
    chasers = [c for c in chasers if c in G]
    presas  = [e for e in presas  if e in G]
    if not chasers or not presas:
        return np.nan, 0, len(chasers)

    # --- adiciona pesos (w=1 nas arestas normais) ---
    # (se já tiver pesos, pode pular; aqui garantimos compatibilidade)
    for u, v in G.edges():
        G.edges[u, v]['w'] = 1

    # --- super-source com peso 0 até todas as presas ---
    SRC = ('SRC', -1)  # qualquer identificador único que não conflite com (x,y)
    G.add_node(SRC)
    for e in presas:
        G.add_edge(SRC, e, w=0)

    # tenta usar shortest_path_length; senão, usa o fallback por caminho
    has_len = hasattr(nx, 'shortest_path_length')

    dists = {}
    if has_len:
        # Um Dijkstra a partir do SRC retorna distâncias mínimas a TODOS os nós
        # com pesos 'w'. Em versões antigas, isso retorna um generator (node,dist).
        try:
            gen = nx.shortest_path_length(G, source=SRC, weight='w', method='dijkstra')
            dists = dict(gen) if not isinstance(gen, dict) else gen
        except TypeError:
            # algumas versões antigas não aceitam 'method'; tenta sem ele
            dists = dict(nx.shortest_path_length(G, source=SRC, weight='w'))
    else:
        # Fallback: calcular por caçador usando nx.shortest_path (retorna a lista do caminho)
        def path_weight(path):
            s = 0
            for a, b in zip(path[:-1], path[1:]):
                s += G.edges[a, b].get('w', 1)
            return s

        for c in chasers:
            try:
                path = nx.shortest_path(G, source=SRC, target=c, weight='w', method='dijkstra')
            except TypeError:
                path = nx.shortest_path(G, source=SRC, target=c, weight='w')
            except nx.NetworkXNoPath:
                continue
            dists[c] = path_weight(path)

    # extrai as distâncias dos caçadores (ignora os inalcançáveis)
    vals = [dists[c] for c in chasers if c in dists]

    # remove o SRC para não vazar para outros runs (opcional se você recria G a cada run)
    G.remove_node(SRC)

    if vals:
        return float(np.mean(vals)), len(vals), len(chasers)
    else:
        return np.nan, 0, len(chasers)

def extrai_num_obs(nome_obs: str) -> int:
    # "obs_XXXX" -> XXXX
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / area

# ---------------------- Agregação por densidade ----------------------
cmap = plt.cm.viridis
plt.figure(figsize=(10, 6))

for base_idx, (label_tex, base_dirname) in enumerate(bases):
    base_path = BASE_ROOT / base_dirname

    xs_phi = []        # densidade de obstáculos φ
    ys_mean_d = []     # média de <d> por densidade (média dos runs)
    ys_std_d  = []     # desvio-padrão de <d> por densidade (sobre os runs)
    frac_reach_list = []  # (opcional) fração média de caçadores alcançáveis

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
                # se a estrutura diferir (ex.: run_i sem zero à esquerda), tente a outra
                alt = pasta_nC / f"run_{i}"
                if alt.exists():
                    pasta_run = alt
                else:
                    continue

            d_med, n_reach, n_tot = d_media_run(pasta_run, L)
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

# linha de referência (se quiser marcar um limiar teórico)
plt.axvline(x=0.59, color='black', linestyle='--', linewidth=1.2, label='$\phi\\,{\sim}\,0.59$')

plt.xlabel("$\\phi$  (densidade de obstáculos)")
plt.ylabel("$\\langle d \\rangle$  (distância média caçador → presa mais próxima)")
plt.title("Distância média $\\langle d \\rangle$ vs $\\phi$  —  $N_C=N_P$  vs  $N_C=0.5\\,N_P$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("mean_distance_vs_phi.pdf", dpi=200)
plt.show()
