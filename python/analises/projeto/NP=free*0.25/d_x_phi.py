from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv

# ---------------------- parâmetros principais ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
            "obs_8192", "obs_9830", "obs_11468", "obs_13107", "obs_14745"]

bases = [
    (r"$N^{C}_{0}=0.5\,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}_{0}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}_{0}=N^{E}_{0}$",      "Nc=Np"),
]

L = 128
area = L**2
NUM_RUNS = 100
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"

# >>> novos controles <<<
MODO = "por_presa"   # "por_cacador" ou "por_presa"
PERIODIC = True      # True = contorno periódico (toro); False = aberto
cmap = plt.get_cmap("flag")
#cores = [cmap(i) for i in (0, 1, 2)]  # 3 cores bem distintas
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
                x = int(float(parts[0])); y = int(float(parts[1]))
                pts.append((x, y))
            except ValueError:
                pass
    return pts

def in_bounds(p, L):
    x, y = p
    return (0 <= x < L) and (0 <= y < L)

def _construir_grafo_grade_periodico_fallback(L, obst_set):
    G = nx.Graph()
    livres = {(x, y) for x in range(L) for y in range(L)} - set(obst_set)
    G.add_nodes_from(livres)
    for x, y in list(livres):
        vizinhos = [((x+1) % L, y), ((x-1) % L, y), (x, (y+1) % L), (x, (y-1) % L)]
        for nb in vizinhos:
            if nb in livres:
                G.add_edge((x, y), nb)
    return G

def construir_grafo_grade(L, obst_set, periodic=False):
    if periodic:
        try:
            G = nx.grid_2d_graph(L, L, periodic=True)
            if obst_set:
                G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
            return G
        except TypeError:
            return _construir_grafo_grade_periodico_fallback(L, obst_set)
    else:
        G = nx.grid_2d_graph(L, L)
        if obst_set:
            G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
        return G

def d_media_run(pasta_run: Path, L: int, modo: str = "por_cacador", periodic: bool = False):
    chasers_fp   = pasta_run / "chasers.txt"
    escapers_fp  = pasta_run / "escapers.txt"
    obstacles_fp = pasta_run / "obstacules.txt"

    chasers  = [p for p in ler_coords_txt(chasers_fp)  if in_bounds(p, L)]
    presas   = [p for p in ler_coords_txt(escapers_fp) if in_bounds(p, L)]
    obst_set = set([p for p in ler_coords_txt(obstacles_fp) if in_bounds(p, L)])

    if not chasers or not presas:
        n_tot = len(chasers) if modo == "por_cacador" else len(presas)
        return np.nan, 0, n_tot

    G = construir_grafo_grade(L, obst_set, periodic=periodic)
    chasers = [c for c in chasers if c in G]
    presas  = [e for e in presas  if e in G]

    if modo == "por_cacador":
        queries, targets = chasers, presas
    elif modo == "por_presa":
        queries, targets = presas, chasers
    else:
        raise ValueError("modo deve ser 'por_cacador' ou 'por_presa'.")

    if not queries or not targets:
        return np.nan, 0, len(queries)

    for u, v in G.edges():
        G.edges[u, v]['w'] = 1

    SRC = ('SRC', -1)
    G.add_node(SRC)
    for t in targets:
        G.add_edge(SRC, t, w=0)

    try:
        gen = nx.shortest_path_length(G, source=SRC, weight='w', method='dijkstra')
        dists = dict(gen) if not isinstance(gen, dict) else gen
    except TypeError:
        dists = dict(nx.shortest_path_length(G, source=SRC, weight='w'))

    vals = [dists[q] for q in queries if q in dists]
    G.remove_node(SRC)

    if vals:
        return float(np.mean(vals)), len(vals), len(queries)
    else:
        return np.nan, 0, len(queries)

def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / area

# ---------------------- Agregação por densidade + CSV ----------------------
plt.figure(figsize=(10, 6))

# coletores para CSVs combinados
rows_all_phi  = []   # agregado por phi
rows_all_runs = []   # por run (opcional)

for base_idx, (label_tex, base_dirname) in enumerate(bases):
    base_path = BASE_ROOT / base_dirname

    xs_phi = []
    ys_mean_d = []
    ys_std_d  = []
    frac_reach_list = []

    # linhas do CSV por cenário
    rows_csv = []

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

        ds = []
        reach_counts = 0
        total_counts = 0
        num_obs = extrai_num_obs(obs_folder)
        phi = densidade_obs(num_obs)

        for i in range(NUM_RUNS):
            pasta_run = pasta_nC / f"run_{i:02d}"
            if not pasta_run.exists():
                alt = pasta_nC / f"run_{i}"
                if alt.exists():
                    pasta_run = alt
                else:
                    continue

            d_med, n_reach, n_tot = d_media_run(pasta_run, L, modo=MODO, periodic=PERIODIC)
            # CSV por run (opcional)
            rows_all_runs.append({
                "scenario": base_dirname, "phi": phi, "num_obs": num_obs,
                "run": i, "d_run": (float(d_med) if not np.isnan(d_med) else np.nan),
                "n_reach": n_reach, "n_total": n_tot, "modo": MODO, "periodic": PERIODIC
            })

            if not np.isnan(d_med):
                ds.append(d_med)
            reach_counts += n_reach
            total_counts += n_tot

        if not ds:
            print(f"[WARN] Sem valores válidos de <d> em {pasta_nC}")
            continue

        xs_phi.append(phi)
        d_mean = float(np.mean(ds))
        d_std  = float(np.std(ds, ddof=1)) if len(ds) > 1 else 0.0
        frac_reach = (reach_counts / total_counts) if total_counts > 0 else np.nan

        ys_mean_d.append(d_mean)
        ys_std_d.append(d_std)
        frac_reach_list.append(frac_reach)

        # linha para CSV por cenário
        rows_csv.append({
            "phi": phi,
            "num_obs": num_obs,
            "n_runs_validos": int(len(ds)),
            "d_mean": d_mean,
            "d_std": d_std,
            "frac_reach": frac_reach,
            "modo": MODO,
            "periodic": PERIODIC
        })

        # linha para CSV combinado (φ agregado)
        rows_all_phi.append({
            "scenario": base_dirname,
            "phi": phi,
            "num_obs": num_obs,
            "n_runs_validos": int(len(ds)),
            "d_mean": d_mean,
            "d_std": d_std,
            "frac_reach": frac_reach,
            "modo": MODO,
            "periodic": PERIODIC
        })

    # ordenar e plotar por cenário
    if ys_mean_d:
        xs_phi, ys_mean_d, ys_std_d, frac_reach_list = zip(
            *sorted(zip(xs_phi, ys_mean_d, ys_std_d, frac_reach_list))
        )
        plt.errorbar(xs_phi, ys_mean_d, yerr=ys_std_d,
                     fmt='o-', capsize=5, markersize=5,
                     label=label_tex,
                     color=cmap(base_idx))

    # salvar CSV por cenário
    if rows_csv:
        rows_csv = sorted(rows_csv, key=lambda r: r["phi"])
        out_csv = f"dist_minima_vs_phi_{base_dirname}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            for r in rows_csv:
                w.writerow(r)
        print(f"[OK] CSV salvo: {out_csv}")

# salvar CSV combinado (φ agregado)
if rows_all_phi:
    rows_all_phi = sorted(rows_all_phi, key=lambda r: (r["scenario"], r["phi"]))
    out_all = "dist_minima_vs_phi__todas_bases.csv"
    with open(out_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all_phi[0].keys()))
        w.writeheader()
        for r in rows_all_phi:
            w.writerow(r)
    print(f"[OK] CSV salvo: {out_all}")

# salvar CSV por run (opcional)
if rows_all_runs:
    rows_all_runs = sorted(rows_all_runs, key=lambda r: (r["scenario"], r["phi"], r["run"]))
    out_runs = "dist_minima_por_run__todas_bases.csv"
    with open(out_runs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all_runs[0].keys()))
        w.writeheader()
        for r in rows_all_runs:
            w.writerow(r)
    print(f"[OK] CSV salvo: {out_runs}")

# ---------------------- Plot ----------------------
plt.axvline(x=0.59, color='black', linestyle='--', linewidth=1.5, label='$\phi_{c} \\approx 0.59$')
plt.xlabel("$\\phi$")
plt.ylabel("$\\langle d \\rangle$")
#suffix = "caçador→presa" if MODO == "por_cacador" else "presa→caçador"
#plt.title(f"$\\langle d \\rangle$ ({suffix}) vs $\\phi$")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("<d>_vs_phi.pdf", dpi=200)
plt.show()
