from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv

# ---------------------- parâmetros ----------------------
obs_list = ["obs_00", "obs_1638", "obs_3276", "obs_4915", "obs_6553",
            "obs_8192", "obs_9830", "obs_11468", "obs_13107", "obs_14745"]

bases = [
    (r"$N_C=0.5\,N_P$", "Nc=Np*0.5"),
    (r"$N_C=0.8\,N_P$", "Nc=Np*0.8"),
    (r"$N_C=N_P$",      "Nc=Np"),
]

L = 128
area = L**2
NUM_RUNS = 100
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"

# limiar de sítios livres na rede quadrada: p_c ≈ 0.592746
PC_SITE_SQUARE = 0.592746
PHI_CRIT_OBST = 1.0 - PC_SITE_SQUARE  # ~0.407254
PHI_CRIT_OBST = 0.592746

# Paleta discreta p/ 3 cenários (cores bem distintas)
cmap = plt.get_cmap("flag")
# --------------------------------------------------------

def ler_coords_txt(fp: Path):
    """Lê coordenadas (x y) de um txt; ignora comentários/linhas vazias."""
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

def grafo_livre(L, obst_set):
    """Grafo 2D LxL (4-vizinhos) com nós livres (obstáculos removidos)."""
    G = nx.grid_2d_graph(L, L)  # nós = (x,y)
    if obst_set:
        G.remove_nodes_from([p for p in obst_set if in_bounds(p, L)])
    return G

def stats_componentes_from_obst(obst_set, L):
    """Calcula métricas de componentes conectadas dado o conjunto de obstáculos."""
    n_obst = len(obst_set)
    n_livres = L*L - n_obst
    if n_livres <= 0:
        return dict(
            n_comp=0, n_livres=0, giant_size=0, giant_frac=np.nan,
            mean_size_no_giant=np.nan, S_susc=np.nan,
        )

    G = grafo_livre(L, obst_set)
    comps = list(nx.connected_components(G))
    sizes = sorted((len(c) for c in comps), reverse=True)

    n_comp = len(sizes)
    giant_size = sizes[0] if sizes else 0
    giant_frac = giant_size / n_livres if n_livres > 0 else np.nan

    others = sizes[1:] if len(sizes) > 1 else []
    mean_size_no_giant = (float(np.mean(others)) if others else np.nan)
    # Susceptibilidade (exclui a gigante): sum s^2 / sum s
    if others:
        sum_s = float(np.sum(others))
        sum_s2 = float(np.sum([s*s for s in others]))
        S_susc = sum_s2 / sum_s if sum_s > 0 else np.nan
    else:
        S_susc = np.nan

    return dict(
        n_comp=n_comp,
        n_livres=n_livres,
        giant_size=giant_size,
        giant_frac=float(giant_frac),
        mean_size_no_giant=mean_size_no_giant,
        S_susc=S_susc,
    )

def extrai_num_obs(nome_obs: str) -> int:
    return int(nome_obs.split("_")[1])

def densidade_obs(num_obs: int) -> float:
    return num_obs / area

# ---------------------- Agregação por ϕ ----------------------
cmap = plt.cm.viridis

# figuras
fig1, ax1 = plt.subplots(figsize=(10, 6))  # giant_frac vs phi
fig2, ax2 = plt.subplots(figsize=(10, 6))  # n_comp vs phi

for base_idx, (label_tex, base_dirname) in enumerate(bases):
    base_path = BASE_ROOT / base_dirname

    xs_phi = []
    mean_giant_frac = []
    std_giant_frac  = []

    xs_phi_n = []
    mean_n_comp = []
    std_n_comp  = []

    # também vamos salvar CSV com mais métricas
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

        phi = densidade_obs(extrai_num_obs(obs_folder))

        vals_giant_frac = []
        vals_n_comp = []
        vals_mean_no_giant = []
        vals_S = []

        for i in range(NUM_RUNS):
            pasta_run = pasta_nC / f"run_{i:02d}"
            if not pasta_run.exists():
                alt = pasta_nC / f"run_{i}"
                if alt.exists():
                    pasta_run = alt
                else:
                    continue

            obst_fp = pasta_run / "obstacules.txt"
            obsts = [p for p in ler_coords_txt(obst_fp) if in_bounds(p, L)]
            obst_set = set(obsts)

            st = stats_componentes_from_obst(obst_set, L)
            if np.isfinite(st["giant_frac"]):
                vals_giant_frac.append(st["giant_frac"])
            if st["n_comp"] is not None:
                vals_n_comp.append(st["n_comp"])
            if np.isfinite(st["mean_size_no_giant"]):
                vals_mean_no_giant.append(st["mean_size_no_giant"])
            if np.isfinite(st["S_susc"]):
                vals_S.append(st["S_susc"])

        if vals_giant_frac:
            xs_phi.append(phi)
            mean_giant_frac.append(float(np.mean(vals_giant_frac)))
            std_giant_frac.append(float(np.std(vals_giant_frac, ddof=1)) if len(vals_giant_frac) > 1 else 0.0)

        if vals_n_comp:
            xs_phi_n.append(phi)
            mean_n_comp.append(float(np.mean(vals_n_comp)))
            std_n_comp.append(float(np.std(vals_n_comp, ddof=1)) if len(vals_n_comp) > 1 else 0.0)

        # adicionar linha ao CSV (médias por ϕ)
        rows_csv.append({
            "phi": phi,
            "num_obs": extrai_num_obs(obs_folder),
            "giant_frac_mean": (np.mean(vals_giant_frac) if vals_giant_frac else np.nan),
            "giant_frac_std": (np.std(vals_giant_frac, ddof=1) if len(vals_giant_frac) > 1 else 0.0),
            "n_comp_mean": (np.mean(vals_n_comp) if vals_n_comp else np.nan),
            "n_comp_std": (np.std(vals_n_comp, ddof=1) if len(vals_n_comp) > 1 else 0.0),
            "mean_size_no_giant_mean": (np.mean(vals_mean_no_giant) if vals_mean_no_giant else np.nan),
            "mean_size_no_giant_std": (np.std(vals_mean_no_giant, ddof=1) if len(vals_mean_no_giant) > 1 else 0.0),
            "S_susc_mean": (np.mean(vals_S) if vals_S else np.nan),
            "S_susc_std": (np.std(vals_S, ddof=1) if len(vals_S) > 1 else 0.0),
        })

    # ordenar e plotar por cenário
    if mean_giant_frac:
        xs_phi, mean_giant_frac, std_giant_frac = zip(*sorted(zip(xs_phi, mean_giant_frac, std_giant_frac)))
        ax1.errorbar(xs_phi, mean_giant_frac, yerr=std_giant_frac,
                     fmt='o-', capsize=5, markersize=5,
                     color=cmap(base_idx),   # em plt.errorbar(...),
                     label=label_tex)

    if mean_n_comp:
        xs_phi_n, mean_n_comp, std_n_comp = zip(*sorted(zip(xs_phi_n, mean_n_comp, std_n_comp)))
        ax2.errorbar(xs_phi_n, mean_n_comp, yerr=std_n_comp,
                     fmt='o-', capsize=5, markersize=5,
                     color=cmap(base_idx),   # em plt.errorbar(...),
                     label=label_tex)

    # salvar CSV do cenário
    out_csv = f"componentes_conectadas_{base_dirname}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        for r in sorted(rows_csv, key=lambda r: r["phi"]):
            w.writerow(r)
    print(f"[OK] CSV salvo: {out_csv}")

# configs dos gráficos
for ax in (ax1, ax2):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_xlabel(r"$\phi$")
    ax.axvline(PHI_CRIT_OBST, color="black", linestyle="--", linewidth=1.2,
               label=r"$\phi_c \approx {:.3f}$".format(PHI_CRIT_OBST))

ax1.set_ylabel(r"$S_1$")
ax1.set_title("$S_1$ vs $\\phi$")

ax2.set_ylabel("number of components(N_c)")
ax2.set_title("$N_c$ vs $\\phi$")

fig1.tight_layout(); fig2.tight_layout()
fig1.savefig("componentes_gigante_vs_phi.pdf", dpi=200)
fig2.savefig("componentes_numero_vs_phi.pdf", dpi=200)
plt.show()
