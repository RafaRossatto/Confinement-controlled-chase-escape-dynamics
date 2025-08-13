import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from typing import Optional, Dict, List

# ===== Parâmetros =====
base_root = Path.home() / "Dados_Doc"
n_obs = 3276          # funciona com s_obs_0, s_obs_00, s_obs_000, etc.
run_desejado = 25 # tente usar este run nos dois cenários

# Aceita variações do nome da pasta do cenário
cenarios = [
    (r"$N_C = N_P$",      ["Nc=Np"]),
    (r"$N_C = 0.5\,N_P$", ["Nc=Np*0.5", "Nc= Np*0.5"]),
]

# ===== Helpers =====
def find_obs_dir(base_root: Path, subpasta: str, n_obs: int) -> Optional[Path]:
    """
    Retorna a pasta s_obs_* cujo número bate com n_obs (aceita s_obs_0, s_obs_00, s_obs_000, ...).
    """
    base = base_root / subpasta
    if not base.exists():
        return None
    for d in base.glob("s_obs_*"):
        if d.is_dir():
            m = re.fullmatch(r"s_obs_(\d+)", d.name)
            if m and int(m.group(1)) == n_obs:
                return d
    # fallbacks diretos, caso não ache via varredura
    for name in (f"s_obs_{n_obs}", f"s_obs_{n_obs:02d}", f"s_obs_{n_obs:03d}"):
        d = base / name
        if d.exists():
            return d
    return None

# não fixamos O_{n_obs} aqui: já estamos dentro da s_obs_* correta
RE_RUN = re.compile(r"_run_(\d+)_presas_por_passo\.csv$")

def listar_por_run(dir_obs: Path) -> Dict[int, List[Path]]:
    """
    Varre dir_obs e devolve {run:int -> [Paths...]} para arquivos ..._run_XXX_presas_por_passo.csv
    """
    por_run: Dict[int, List[Path]] = {}
    for p in dir_obs.rglob("**/NC_*_O_*_run_*_presas_por_passo.csv"):
        m = RE_RUN.search(p.name)
        if not m:
            continue
        r = int(m.group(1))
        por_run.setdefault(r, []).append(p)
    return por_run

def escolher_mais_recente(paths: List[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)

def escolher_arquivo_por_run(por_run: Dict[int, List[Path]], r: int) -> Optional[Path]:
    """Devolve 1 Path para o run r (o mais recente, se houver múltiplos)."""
    if r in por_run and por_run[r]:
        return escolher_mais_recente(por_run[r])
    return None

def carregar_pacote(label: str, subpastas: List[str], n_obs: int, run_alvo: Optional[int] = None):
    """
    Tenta carregar arquivos de um cenário (podem existir variantes de subpasta).
    Retorna dict com: {label, dir_obs, por_run, escolhido(Path|None), run_escolhido(int|None)}
    """
    dir_obs = None
    for sub in subpastas:
        tentativa = find_obs_dir(base_root, sub, n_obs)
        if tentativa is not None:
            dir_obs = tentativa
            break
    if dir_obs is None:
        return {"label": label, "dir_obs": None, "por_run": {}, "escolhido": None, "run_escolhido": None}

    por_run = listar_por_run(dir_obs)
    escolhido = None
    usado = None
    if run_alvo is not None:
        escolhido = escolher_arquivo_por_run(por_run, run_alvo)
        usado = run_alvo if escolhido is not None else None

    return {"label": label, "dir_obs": dir_obs, "por_run": por_run, "escolhido": escolhido, "run_escolhido": usado}

def ajustar_lnC_vs_t(df: pd.DataFrame):
    mask = df["presas_vivas"] > 0
    t = df.loc[mask, "passo"].to_numpy()
    lnC = np.log(df.loc[mask, "presas_vivas"].to_numpy())
    if len(t) < 2:
        return None
    slope, intercept, r_value, p_value, std_err = linregress(t, lnC)
    tau = -1.0 / slope if slope != 0 else np.inf
    return {"a": intercept, "b": slope, "tau": tau, "R2": r_value**2, "n": len(t), "t": t, "lnC": lnC}

# ===== Carregar info dos dois cenários =====
packs = []
for label, subs in cenarios:
    pack = carregar_pacote(label, subs, n_obs, run_alvo=run_desejado)
    packs.append(pack)

# Debug: onde procurou e o que achou
print("\n[debug] Raízes e runs disponíveis:")
for p in packs:
    print(f"  {p['label']}: dir={p['dir_obs']}")
    if p["por_run"]:
        print(f"    runs: {sorted(p['por_run'].keys())[:20]}{' ...' if len(p['por_run'])>20 else ''}")
    else:
        print("    (nenhum arquivo encontrado)")

# ===== Escolher runs para garantir DUAS curvas =====
tem_ambos_desejado = all(p["escolhido"] is not None for p in packs)
if not tem_ambos_desejado:
    conjuntos = [set(p["por_run"].keys()) for p in packs if p["por_run"]]
    comum = set.intersection(*conjuntos) if len(conjuntos) == 2 else set()
    if comum:
        run_comum = max(comum)  # usa o maior em comum
        for p in packs:
            p["escolhido"] = escolher_arquivo_por_run(p["por_run"], run_comum)
            p["run_escolhido"] = run_comum
    else:
        # sem interseção: usa o mais recente de cada cenário
        for p in packs:
            if not p["por_run"]:
                continue
            candidatos = [escolher_mais_recente(v) for v in p["por_run"].values()]
            if candidatos:
                escolhido = escolher_mais_recente(candidatos)
                m = RE_RUN.search(escolhido.name)
                p["escolhido"] = escolhido
                p["run_escolhido"] = int(m.group(1)) if m else None

packs = [p for p in packs if p["escolhido"] is not None]
if len(packs) < 2:
    raise FileNotFoundError("Não consegui formar par de curvas — um dos cenários não tem arquivo para este obs.")

# ===== Plot 1: Presas vivas x Passo =====
plt.figure(figsize=(8,5))
for p in packs:
    df = pd.read_csv(p["escolhido"])
    p["df"] = df
    rot = f"{p['label']} (run={p['run_escolhido']})"
    plt.plot(df["passo"], df["presas_vivas"], marker='o', markersize=2, linewidth=1, label=rot)
plt.xlabel("Passos")
plt.ylabel("Presas vivas")
plt.title(f"Presas vivas por passo | $\phi = ${round(n_obs / (128**2), 1)}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== Plot 2: ln(C) x t + ajuste =====
plt.figure(figsize=(8,5))
for p in packs:
    df = p["df"]
    mask = df["presas_vivas"] > 0
    t = df.loc[mask, "passo"].to_numpy()
    lnC = np.log(df.loc[mask, "presas_vivas"].to_numpy())
    rot = f"Dados {p['label']} (run={p['run_escolhido']})"
    plt.scatter(t, lnC, s=10, label=rot)

    fit = ajustar_lnC_vs_t(df)
    if fit is not None:
        a, b, tau, R2 = fit["a"], fit["b"], fit["tau"], fit["R2"]
        plt.plot(t, a + b*t, linewidth=1.2,
                 label=f"Ajuste {p['label']}: $\\tau$={tau:.2f}, $R^2$={R2:.3f}")

plt.xlabel("Passos (t)")
plt.ylabel("ln(C)")
plt.title(f"Ajuste ln(C) = ln(C0) - t/τ | $\phi = ${round(n_obs / (128**2),1)}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
