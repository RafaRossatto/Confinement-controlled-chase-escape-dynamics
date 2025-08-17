from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# === Parâmetros ===
L = 128
roots = [
    ("Nc=Np",       Path("/home/rafarossatto/Dados_Doc/Nc=Np")),
    ("Nc=Np*0.5",   Path("/home/rafarossatto/Dados_Doc/Nc=Np*0.5")),  # ajuste se o nome for diferente
]

# === Utilitários ===
def ler_dat(fp: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(fp, sep=None, engine="python", comment="#")
        ok = {"run","steps","escapers","seed"}.issubset(df.columns)
        if not ok:
            raise ValueError("Cabeçalho inesperado")
    except Exception:
        df = pd.read_csv(
            fp, delim_whitespace=True, header=None,
            names=["run","steps","escapers","seed"], comment="#",
        )
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    return df.dropna(subset=["steps"])

def extrai(padrao: str, texto: str, default=None, cast=int):
    m = re.search(padrao, texto)
    return cast(m.group(1)) if m else default

def coletar(base_root: Path, label: str) -> pd.DataFrame:
    linhas = []
    subdirs = [d for d in base_root.iterdir() if d.is_dir() and d.name.startswith("s_obs_")]
    for sdir in sorted(subdirs):
        cand = list(sdir.glob("NC_*_NE_*_O_*_TCC_*_SR_*_TCT_*_SR_*.dat"))
        if not cand:
            print(f"[{label}] [!] Sem .dat no formato em {sdir}")
            continue
        cand.sort()
        fp = cand[0]

        df = ler_dat(fp)
        media = df["steps"].mean()
        std   = df["steps"].std(ddof=1)
        n     = len(df)

        O = extrai(r"_O_(\d+)_", fp.name, default=extrai(r"s_obs_(\d+)$", sdir.name, default=None))
        if O is None:
            print(f"[{label}] [!] Não consegui obter O em {fp.name}; pulando…")
            continue

        linhas.append({
            "cenario": label,
            "pasta": sdir.name,
            "O": int(O),
            "frac_O_L2": O / float(L**2),
            "n": n,
            "mean_steps": media,
            "std_steps": std,
            "arquivo": str(fp),
        })
    return pd.DataFrame(linhas).sort_values("frac_O_L2").reset_index(drop=True)

# === Coleta para os dois cenários ===
tabs = []
for label, root in roots:
    if root.exists():
        tabs.append(coletar(root, label))
    else:
        print(f"[!] Raiz não encontrada: {root} (pulando {label})")

if not tabs:
    raise SystemExit("[!] Nada encontrado em nenhuma raiz.")
df = pd.concat(tabs, ignore_index=True)

# === Normalização pelo MÁXIMO de cada curva (por cenário) ===
baselines = df.groupby("cenario")["mean_steps"].max().to_dict()
print("[info] Máximos por cenário:", baselines)

def norm_row(r):
    b = baselines.get(r["cenario"], np.nan)
    if not (isinstance(b, (int,float)) and b > 0):
        return pd.Series({"mean_norm": np.nan, "std_norm": np.nan})
    return pd.Series({
        "mean_norm": r["mean_steps"] / b,
        "std_norm":  r["std_steps"]  / b,
    })

df[["mean_norm","std_norm"]] = df.apply(norm_row, axis=1)

# === Plot com barras de erro ===
plt.figure(figsize=(8,5))

# leve deslocamento no x pra não sobrepor barras idênticas
labels = list(df["cenario"].unique())

for lab in labels:
    dfl = df[df["cenario"] == lab].sort_values("frac_O_L2")
    x = dfl["frac_O_L2"].values 
    plt.errorbar(x, dfl["mean_norm"], yerr=dfl["std_norm"],
                 fmt="o-", capsize=4, markersize=5, label=lab)

#plt.axhline(1.0, linestyle="--", linewidth=1)
plt.ylim(bottom=0)
plt.xlabel(r"$\phi$")
plt.ylabel("$TT$")
plt.title(r"$TT x \phi $")
plt.grid(True, alpha=0.3)
plt.legend(title="Cenário")
plt.tight_layout()

out_dir = Path("/home/rafarossatto/Dados_Doc")
out_png = out_dir / f"steps_vs_frac_obst_norm_por_curva_L{L}.png"
out_csv = out_dir / f"steps_norm_por_curva_L{L}.csv"
plt.savefig(out_png, dpi=150)
df.to_csv(out_csv, index=False)
print(f"[ok] Figura: {out_png}")
print(f"[ok] Tabela: {out_csv}")

plt.show()
