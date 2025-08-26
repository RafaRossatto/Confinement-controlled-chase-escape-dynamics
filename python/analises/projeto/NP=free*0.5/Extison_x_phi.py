from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# === Parâmetros ===
L = 128
base_path = Path.home() / "Dados_Doc"

roots = [
    ("Nc=Np",     base_path / "Nc=Np"),
    ("Nc=Np*0.5", base_path / "Nc=Np*0.5"),
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
    df["escapers"] = pd.to_numeric(df["escapers"], errors="coerce")
    return df.dropna(subset=["steps","escapers"])

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
        zeros_esc = (df["escapers"] == 0).sum() / 100.0   # <<=== conta zeros e divide

        O = extrai(r"_O_(\d+)_", fp.name, default=extrai(r"s_obs_(\d+)$", sdir.name, default=None))
        if O is None:
            print(f"[{label}] [!] Não consegui obter O em {fp.name}; pulando…")
            continue

        linhas.append({
            "cenario": label,
            "pasta": sdir.name,
            "O": int(O),
            "frac_O_L2": O / float(L**2),
            "zeros_escapers": zeros_esc,
            "arquivo": str(fp),
        })
    return pd.DataFrame(linhas).sort_values("frac_O_L2").reset_index(drop=True)

# === Coleta ===
tabs = []
for label, root in roots:
    if root.exists():
        tabs.append(coletar(root, label))
    else:
        print(f"[!] Raiz não encontrada: {root} (pulando {label})")

if not tabs:
    raise SystemExit("[!] Nada encontrado em nenhuma raiz.")
df = pd.concat(tabs, ignore_index=True)

# === Plot: número de zeros de escapers/100 vs φ ===
plt.figure(figsize=(8,5))
for lab in df["cenario"].unique():
    dfl = df[df["cenario"] == lab].sort_values("frac_O_L2")
    plt.plot(dfl["frac_O_L2"], dfl["zeros_escapers"], "o-", label=lab)

plt.xlabel(r"$\phi$")
plt.ylabel("Zeros escapers / 100")
#plt.title("Número de vezes que escapers = 0 (por arquivo, dividido por 100)")
plt.legend(title="Cenário")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
