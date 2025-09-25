#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EA-MSD 2D com TrajPy
---------------------
- Lê *_hunter_trajectories.csv
- Unwrap CPC (Lx=Ly=128)
- Junta todas as partículas em um Trajectory único
- Calcula o EA-MSD com trajpy.Trajectory.msd_ensemble_averaged
- Salva CSV: tau, msd, n_contrib
"""

from pathlib import Path
import numpy as np
import pandas as pd
from trajpy.trajpy import Trajectory

# ================= Configurações =================
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
PATTERN = "*_hunter_trajectories.csv"
OUT_DIR = BASE_ROOT / "resultados_modelos" / "msd_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Lx = Ly = 128
MAX_TAU = 50
MIN_TRAJ_LENGTH = 5

# ================= Funções =================
def unwrap_1d(x, L):
    x = np.asarray(x, float)
    dx = np.diff(x)
    dx -= np.round(dx / L) * L
    xu = np.empty_like(x)
    xu[0] = x[0]
    xu[1:] = xu[0] + np.cumsum(dx)
    return xu

def unwrap_xy(x, y, Lx, Ly):
    return unwrap_1d(x, Lx), unwrap_1d(y, Ly)

def processar_arquivo(arq: Path):
    print(f"[INFO] Processando {arq.name}")
    df = pd.read_csv(arq, usecols=["timestep", "cell_id", "x", "y"])
    df = df.sort_values(["cell_id", "timestep"])

    # Pivotar: garantir mesmo nº de steps por partícula
    trajetorias = []
    for cid, g in df.groupby("cell_id"):
        x, y = unwrap_xy(g["x"].to_numpy(float), g["y"].to_numpy(float), Lx, Ly)
        if len(x) < MIN_TRAJ_LENGTH:
            continue
        traj = np.column_stack([x, y])  # shape (Nsteps, 2)
        trajetorias.append(traj)

    if not trajetorias:
        print(f"[WARN] Nenhuma trajetória válida em {arq.name}")
        return None

    # Padronizar tamanho (usar min_length comum)
    min_len = min(len(tr) for tr in trajetorias)
    trajs_cut = [tr[:min_len] for tr in trajetorias]

    # Empilhar partículas em um único array: (Nsteps, Npart*2)
    big_traj = np.hstack(trajs_cut)  # concatena colunas
    tr = Trajectory(big_traj)
    msd = tr.msd_ensemble_averaged_(maxtau=MAX_TAU)

    taus = np.arange(1, len(msd) + 1)
    df_out = pd.DataFrame({"tau": taus, "msd": msd})
    out_csv = OUT_DIR / f"{arq.stem}_EAmsd.csv"
    df_out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[OK] Salvo: {out_csv}")
    return out_csv

# ================= Main =================
def main():
    arquivos = sorted(BASE_ROOT.rglob(PATTERN))
    if not arquivos:
        print("[ERRO] Nenhum arquivo encontrado!")
        return

    for arq in arquivos:
        processar_arquivo(arq)

if __name__ == "__main__":
    main()
