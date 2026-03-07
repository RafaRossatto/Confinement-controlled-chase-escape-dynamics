import numpy as np, pandas as pd
df = pd.read_csv("/home/rrossatto/Dados_Doc/Np=free*0.25/Nc=Np*0.5/s_obs_00/NC_2048_NE_4096_O_0_TCC_1.000000_SR_2_TCT_1.000000_SR_2.dat_run_35_hunter_trajectories.csv")
df = df.sort_values(["cell_id","timestep"])

dx = df.groupby("cell_id")["x"].diff().to_numpy()
dy = df.groupby("cell_id")["y"].diff().to_numpy()

L = 128
dx -= np.round(dx / L) * L
dy -= np.round(dy / L) * L

step = np.sqrt(dx**2 + dy**2)
print("max step =", np.nanmax(step))

# --- debug: localizar saltos maiores que 1.5 ---
bad = np.where(step > 1.5)[0]
print("Saltos suspeitos encontrados:", bad.size)
print(df.iloc[bad[:10]])  # primeiras linhas suspeitas
print(dx[bad[:10]], dy[bad[:10]], step[bad[:10]])
