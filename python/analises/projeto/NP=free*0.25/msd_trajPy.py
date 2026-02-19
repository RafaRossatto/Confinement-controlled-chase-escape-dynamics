from pathlib import Path
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from trajpy.trajpy import Trajectory

# ---------------------- Logging configuration ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- General Configurations --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"
pattern = "*_hunter_trajectories.csv"
Lx = Ly = 128
min_traj_length = 5

# Output
OUT_DIR = BASE_ROOT / "resultados_modelos" / "paper_response"/"msd_trajPy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Concentrations
CONCENTRACOES = ["Nc=Np"]#$, "Nc=Np*0.8", "Nc=Np"]

# Obstacles
OBSTACULOS = [
    "s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915",
    "s_obs_6553", "s_obs_8028","s_obs_8192","s_obs_8355","s_obs_9666", "s_obs_9830", "s_obs_9994",
    "s_obs_11468", "s_obs_13107"
]

# ---------------------- Load file into DataFrame -------------------
def load_trajectory(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        #logger.info(f"File loaded successfully: {file_path.name}")
        #logger.info(f"DataFrame shape: {df.shape}")
        #logger.info(f"Columns detected: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load file {file_path.name}: {e}")
        raise

# ---------------------- Unwrap function -----------------------------
def unwrap_trajectory(df_particle: pd.DataFrame, L: int) -> pd.DataFrame:
    df_particle = df_particle.sort_values("timestep").copy()

    x_unwrapped = [df_particle["x"].iloc[0]]
    y_unwrapped = [df_particle["y"].iloc[0]]

    for i in range(1, len(df_particle)):
        dx = df_particle["x"].iloc[i] - df_particle["x"].iloc[i - 1]
        dy = df_particle["y"].iloc[i] - df_particle["y"].iloc[i - 1]

        if dx > L / 2:
            dx -= L
        elif dx < -L / 2:
            dx += L

        if dy > L / 2:
            dy -= L
        elif dy < -L / 2:
            dy += L

        x_unwrapped.append(x_unwrapped[-1] + dx)
        y_unwrapped.append(y_unwrapped[-1] + dy)

    df_particle["x_unwrapped"] = x_unwrapped
    df_particle["y_unwrapped"] = y_unwrapped

    return df_particle

if __name__ == "__main__":
    #logger.info("Starting ensemble MSD computation for all scenarios...")

    for conc in CONCENTRACOES:
        for obs in OBSTACULOS:
            # Input directory
            input_dir = BASE_ROOT / conc / obs
            files = sorted(input_dir.glob(pattern))

            if not files:
                logger.warning(f"No trajectory files found in {input_dir}")
                continue

            # Output directory (mirror structure)
            output_dir = OUT_DIR / conc / obs
            output_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                #logger.info(f"Processing {file_path.name}...")

                # Load trajectory data
                df = load_trajectory(file_path)

                # Time lags
                tau_values = np.arange(1, df["timestep"].nunique())

                msd_list = []

                # Loop over particles
                for pid, group in df.groupby("cell_id"):
                    df_unwrapped = unwrap_trajectory(group, Lx)
                    positions = df_unwrapped[["x_unwrapped", "y_unwrapped"]].values
                    if len(positions) < min_traj_length:
                        continue  # skip very short trajectories

                    max_tau = min(len(positions), len(tau_values) + 1)
                    msd_pid = Trajectory.msd_time_averaged_(
                        positions, tau_values[:max_tau-1]
                    )
                    msd_list.append(msd_pid)

                if not msd_list:
                    logger.warning(f"No valid trajectories in {file_path}")
                    continue

                # Ensemble average
                msd_array = np.vstack(msd_list)
                msd_ensemble = msd_array.mean(axis=0)

                # Save results
                out_file = output_dir / f"{file_path.stem}_msd.csv"
                df_out = pd.DataFrame({
                    "tau": tau_values[:len(msd_ensemble)],
                    "msd_ensemble": msd_ensemble
                })
                df_out.to_csv(out_file, index=False)
                #logger.info(f"Saved MSD results to {out_file}")
