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

# ---------------------- CONFIGURAÇÕES PRINCIPAIS ----------------------
# Diferentes tamanhos de rede - ⬅️ ALTERE OS CAMINHOS AQUI
REDE_CONFIGS = [
    #(64,  "L_64",  Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE AQUI
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"),   # ⬅️ ALTERE AQUI  
    #(256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")    # ⬅️ ALTERE AQUI
]

# Apenas a proporção 0.5
FRAC_C = "Nc=Np*0.5"

# Lista de obstáculos para cada tamanho de rede (ajuste conforme necessário)
OBS_TEMPLATES = {
    #64: ["s_obs_00", "s_obs_409", "s_obs_819", "s_obs_1228", "s_obs_1638", "s_obs_2048","s_obs_2416",
    # "s_obs_2457","s_obs_2498","s_obs_2867","s_obs_3276"],
    128: ["s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", "s_obs_6553", "s_obs_8192", "s_obs_9666",
    "s_obs_9830", "s_obs_9994", "s_obs_11468", "s_obs_13107"],
    #256: ["s_obs_00", "s_obs_6553", "s_obs_13107", "s_obs_19660", "s_obs_26214", "s_obs_32768", "s_obs_38666",
    #"s_obs_39321", "s_obs_39976", "s_obs_45875", "s_obs_52428"]  # exemplo
}

pattern = "*_hunter_trajectories.csv"
min_traj_length = 5

# Output principal
OUT_BASE = Path.home() / "Dados_Doc" / "resultados_modelos" / "msd_trajPy_multiple_L"
OUT_BASE.mkdir(parents=True, exist_ok=True)

# ---------------------- Funções (mantidas) -------------------
def load_trajectory(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Failed to load file {file_path.name}: {e}")
        raise

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

# ---------------------- Pipeline principal -------------------
if __name__ == "__main__":
    logger.info("Starting ensemble MSD computation for multiple network sizes...")

    for L, rede_nome, base_path in REDE_CONFIGS:
        # Verifica se a pasta existe
        if not base_path.exists():
            logger.error(f"Pasta não encontrada: {base_path}")
            logger.error("Por favor, ajuste os caminhos em REDE_CONFIGS")
            continue
            
        logger.info(f"🎯 Processando rede: {rede_nome} (L={L})")
        logger.info(f"📁 Caminho: {base_path}")

        # Lista de obstáculos para este L
        obs_list = OBS_TEMPLATES.get(L, [f"s_obs_{i}" for i in range(0, L**2, L**2//8)])

        for obs in obs_list:
            # Input directory - estrutura: base_path / FRAC_C / obs
            input_dir = base_path / FRAC_C / obs
            files = sorted(input_dir.glob(pattern))

            if not files:
                logger.warning(f"No trajectory files found in {input_dir}")
                continue

            # Output directory - estrutura: OUT_BASE / rede_nome / FRAC_C / obs
            output_dir = OUT_BASE / rede_nome / FRAC_C / obs
            output_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                logger.info(f"Processing {file_path.name}...")

                # Load trajectory data
                df = load_trajectory(file_path)

                # Time lags
                tau_values = np.arange(1, df["timestep"].nunique())

                msd_list = []

                # Loop over particles
                for pid, group in df.groupby("cell_id"):
                    df_unwrapped = unwrap_trajectory(group, L)  # ⚠️ Usa L da rede atual
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
                logger.info(f"Saved MSD results to {out_file}")

    logger.info("✅ Processamento concluído para todas as redes!")