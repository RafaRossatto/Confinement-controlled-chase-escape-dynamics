import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path

# -------- Config --------
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"/"L_128"
MSD_DIR = BASE_ROOT / "resultados_modelos" /"paper_response"/ "msd_trajPy"

OBSTACULOS = [
    "s_obs_00", "s_obs_1638", "s_obs_3276",
    "s_obs_4915", "s_obs_6553", "s_obs_8192","s_obs_9666","s_obs_9994"
    "s_obs_9830", "s_obs_11468", "s_obs_13107"
]

def get_input_dir(agent_density: float, obs_tag: str) -> Path:
    """
    Build the path to MSD results given agent density and obstacle density.
    - agent_density: 1.0, 0.8, 0.5
    - obs_tag: e.g. "s_obs_4915"
    """
    conc_dir = "Nc=Np" if agent_density == 1.0 else f"Nc=Np*{agent_density}"
    return MSD_DIR / conc_dir / obs_tag

def plot_random_curves(agent_density: float, obs_tag: str, n_curves: int = 5,
                       loglog: bool = True, vline_x: float = None):
    """
    Plot up to n_curves MSD curves for a given agent density and obstacle tag.
    If there are more than n_curves files, a random subset is chosen.

    Parameters
    ----------
    agent_density : float
        1.0, 0.8, or 0.5
    obs_tag : str
        e.g. "s_obs_4915"
    n_curves : int
        Maximum number of curves to plot (randomly sampled if too many).
    loglog : bool
        If True, set x and y axes to log scale.
    vline_x : float, optional
        If provided, draw a vertical line at this x value.
    """
    input_dir = get_input_dir(agent_density, obs_tag)
    files = sorted(input_dir.glob("*_msd.csv"))

    print(f"Found {len(files)} MSD files in {input_dir}")
    if not files:
        return

    if n_curves and len(files) > n_curves:
        files = random.sample(files, n_curves)

    plt.figure(figsize=(7,5))
    for f in files:
        df = pd.read_csv(f)
        plt.plot(df["tau"], df["msd_ensemble"], alpha=0.6)

    plt.xlabel("Time lag Δt")
    plt.ylabel("MSD (ensemble)")
    plt.title(f"MSD curves — Nc=Np*{agent_density}, {obs_tag}")

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    if vline_x is not None:
        # force x-limits to include vline_x if necessário
        xmin, xmax = plt.xlim()
        new_xmin = min(xmin, vline_x * 0.9)
        new_xmax = max(xmax, vline_x * 1.1)
        plt.xlim(new_xmin, new_xmax)

        plt.axvline(x=vline_x, color="black", linestyle="--", linewidth=1.2)

    plt.tight_layout()
    plt.show()

# -------- Run example --------
if __name__ == "__main__":
    n_curves=100
    obs_tag="s_obs_8355" #00, 1638, 3276, 4915, 6553,8028, 8192,8355, 9830, 11468, 13107
    agent_density=0.8
    vline_x=13
    loglog = True
    plot_random_curves(agent_density, obs_tag, n_curves,loglog,vline_x)
