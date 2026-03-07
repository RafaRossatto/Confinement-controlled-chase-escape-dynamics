import matplotlib.pyplot as plt
import pandas as pd
import random
from pathlib import Path

# -------- CONFIGURAÇÕES PRINCIPAIS --------
# Diferentes tamanhos de rede - ⬅️ ALTERE OS CAMINHOS AQUI
REDE_CONFIGS = [
    (64,  "L_64",  Path.home() / "Dados_Doc/Np=free*0.25/L_64"),    # ⬅️ ALTERE
    (128, "L_128", Path.home() / "Dados_Doc/Np=free*0.25/L_128"),   # ⬅️ ALTERE  
    (256, "L_256", Path.home() / "Dados_Doc/Np=free*0.25/L_256")    # ⬅️ ALTERE
]

# Apenas a proporção 0.5
FRAC_C = "Nc=Np*0.5"

# Diretório base dos resultados MSD
BASE_ROOT = Path.home() / "Dados_Doc" / "resultados_modelos"
MSD_DIR_BASE = BASE_ROOT / "msd_trajPy_multiple_L"

# Lista de obstáculos para cada tamanho de rede (ajuste conforme necessário)
OBS_TEMPLATES = {
    64: ["s_obs_00", "s_obs_819", "s_obs_1638", "s_obs_2457", "s_obs_3276"],
    128: ["s_obs_00", "s_obs_1638", "s_obs_3276", "s_obs_4915", "s_obs_6553", 
          "s_obs_8192", "s_obs_9830", "s_obs_11468", "s_obs_13107"],
    256: ["s_obs_00", "s_obs_6553", "s_obs_13107", "s_obs_19660", "s_obs_26214"]
}

def get_input_dir(rede_nome: str, num_obs: int) -> Path:
    """
    Build the path to MSD results given network name and number of obstacles.
    
    Parameters
    ----------
    rede_nome : str
        "L64", "L128", or "L256"
    num_obs : int
        Number of obstacles (e.g., 0, 1638, 3276, etc.)
    """
    # Encontra o nome da pasta do obstáculo correspondente
    for L, nome, path in REDE_CONFIGS:
        if nome == rede_nome:
            obs_list = OBS_TEMPLATES.get(L, [])
            # Procura pelo obstáculo com o número especificado
            for obs_tag in obs_list:
                if str(num_obs) in obs_tag:
                    return MSD_DIR_BASE / rede_nome / FRAC_C / obs_tag
    
    # Se não encontrou, tenta construir o nome
    obs_tag = f"s_obs_{num_obs}"
    return MSD_DIR_BASE / rede_nome / FRAC_C / obs_tag

def plot_random_curves(rede_nome: str, num_obs: int, n_curves: int = 5,
                       loglog: bool = True, vline_x: float = None):
    """
    Plot MSD curves for a specific network and number of obstacles.

    Parameters
    ----------
    rede_nome : str
        "L64", "L128", or "L256"
    num_obs : int
        Number of obstacles (e.g., 0, 1638, 3276, etc.)
    n_curves : int
        Maximum number of curves to plot (randomly sampled if too many).
    loglog : bool
        If True, set x and y axes to log scale.
    vline_x : float, optional
        If provided, draw a vertical line at this x value.
    """
    input_dir = get_input_dir(rede_nome, num_obs)
    
    if not input_dir.exists():
        print(f"❌ Diretório não encontrado: {input_dir}")
        print("   Verifique:")
        print(f"   - Se a rede '{rede_nome}' existe em REDE_CONFIGS")
        print(f"   - Se o número de obstáculos {num_obs} é válido para {rede_nome}")
        print(f"   - Se os dados MSD foram processados para esta configuração")
        return

    files = sorted(input_dir.glob("*_msd.csv"))

    print(f"📁 Diretório: {input_dir}")
    print(f"📊 Encontrados {len(files)} arquivos MSD")
    
    if not files:
        print("❌ Nenhum arquivo MSD encontrado")
        return

    if n_curves and len(files) > n_curves:
        files = random.sample(files, n_curves)
        print(f"🎲 Plotando {n_curves} curvas (amostra aleatória)")
    else:
        print(f"📈 Plotando todas as {len(files)} curvas")

    plt.figure(figsize=(10, 6))
    
    for f in files:
        df = pd.read_csv(f)
        plt.plot(df["tau"], df["msd_ensemble"], alpha=0.6, linewidth=1)

    # Calcular phi (fração de obstáculos)
    for L, nome, path in REDE_CONFIGS:
        if nome == rede_nome:
            area = L**2
            phi = num_obs / area
            break
    else:
        phi = num_obs / (128**2)  # fallback

    plt.xlabel("Time lag Δt", fontsize=12)
    plt.ylabel("MSD (ensemble)", fontsize=12)
    plt.title(f"MSD curves — {rede_nome}, Nc=0.5Np, φ={phi:.3f} (obs={num_obs})", fontsize=14)

    if loglog:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Time lag Δt (log)", fontsize=12)
        plt.ylabel("MSD (ensemble, log)", fontsize=12)

    if vline_x is not None:
        plt.axvline(x=vline_x, color="black", linestyle="--", linewidth=1.5, 
                   alpha=0.7, label=f"Δt = {vline_x}")
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def listar_configuracoes_disponiveis():
    """Lista todas as configurações de rede e obstáculos disponíveis."""
    print("🔍 Configurações disponíveis:")
    
    for L, rede_nome, base_path in REDE_CONFIGS:
        print(f"\n📋 Rede: {rede_nome} (L={L})")
        
        # Verifica se o diretório base existe
        rede_dir = MSD_DIR_BASE / rede_nome / FRAC_C
        if not rede_dir.exists():
            print(f"   ❌ Diretório não encontrado: {rede_dir}")
            continue
            
        obs_list = OBS_TEMPLATES.get(L, [])
        disponiveis = []
        
        for obs_tag in obs_list:
            obs_dir = rede_dir / obs_tag
            if obs_dir.exists():
                num_arquivos = len(list(obs_dir.glob("*_msd.csv")))
                if num_arquivos > 0:
                    num_obs = obs_tag.replace("s_obs_", "")
                    disponiveis.append(f"{num_obs} ({num_arquivos} arquivos)")
        
        if disponiveis:
            print(f"   ✅ Obstáculos disponíveis: {', '.join(disponiveis)}")
        else:
            print("   ❌ Nenhum dado MSD encontrado")

# -------- EXEMPLO DE USO --------
if __name__ == "__main__":
    # Primeiro, liste as configurações disponíveis
    listar_configuracoes_disponiveis()
    
    print("\n" + "="*50)
    print("🎯 EXEMPLOS DE USO:")
    print("="*50)
    """
    # Exemplo 1: L128 com 1638 obstáculos
    print("\n📊 Exemplo 1: L128 com 1638 obstáculos")
    plot_random_curves(
        rede_nome="L_128",
        num_obs=0,
        n_curves=100,
        loglog=True,
        vline_x=5
    )
   
    # Exemplo 2: L64 com 819 obstáculos  
    print("\n📊 Exemplo 2: L64 com 819 obstáculos")
    plot_random_curves(
        rede_nome="L_64", 
        num_obs=3276,
        n_curves=100,
        loglog=True,
        vline_x=5
    )
    """
    # Exemplo 3: L256 com 6553 obstáculos
    print("\n📊 Exemplo 3: L256 com 6553 obstáculos")
    plot_random_curves(
        rede_nome="L_256",
        num_obs=39321, 
        n_curves=100,
        loglog=True,
        vline_x=25
    )
  