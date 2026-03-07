from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# ---------------------- parâmetros ----------------------
L = 128
AREA = L**2
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
HAZARD_DIR = BASE_ROOT /"L_128"/ "resultados_modelos" /"paper_response"/ "survival_analysis"
OUT_DIR = BASE_ROOT /"L_128"/ "resultados_modelos" /"paper_response"/ "tau_x_phi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

bases = [
    (r"$N^{C}=0.5 \,N^{E}_{0}$", "Nc=Np*0.5"),
    (r"$N^{C}=0.8\,N^{E}_{0}$", "Nc=Np*0.8"),
    (r"$N^{C}=N^{E}_{0}$",      "Nc=Np"),
]


OBSTACULOS = [
    "s_obs_00", "s_obs_1638", "s_obs_3276",
    "s_obs_4915", "s_obs_6553","s_obs_8028", "s_obs_8192", "s_obs_8355",
    "s_obs_9666","s_obs_9994","s_obs_9830", 
    "s_obs_11468", "s_obs_13107"
]

# Dicionário para armazenar todos os resultados
resultados = {}

for base_label, base_nome in bases:
    resultados[base_nome] = {}
    
    for obstaculo in OBSTACULOS:
        ob_num = obstaculo.split('_')[-1]
        nome_arquivo = f"{base_nome}_ob{ob_num}_fit_parameters.csv"
        arquivo = HAZARD_DIR / nome_arquivo
        
        if arquivo.exists():
            df = pd.read_csv(arquivo)
            
            if 'tau' in df.columns and 'tau_err' in df.columns:
                # Armazenar o DataFrame completo ou apenas as colunas desejadas
                resultados[base_nome][ob_num] = {
                    'df': df,
                    'tau': df['tau'].values,
                    'tau_err': df['tau_err'].values,
                    'estatisticas': {
                        'media_tau': df['tau'].mean(),
                        'std_tau': df['tau'].std(),
                        'media_tau_err': df['tau_err'].mean()
                    }
                }
                print(f"✓ {base_nome}_ob{ob_num}: {len(df)} linhas")
            else:
                resultados[base_nome][ob_num] = None
                print(f"✗ {base_nome}_ob{ob_num}: colunas não encontradas")
        else:
            resultados[base_nome][ob_num] = None
            print(f"✗ {base_nome}_ob{ob_num}: arquivo não encontrado")

# Exemplo de como acessar os resultados
print("\n" + "="*60)
print("EXEMPLO DE ACESSO AOS RESULTADOS")
print("="*60)

# ---------------------- plot ----------------------
plt.figure(figsize=(10,6), dpi=150)
cmap = colormaps.get_cmap("flag")

for idx, (label_tex, nc_tag) in enumerate(bases):
    print(f"\nProcessando base: {label_tex}")
    
    # Listas para armazenar dados desta base
    phis = []
    taus = []
    tau_errs = []
    
    # Percorrer todos os obstáculos para esta base
    for obstaculo in OBSTACULOS:
        ob_num = obstaculo.split('_')[-1]
        
        # Construir nome do arquivo específico
        fpath = HAZARD_DIR / f"{nc_tag}_ob{ob_num}_fit_parameters.csv"
        
        if not fpath.exists():
            print(f"  Arquivo não encontrado: {fpath.name}")
            continue
        
        # Ler o arquivo
        df = pd.read_csv(fpath)
        
        # Verificar se as colunas necessárias existem
        if 'tau' not in df.columns or 'tau_err' not in df.columns:
            print(f"  Colunas tau/tau_err não encontradas em: {fpath.name}")
            continue
        
        # Calcular phi (fração de obstáculos)
        n_obs = int(ob_num)  # Converter para inteiro
        phi = n_obs / AREA
        
        # Pegar os valores (média se houver múltiplos valores)
        tau_mean = df['tau'].mean()
        tau_err_mean = df['tau_err'].mean()
        
        phis.append(phi)
        taus.append(tau_mean)
        tau_errs.append(tau_err_mean)
        
        print(f"  Obstáculo {ob_num}: phi={phi:.4f}, tau={tau_mean:.4f}")
    
    # Ordenar por phi
    if phis:  # Verificar se há dados
        # Criar arrays numpy para ordenação
        phis_array = np.array(phis)
        taus_array = np.array(taus)
        tau_errs_array = np.array(tau_errs)
        
        # Ordenar por phi
        sorted_indices = np.argsort(phis_array)
        phis_sorted = phis_array[sorted_indices]
        taus_sorted = taus_array[sorted_indices]
        tau_errs_sorted = tau_errs_array[sorted_indices]
        
        # Plotar
        plt.errorbar(
            phis_sorted, taus_sorted,
            yerr=tau_errs_sorted,
            fmt="o-", capsize=4, markersize=6,
            color=cmap(idx ),
            linewidth=2,
            label=label_tex
        )

# ---------------------- decoração ----------------------
plt.axvline(x=0.60, color="black", linestyle="--", linewidth=1.5,
            label=r"$\phi = 0.60$")

plt.xlabel(r"$\phi$", fontsize=18)
plt.ylabel(r"$ \langle \tau \rangle $", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3, linestyle='--')
plt.legend(fontsize=14, loc='best')
plt.tight_layout()

# salvar
out_file = OUT_DIR / "tau_vs_phi.pdf"
plt.savefig(out_file, bbox_inches="tight", dpi=300)
plt.show()

print(f"[OK] Gráfico salvo em: {out_file}")