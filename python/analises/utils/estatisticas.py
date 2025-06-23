import numpy as np
import pandas as pd
from scipy.optimize import curve_fit # type: ignore
from scipy.stats import linregress # type: ignore


def gerar_matriz_medias_steps(dfs, limite=1000, verbose=True):
    """
    Gera uma matriz 50x50 (como DataFrame) contendo a média dos valores da coluna 'steps'
    de cada DataFrame do dicionário fornecido e retorna também o maior valor abaixo de um limite.

    Parâmetros:
    - dfs: dicionário com chaves do tipo 'SR_TCC_x_SR_TCT_y' e valores DataFrames contendo coluna 'steps'
    - limite: valor limite superior para buscar o maior valor abaixo dele
    - verbose: se True, imprime mensagens informativas

    Retorna:
    - df_matriz: DataFrame 50x50 com as médias, onde índice = SR_TCC e colunas = SR_TCT
    - max_abaixo: maior valor da matriz abaixo do limite (ou None)
    """
    matriz_medias = np.full((50, 50), np.nan)

    for chave, df in dfs.items():
        try:
            sr_tcc = int(chave.split("_")[2])
            sr_tct = int(chave.split("_")[5])

            # Remove espaços extras dos nomes de colunas (caso existam)
            df.columns = df.columns.str.strip()

            if "steps" in df.columns:
                media = pd.to_numeric(df["steps"], errors='coerce').mean()
                matriz_medias[sr_tcc - 1, sr_tct - 1] = media
            else:
                if verbose:
                    print(f"⚠️ Coluna 'steps' não encontrada no DataFrame {chave}")
        except Exception as e:
            if verbose:
                print(f"Erro ao processar {chave}: {e}")

    df_matriz = pd.DataFrame(
        matriz_medias,
        index=[f"SR_TCC_{i+1}" for i in range(50)],
        columns=[f"SR_TCT_{j+1}" for j in range(50)]
    )

    valores_abaixo = df_matriz.values[df_matriz.values < limite]

    if len(valores_abaixo) > 0:
        max_abaixo = np.max(valores_abaixo)
        if verbose:
            print(f"Maior valor abaixo de {limite}: {max_abaixo}")
    else:
        max_abaixo = None
        if verbose:
            print(f"Nenhum valor abaixo de {limite} encontrado.")

    return df_matriz, max_abaixo

def modelo_exponencial(x, tau, NE):
    """Modelo NE * exp(-x / tau), com NE fixo"""
    return NE * np.exp(-x / tau)

def ajustar_tau(curva_media: np.ndarray, passos: np.ndarray, NE: float):
    """
    Ajusta o parâmetro tau na curva NE * exp(-steps / tau).
    
    Parâmetros:
    - curva_media: vetor com os valores médios da curva ao longo dos passos.
    - passos: vetor com os valores do eixo x.
    - NE: valor inicial dos escapers.
    
    Retorna:
    - tau ajustado
    - erro padrão da estimativa
    - R² do ajuste
    """
    try:
        # Ajuste do modelo
        popt, pcov = curve_fit(lambda x, tau: modelo_exponencial(x, tau, NE),
                               passos, curva_media, p0=(100.0), maxfev=5000)

        tau_otimo = popt[0]
        erro_tau = np.sqrt(np.diag(pcov))[0]

        # Calcular R² do ajuste
        ajuste = modelo_exponencial(passos, tau_otimo, NE)
        ss_res = np.sum((curva_media - ajuste) ** 2)
        ss_tot = np.sum((curva_media - np.mean(curva_media)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return tau_otimo, erro_tau, r2

    except Exception as e:
        print(f"[Erro no ajuste exponencial]: {e}")
        return None, None, None