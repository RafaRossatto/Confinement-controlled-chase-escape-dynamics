import numpy as np
import pandas as pd

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
