import pandas as pd
import os

def carregar_dataframes_com_runs(pasta, NE, TCC, TCT, O, NC=500, num_runs=100):
    dataframes = {}
    for sr_tcc in range(1, 51):
        for sr_tct in range(1, 51):
            chave_config = f"SR_TCC_{sr_tcc}_SR_TCT_{sr_tct}"
            dataframes[chave_config] = {}

            for run in range(1, num_runs + 1):
                nome_arquivo = (
                    f"NC_{NC}_NE_{NE}_O_{O}_TCC_{TCC}_SR_{sr_tcc}"
                    f"_TCT_{TCT}_SR_{sr_tct}.dat_run_{run}_presas_por_passo.csv")
                caminho_arquivo = os.path.join(pasta, nome_arquivo)

                if os.path.exists(caminho_arquivo):
                    try:
                        df = pd.read_csv(caminho_arquivo)
                        dataframes[chave_config][run] = df
                    except Exception as e:
                        print(f" Erro ao ler {nome_arquivo}: {e}")
    return dataframes