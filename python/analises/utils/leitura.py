import os
import pandas as pd
import pickle
import hashlib

def gerar_nome_cache(pasta, NE, TCC, TCT, O, NC, tipo_dado="csv"):
    """Gera nome de cache único com base nos parâmetros e na pasta."""
    hash_pasta = hashlib.md5(str(pasta).encode()).hexdigest()[:8]
    return f"cache_{tipo_dado}_{hash_pasta}_NE{NE}_TCC{TCC}_TCT{TCT}_O{O}_NC{NC}.pkl"

def carregar_dataframes_com_runs(
    pasta, NE, TCC, TCT, O,
    NC, num_runs= 100,
    usar_cache=True, forcar_recarregar=False
):
    tipo_dado = "csv"
    cache_name = gerar_nome_cache(pasta, NE, TCC, TCT, O, NC, tipo_dado=tipo_dado)
    cache_file = os.path.join(pasta, cache_name)
    sr_tct = 2
    sr_tcc = 2

    if usar_cache and not forcar_recarregar and os.path.exists(cache_file):
        print(f"✔️ Carregando do cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("📂 Lendo arquivos CSV (isso pode demorar)...")
    dataframes = {}
           
    chave = f"SR_TCC_{sr_tcc}_SR_TCT_{sr_tct}"
    dataframes[chave] = {}

    for run in range(1, num_runs + 1):
        nome_arquivo = (
            f"NC_{NC}_NE_{NE}_O_{O}_TCC_{TCC:.2f}_SR_{sr_tcc}"
            f"_TCT_{TCT:.2f}_SR_{sr_tct}.dat_run_{run}_presas_por_passo.csv"
        )
        caminho_arquivo = os.path.join(pasta, nome_arquivo)
        if os.path.exists(caminho_arquivo):
            try:
                df = pd.read_csv(caminho_arquivo, sep=",")
                dataframes[chave][run] = df
            except Exception as e:
                print(f"❌ Erro ao ler {nome_arquivo}: {e}")
        else:
            print(f"❌ Arquivo NÃO encontrado: {nome_arquivo}")

    if usar_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dataframes, f)
            print(f"💾 Cache salvo em: {cache_file}")
        except Exception as e:
            print(f"⚠️ Não foi possível salvar o cache: {e}")

    return dataframes


def carregar_dataframes(
    pasta, NE, TCC, TCT, O,
    NC,
    usar_cache=True,
    forcar_recarregar=False
):
    """
    Lê arquivos .dat e usa cache para evitar leituras repetidas.
    """
    cache_name = gerar_nome_cache(pasta, NE, TCC, TCT, O, NC, tipo_dado="dat")
    cache_file = os.path.join(pasta, cache_name)

    if usar_cache and not forcar_recarregar and os.path.exists(cache_file):
        print(f"Carregando do cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Lendo arquivos .dat...")
    dataframes = {}

    for sr_tcc in range(1, 51):
        for sr_tct in range(1, 51):
            nome_arquivo = f"NC_{NC}_NE_{NE}_O_{O}_TCC_{TCC}_SR_{sr_tcc}_TCT_{TCT}_SR_{sr_tct}.dat"
            caminho_arquivo = os.path.join(pasta, nome_arquivo)

            if os.path.exists(caminho_arquivo):
                try:
                    df = pd.read_csv(caminho_arquivo, sep=',')
                    chave = f"SR_TCC_{sr_tcc}_SR_TCT_{sr_tct}"
                    dataframes[chave] = df
                except Exception as e:
                    print(f" Erro ao ler {nome_arquivo}: {e}")

    if usar_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dataframes, f)
            print(f"Cache salvo em: {cache_file}")
        except Exception as e:
            print(f" Não foi possível salvar o cache: {e}")

    return dataframes

def carregar_dataframes(
    pasta, NE, TCC, TCT, O,
    NC=500,
    usar_cache=True,
    forcar_recarregar=False
):
    """
    Lê arquivos .dat e usa cache para evitar leituras repetidas.
    """
    cache_name = gerar_nome_cache(pasta, NE, TCC, TCT, O, NC, tipo_dado="dat")
    cache_file = os.path.join(pasta, cache_name)

    if usar_cache and not forcar_recarregar and os.path.exists(cache_file):
        print(f" Carregando do cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Lendo arquivos .dat...")
    dataframes = {}

    for sr_tcc in [100]:
        for sr_tct in [100]:
            nome_arquivo = f"NC_{NC}_NE_{NE}_O_{O}_TCC_{TCC:.2f}_SR_{sr_tcc}_TCT_{TCT:.2f}_SR_{sr_tct}.dat"


            caminho_arquivo = os.path.join(pasta, nome_arquivo)

            if os.path.exists(caminho_arquivo):
                try:
                    df = pd.read_csv(caminho_arquivo, sep=',')
                    chave = f"SR_TCC_{sr_tcc}_SR_TCT_{sr_tct}"
                    dataframes[chave] = df
                except Exception as e:
                    print(f" Erro ao ler {nome_arquivo}: {e}")

    if usar_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dataframes, f)
            print(f" Cache salvo em: {cache_file}")
        except Exception as e:
            print(f" Não foi possível salvar o cache: {e}")

    return dataframes