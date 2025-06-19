import os
import pandas as pd #type: ignore
import pickle
import hashlib

def gerar_nome_cache(pasta, NE, TCC, TCT, O, NC):
    """Gera um nome de arquivo de cache único com base nos parâmetros e caminho."""
    hash_pasta = hashlib.md5(str(pasta).encode()).hexdigest()[:8]
    return f"cache_{hash_pasta}_NE{NE}_TCC{TCC}_TCT{TCT}_O{O}_NC{NC}.pkl"

def carregar_dataframes_com_runs(pasta, NE, TCC, TCT, O, NC=500, num_runs=100, usar_cache=True, forcar_recarregar=False):
    """
    Carrega DataFrames a partir dos arquivos CSV ou do cache (.pkl).

    - usar_cache: se True, tenta carregar do cache.
    - forcar_recarregar: se True, ignora o cache e recarrega dos arquivos .csv.
    """
    cache_name = gerar_nome_cache(pasta, NE, TCC, TCT, O, NC)
    cache_file = os.path.join(pasta, cache_name)

    if usar_cache and not forcar_recarregar and os.path.exists(cache_file):
        print(f"✔️ Carregando do cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("📂 Carregando arquivos CSV (isso pode demorar)...")
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
                        print(f"❌ Erro ao ler {nome_arquivo}: {e}")

    # Salvar no cache
    if usar_cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(dataframes, f)
            print(f"💾 Cache salvo em: {cache_file}")
        except Exception as e:
            print(f"⚠️ Não foi possível salvar o cache: {e}")

    return dataframes
