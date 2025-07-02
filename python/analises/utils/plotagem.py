import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.estatisticas import ajustar_tau # type: ignore

def plotar_curvas_media_e_desvio(dataframes, chaves_escolhidas, titulo="Média das Presas Vivas por Configuração",
                                  salvar_pdf=False, nome_pdf="grafico.pdf", mostrar=False, valor_inicial=None):
    plt.figure(figsize=(12, 7))
    for chave in chaves_escolhidas:
        if chave in dataframes:
            runs_dict = dataframes[chave]
            if not runs_dict:
                print(f"[Aviso] Nenhuma run encontrada para: {chave}")
                continue

            max_passo = max(df['passo'].iloc[-1] for df in runs_dict.values() if not df.empty)
            intervalo = runs_dict[next(iter(runs_dict))]['passo'].iloc[1] - runs_dict[next(iter(runs_dict))]['passo'].iloc[0]

            curvas = []
            for run_df in runs_dict.values():
                if not run_df.empty:
                    curva = run_df.set_index('passo').reindex(range(0, max_passo + 1, intervalo))['presas_vivas'].fillna(method='ffill').fillna(method='bfill')
                    curva_np = curva.to_numpy()
                    if valor_inicial is not None:
                        curva_np = np.insert(curva_np, 0, valor_inicial)
                    curvas.append(curva_np)

            if curvas:
                matriz = np.array(curvas)
                media = matriz.mean(axis=0)
                desvio = matriz.std(axis=0)

                passos = np.arange(len(media)) * intervalo
                if valor_inicial is not None:
                    passos = passos - intervalo
                    passos[0] = 0

                # Curva média e desvio padrão
            plt.plot(passos, media, label=chave)
            plt.fill_between(passos, media - desvio, media + desvio, alpha=0.3)

            # 👉 Ajuste da curva média ao modelo NE * exp(-x / tau)
            tau, erro_tau, r2 = ajustar_tau(media, passos, valor_inicial)
            if tau is not None:
                print(f"[{chave}] τ ajustado: {tau:.2f} ± {erro_tau:.2f}, R² = {r2:.4f}")
                curva_fit = valor_inicial * np.exp(-passos / tau)
                plt.plot(passos, curva_fit, '--', label=f'Fit {chave} (τ={tau:.1f})')
        else:
            print(f"[Aviso] Chave não encontrada: {chave}")

    plt.xlabel('Steps')
    plt.ylabel('Escapers_Alive')
    plt.title(titulo)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    if salvar_pdf:
        plt.savefig(nome_pdf, format='pdf')
        print(f"Gráfico salvo em: {nome_pdf}")

    if mostrar:
        plt.show()

def gerar_heatmap_matriz_media(
    dfs,
    limite=680,
    titulo="Média de Escapers (valores > {limite} marcados como 'infinito')",
    salvar_pdf=False,
    nome_pdf="heatmap_escapers.pdf"
):
    """
    Gera um mapa de calor a partir de um dicionário de DataFrames com colunas 'steps'.

    Parâmetros:
    - dfs: dicionário no formato dfs["SR_TCC_x_SR_TCT_y"] = DataFrame
    - limite: valor máximo que será exibido no heatmap (valores maiores serão limitados)
    - titulo: título do gráfico (você pode usar {limite} para inserir o valor automaticamente)
    - salvar_pdf: se True, salva o gráfico como PDF
    - nome_pdf: nome do arquivo PDF a ser salvo
    """
    matriz_medias = np.full((50, 50), np.nan)

    for chave, df in dfs.items():
        try:
            sr_tcc = int(chave.split("_")[2])
            sr_tct = int(chave.split("_")[5])

            df.columns = df.columns.str.strip()

            if "steps" in df.columns:
                media = pd.to_numeric(df["steps"], errors='coerce').mean()
                matriz_medias[sr_tcc - 1, sr_tct - 1] = media
            else:
                print(f"Coluna 'steps' não encontrada em {chave}")
        except Exception as e:
            print(f"Erro ao processar {chave}: {e}")

    df_matriz = pd.DataFrame(
        matriz_medias,
        index=[f"SR_TCC_{i+1}" for i in range(50)],
        columns=[f"SR_TCT_{j+1}" for j in range(50)]
    )

    df_limitada = df_matriz.clip(upper=limite)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        df_limitada,
        cmap='viridis',
        fmt='',
        linewidths=0.5,
        cbar_kws={'label': 'Steps'}
    )

    plt.title(titulo.format(limite=limite))
    plt.xlabel("SR_TCT")
    plt.ylabel("SR_TCC")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if salvar_pdf:
        plt.savefig(nome_pdf, format='pdf')
        print(f"Gráfico salvo em: {nome_pdf}")




def plotar_media_steps_por_NC(dfs_por_nc, titulo="Média de steps por NC", salvar=False, nome_arquivo="grafico.pdf"):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    ncs = []
    medias = []

    for nc, dfs_dict in dfs_por_nc.items():
        todas_as_steps = []

        for df in dfs_dict.values():
            if "steps" in df.columns:
                steps_validos = pd.to_numeric(df["steps"], errors="coerce").dropna()
                todas_as_steps.extend(steps_validos)

        if todas_as_steps:
            media = np.mean(todas_as_steps)
            ncs.append(nc)
            medias.append(media)
        else:
            print(f"[Aviso] Nenhuma coluna 'steps' válida para NC = {nc}")

    # Ordenar os pontos por NC para o gráfico
    ncs, medias = zip(*sorted(zip(ncs, medias)))

    plt.figure(figsize=(10, 6))
    plt.plot(ncs, medias, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("the number of chasers $N_{C}$")
    plt.ylabel("trapping time $T$")
    plt.title(titulo)
    plt.grid(True)

    if salvar:
        plt.savefig(nome_arquivo)
        print(f"Gráfico salvo em: {nome_arquivo}")
    else:
        plt.show()

    return list(ncs), list(medias)
