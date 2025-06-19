import numpy as np
import matplotlib.pyplot as plt

def plotar_curvas_media_e_desvio(dataframes, chaves_escolhidas, titulo="Média das Presas Vivas por Configuração",
                                  salvar_pdf=False, nome_pdf="grafico.pdf", mostrar=True, valor_inicial=None):
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

                plt.plot(passos, media, label=chave)
                plt.fill_between(passos, media - desvio, media + desvio, alpha=0.3)
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

    if mostrar:
        plt.show()
