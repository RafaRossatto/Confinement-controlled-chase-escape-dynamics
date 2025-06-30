import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.plotagem import gerar_heatmap_matriz_media # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore



import matplotlib.pyplot as plt




import re

def coletar_media_varrendo_sr_tcc_com_sr_tct_fixado(dfs_dict, sr_tct_alvo):
    """
    Coleta as médias e desvios dos DataFrames cujo nome segue o padrão:
    SR_TCC_<x>_SR_TCT_<y>, fixando SR_TCT = sr_tct_alvo, e variando SR_TCC.

    Retorna listas: sr_tccs, medias, desvios
    """
    import re

    sr_tccs = []
    medias = []
    desvios = []

    padrao = r"SR_TCC_(\d+)_SR_TCT_(\d+)"

    for nome, df in dfs_dict.items():
        match = re.match(padrao, nome)
        if match:
            sr_tcc, sr_tct = map(int, match.groups())

            if sr_tct == sr_tct_alvo:
                if 'steps' in df.columns:
                    sr_tccs.append(sr_tcc)
                    medias.append(df['steps'].mean())
                    desvios.append(df['steps'].std())

    # Ordena por SR_TCC
    ordenado = sorted(zip(sr_tccs, medias, desvios))
    sr_tccs, medias, desvios = zip(*ordenado) if ordenado else ([], [], [])
    return list(sr_tccs), list(medias), list(desvios)








def plotar_medias_sr_tcc(sr_tccs, medias, desvios, 
                          titulo="Média dos steps variando SR_TCC",
                          legenda=None,
                          salvar_em=None):
    """
    Plota a média e desvio padrão dos steps em função do SR_TCC.

    Parâmetros:
    - sr_tccs: lista com os valores de SR_TCC (eixo x)
    - medias: lista com as médias de steps
    - desvios: lista com os desvios padrão
    - titulo: título do gráfico
    - legenda: string opcional para a legenda
    - salvar_em: se fornecido, salva o gráfico no caminho especificado
    """

    plt.figure(figsize=(10, 6))
    plt.errorbar(sr_tccs, medias, yerr=desvios, fmt='o-', capsize=4, label=legenda)

    plt.xlabel("SR_TCC", fontsize=14)
    plt.ylabel("Steps médios", fontsize=14)
    plt.title(titulo, fontsize=16)
    plt.grid(True)
    plt.xticks(sr_tccs)  # Garante marcação de cada ponto inteiro
    #plt.yscale("log")    # Usar escala logarítmica para steps (se desejar)

    if legenda:
        plt.legend()

    if salvar_em:
        plt.tight_layout()
        plt.savefig(salvar_em)

    plt.show()







def plotar_multiplas_curvas_sr_tct(dfs_dict, sr_tct_valores, salvar_em=None):
    """
    Plota múltiplas curvas variando SR_TCC para diferentes valores de SR_TCT fixados.
    """
    plt.figure(figsize=(10, 6))

    for sr_tct in sr_tct_valores:
        sr_tccs, medias, desvios = coletar_media_varrendo_sr_tcc_com_sr_tct_fixado(dfs_dict, sr_tct)

        if len(sr_tccs) == 0:
            print(f"[Aviso] Nenhum dado encontrado para SR_TCT = {sr_tct}")
            continue

        plt.errorbar(sr_tccs, medias, yerr=desvios, fmt='o-', capsize=4, label=f"SR_TCT = {sr_tct}")

    plt.xlabel("SR_TCC", fontsize=14)
    plt.ylabel("Steps médios", fontsize=14)
    plt.title("Steps médios variando SR_TCC para diferentes SR_TCT", fontsize=16)
    plt.grid(True)
    plt.xticks(range(1, max(sr_tccs) + 1))
    plt.legend()
    plt.tight_layout()
    
    if salvar_em:
        plt.savefig(salvar_em)

    plt.show()







#
base_path = Path.home() / "Dados_Doc" / "TCC_095" / "TCT_005" / "Distribuição_quadrada" / "NC_500_NE_100_TCC_095_TCT_005"

NE = 100
TCC = 0.95
TCT = 0.05
O = 51


# Gera o gráfico do numero de escapers por tempo
NC_500_NE_500_TCC_005_TCT_005_O_0_passos = carregar_dataframes_com_runs(base_path, NE, TCC, TCT,O, usar_cache=True, forcar_recarregar=False)
chaves_escolhidas = ["SR_TCC_1_SR_TCT_5"]

valor_inicial = 100
nome_pdf=f"H_M_NC_500_NE_{NE}_TCC_{TCC}_TCT_{TCT}_O_{O}_passos.pdf"


plotar_curvas_media_e_desvio(
    dataframes=NC_500_NE_500_TCC_005_TCT_005_O_0_passos,
    chaves_escolhidas=chaves_escolhidas,
    salvar_pdf=True,
    nome_pdf=nome_pdf,
    valor_inicial=valor_inicial,
    mostrar= False  # se quiser exibir também
)
""""
# Gera o gráfico de mapa de calor.
df = carregar_dataframes(base_path,NE,TCC,TCT,O,usar_cache=True,forcar_recarregar=False)
df_matriz, max_abaixo = gerar_matriz_medias_steps(df, limite=2800)
limite = 2800
titulo = f"NC_500_NE_{NE}_TCC_{TCC}_TCT_{TCT}_O_{O}"
salvar_pdf=True
nome_pdf=f"H_M_NC_500_NE_{NE}_TCC_{TCC}_TCT_{TCT}_O_{O}.pdf"


gerar_heatmap_matriz_media(
    dfs = df,
    limite=limite,
    titulo=titulo,
    salvar_pdf=salvar_pdf,
    nome_pdf=nome_pdf
)



#sr_tccs, medias, desvios = coletar_media_varrendo_sr_tcc_com_sr_tct_fixado(
#    dfs_dict=df,
#    sr_tct_alvo=10
#)


print("=== Dados para plotagem ===")
for sr, media, desvio in zip(sr_tccs, medias, desvios):
    print(f"SR_TCC = {sr:2d} -> Média = {media:.2f}, Desvio = {desvio:.2f}")
# Verificar os dados ANTES de plotar
print("\n=== Dados para plotagem ===")
for sr, media, desvio in zip(sr_tccs, medias, desvios):
    print(f"SR_TCC = {sr:2d} -> Média = {media:.2f}, Desvio = {desvio:.2f}")

print("Tamanhos -> sr_tccs:", len(sr_tccs), "| medias:", len(medias), "| desvios:", len(desvios))
# Agora vamos plotar
plotar_medias_sr_tcc(
    sr_tccs, medias, desvios,
    titulo="Steps médios variando SR_TCC (TCT_SR=1)",
    legenda="TCT_SR = 10",
    salvar_em="grafico_steps_vs_sr_tcc.pdf"
)

"""
