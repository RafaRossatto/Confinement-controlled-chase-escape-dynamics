import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.plotagem import gerar_heatmap_matriz_media # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore



#
base_path = Path.home() / "Dados_Doc" / "TCC_095" / "TCT_095" / "Distribuição_quadrada" / "NC_500_NE_500_TCC_095_TCT_095"

NE = 500
TCC = 0.95
TCT = 0.95
O = 51

'''
# Gera o gráfico do numero de escapers por tempo
NC_500_NE_500_TCC_005_TCT_005_O_0_passos = carregar_dataframes_com_runs(base_path, NE, TCC, TCT,O, usar_cache=True, forcar_recarregar=False)
chaves_escolhidas = ["SR_TCC_4_SR_TCT_5"]

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
'''

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
