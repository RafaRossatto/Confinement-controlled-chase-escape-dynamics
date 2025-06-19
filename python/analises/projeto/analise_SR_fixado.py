import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.leitura import carregar_dataframes_com_runs # type: ignore
from utils.leitura import carregar_dataframes # type: ignore
from utils.plotagem import plotar_curvas_media_e_desvio # type: ignore
from utils.plotagem import gerar_heatmap_matriz_media # type: ignore
from utils.estatisticas import gerar_matriz_medias_steps # type: ignore

base_path = Path.home() / "Dados_Doc" / "TCC_095" / "TCT_095" / "Distribuição_quadrada" / "NC_500_NE_100_TCC_095_TCT_095"

NE = 100
TCC = 0.95
TCT = 0.95
O = 0

chaves_escolhidas = ["SR_TCC_5_SR_TCT_5", "SR_TCC_10_SR_TCT_5", "SR_TCC_15_SR_TCT_5",
                    "SR_TCC_20_SR_TCT_5", "SR_TCC_30_SR_TCT_5", "SR_TCC_35_SR_TCT_5",
                    "SR_TCC_40_SR_TCT_5", "SR_TCC_45_SR_TCT_5", "SR_TCC_50_SR_TCT_5"]

df_dados = carregar_dataframes_com_runs(base_path, NE, TCC, TCT,O, usar_cache=True, forcar_recarregar=False  # Coloque True se quiser ignorar o cache
)
#plotar_curvas_media_e_desvio(df_dados, chaves_escolhidas, valor_inicial=100)

dfs = carregar_dataframes(
    pasta=base_path,
    NE=100,
    TCC=0.95,
    TCT=0.95,
    O=0,
    usar_cache=True,
    forcar_recarregar=False)

""""
gerar_heatmap_matriz_media(
    dfs,
    limite=680,
    titulo="Média de Escapers (valores > {limite} marcados como 'infinito')",
    salvar_pdf=True,
    nome_pdf="mapa_escapers.pdf"
)
"""
df_matriz, max_abaixo = gerar_matriz_medias_steps(dfs, limite=1000)
