#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD Ensemble-Averaged com TrajPy
- Varre todas as concentrações e obstáculos
- Fit em log–log por arquivo, com janela opcional
- Salva CSV por par e resumo geral
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= TrajPy obrigatório =========
try:
    import trajpy.trajpy as tj
except Exception as e:
    raise ImportError(
        "TrajPy não está instalado/visível. "
        "Instale-o e garanta 'import trajpy.trajpy as tj' funcionando."
    ) from e

# ========= Configurações Gerais =========
BASE_ROOT = Path.home() / "Dados_Doc" / "Np=free*0.25"
PATTERN = "*_hunter_trajectories.csv"

Lx = Ly = 128              # caixa para unwrapping (PBC)
MIN_TRAJ_LENGTH = 5        # tamanho mínimo da trajetória (frames)
MAX_TAU = 50               # número de lags (τ) para calcular MSD
MIN_CONTRIB_POINT = 5      # min. contribuições para considerar MSD(τ) válido

# JANELA DE FIT (em τ). None => usa todos os τ válidos
JANELA_FIT = None          # Ex.: None, [5, 30], [3, 20]

# Saídas
OUT_DIR = BASE_ROOT / "resultados_modelos" / "msd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= Lista de Concentrações e Obstáculos =========
CONCENTRACOES = [
    "Nc=Np*0.5",
    "Nc=Np*0.8",
    "Nc=Np",
]

OBSTACULOS = [
    "s_obs_00",
    "s_obs_1638",
    "s_obs_3276",
    "s_obs_4915",
    "s_obs_6553",
    "s_obs_8192",
    "s_obs_9830",
    "s_obs_11468",
    "s_obs_13107",
]

# ========= Funções utilitárias =========
def unwrap_1d(x, L):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    dx -= np.round(dx / L) * L
    xu = np.empty_like(x, dtype=float)
    xu[0] = x[0]
    xu[1:] = xu[0] + np.cumsum(dx)
    return xu

def unwrap_xy(x, y, Lx, Ly):
    return unwrap_1d(x, Lx), unwrap_1d(y, Ly)

from functools import lru_cache

@lru_cache(maxsize=1)
def _resolve_trajpy_msd_caller():
    """
    Descobre uma função de MSD do TrajPy (por trajetória) que exista na sua versão.
    Retorna um 'caller(traj, max_tau) -> msd_vec' que:
      - usa TrajPy de fato;
      - aceita traj Nx2 (float);
      - devolve um vetor 1D de MSD para τ>=1 (sem o ponto τ=0);
      - corta ou preenche até max_tau quando necessário.
    Levanta RuntimeError se nada compatível for encontrado.
    """
    # candidatos por ordem de preferência
    name_candidates = [
        # nomes típicos (trocas de underscore/minúsculas em versões diferentes)
        ("Trajectory", "msd_ensemble_averaged"),
        ("Trajectory", "msd_ensemble_average"),
        ("Trajectory", "msd_ensemble"),
        ("Trajectory", "msd_time_averaged"),   # TA-MSD também serve; preferimos EA se houver
        ("Trajectory", "msd_time_average"),
        ("Trajectory", "time_averaged_msd"),
        (None,        "msd_ensemble_averaged"),
        (None,        "msd_ensemble_average"),
        (None,        "msd_ensemble"),
        (None,        "msd_time_averaged"),
        (None,        "msd_time_average"),
        (None,        "time_averaged_msd"),
    ]

    # tenta obter o objeto função
    funcs = []
    for owner, attr in name_candidates:
        try:
            obj = getattr(tj.Trajectory, attr) if owner == "Trajectory" else getattr(tj, attr)
            if callable(obj):
                funcs.append((f"{owner or 'module'}.{attr}", obj))
        except Exception:
            pass

    if not funcs:
        raise RuntimeError(
            "TrajPy não expõe nenhuma função de MSD conhecida nesta versão "
            "(procurado: msd_ensemble_averaged / msd_time_averaged, etc.)."
        )

    def _first_successful_call(traj, max_tau):
        """
        Tenta assinaturas diferentes na função 'func' até uma funcionar:
          1) func(traj)
          2) func(traj, taus)  com taus = [0..max_tau]
          3) func(traj, max_tau)
        Valida saída 1D e remove τ=0 se vier.
        """
        traj = np.asarray(traj, float)
        taus0 = np.arange(0, max_tau + 1, dtype=int)  # algumas APIs pedem τ incluindo 0

        last_exc = None
        for name, func in funcs:
            # tentativas de assinatura
            for signature in ("traj", "traj_taus", "traj_maxtau"):
                try:
                    if signature == "traj":
                        out = func(traj)
                    elif signature == "traj_taus":
                        out = func(traj, taus0)
                    else:  # "traj_maxtau"
                        out = func(traj, max_tau)

                    arr = np.asarray(out, float).ravel()
                    if arr.size < 1:
                        raise ValueError("função retornou vetor vazio")

                    # muitas implementações retornam τ=0 na primeira posição; removemos
                    # (se não for o caso, isso mantém os dados)
                    if arr.size >= 2 and arr[0] == 0.0:
                        arr = arr[1:]

                    # corta/preenche para max_tau
                    if arr.size >= max_tau:
                        return name, arr[:max_tau]
                    else:
                        v = np.full(max_tau, np.nan, float)
                        v[:arr.size] = arr
                        return name, v
                except Exception as e:
                    last_exc = e
                    continue
        raise RuntimeError(f"Não foi possível chamar função de MSD do TrajPy; último erro: {last_exc}")

    # criamos um closure estável que também devolve o nome (primeira vez)
    resolved = {"used": None}

    def caller(traj, max_tau):
        name, vec = _first_successful_call(traj, max_tau)
        if resolved["used"] is None:
            resolved["used"] = name
            print(f"[TrajPy] Usando função de MSD: {name}")
        return vec

    return caller


def calcular_msd_por_arquivo(trajetorias, max_tau, contexto=""):
    """
    Calcula MSD(τ) POR ARQUIVO usando TrajPy (função de MSD por trajetória,
    descoberta dinamicamente). Depois faz a média entre trajetórias para cada τ.
    Retorna (msd_ensemble, n_contrib) para τ=1..max_tau.
    Levanta erro se TrajPy não fornecer nenhuma função compatível.
    """
    if not trajetorias:
        raise RuntimeError(f"Sem trajetórias válidas. {contexto}")

    caller = _resolve_trajpy_msd_caller()

    # MSD de cada trajetória
    msd_mat = np.full((len(trajetorias), max_tau), np.nan, float)
    for i, tr in enumerate(trajetorias):
        vec = caller(tr, max_tau)  # usa TrajPy de verdade
        msd_mat[i, :len(vec)] = vec

    # média por τ ignorando NaN; n_contrib = quantos traj têm valor finito naquele τ
    msd_ensemble = np.nanmean(msd_mat, axis=0)
    n_contrib = np.sum(np.isfinite(msd_mat), axis=0).astype(int)

    # checagem mínima
    if not np.any(np.isfinite(msd_ensemble[:max(3, min(10, max_tau))])):
        raise RuntimeError(f"MSD ficou inválido para {contexto} (verifique TrajPy/entradas).")

    return msd_ensemble, n_contrib

def calcular_alpha(taus, msd, janela_fit=None):
    """
    Fit em log–log: log(MSD) ~ alpha*log(tau) + log(K).
    Retorna (alpha, K, R2, mask_usada_em_taus).
    """
    mask = (msd > 0) & np.isfinite(msd)
    if janela_fit is not None:
        tau_min, tau_max = janela_fit
        mask &= (taus >= tau_min) & (taus <= tau_max)

    if np.sum(mask) < 3:
        return np.nan, np.nan, np.nan, mask

    x = np.log(taus[mask])
    y = np.log(msd[mask])
    a1, a0 = np.polyfit(x, y, 1)

    y_pred = a1 * x + a0
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return float(a1), float(np.exp(a0)), float(r2), mask

def processar_concentracao_obstaculo(conc, obst, janela_fit=None):
    """
    Processa um par concentração/obstáculo:
      - Lê todos os arquivos .csv
      - Calcula MSD por arquivo (TrajPy)
      - Faz fit em log–log por arquivo (janela opcional)
      - Salva CSV por par e um plot por par
    Retorna dict resumo com média/DP de α para o par.
    """
    base = BASE_ROOT / conc / obst
    out_prefix = f"msd_ensemble_hunter_{obst}_{conc.replace('*','_')}"

    print("\n" + "="*70)
    print(f"PROCESSANDO: {conc} / {obst}")
    print("="*70)

    if not base.exists():
        print(f"  [AVISO] Diretório não encontrado: {base}")
        return None

    arquivos = sorted(base.glob(PATTERN))
    if not arquivos:
        print(f"  [AVISO] Nenhum arquivo encontrado em {base}")
        return None

    dados_por_arquivo = []
    todos_msd = []  # (taus_valid, msd_valid, alpha, nome, mask_fit)

    for arq in arquivos:
        try:
            df = pd.read_csv(arq, usecols=["timestep", "cell_id", "x", "y"])
            df = df.sort_values(["cell_id", "timestep"])

            trajetorias = []
            comprimentos = []

            for _, g in df.groupby("cell_id"):
                x = g["x"].to_numpy(float)
                y = g["y"].to_numpy(float)
                x, y = unwrap_xy(x, y, Lx, Ly)
                traj = np.column_stack([x, y])
                if len(traj) >= MIN_TRAJ_LENGTH:
                    trajetorias.append(traj)
                    comprimentos.append(len(traj))

            if not trajetorias:
                print(f"  [AVISO] {arq.name}: nenhuma trajetória válida.")
                continue

            msd_ensemble, n_contrib = calcular_msd_por_arquivo(
                trajetorias, MAX_TAU, contexto=f"{conc}/{obst}/{arq.name}"
            )
            taus = np.arange(1, MAX_TAU + 1)
            mask_valid = (n_contrib >= MIN_CONTRIB_POINT) & np.isfinite(msd_ensemble)
            taus_valid = taus[mask_valid]
            msd_valid = msd_ensemble[mask_valid]

            alpha, K, r2, mask_fit = calcular_alpha(taus_valid, msd_valid, janela_fit)

            dados_por_arquivo.append({
                "concentracao": conc,
                "obstaculo": obst,
                "arquivo": arq.name,
                "n_trajetorias": len(trajetorias),
                "comprimento_medio": float(np.mean(comprimentos)),
                "alpha": alpha,
                "K": K,
                "r2": r2,
                "pontos_fit": int(np.sum(mask_fit)) if mask_fit is not None else int(np.sum(mask_valid)),
            })

            todos_msd.append((taus_valid, msd_valid, alpha, arq.name, mask_fit))

        except Exception as e:
            print(f"  [ERRO] Falha em {arq.name}: {e}")
            continue

    if not dados_por_arquivo:
        print(f"  [AVISO] Nenhum dado válido em {base}")
        return None

    # Salva CSV por par
    df_par = pd.DataFrame(dados_por_arquivo)
    csv_par = OUT_DIR / f"{out_prefix}.csv"
    df_par.to_csv(csv_par, index=False, float_format="%.6f")
    print(f"  CSV salvo: {csv_par}")

    # Plot por par (curvas + região de fit marcada e hist de α)
    alphas_valid = df_par["alpha"].dropna()
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(todos_msd)))
    for i, (t_v, m_v, a, nome, mfit) in enumerate(todos_msd):
        c = colors[i]
        ax1.plot(t_v, m_v, color=c, alpha=0.6, lw=1.5)
        if mfit is not None and np.any(mfit):
            ax1.plot(t_v[mfit], m_v[mfit], color=c, alpha=0.95, lw=2.2)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Time Lag τ (frames)")
    ax1.set_ylabel("MSD(τ)")
    jan_txt = f"[{JANELA_FIT[0]},{JANELA_FIT[1]}]" if JANELA_FIT else "todos os válidos"
    ax1.set_title(f"MSD por arquivo • {conc} / {obst}\nRegião do fit destacada (janela: {jan_txt})")

    if len(alphas_valid) > 0:
        ax3.hist(alphas_valid, bins=30, alpha=0.7, edgecolor="black", density=True)
        ax3.axvline(alphas_valid.mean(), color="blue", lw=2,
                    label=f"Média: α={alphas_valid.mean():.3f}")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Nenhum fit válido", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Expoente α")
    ax3.set_ylabel("Densidade")
    ax3.set_title(f"Distribuição de α • {conc} / {obst}")

    plt.tight_layout()
    fig_par = OUT_DIR / f"{out_prefix}_plot.png"
    plt.savefig(fig_par, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura salva: {fig_par}")

    return {
        "concentracao": conc,
        "obstaculo": obst,
        "media_alpha": float(alphas_valid.mean()) if len(alphas_valid) > 0 else np.nan,
        "std_alpha": float(alphas_valid.std()) if len(alphas_valid) > 0 else np.nan,
        "n_arquivos": len(dados_por_arquivo),
        "n_validos": int(len(alphas_valid)),
    }

# ========= Função Principal =========
def main():
    print("INICIANDO VARREDURA")
    print(f"Base: {BASE_ROOT}")
    print(f"Janela de fit: {JANELA_FIT}")
    print(f"Saídas: {OUT_DIR}")

    resultados_gerais = []

    for conc in CONCENTRACOES:
        for obst in OBSTACULOS:
            res = processar_concentracao_obstaculo(conc, obst, JANELA_FIT)
            if res is not None:
                resultados_gerais.append(res)

    if not resultados_gerais:
        raise RuntimeError("Nenhum resultado gerado. Verifique paths e listas.")

    df_resumo = pd.DataFrame(resultados_gerais)
    resumo_csv = OUT_DIR / "resumo_geral_completo.csv"
    df_resumo.to_csv(resumo_csv, index=False, float_format="%.6f")
    print(f"\nResumo geral salvo: {resumo_csv}")

    # Gráfico comparativo por concentração
    fig, axes = plt.subplots(1, len(CONCENTRACOES), figsize=(6*len(CONCENTRACOES), 5))
    if len(CONCENTRACOES) == 1:
        axes = [axes]

    for i, conc in enumerate(CONCENTRACOES):
        sub = df_resumo[df_resumo["concentracao"] == conc]
        if sub.empty:
            axes[i].text(0.5, 0.5, "Sem dados", ha="center", va="center",
                         transform=axes[i].transAxes)
            continue
        obsts = sub["obstaculo"].tolist()
        medias = sub["media_alpha"].to_numpy()
        stds = sub["std_alpha"].to_numpy()
        x = np.arange(len(obsts))
        bars = axes[i].bar(x, medias, yerr=stds, alpha=0.8, capsize=5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(obsts, rotation=45, ha="right")
        axes[i].set_title(f"Concentração: {conc}")
        axes[i].set_ylabel("α médio por obstáculo")
        axes[i].grid(True, axis="y", alpha=0.3)
        for b, m in zip(bars, medias):
            axes[i].text(b.get_x()+b.get_width()/2, m + 0.01, f"{m:.3f}",
                         ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_comp = OUT_DIR / "comparativo_completo.png"
    plt.savefig(fig_comp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico comparativo salvo: {fig_comp}")

    print("\nPROCESSAMENTO CONCLUÍDO!")

if __name__ == "__main__":
    main()
