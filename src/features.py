"""Cálculo das features — **o mesmo código no treino e no serviço ao vivo**.

Regra de ouro (seção 5.3): toda feature usada no treino precisa existir ao vivo, com a mesma
fonte e a mesma definição. Por isso o serviço (``app/predictor.py``) não recalcula nada por
conta própria: chama ``features_ao_vivo()``, que usa exatamente ``calcular_features_chuva()``
e ``calcular_features_tempo()``, as mesmas funções usadas por ``src/dataset.py``.

Convenção da série de chuva
---------------------------
Série horária em UTC; o valor no instante ``T`` é a chuva (mm) entre ``T-1h`` e ``T``
(padrão do Open-Meteo). A linha da hora ``t`` só usa valores com índice ``<= t``, exceto
``chuva_prevista_3h`` (índices ``t+1h..t+3h``), que fica FORA do modelo principal.

Features
--------
Chuva (passado, até ``t``):
    ``chuva_1h``            chuva da última hora
    ``chuva_3h``            soma das últimas 3 h
    ``chuva_6h``            soma das últimas 6 h
    ``chuva_24h``           soma das últimas 24 h
    ``chuva_72h``           soma das últimas 72 h (solo encharcado / sistema carregado)
    ``max_chuva_1h_24h``    maior chuva horária nas últimas 24 h (intensidade de pico)
    ``horas_desde_ultima_chuva``  horas desde a última hora com chuva >= ``LIMIAR_CHUVA_MM``
                            (0 = chovendo agora; teto ``MAX_HORAS_SEM_CHUVA``)
Chuva (futuro — experimental, desligada por padrão):
    ``chuva_prevista_3h``   soma de ``t+1h..t+3h``. No treino com o arquivo de reanálise isso
                            é chuva OBSERVADA no futuro (vazamento). Só é legítima com a API de
                            previsões históricas. Ver docs/recomendacoes.md (R1).
Tempo (hora LOCAL de São Paulo):
    ``hora_dia`` (0–23), ``mes`` (1–12), ``dia_semana`` (0 = segunda), ``estacao_chuvosa`` (out–mar)
Ponto (fixas ou anuais):
    ``lat``, ``lon``, ``elevacao``, ``dist_corrego_m``, ``zona``, ``freq_hist_alagamento``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    LIMIAR_CHUVA_MM,
    MAX_HORAS_SEM_CHUVA,
    MESES_ESTACAO_CHUVOSA,
    USAR_CHUVA_PREVISTA,
    ZONAS,
)
from src.tempo import hora_referencia, utc_para_local

COLUNAS_CHUVA = [
    "chuva_1h", "chuva_3h", "chuva_6h", "chuva_24h", "chuva_72h",
    "max_chuva_1h_24h", "horas_desde_ultima_chuva",
]
COLUNAS_CHUVA_PREVISTA = ["chuva_prevista_3h"]
COLUNAS_TEMPO = ["hora_dia", "mes", "dia_semana", "estacao_chuvosa"]
COLUNAS_PONTO = ["lat", "lon", "elevacao", "dist_corrego_m", "zona", "freq_hist_alagamento"]
COLUNAS_CATEGORICAS = ["zona"]
HORAS_MINIMAS_SERIE = 72  # a maior janela (chuva_72h)


def lista_features(usar_chuva_prevista: bool = USAR_CHUVA_PREVISTA) -> list[str]:
    """Lista ordenada das features do modelo."""
    chuva = COLUNAS_CHUVA + (COLUNAS_CHUVA_PREVISTA if usar_chuva_prevista else [])
    return chuva + COLUNAS_TEMPO + COLUNAS_PONTO


# ---------------------------------------------------------------------------
# Chuva
# ---------------------------------------------------------------------------
def regularizar_serie(serie: pd.Series) -> pd.Series:
    """Garante índice horário contínuo em UTC (horas faltantes viram NaN) e valores >= 0."""
    s = serie.copy()
    indice = pd.DatetimeIndex(s.index)
    indice = indice.tz_localize("UTC") if indice.tz is None else indice.tz_convert("UTC")
    s.index = indice.floor("h")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s) == 0:
        return s.astype("float64")
    completo = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return pd.to_numeric(s, errors="coerce").astype("float64").clip(lower=0).reindex(completo)


def horas_desde_ultima_chuva(valores: np.ndarray, limiar: float = LIMIAR_CHUVA_MM,
                             teto: int = MAX_HORAS_SEM_CHUVA) -> np.ndarray:
    """Horas desde a última hora chuvosa (0 se chove na hora), limitado a ``teto``.

    Antes da primeira chuva da série, o valor só é conhecido se a série já cobre ``teto``
    horas secas até ali (aí vale ``teto``); caso contrário fica ``NaN``. Por isso uma série ao
    vivo de exatamente 72 h secas dá 72, igual ao treino.
    """
    valores = np.asarray(valores, dtype="float64")
    n = len(valores)
    posicoes = np.arange(n)
    chuvosa = np.nan_to_num(valores, nan=0.0) >= limiar
    ultima = np.maximum.accumulate(np.where(chuvosa, posicoes, -1)) if n else np.array([], dtype=int)
    horas = (posicoes - ultima).astype("float64")
    sem_historico = ultima < 0
    horas[sem_historico] = np.where(posicoes[sem_historico] + 1 >= teto, teto, np.nan)
    return np.minimum(horas, teto)


def calcular_features_chuva(serie: pd.Series) -> pd.DataFrame:
    """Todas as features de chuva para cada hora da série (mesma função no treino e ao vivo).

    Janelas exigem a janela completa (``min_periods`` = tamanho): se faltar alguma hora, o
    resultado é ``NaN`` em vez de uma soma subestimada.
    """
    s = regularizar_serie(serie)
    somar = lambda n: s.rolling(n, min_periods=n).sum().clip(lower=0)  # noqa: E731
    df = pd.DataFrame(index=s.index)
    df["chuva_1h"] = s
    df["chuva_3h"] = somar(3)
    df["chuva_6h"] = somar(6)
    df["chuva_24h"] = somar(24)
    df["chuva_72h"] = somar(72)
    df["max_chuva_1h_24h"] = s.rolling(24, min_periods=24).max()
    df["horas_desde_ultima_chuva"] = horas_desde_ultima_chuva(s.to_numpy())
    # Futuro (t+1..t+3): só para experimento com previsões históricas / serviço ao vivo
    df["chuva_prevista_3h"] = somar(3).shift(-3)
    return df.round(3)


def completar_resumo_chuva(resumo: dict, limiar: float = LIMIAR_CHUVA_MM) -> dict:
    """Completa um resumo parcial (ex.: só ``chuva_1h``/``chuva_3h``/``chuva_24h``).

    Usado quando o backend/simulador INFORMA a chuva em vez de deixar o serviço buscar.
    É uma aproximação conservadora baseada em monotonicidade
    (``chuva_1h <= chuva_3h <= chuva_6h <= chuva_24h <= chuva_72h``); documentada no README.
    Para cenários de demo prefira ``chuvaHoraria`` (série completa), que usa o caminho exato.
    """
    r = {k: (None if v is None else float(v)) for k, v in resumo.items()}
    c1 = r.get("chuva_1h")
    c3 = r.get("chuva_3h")
    if c1 is None:
        c1 = (c3 / 3.0) if c3 is not None else 0.0
    c3 = max(c3 if c3 is not None else c1, c1)
    c6 = max(r.get("chuva_6h") or 0.0, c3)
    c24 = max(r.get("chuva_24h") or 0.0, c6)
    c72 = max(r.get("chuva_72h") or 0.0, c24)
    pico = r.get("max_chuva_1h_24h")
    pico = max(pico if pico is not None else 0.0, c1, c3 / 3.0)
    horas = r.get("horas_desde_ultima_chuva")
    if horas is None:
        if c1 >= limiar:
            horas = 0.0
        elif c3 >= limiar:
            horas = 1.0
        elif c6 >= limiar:
            horas = 3.0
        elif c24 >= limiar:
            horas = 6.0
        elif c72 >= limiar:
            horas = 24.0
        else:
            horas = float(MAX_HORAS_SEM_CHUVA)
    return {
        "chuva_1h": c1, "chuva_3h": c3, "chuva_6h": c6, "chuva_24h": c24, "chuva_72h": c72,
        "max_chuva_1h_24h": pico, "horas_desde_ultima_chuva": float(horas),
        "chuva_prevista_3h": r.get("chuva_prevista_3h"),
    }


def aplicar_sobrescritas_chuva(base: dict, sobrescritas: dict, limiar: float = LIMIAR_CHUVA_MM) -> dict:
    """Troca valores calculados pelos informados e restaura a coerência entre as janelas."""
    saida = dict(base)
    for chave, valor in sobrescritas.items():
        if valor is not None and chave in saida:
            saida[chave] = float(valor)
    ordem = ["chuva_1h", "chuva_3h", "chuva_6h", "chuva_24h", "chuva_72h"]
    for menor, maior in zip(ordem, ordem[1:], strict=False):
        if saida.get(menor) is not None and saida.get(maior) is not None:
            saida[maior] = max(saida[maior], saida[menor])
    if saida.get("chuva_1h") is not None:
        saida["max_chuva_1h_24h"] = max(saida.get("max_chuva_1h_24h") or 0.0, saida["chuva_1h"])
        if saida["chuva_1h"] >= limiar:
            saida["horas_desde_ultima_chuva"] = 0.0
    return saida


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------
def calcular_features_tempo(indice_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Features de calendário em hora LOCAL de São Paulo."""
    local = utc_para_local(pd.DatetimeIndex(indice_utc))
    return pd.DataFrame({
        "hora_dia": local.hour.astype("int8"),
        "mes": local.month.astype("int8"),
        "dia_semana": local.dayofweek.astype("int8"),
        "estacao_chuvosa": np.isin(local.month, MESES_ESTACAO_CHUVOSA).astype("int8"),
    }, index=indice_utc)


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def preparar_matriz(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Seleciona as features na ordem certa e padroniza os tipos (igual no treino e ao vivo)."""
    faltando = [c for c in features if c not in df.columns]
    if faltando:
        raise KeyError(f"Features ausentes: {faltando}")
    X = df[features].copy()
    for coluna in features:
        if coluna in COLUNAS_CATEGORICAS:
            X[coluna] = pd.Categorical(X[coluna].astype("object"), categories=ZONAS)
        else:
            X[coluna] = pd.to_numeric(X[coluna], errors="coerce").astype("float32")
    return X


def features_ao_vivo(serie_chuva: pd.Series | None, instante, estaticas: dict,
                     chuva_informada: dict | None = None,
                     permitir_hora_anterior: bool = True) -> tuple[pd.DataFrame, pd.Timestamp, list[str]]:
    """Uma linha de features para ``instante`` — o caminho usado pelo serviço.

    Args:
        serie_chuva: série horária (UTC) com pelo menos 72 h até a hora de referência.
            Pode ser ``None`` se ``chuva_informada`` trouxer o resumo.
        instante: horário da previsão (qualquer fuso); a hora de referência é a hora cheia anterior.
        estaticas: ``lat, lon, elevacao, dist_corrego_m, zona, freq_hist_alagamento``.
        chuva_informada: resumo opcional (``chuva_1h`` ...) que sobrescreve/complementa a série.
        permitir_hora_anterior: se a última hora ainda não tem valor, usa ``t - 1h``.

    Returns:
        (linha de features, hora de referência usada, avisos)
    """
    avisos: list[str] = []
    t_ref = hora_referencia(instante)
    valores: dict
    if serie_chuva is not None and len(serie_chuva):
        tabela = calcular_features_chuva(serie_chuva)
        linha = tabela.loc[t_ref] if t_ref in tabela.index else None
        incompleta = linha is None or linha[COLUNAS_CHUVA].isna().any()
        if incompleta and permitir_hora_anterior and (t_ref - pd.Timedelta(hours=1)) in tabela.index:
            anterior = tabela.loc[t_ref - pd.Timedelta(hours=1)]
            if not anterior[COLUNAS_CHUVA].isna().any():
                linha = anterior
                t_ref = t_ref - pd.Timedelta(hours=1)
                avisos.append("chuva da última hora indisponível; usada a hora anterior")
        if linha is None or linha[COLUNAS_CHUVA].isna().any():
            raise ValueError(
                f"Série de chuva insuficiente para {t_ref.isoformat()} "
                f"(são necessárias {HORAS_MINIMAS_SERIE} h completas até a hora de referência)"
            )
        valores = {k: (None if pd.isna(v) else float(v)) for k, v in linha.to_dict().items()}
        if chuva_informada:
            valores = aplicar_sobrescritas_chuva(valores, chuva_informada)
    elif chuva_informada:
        valores = completar_resumo_chuva(chuva_informada)
        avisos.append("features de chuva completadas a partir do resumo informado (aproximação)")
    else:
        raise ValueError("Sem série de chuva e sem chuva informada")

    tempo = calcular_features_tempo(pd.DatetimeIndex([t_ref])).iloc[0].to_dict()
    registro = {**valores, **tempo, **estaticas}
    return pd.DataFrame([registro], index=pd.DatetimeIndex([t_ref], name="data_hora_utc")), t_ref, avisos
