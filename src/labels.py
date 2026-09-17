"""Rótulo ``alagou_3h`` e frequência histórica de alagamento (seções 3, 6.2 e 6.3).

Definições
----------
- ``alagou_3h(p, t) = 1`` se existe alagamento registrado a até ``R`` metros do ponto ``p``
  com início no intervalo **(t, t + 3h]** (aberto à esquerda, fechado à direita).
- ``ja_alagado(p, t) = True`` se existe alagamento a até ``R`` metros com
  ``inicio <= t < fim``. Essas horas saem do dataset: nelas o ponto já está alagado, então
  não há o que prever. Sem horário de término (ou com duração acima de
  ``DURACAO_MAXIMA_EVENTO_H``, provável erro de registro), assume-se ``DURACAO_PADRAO_EVENTO_H``.
- ``evento_alvo_id``: o primeiro evento que torna a linha positiva (para métricas por evento).

É um **rótulo proxy**: o evento operacional é o bueiro transbordar, mas não existe histórico
disso; usamos o alagamento registrado nas proximidades (limitação declarada no TCC).

Frequência histórica sem vazamento (``freq_hist_alagamento``)
-------------------------------------------------------------
Para uma linha do ano ``Y``, usa só eventos de anos **anteriores** a ``Y``:
``freq = eventos(ano < Y) / (Y - ano_inicio_dados)``, em eventos por ano. No primeiro ano da
série fica ``NaN`` (sem histórico). Em produção usa-se ``Y = último ano + 1``, ou seja, todo o
histórico disponível. Ver docs/adr/0003 — é mais rigoroso que "calcular só no treino", porque
nem as linhas de treino enxergam o próprio evento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DURACAO_MAXIMA_EVENTO_H,
    DURACAO_PADRAO_EVENTO_H,
    JANELA_PREVISAO_HORAS,
    RAIO_ROTULO_M,
)
from src.geo import indices_no_raio
from src.tempo import utc_para_local

COLUNAS_EVENTO = ["evento_id", "inicio_utc", "fim_utc"]


def _segundos(valores) -> np.ndarray:
    """Datetimes UTC -> inteiros em segundos (independe da resolução ns/us do pandas)."""
    indice = pd.DatetimeIndex(valores)
    return indice.as_unit("s").asi8


def eventos_por_ponto(pontos: pd.DataFrame, eventos: pd.DataFrame,
                      raio_m: float = RAIO_ROTULO_M) -> dict[str, pd.DataFrame]:
    """Para cada ``ponto_id``, os eventos a até ``raio_m`` metros, ordenados pelo início."""
    indices = indices_no_raio(pontos["lat"].to_numpy(), pontos["lon"].to_numpy(),
                              eventos["lat"].to_numpy(), eventos["lon"].to_numpy(), raio_m)
    saida = {}
    for ponto_id, idx in zip(pontos["ponto_id"], indices, strict=True):
        sub = eventos.iloc[np.asarray(idx, dtype=int)][COLUNAS_EVENTO]
        saida[ponto_id] = sub.sort_values("inicio_utc").reset_index(drop=True)
    return saida


def rotular_ponto(horas: pd.DatetimeIndex, eventos_ponto: pd.DataFrame,
                  janela_h: int = JANELA_PREVISAO_HORAS,
                  duracao_padrao_h: float = DURACAO_PADRAO_EVENTO_H,
                  duracao_maxima_h: float = DURACAO_MAXIMA_EVENTO_H) -> pd.DataFrame:
    """Rótulo e exclusão para todas as horas de um ponto (vetorizado com ``searchsorted``).

    Args:
        horas: horas cheias em UTC (``t``).
        eventos_ponto: eventos próximos ao ponto, com ``inicio_utc``/``fim_utc``.

    Returns:
        DataFrame indexado por ``horas`` com ``alagou_3h`` (int8), ``ja_alagado`` (bool)
        e ``evento_alvo_id`` (str ou None).
    """
    t = _segundos(horas)
    n = len(t)
    if eventos_ponto is None or len(eventos_ponto) == 0:
        return pd.DataFrame({
            "alagou_3h": np.zeros(n, dtype="int8"),
            "ja_alagado": np.zeros(n, dtype=bool),
            "evento_alvo_id": pd.Series([None] * n, dtype="object").to_numpy(),
        }, index=horas)

    ev = eventos_ponto.sort_values("inicio_utc")
    inicios = _segundos(ev["inicio_utc"])
    fins = pd.to_datetime(ev["fim_utc"], utc=True)
    inicios_ts = pd.to_datetime(ev["inicio_utc"], utc=True)
    fim_padrao = inicios_ts + pd.Timedelta(hours=duracao_padrao_h)
    # Sem término, término antes do início ou duração absurda (> duracao_maxima_h) -> duração padrão
    plausivel = fins.notna() & (fins > inicios_ts) & (fins - inicios_ts <= pd.Timedelta(hours=duracao_maxima_h))
    fins_validos = fins.where(plausivel, fim_padrao)
    fins_s = _segundos(fins_validos)
    ids = ev["evento_id"].to_numpy()

    janela_s = int(janela_h * 3600)
    esquerda = np.searchsorted(inicios, t, side="right")  # 1º evento com início > t
    direita = np.searchsorted(inicios, t + janela_s, side="right")  # 1º com início > t+janela
    alagou = direita > esquerda

    # Já alagado: entre os eventos com início <= t, o maior "fim" ainda está depois de t?
    maior_fim = np.maximum.accumulate(fins_s)
    ultimo_iniciado = esquerda - 1
    ja_alagado = np.zeros(n, dtype=bool)
    tem = ultimo_iniciado >= 0
    ja_alagado[tem] = maior_fim[ultimo_iniciado[tem]] > t[tem]

    alvo = np.full(n, None, dtype=object)
    alvo[alagou] = ids[esquerda[alagou]]
    return pd.DataFrame({"alagou_3h": alagou.astype("int8"), "ja_alagado": ja_alagado,
                         "evento_alvo_id": alvo}, index=horas)


def taxa_historica_anual(eventos_ponto: dict[str, pd.DataFrame], anos: list[int],
                         ano_inicio_dados: int) -> pd.DataFrame:
    """``freq_hist_alagamento`` por ponto × ano, usando só anos anteriores (eventos/ano).

    Args:
        eventos_ponto: saída de ``eventos_por_ponto``.
        anos: anos para os quais calcular (inclua ``último + 1`` para o valor de produção).
        ano_inicio_dados: primeiro ano com eventos coletados.
    """
    linhas = []
    for ponto_id, ev in eventos_ponto.items():
        anos_ev = (utc_para_local(pd.to_datetime(ev["inicio_utc"], utc=True)).dt.year.to_numpy()
                   if len(ev) else np.array([], dtype=int))
        for ano in anos:
            anos_hist = ano - ano_inicio_dados
            if anos_hist < 1:
                freq = np.nan
            else:
                n = int(((anos_ev < ano) & (anos_ev >= ano_inicio_dados)).sum())
                freq = n / anos_hist
            linhas.append({"ponto_id": ponto_id, "ano": ano, "freq_hist_alagamento": freq})
    return pd.DataFrame(linhas)
