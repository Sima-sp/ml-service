"""Conversões de data/hora usadas em todo o pipeline.

Por que UTC internamente?
    São Paulo teve horário de verão até fevereiro de 2019. Em hora local existem horas
    repetidas (fim do horário de verão) e horas que não existem (início). Guardando tudo em
    UTC as séries horárias ficam contínuas, e o horário local só é calculado quando precisamos
    dele (features de calendário e leitura das fontes que publicam em hora local, como o CGE).

Hora de referência (``t``)
    O dataset tem uma linha por ponto × hora cheia. Ao vivo, uma previsão pedida às 14:20
    usa ``t = 14:00`` (hora cheia anterior), porque o valor de chuva "14:00" do Open-Meteo é o
    acumulado de 13:00 a 14:00 — o último intervalo completo. Assim o serviço calcula as
    features exatamente como no treino.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import FUSO_LOCAL


def agora_utc() -> pd.Timestamp:
    """Instante atual em UTC (tz-aware)."""
    return pd.Timestamp(datetime.now(UTC))


def garantir_utc(valor) -> pd.Timestamp:
    """Converte um datetime/str para ``pd.Timestamp`` em UTC.

    Horários sem fuso são interpretados como hora LOCAL de São Paulo (é o que as fontes e o
    backend enviam quando omitem o fuso).
    """
    ts = pd.Timestamp(valor)
    if ts.tzinfo is None:
        ts = ts.tz_localize(FUSO_LOCAL, ambiguous=True, nonexistent="shift_forward")
    return ts.tz_convert("UTC")


def hora_referencia(valor) -> pd.Timestamp:
    """Hora cheia (UTC) usada como ``t`` nas features. Ex.: 14:20 local -> 14:00 local em UTC."""
    return garantir_utc(valor).floor("h")


def local_para_utc(serie: pd.Series) -> pd.Series:
    """Converte uma série de datetimes locais "ingênuos" (sem fuso) para UTC.

    - Hora ambígua (fim do horário de verão): assume a primeira ocorrência (ainda em verão).
    - Hora inexistente (início do horário de verão): empurra para a próxima hora válida.
    """
    serie = pd.to_datetime(serie)
    if serie.dt.tz is not None:
        return serie.dt.tz_convert("UTC")
    ambiguas = np.ones(len(serie), dtype=bool)
    return serie.dt.tz_localize(
        FUSO_LOCAL, ambiguous=ambiguas, nonexistent="shift_forward"
    ).dt.tz_convert("UTC")


def utc_para_local(valor):
    """Converte um Timestamp/Series/DatetimeIndex em UTC para America/Sao_Paulo."""
    if isinstance(valor, pd.Series):
        return valor.dt.tz_convert(FUSO_LOCAL)
    return valor.tz_convert(FUSO_LOCAL)


def para_iso_local(ts: pd.Timestamp) -> str:
    """ISO-8601 com o deslocamento de São Paulo, sem microssegundos (ex.: 2026-09-16T14:20:03-03:00)."""
    return utc_para_local(garantir_utc(ts)).floor("s").isoformat()
