"""Chuva horária do Open-Meteo — a MESMA fonte no treino e no serviço ao vivo.

APIs usadas (todas sem chave, uso não comercial):

- **Arquivo (reanálise)**      ``archive-api.open-meteo.com/v1/archive``
- **Previsões históricas**     ``historical-forecast-api.open-meteo.com/v1/forecast``
  (mais parecida com o dado ao vivo; verificar a cobertura de datas no portão de decisão)
- **Ao vivo**                  ``api.open-meteo.com/v1/forecast`` com ``past_hours``/``forecast_hours``
- **Elevação**                 ``api.open-meteo.com/v1/elevation`` (MDE de 90 m, até 100 pontos/chamada)

Detalhes que importam para não haver divergência treino × produção:

1. ``precipitation`` no instante ``T`` é a soma da hora anterior (``T-1h`` a ``T``).
   Por isso a hora de referência ao vivo é a hora cheia anterior (ver ``src/tempo.py``).
2. Pedimos tudo em **UTC** (``timezone=GMT``) e com ``timeformat=unixtime``.
3. Os pontos são agrupados em **células** de ``PASSO_GRADE_CACHE_GRAUS`` e a chuva é pedida
   no **centro da célula**, tanto no treino quanto ao vivo. Assim pontos da mesma célula
   recebem a mesma série nos dois momentos, e o cache ao vivo reaproveita as chamadas.
4. Limites do plano gratuito (2026): ~10.000 chamadas/dia, 5.000/hora, 600/minuto; pedidos
   longos contam como várias chamadas. O download histórico é feito por célula × ano, com
   pausa e retomada (arquivos já baixados são pulados).

Uso pela linha de comando (download do histórico para os pontos escolhidos)::

    python -m src.collect.open_meteo --pontos data/processed/pontos.csv \
        --inicio 2017-01-01 --fim 2025-12-31 --fonte arquivo
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    ARQ_CHUVA_HORARIA,
    DIR_INTERIM,
    DIR_RAW,
    HORAS_FUTURAS_AO_VIVO,
    HORAS_PASSADAS_AO_VIVO,
    OPEN_METEO_MODELO,
    PASSO_GRADE_CACHE_GRAUS,
    URL_OPEN_METEO_AO_VIVO,
    URL_OPEN_METEO_ARQUIVO,
    URL_OPEN_METEO_PREVISOES_HISTORICAS,
)

log = logging.getLogger(__name__)

URL_ELEVACAO = "https://api.open-meteo.com/v1/elevation"
FONTES_HISTORICAS = {
    "arquivo": URL_OPEN_METEO_ARQUIVO,
    "previsoes_historicas": URL_OPEN_METEO_PREVISOES_HISTORICAS,
}
DIR_RAW_OPEN_METEO = DIR_RAW / "open_meteo"
ARQ_CELULAS = DIR_INTERIM / "celulas_chuva.csv"


@dataclass
class SerieChuva:
    """Resultado de uma coordenada: série horária em UTC (mm na hora que TERMINA no índice)."""

    latitude: float
    longitude: float
    serie: pd.Series
    latitude_grade: float | None = None
    longitude_grade: float | None = None
    elevacao: float | None = None
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Células de grade
# ---------------------------------------------------------------------------
def celula_grade(lat: float, lon: float, passo: float = PASSO_GRADE_CACHE_GRAUS) -> tuple[float, float]:
    """Centro da célula que contém (lat, lon). Ex.: passo 0,05 -> (-23.5, -46.6)."""
    lat_c = round(round(lat / passo) * passo, 4)
    lon_c = round(round(lon / passo) * passo, 4)
    return lat_c, lon_c


def id_celula(lat: float, lon: float, passo: float = PASSO_GRADE_CACHE_GRAUS) -> str:
    """Identificador textual estável da célula, ex.: ``c_-23.50_-46.60``."""
    lat_c, lon_c = celula_grade(lat, lon, passo)
    return f"c_{lat_c:.2f}_{lon_c:.2f}"


# ---------------------------------------------------------------------------
# Conversão da resposta
# ---------------------------------------------------------------------------
def _bloco_para_serie(bloco: dict, lat: float, lon: float) -> SerieChuva:
    horario = bloco.get("hourly") or {}
    tempos = pd.to_datetime(np.asarray(horario.get("time", []), dtype="int64"), unit="s", utc=True)
    valores = pd.to_numeric(pd.Series(horario.get("precipitation", []), dtype="float64"))
    serie = pd.Series(valores.to_numpy(), index=pd.DatetimeIndex(tempos, name="data_hora_utc"),
                      name="chuva_mm")
    serie = serie[~serie.index.duplicated(keep="last")].sort_index()
    return SerieChuva(
        latitude=lat, longitude=lon, serie=serie,
        latitude_grade=bloco.get("latitude"), longitude_grade=bloco.get("longitude"),
        elevacao=bloco.get("elevation"),
    )


def _requisitar(url: str, params: dict, sessao: requests.Session | None, timeout: float) -> list[dict]:
    cliente = sessao or requests
    resp = cliente.get(url, params=params, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"Open-Meteo {resp.status_code}: {resp.text[:300]}")
    dados = resp.json()
    return dados if isinstance(dados, list) else [dados]


def _params_coordenadas(coordenadas: list[tuple[float, float]]) -> dict:
    return {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in coordenadas),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in coordenadas),
    }


def _params_comuns(modelo: str | None) -> dict:
    params = {"hourly": "precipitation", "timezone": "GMT", "timeformat": "unixtime"}
    if modelo:
        params["models"] = modelo
    return params


# ---------------------------------------------------------------------------
# Histórico (treino)
# ---------------------------------------------------------------------------
def buscar_historico(
    coordenadas: list[tuple[float, float]],
    inicio: str,
    fim: str,
    fonte: str = "arquivo",
    sessao: requests.Session | None = None,
    modelo: str | None = OPEN_METEO_MODELO,
    timeout: float = 120,
) -> list[SerieChuva]:
    """Chuva horária entre ``inicio`` e ``fim`` (datas AAAA-MM-DD, inclusivas) para várias coordenadas."""
    if fonte not in FONTES_HISTORICAS:
        raise ValueError(f"fonte deve ser uma de {list(FONTES_HISTORICAS)}")
    params = {**_params_comuns(modelo), **_params_coordenadas(coordenadas),
              "start_date": inicio, "end_date": fim}
    blocos = _requisitar(FONTES_HISTORICAS[fonte], params, sessao, timeout)
    return [_bloco_para_serie(b, lat, lon) for b, (lat, lon) in zip(blocos, coordenadas, strict=True)]


# ---------------------------------------------------------------------------
# Ao vivo (serviço)
# ---------------------------------------------------------------------------
def buscar_recente(
    coordenadas: list[tuple[float, float]],
    horas_passadas: int = HORAS_PASSADAS_AO_VIVO,
    horas_futuras: int = HORAS_FUTURAS_AO_VIVO,
    sessao: requests.Session | None = None,
    modelo: str | None = OPEN_METEO_MODELO,
    timeout: float = 5,
) -> list[SerieChuva]:
    """Últimas ``horas_passadas`` horas + próximas ``horas_futuras`` para várias coordenadas.

    Uma única chamada HTTP atende todas as coordenadas (a API aceita listas separadas por
    vírgula), o que economiza a cota no ``/predict/batch``.
    """
    params = {**_params_comuns(modelo), **_params_coordenadas(coordenadas),
              "past_hours": horas_passadas, "forecast_hours": horas_futuras}
    blocos = _requisitar(URL_OPEN_METEO_AO_VIVO, params, sessao, timeout)
    return [_bloco_para_serie(b, lat, lon) for b, (lat, lon) in zip(blocos, coordenadas, strict=True)]


def buscar_elevacao(
    coordenadas: list[tuple[float, float]],
    sessao: requests.Session | None = None,
    timeout: float = 30,
) -> list[float | None]:
    """Elevação (m) do MDE de 90 m do Open-Meteo, em lotes de 100 coordenadas."""
    saida: list[float | None] = []
    for i in range(0, len(coordenadas), 100):
        lote = coordenadas[i : i + 100]
        cliente = sessao or requests
        resp = cliente.get(URL_ELEVACAO, params=_params_coordenadas(lote), timeout=timeout)
        resp.raise_for_status()
        saida.extend(resp.json().get("elevation", [None] * len(lote)))
    return saida


# ---------------------------------------------------------------------------
# Download em lote para os pontos do dataset
# ---------------------------------------------------------------------------
def baixar_chuva_pontos(
    pontos: pd.DataFrame,
    inicio: str,
    fim: str,
    fonte: str = "arquivo",
    pausa_s: float = 1.0,
    dir_saida: Path = DIR_RAW_OPEN_METEO,
) -> pd.DataFrame:
    """Baixa a chuva de todas as células que contêm os pontos, um arquivo por célula × ano.

    Retorna a tabela longa ``celula_id, data_hora_utc, chuva_mm`` e grava:
    - ``data/raw/open_meteo/<fonte>/<celula>_<ano>.parquet`` (bruto, retomável)
    - ``data/interim/chuva_horaria.parquet`` (união)
    - ``data/interim/celulas_chuva.csv`` (centro e grade de cada célula)
    """
    from src.collect.http import criar_sessao

    sessao = criar_sessao(usar_cache=True)
    destino = Path(dir_saida) / fonte
    destino.mkdir(parents=True, exist_ok=True)

    celulas = (
        pontos.assign(celula_id=[id_celula(a, b) for a, b in zip(pontos.lat, pontos.lon, strict=True)])
        .groupby("celula_id").first()[["lat", "lon"]]
    )
    anos = range(pd.Timestamp(inicio).year, pd.Timestamp(fim).year + 1)
    info_celulas = []
    partes = []
    for celula_id, linha in celulas.iterrows():
        lat_c, lon_c = celula_grade(linha.lat, linha.lon)
        for ano in anos:
            arq = destino / f"{celula_id}_{ano}.parquet"
            if arq.exists():
                partes.append(pd.read_parquet(arq))
                continue
            ini = max(pd.Timestamp(inicio), pd.Timestamp(f"{ano}-01-01")).strftime("%Y-%m-%d")
            fi = min(pd.Timestamp(fim), pd.Timestamp(f"{ano}-12-31")).strftime("%Y-%m-%d")
            log.info("Open-Meteo %s: %s %s..%s", fonte, celula_id, ini, fi)
            resultado = buscar_historico([(lat_c, lon_c)], ini, fi, fonte=fonte, sessao=sessao)[0]
            df = resultado.serie.rename("chuva_mm").reset_index()
            df.insert(0, "celula_id", celula_id)
            # A grade usada pela API fica registrada no próprio arquivo (retomada sem perder a info)
            df["lat_grade"] = resultado.latitude_grade
            df["lon_grade"] = resultado.longitude_grade
            df.to_parquet(arq, index=False)
            partes.append(df)
            time.sleep(pausa_s)
        info_celulas.append({"celula_id": celula_id, "lat": lat_c, "lon": lon_c})

    chuva = pd.concat(partes, ignore_index=True).drop_duplicates(["celula_id", "data_hora_utc"])
    grades = chuva.groupby("celula_id")[["lat_grade", "lon_grade"]].first().reset_index()
    chuva = chuva[["celula_id", "data_hora_utc", "chuva_mm"]].copy()
    chuva["chuva_mm"] = chuva["chuva_mm"].astype("float32")
    ARQ_CHUVA_HORARIA.parent.mkdir(parents=True, exist_ok=True)
    chuva.to_parquet(ARQ_CHUVA_HORARIA, index=False)
    pd.DataFrame(info_celulas).merge(grades, on="celula_id", how="left").to_csv(ARQ_CELULAS, index=False)
    log.info("Chuva salva em %s (%d linhas, %d células)", ARQ_CHUVA_HORARIA, len(chuva),
             chuva.celula_id.nunique())
    return chuva


def comparar_historico_e_ao_vivo(lat: float, lon: float, dias: int = 3) -> pd.DataFrame:
    """Tarefa do portão de decisão: compara, nas mesmas horas, arquivo × previsões históricas × ao vivo.

    Retorna uma tabela lado a lado para as últimas ``dias`` × 24 horas (o arquivo de reanálise
    tem alguns dias de atraso, então as horas mais recentes podem vir vazias nele).
    """
    fim = pd.Timestamp.now(tz="UTC").normalize()
    ini = (fim - pd.Timedelta(days=dias + 7)).strftime("%Y-%m-%d")
    fim_s = fim.strftime("%Y-%m-%d")
    colunas = {}
    for fonte in FONTES_HISTORICAS:
        try:
            colunas[fonte] = buscar_historico([(lat, lon)], ini, fim_s, fonte=fonte)[0].serie
        except Exception as erro:  # noqa: BLE001 - diagnóstico
            log.warning("Falha em %s: %s", fonte, erro)
    colunas["ao_vivo"] = buscar_recente([(lat, lon)], horas_passadas=dias * 24, horas_futuras=0)[0].serie
    return pd.DataFrame(colunas)


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pontos", type=Path, required=True, help="CSV com colunas lat, lon")
    parser.add_argument("--inicio", required=True, help="AAAA-MM-DD (o ECMWF IFS do arquivo começa em 2017-01-01)")
    parser.add_argument("--fim", required=True, help="AAAA-MM-DD")
    parser.add_argument("--fonte", choices=list(FONTES_HISTORICAS), default="arquivo")
    parser.add_argument("--pausa", type=float, default=1.0, help="segundos entre chamadas reais")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pontos = pd.read_csv(args.pontos)
    baixar_chuva_pontos(pontos, args.inicio, args.fim, fonte=args.fonte, pausa_s=args.pausa)


if __name__ == "__main__":
    _principal()
