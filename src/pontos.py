"""Eventos unificados e escolha dos pontos monitorados (seção 6.1 do briefing).

Etapas:

1. ``geocodificar_cge``   coordenadas para os registros do CGE (só têm endereço).
2. ``unificar_eventos``   CGE + GeoSampa numa tabela única ``data/interim/eventos.parquet``.
3. ``atribuir_zona``      subprefeitura/zona de cada evento (para as contagens).
4. ``identificar_pontos`` DBSCAN (~200 m) -> agrupamentos = **pontos recorrentes**.
5. ``selecionar_pontos``  os N mais recorrentes da cidade + TODOS os recorrentes da Zona Norte.
6. ``enriquecer_pontos``  subprefeitura, zona, distância ao córrego, elevação, célula de chuva.

Decisão registrada (docs/adr/0003): os pontos são escolhidos só com eventos até
``ANO_LIMITE_SELECAO_PONTOS`` (fim do treino). Usar os anos de teste para escolher onde medir
favoreceria o modelo no teste (viés de seleção).

Linha de comando::

    python -m src.pontos geocodificar
    python -m src.pontos unificar      # usa data/raw/geosampa/subprefeitura.geojson
    python -m src.pontos selecionar    # + data/raw/geosampa/drenagem.geojson
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ANO_LIMITE_SELECAO_PONTOS,
    ARQ_EVENTOS,
    ARQ_EVENTOS_CGE,
    ARQ_EVENTOS_GEOSAMPA,
    ARQ_PONTOS,
    DIR_INTERIM,
    MIN_EVENTOS_PONTO,
    N_PONTOS_CIDADE,
    RAIO_AGRUPAMENTO_M,
    ZONA_FOCO,
)
from src.geo import agrupar_dbscan, dentro_de_sp, zona_da_subprefeitura

log = logging.getLogger(__name__)

ARQ_EVENTOS_CGE_GEO = DIR_INTERIM / "eventos_cge_geocodificados.parquet"
PRECISOES_ACEITAS = ("coordenada_oficial", "numero", "cruzamento", "referencia", "logradouro_curto")
COLUNAS_EVENTOS = [
    "evento_id", "fonte", "inicio_utc", "fim_utc", "lat", "lon", "precisao_geocod",
    "logradouro", "bairro", "subprefeitura", "transitavel", "hora_conhecida", "tipo_ocorrencia",
]
SUBPREF_PADRAO = Path("data/raw/geosampa/subprefeitura.geojson")
HIDRO_PADRAO = Path("data/raw/geosampa/drenagem.geojson")


def geocodificar_cge(cge: pd.DataFrame, geocodificador=None) -> pd.DataFrame:
    """Adiciona ``lat``, ``lon`` e ``precisao_geocod`` aos eventos do CGE (com cache)."""
    from src.geo import Geocodificador, texto_ou_none

    geocodificador = geocodificador or Geocodificador()
    # Campos vazios viram None: NaN != NaN, e com NaN a chave do dicionário nunca seria encontrada
    chaves = [tuple(texto_ou_none(v) for v in trio)
              for trio in zip(cge["logradouro"], cge["referencia"], cge["regiao_cge"], strict=True)]
    unicos = list(dict.fromkeys(chaves))
    log.info("Geocodificando %d endereços distintos (1 req/s; o cache evita repetir)", len(unicos))
    resultados = {}
    falhas = 0
    try:
        for i, chave in enumerate(unicos, start=1):
            try:
                resultados[chave] = geocodificador.geocodificar_registro(*chave)
            except Exception as erro:  # noqa: BLE001 - um endereço ruim não derruba horas de coleta
                falhas += 1
                log.warning("Erro ao geocodificar %s: %s", chave, erro)
                resultados[chave] = (None, None, "falhou")
            if i % 100 == 0:
                log.info("... %d/%d (erros: %d)", i, len(unicos), falhas)
    finally:
        geocodificador.salvar_cache()
    saida = cge.copy()
    trio = [resultados.get(chave, (None, None, "falhou")) for chave in chaves]
    saida["lat"] = [t[0] for t in trio]
    saida["lon"] = [t[1] for t in trio]
    saida["precisao_geocod"] = [t[2] for t in trio]
    log.info("Precisão da geocodificação:\n%s", saida["precisao_geocod"].value_counts().to_string())
    return saida


def unificar_eventos(cge_geo: pd.DataFrame | None, geosampa: pd.DataFrame | None,
                     precisoes: tuple[str, ...] = PRECISOES_ACEITAS) -> pd.DataFrame:
    """Junta as fontes num formato único e descarta coordenadas imprecisas ou fora de SP."""
    partes = []
    if cge_geo is not None and len(cge_geo):
        from src.geo import bairro_entre_parenteses

        c = cge_geo.copy()
        c["subprefeitura"] = c["regiao_cge"]  # o CGE agrupa por subprefeitura
        c["bairro"] = c["logradouro"].map(bairro_entre_parenteses)
        c["hora_conhecida"] = True
        c["tipo_ocorrencia"] = "ALAGAMENTO"
        partes.append(c)
    if geosampa is not None and len(geosampa):
        g = geosampa.copy()
        g["transitavel"] = None
        partes.append(g)
    if not partes:
        raise ValueError("Nenhuma fonte de eventos disponível")
    ev = pd.concat([p.reindex(columns=COLUNAS_EVENTOS) for p in partes], ignore_index=True)
    ev["inicio_utc"] = pd.to_datetime(ev["inicio_utc"], utc=True)
    ev["fim_utc"] = pd.to_datetime(ev["fim_utc"], utc=True)
    antes = len(ev)
    ev = ev.dropna(subset=["lat", "lon", "inicio_utc"])
    ev = ev[ev["precisao_geocod"].isin(precisoes)]
    ev = ev[dentro_de_sp(ev["lat"], ev["lon"])]
    log.info("Eventos: %d -> %d após filtros de coordenada/precisão", antes, len(ev))
    return ev.sort_values("inicio_utc").reset_index(drop=True)


def atribuir_zona(tabela: pd.DataFrame, subprefeituras=None, coluna_subpref: str | None = "nm_subprefeitura",
                  coluna_regiao: str = "nm_regiao_05") -> pd.DataFrame:
    """Preenche ``subprefeitura`` e ``zona``.

    Com o polígono do GeoSampa, usa a região oficial (``nm_regiao_05``: Norte, Sul, Leste,
    Oeste, Centro). Sem polígono (ou fora dele), deduz a zona pelo nome da subprefeitura.
    """
    from src.config import ZONAS
    from src.geo import atribuir_poligono, normalizar_texto

    saida = tabela.copy()
    if "subprefeitura" not in saida:
        saida["subprefeitura"] = None
    zona_mapa = pd.Series(None, index=saida.index, dtype="object")
    if subprefeituras is not None and coluna_subpref and coluna_subpref in subprefeituras:
        pelo_mapa = atribuir_poligono(saida, subprefeituras, coluna_subpref)
        saida["subprefeitura"] = pelo_mapa.where(pelo_mapa.notna(), saida["subprefeitura"])
        if coluna_regiao in subprefeituras:
            regiao = atribuir_poligono(saida, subprefeituras, coluna_regiao).map(normalizar_texto)
            zona_mapa = regiao.where(regiao.isin(ZONAS))
    pelo_nome = saida["subprefeitura"].map(zona_da_subprefeitura)
    saida["zona"] = zona_mapa.where(zona_mapa.notna(), pelo_nome)
    return saida


def contar_eventos_por_ano(eventos: pd.DataFrame) -> pd.DataFrame:
    """Tarefa de comprovação (seção 4): eventos por ano, cidade inteira × Zona Norte, por fonte."""
    ev = eventos.assign(ano=eventos["inicio_utc"].dt.tz_convert("America/Sao_Paulo").dt.year)
    cidade = ev.pivot_table(index="ano", columns="fonte", values="evento_id", aggfunc="count", fill_value=0)
    norte = ev[ev["zona"] == ZONA_FOCO].pivot_table(index="ano", columns="fonte", values="evento_id",
                                                     aggfunc="count", fill_value=0)
    tabela = cidade.add_prefix("cidade_").join(norte.add_prefix("norte_"), how="left").fillna(0).astype(int)
    tabela["pct_norte"] = (tabela.filter(like="norte_").sum(axis=1)
                           / tabela.filter(like="cidade_").sum(axis=1).clip(lower=1)).round(3)
    return tabela


def _moda(serie: pd.Series):
    serie = serie.dropna()
    return serie.mode().iloc[0] if len(serie) else None


def identificar_pontos(eventos: pd.DataFrame, raio_m: float = RAIO_AGRUPAMENTO_M,
                       min_eventos: int = MIN_EVENTOS_PONTO,
                       ano_limite: int | None = ANO_LIMITE_SELECAO_PONTOS) -> pd.DataFrame:
    """Agrupa eventos próximos e devolve um candidato a ponto por agrupamento."""
    ev = eventos
    if ano_limite is not None:
        ev = ev[ev["inicio_utc"].dt.year <= ano_limite]
    ev = ev.copy()
    ev["grupo"] = agrupar_dbscan(ev["lat"].to_numpy(), ev["lon"].to_numpy(), raio_m, min_eventos)
    ev = ev[ev["grupo"] >= 0]
    agregados = ev.groupby("grupo").agg(
        lat=("lat", "median"), lon=("lon", "median"),
        n_eventos=("evento_id", "count"),
        n_cge=("fonte", lambda s: int((s == "CGE").sum())),
        n_geosampa=("fonte", lambda s: int((s == "GEOSAMPA").sum())),
        primeiro_evento=("inicio_utc", "min"), ultimo_evento=("inicio_utc", "max"),
        bairro=("bairro", _moda), subprefeitura=("subprefeitura", _moda),
        logradouro=("logradouro", _moda),
    ).reset_index(drop=True)
    log.info("%d agrupamentos com >= %d eventos (raio %d m)", len(agregados), min_eventos, raio_m)
    return agregados.sort_values("n_eventos", ascending=False).reset_index(drop=True)


def selecionar_pontos(candidatos: pd.DataFrame, n_cidade: int = N_PONTOS_CIDADE,
                      zona_foco: str = ZONA_FOCO) -> pd.DataFrame:
    """Os ``n_cidade`` mais recorrentes + todos os da zona de foco. Gera ``ponto_id``."""
    if "zona" not in candidatos:
        raise ValueError("Rode atribuir_zona() nos candidatos antes de selecionar")
    top = candidatos.nlargest(n_cidade, "n_eventos")
    foco = candidatos[candidatos["zona"] == zona_foco]
    escolhidos = pd.concat([top, foco]).drop_duplicates(subset=["lat", "lon"])
    escolhidos = escolhidos.sort_values(["zona", "n_eventos"], ascending=[True, False]).reset_index(drop=True)
    escolhidos.insert(0, "ponto_id", [f"P{i:04d}" for i in range(1, len(escolhidos) + 1)])
    log.info("Pontos escolhidos: %d (zona %s: %d)", len(escolhidos), zona_foco,
             int((escolhidos["zona"] == zona_foco).sum()))
    return escolhidos


def enriquecer_pontos(pontos: pd.DataFrame, hidrografia=None, buscar_elevacao: bool = True) -> pd.DataFrame:
    """Adiciona ``dist_corrego_m``, ``elevacao`` e ``celula_id`` (célula de chuva do Open-Meteo)."""
    from src.collect.open_meteo import buscar_elevacao as _elev
    from src.collect.open_meteo import id_celula

    saida = pontos.copy()
    if hidrografia is not None:
        from src.geo import distancia_a_linhas_m

        saida["dist_corrego_m"] = distancia_a_linhas_m(saida, hidrografia).round(1)
    elif "dist_corrego_m" not in saida:
        saida["dist_corrego_m"] = pd.NA
    if buscar_elevacao:
        saida["elevacao"] = _elev(list(zip(saida["lat"], saida["lon"], strict=True)))
    elif "elevacao" not in saida:
        saida["elevacao"] = pd.NA
    saida["celula_id"] = [id_celula(a, b) for a, b in zip(saida["lat"], saida["lon"], strict=True)]
    return saida


COLUNAS_PONTOS = [
    "ponto_id", "lat", "lon", "bairro", "subprefeitura", "zona", "n_eventos", "n_cge", "n_geosampa",
    "logradouro", "dist_corrego_m", "elevacao", "celula_id", "primeiro_evento", "ultimo_evento",
]


def salvar_pontos(pontos: pd.DataFrame, caminho: Path = ARQ_PONTOS) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    pontos.reindex(columns=COLUNAS_PONTOS).to_csv(caminho, index=False)
    log.info("pontos salvos em %s", caminho)


def _ler_opcional(caminho: Path | None):
    if not caminho or not Path(caminho).exists():
        if caminho:
            log.warning("%s não encontrado — seguindo sem essa camada", caminho)
        return None
    from src.collect.geosampa_wfs import ler_geojson_wgs84

    return ler_geojson_wgs84(caminho)


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etapa", choices=["geocodificar", "unificar", "selecionar"])
    parser.add_argument("--subprefeituras", type=Path, default=SUBPREF_PADRAO,
                        help="GeoJSON da camada geoportal:subprefeitura (padrão: %(default)s)")
    parser.add_argument("--coluna-subpref", default="nm_subprefeitura")
    parser.add_argument("--hidrografia", type=Path, default=HIDRO_PADRAO,
                        help="GeoJSON da camada geoportal:drenagem (padrão: %(default)s)")
    parser.add_argument("--sem-elevacao", action="store_true", help="não consulta a API de elevação")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.etapa == "geocodificar":
        cge = pd.read_parquet(ARQ_EVENTOS_CGE)
        geocodificar_cge(cge).to_parquet(ARQ_EVENTOS_CGE_GEO, index=False)
        return

    subpref = _ler_opcional(args.subprefeituras)
    if args.etapa == "unificar":
        cge = pd.read_parquet(ARQ_EVENTOS_CGE_GEO) if ARQ_EVENTOS_CGE_GEO.exists() else None
        geo = pd.read_parquet(ARQ_EVENTOS_GEOSAMPA) if ARQ_EVENTOS_GEOSAMPA.exists() else None
        eventos = atribuir_zona(unificar_eventos(cge, geo), subpref, args.coluna_subpref)
        eventos.to_parquet(ARQ_EVENTOS, index=False)
        print(contar_eventos_por_ano(eventos).to_string())
        return

    eventos = pd.read_parquet(ARQ_EVENTOS)
    candidatos = atribuir_zona(identificar_pontos(eventos), subpref, args.coluna_subpref)
    pontos = selecionar_pontos(candidatos)
    pontos = enriquecer_pontos(pontos, _ler_opcional(args.hidrografia), not args.sem_elevacao)
    salvar_pontos(pontos)


if __name__ == "__main__":
    _principal()
