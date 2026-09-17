"""Camadas do GeoSampa (Prefeitura de São Paulo) via WFS.

Fonte B do briefing — "Alagamento (dados de Ocorrências)" da Defesa Civil (SIGRC), 2013 em
diante — usada como **rótulo complementar**, validação espacial e fonte de coordenadas.
O mesmo serviço fornece as camadas auxiliares de **hidrografia** (distância ao córrego) e
**subprefeituras/distritos** (zona do ponto).

Conferido em 16/09/2026 (WFS, 477 camadas):

| Camada | Registros | Observação |
|---|---|---|
| ``geoportal:risco_ocorrencia_alagamento`` | 197 | **só 2026** (jan–ago), **só data** (``dt_ocorrencia = "2026-04-01Z"``) |
| ``geoportal:risco_ocorrencia_inundacao`` | 260 | idem |
| ``geoportal:drenagem`` | 28.366 linhas | cursos d'água (abertos e canalizados) → ``dist_corrego_m`` |
| ``geoportal:subprefeitura`` | 32 polígonos | ``nm_subprefeitura`` e ``nm_regiao_05`` (Norte/Sul/Leste/Oeste/Centro) |
| ``geoportal:risco_hidrologico`` | 793 polígonos | áreas de risco (feature fixa opcional) |

Consequência: pelo WFS, o GeoSampa **não serve como rótulo horário nem histórico** (a série
2013+ citada no catálogo não está no WFS). As ocorrências entram com ``hora_conhecida=False``
e ficam fora do rótulo; servem para validação espacial. Ver docs/recomendacoes.md (R16).

Passo a passo (dia 1):

1. Descobrir os nomes das camadas::

       python -m src.collect.geosampa_wfs camadas --filtro alag
       python -m src.collect.geosampa_wfs camadas --filtro hidrograf
       python -m src.collect.geosampa_wfs camadas --filtro subprefeitura

2. Baixar cada camada (GeoJSON no sistema de coordenadas nativo, com paginação)::

       python -m src.collect.geosampa_wfs baixar --camada geoportal:<nome_da_camada>

3. Ler o **Dicionário de Dados** do catálogo antes de usar as colunas e preencher
   ``COLUNAS_OCORRENCIA`` abaixo.

4. Extrair a tabela de eventos::

       python -m src.collect.geosampa_wfs extrair --arquivo data/raw/geosampa/<arquivo>.geojson

Cuidados:
- A hora é a do **registro do chamado**, não a do início do alagamento; pode haver duplicatas.
- Baixamos no sistema nativo (provavelmente SIRGAS 2000 / UTM 23S, EPSG:31983) e convertemos
  localmente para WGS84 com geopandas. Pedir EPSG:4326 direto no WFS 2.0 pode inverter a
  ordem lat/lon, então evitamos.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.config import ARQ_EVENTOS_GEOSAMPA, DIR_RAW, LIMITES_SP
from src.tempo import local_para_utc

log = logging.getLogger(__name__)

URL_WFS = "http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs"
DIR_GEOSAMPA = DIR_RAW / "geosampa"
TAMANHO_PAGINA = 5000
DATA_DIA_PRIMEIRO = False  # True se as datas vierem como DD/MM/AAAA

# Nomes candidatos das colunas (os primeiros de cada lista foram conferidos em 16/09/2026).
# A primeira coluna existente de cada lista é usada.
COLUNAS_OCORRENCIA = {
    "data_hora": ["dt_ocorrencia", "data_hora", "dt_hr_ocorrencia", "dt_registro", "data_ocorrencia", "data"],
    "hora": ["hr_ocorrencia", "hora", "hora_ocorrencia"],
    "logradouro": ["nm_logradouro", "logradouro", "endereco", "tx_endereco"],
    "bairro": ["nm_bairro", "bairro"],
    "distrito": ["nm_distrito", "distrito"],
    "subprefeitura": ["nm_subprefeitura", "subprefeitura", "sg_subprefeitura"],
    "id_origem": ["cd_identificador", "cd_ocorrencia", "id_ocorrencia", "id", "fid"],
    "tipo": ["dc_tipo_ocorrencia", "tipo_ocorrencia"],
}


def listar_camadas(filtro: str | None = None, sessao=None) -> pd.DataFrame:
    """Lê o GetCapabilities e devolve nome/título/resumo das camadas (filtradas por texto)."""
    from src.collect.http import criar_sessao

    sessao = sessao or criar_sessao(usar_cache=True, expira_dias=7)
    resp = sessao.get(URL_WFS, params={"service": "WFS", "request": "GetCapabilities"}, timeout=60)
    resp.raise_for_status()
    raiz = ET.fromstring(resp.content)
    linhas = []
    for ft in raiz.iter():
        if not ft.tag.endswith("FeatureType"):
            continue
        campos = {filho.tag.split("}")[-1]: (filho.text or "").strip() for filho in ft}
        linhas.append({"nome": campos.get("Name"), "titulo": campos.get("Title"),
                       "resumo": campos.get("Abstract", "")[:200], "crs": campos.get("DefaultCRS")})
    df = pd.DataFrame(linhas)
    if filtro:
        padrao = re.compile(filtro, re.IGNORECASE)
        mascara = df.apply(lambda r: bool(padrao.search(f"{r.nome} {r.titulo} {r.resumo}")), axis=1)
        df = df[mascara]
    return df.reset_index(drop=True)


def baixar_camada(nome: str, destino: Path | None = None, sessao=None) -> Path:
    """Baixa todas as feições de uma camada em GeoJSON (paginando com ``startIndex``)."""
    from src.collect.http import criar_sessao

    sessao = sessao or criar_sessao(usar_cache=True)
    destino = destino or DIR_GEOSAMPA / f"{nome.split(':')[-1]}.geojson"
    destino.parent.mkdir(parents=True, exist_ok=True)
    feicoes: list[dict] = []
    crs = None
    inicio = 0
    ordenacao = _campo_ordenacao(nome, sessao)  # descoberto uma vez só
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": nome, "outputFormat": "application/json",
            "count": TAMANHO_PAGINA, "startIndex": inicio, "sortBy": ordenacao,
        }
        params = {k: v for k, v in params.items() if v}
        resp = sessao.get(URL_WFS, params=params, timeout=300)
        resp.raise_for_status()
        pagina = resp.json()
        crs = crs or pagina.get("crs")
        novas = pagina.get("features", [])
        feicoes.extend(novas)
        log.info("%s: +%d feições (total %d)", nome, len(novas), len(feicoes))
        if len(novas) < TAMANHO_PAGINA:
            break
        inicio += TAMANHO_PAGINA
    colecao = {"type": "FeatureCollection", "features": feicoes}
    if crs:
        colecao["crs"] = crs
    destino.write_text(json.dumps(colecao), encoding="utf-8")
    log.info("Camada salva em %s", destino)
    return destino


def _campo_ordenacao(nome: str, sessao) -> str | None:
    """Paginação estável exige ordenação; usa o primeiro atributo da camada, se conseguir descobrir."""
    try:
        resp = sessao.get(URL_WFS, params={"service": "WFS", "version": "2.0.0",
                                            "request": "DescribeFeatureType", "typeNames": nome}, timeout=60)
        raiz = ET.fromstring(resp.content)
        for el in raiz.iter():
            if el.tag.endswith("element") and el.get("name") and "geom" not in el.get("name", "").lower():
                tipo = el.get("type", "")
                if "Geometry" not in tipo and el.get("name") != nome.split(":")[-1]:
                    return el.get("name")
    except Exception:  # noqa: BLE001 - ordenação é só uma melhoria
        return None
    return None


def ler_geojson_wgs84(caminho: Path):
    """Lê um GeoJSON baixado e devolve um GeoDataFrame em WGS84 (EPSG:4326)."""
    import geopandas as gpd

    gdf = gpd.read_file(caminho)
    if gdf.crs is None:
        # Sem CRS declarado: decide pelo intervalo dos números (UTM tem valores na casa de 10^5)
        x = gdf.geometry.representative_point().x.abs().median()
        gdf = gdf.set_crs(31983 if x > 1000 else 4326)
    return gdf.to_crs(4326)


def _primeira_coluna(df: pd.DataFrame, candidatas: list[str]) -> str | None:
    minusculas = {c.lower(): c for c in df.columns}
    for nome in candidatas:
        if nome.lower() in minusculas:
            return minusculas[nome.lower()]
    return None


def _ler_ocorrencias(caminho: Path) -> pd.DataFrame:
    """Uma camada de ocorrências → tabela padronizada (antes de filtros e duplicatas)."""
    gdf = ler_geojson_wgs84(caminho)
    ponto = gdf.geometry.representative_point()
    df = pd.DataFrame({"lat": ponto.y, "lon": ponto.x})

    col = {chave: _primeira_coluna(gdf, cands) for chave, cands in COLUNAS_OCORRENCIA.items()}
    log.info("%s — colunas mapeadas: %s", Path(caminho).name, col)
    if col["data_hora"] is None:
        raise ValueError(
            "Não achei a coluna de data/hora. Veja as colunas disponíveis e atualize "
            f"COLUNAS_OCORRENCIA: {list(gdf.columns)}"
        )
    data_hora = gdf[col["data_hora"]].astype(str).str.replace(r"Z$", "", regex=True)
    if col["hora"] is not None:
        data_hora = data_hora.str.slice(0, 10) + " " + gdf[col["hora"]].astype(str)
    # "2026-04-01" (só data) -> hora desconhecida: não pode virar rótulo horário
    df["hora_conhecida"] = data_hora.str.contains(":", regex=False).to_numpy()
    df["inicio_local"] = pd.to_datetime(data_hora, errors="coerce", dayfirst=DATA_DIA_PRIMEIRO).to_numpy()
    for chave in ("logradouro", "bairro", "distrito", "subprefeitura", "id_origem", "tipo"):
        df[chave] = gdf[col[chave]].astype(str).to_numpy() if col[chave] else None
    df["tipo_ocorrencia"] = df.pop("tipo") if col["tipo"] else Path(caminho).stem
    return df


def extrair_ocorrencias(caminhos, destino: Path = ARQ_EVENTOS_GEOSAMPA) -> pd.DataFrame:
    """Junta uma ou mais camadas de ocorrências (alagamento, inundação) numa tabela de eventos."""
    caminhos = [caminhos] if isinstance(caminhos, (str, Path)) else list(caminhos)
    df = pd.concat([_ler_ocorrencias(Path(c)) for c in caminhos], ignore_index=True)

    df = df.dropna(subset=["inicio_local"])
    dentro = df.lat.between(LIMITES_SP["lat_min"], LIMITES_SP["lat_max"]) & df.lon.between(
        LIMITES_SP["lon_min"], LIMITES_SP["lon_max"])
    if (~dentro).any():
        log.warning("%d ocorrências fora do município foram descartadas", int((~dentro).sum()))
    df = df[dentro].copy()

    df["inicio_utc"] = local_para_utc(df["inicio_local"])
    df["fim_utc"] = pd.NaT
    df["fonte"] = "GEOSAMPA"
    df["precisao_geocod"] = "coordenada_oficial"
    base_id = df["id_origem"].fillna("").astype(str) if df["id_origem"].notna().any() else \
        pd.Series(df.index.astype(str), index=df.index)
    df["evento_id"] = ("geo_" + df["tipo_ocorrencia"].astype(str).str.lower() + "_" + base_id
                       + "_" + df["inicio_local"].dt.strftime("%Y%m%d%H%M"))
    # Duplicatas: mesmo lugar (~50 m), mesmo tipo e mesma hora (ou dia, se a hora é desconhecida)
    df["_chave"] = (df.lat.round(3).astype(str) + df.lon.round(3).astype(str)
                    + df["tipo_ocorrencia"].astype(str) + df.inicio_local.dt.floor("h").astype(str))
    antes = len(df)
    df = df.drop_duplicates("_chave").drop(columns="_chave")
    log.info("%d duplicatas removidas; %d ocorrências (%d com hora conhecida)",
             antes - len(df), len(df), int(df["hora_conhecida"].sum()))
    if not df["hora_conhecida"].any():
        log.warning("Nenhuma ocorrência do GeoSampa tem hora: elas NÃO entram no rótulo horário "
                    "(só validação espacial).")
    df["inicio_local"] = df["inicio_local"].dt.strftime("%Y-%m-%d %H:%M")
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return df


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="etapa", required=True)
    p_cam = sub.add_parser("camadas", help="lista camadas do WFS")
    p_cam.add_argument("--filtro", default=None, help="regex, ex.: alag|hidrograf|subpref")
    p_baixar = sub.add_parser("baixar", help="baixa uma camada inteira")
    p_baixar.add_argument("--camada", required=True, nargs="+",
                          help="ex.: geoportal:risco_ocorrencia_alagamento geoportal:drenagem")
    p_ext = sub.add_parser("extrair", help="gera eventos_geosampa.parquet")
    p_ext.add_argument("--arquivo", type=Path, required=True, nargs="+",
                       help="um ou mais GeoJSON de ocorrências (alagamento, inundação)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.etapa == "camadas":
        with pd.option_context("display.max_rows", 200, "display.max_colwidth", 80):
            print(listar_camadas(args.filtro))
    elif args.etapa == "baixar":
        for camada in args.camada:
            baixar_camada(camada)
    else:
        extrair_ocorrencias(args.arquivo)


if __name__ == "__main__":
    _principal()
