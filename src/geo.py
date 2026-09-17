"""Funções geoespaciais: distâncias, vizinhança, agrupamento e geocodificação.

Partes leves (usadas também pelo serviço): ``haversine_m``, ``mais_proximo``,
``normalizar_texto``, ``zona_da_subprefeitura``.

Partes pesadas (só no desenvolvimento, importam geopandas/sklearn dentro da função):
``agrupar_dbscan``, ``distancia_a_linhas_m``, ``atribuir_poligono`` e ``Geocodificador``.

Sistemas de coordenadas:
- WGS84 (EPSG:4326) para guardar lat/lon.
- SIRGAS 2000 / UTM 23S (EPSG:31983) para medir distâncias em metros com geopandas
  (é o sistema oficial dos dados da Prefeitura).
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DIR_RAW, LIMITES_SP, ZONA_POR_SUBPREFEITURA

log = logging.getLogger(__name__)

RAIO_TERRA_M = 6_371_008.8
CRS_METRICO = 31983


# ---------------------------------------------------------------------------
# Distâncias
# ---------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distância em metros entre coordenadas (aceita escalares ou arrays, com broadcasting)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RAIO_TERRA_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mais_proximo(lat: float, lon: float, lats, lons) -> tuple[int, float]:
    """Índice e distância (m) do ponto mais próximo numa lista (busca linear; ok até milhares)."""
    dist = haversine_m(lat, lon, lats, lons)
    i = int(np.argmin(dist))
    return i, float(dist[i])


def indices_no_raio(lats_ref, lons_ref, lats_alvo, lons_alvo, raio_m: float) -> list[np.ndarray]:
    """Para cada ponto de referência, os índices dos alvos a até ``raio_m`` metros (BallTree)."""
    from sklearn.neighbors import BallTree

    alvo = np.radians(np.column_stack([lats_alvo, lons_alvo]))
    ref = np.radians(np.column_stack([lats_ref, lons_ref]))
    if len(alvo) == 0:
        return [np.array([], dtype=int) for _ in range(len(ref))]
    arvore = BallTree(alvo, metric="haversine")
    return list(arvore.query_radius(ref, r=raio_m / RAIO_TERRA_M))


def dentro_de_sp(lat, lon) -> np.ndarray:
    """Máscara booleana: coordenada dentro da caixa aproximada do município."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    return ((lat >= LIMITES_SP["lat_min"]) & (lat <= LIMITES_SP["lat_max"])
            & (lon >= LIMITES_SP["lon_min"]) & (lon <= LIMITES_SP["lon_max"]))


# ---------------------------------------------------------------------------
# Agrupamento
# ---------------------------------------------------------------------------
def agrupar_dbscan(lats, lons, raio_m: float, min_amostras: int) -> np.ndarray:
    """DBSCAN com distância haversine. Retorna o rótulo do grupo de cada registro (-1 = isolado).

    ``raio_m`` é o ``eps``: registros a até essa distância são vizinhos. ``min_amostras`` é o
    mínimo de registros para formar um núcleo — na prática, "quantas vezes alagou ali".
    """
    from sklearn.cluster import DBSCAN

    coords = np.radians(np.column_stack([lats, lons]))
    modelo = DBSCAN(eps=raio_m / RAIO_TERRA_M, min_samples=min_amostras,
                    metric="haversine", algorithm="ball_tree")
    return modelo.fit_predict(coords)


# ---------------------------------------------------------------------------
# Texto e zonas
# ---------------------------------------------------------------------------
def normalizar_texto(texto) -> str:
    """Maiúsculas, sem acento, só letras/números/espaço. ``None`` vira ''."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", sem_acento.upper())).strip()


_CHAVES_ZONA = sorted(ZONA_POR_SUBPREFEITURA, key=len, reverse=True)


def zona_da_subprefeitura(nome) -> str | None:
    """Zona (CENTRO/LESTE/NORTE/OESTE/SUL) a partir do nome da subprefeitura.

    Compara por nome exato ou prefixo seguido de espaço, testando as chaves mais longas
    primeiro: "VILA MARIANA" não cai em "VILA MARIA", e "SENADOR..." não cai em "SE".
    """
    if nome is None or (isinstance(nome, float) and np.isnan(nome)):
        return None
    # GeoSampa escreve "VP - VILA PRUDENTE", "MB - MBoi Mirim": tira a sigla do começo
    sem_sigla = re.sub(r"^\s*[A-Za-z]{1,3}\s+-\s+", "", str(nome))
    norm = normalizar_texto(sem_sigla)
    for chave in _CHAVES_ZONA:
        if norm == chave or norm.startswith(chave + " "):
            return ZONA_POR_SUBPREFEITURA[chave]
    return None


# ---------------------------------------------------------------------------
# Camadas vetoriais (geopandas)
# ---------------------------------------------------------------------------
def _para_gdf(df: pd.DataFrame):
    import geopandas as gpd

    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)


def distancia_a_linhas_m(pontos: pd.DataFrame, linhas) -> pd.Series:
    """Distância (m) de cada ponto à feição mais próxima (ex.: córregos da hidrografia)."""
    import geopandas as gpd

    gp = _para_gdf(pontos).to_crs(CRS_METRICO)
    gl = linhas.to_crs(CRS_METRICO)[["geometry"]]
    junto = gpd.sjoin_nearest(gp, gl, how="left", distance_col="_dist")
    junto = junto[~junto.index.duplicated(keep="first")]
    return junto["_dist"].reindex(gp.index)


def atribuir_poligono(pontos: pd.DataFrame, poligonos, coluna: str) -> pd.Series:
    """Valor de ``coluna`` do polígono que contém cada ponto (ex.: nome da subprefeitura)."""
    import geopandas as gpd

    gp = _para_gdf(pontos)
    pol = poligonos.to_crs(4326)[[coluna, "geometry"]]
    junto = gpd.sjoin(gp, pol, how="left", predicate="within")
    junto = junto[~junto.index.duplicated(keep="first")]
    return junto[coluna].reindex(gp.index)


# ---------------------------------------------------------------------------
# Geocodificação (Nominatim / OpenStreetMap)
# ---------------------------------------------------------------------------
# O CGE escreve em maiúsculas e sem ponto: "AV CARD MOTTA", "R CAP RABELO", "AL VICENTE PINZON".
# Tipos de via só são expandidos no INÍCIO do nome; títulos, em qualquer posição.
TIPOS_VIA = {
    "AV": "Avenida", "R": "Rua", "AL": "Alameda", "TV": "Travessa", "PCA": "Praça", "PÇA": "Praça",
    "PC": "Praça", "LGO": "Largo", "EST": "Estrada", "ESTR": "Estrada", "VD": "Viaduto",
    "PTE": "Ponte", "PQ": "Parque", "ROD": "Rodovia", "PASS": "Passagem", "TUN": "Túnel",
    "VL": "Vila", "JD": "Jardim",
}
TITULOS = {
    "CARD": "Cardeal", "CAP": "Capitão", "STO": "Santo", "STA": "Santa", "DR": "Doutor",
    "DRA": "Doutora", "ENG": "Engenheiro", "ENGA": "Engenheira", "PROF": "Professor", "PROFA": "Professora",
    "PRES": "Presidente", "GOV": "Governador", "CEL": "Coronel", "GAL": "General",
    "GEN": "General", "BRIG": "Brigadeiro", "SEN": "Senador", "DEP": "Deputado",
    "COM": "Comendador", "MAL": "Marechal", "TEN": "Tenente", "MQ": "Marquês",
    "VISC": "Visconde", "CONS": "Conselheiro", "DES": "Desembargador", "MONS": "Monsenhor",
    "PE": "Padre", "FR": "Frei", "ALM": "Almirante", "MIN": "Ministro", "MAJ": "Major",
    "SGT": "Sargento", "VER": "Vereador", "ARQ": "Arquiteto", "JORN": "Jornalista",
}
PREFIXOS_VIA = re.compile(
    r"^(" + "|".join(sorted(TIPOS_VIA, key=len, reverse=True))
    + r"|AVENIDA|RUA|ALAMEDA|TRAVESSA|PRACA|PRAÇA|LARGO|ESTRADA|VIADUTO|PONTE|VIA|RODOVIA"
    + r"|MARGINAL|TUNEL|TÚNEL|COMPLEXO|PA)\.?(\s|$)",
    re.IGNORECASE,
)
# "461", "Nº 461", "ALTURA DO N. 66", "ALTURA DO NÚMERO 196"
RE_NUMERO = re.compile(
    r"^\s*(?:ALTURA\s+)?(?:D[OA]\s+)?(?:N(?:[º°O]|UMERO|ÚMERO)?\.?\s*)?(\d{1,5})\s*$", re.IGNORECASE)
RE_PARENTESES = re.compile(r"\(([^)]*)\)")


def expandir_abreviacoes(texto: str | None) -> str:
    """"AV CARD MOTTA" -> "Avenida Cardeal MOTTA"; "Av. Eng. X" -> "Avenida Engenheiro X"."""
    if not texto:
        return ""
    saida = RE_PARENTESES.sub(" ", str(texto)).strip()
    palavras = saida.split()
    if palavras:
        chave = palavras[0].rstrip(".").upper()
        if chave in TIPOS_VIA:
            palavras[0] = TIPOS_VIA[chave]
    for i, palavra in enumerate(palavras[1:], start=1):
        chave = palavra.rstrip(".").upper()
        if chave in TITULOS and (palavra.endswith(".") or palavra.isupper()):
            palavras[i] = TITULOS[chave]
    return re.sub(r"\s+", " ", " ".join(palavras)).strip()


def candidatos_via(texto: str | None) -> list[str]:
    """Variações de busca para um nome de via, da mais completa para a mais simples.

    Ex.: "AV JACU PESSEGO N TRABALHADORES" -> ["Avenida JACU PESSEGO N TRABALHADORES",
    "Avenida JACU PESSEGO"]; "PA PASCOAL GIANNETTI" -> [..., "PASCOAL GIANNETTI"].
    """
    base = expandir_abreviacoes(texto)
    if not base:
        return []
    candidatos = [base]
    if " N " in f" {base} ":
        candidatos.append(base.split(" N ")[0].strip())
    primeira = base.split()[0]
    if primeira.isupper() and len(primeira) <= 3 and len(base.split()) > 1:
        candidatos.append(" ".join(base.split()[1:]))  # tipo de via desconhecido
    return list(dict.fromkeys(c for c in candidatos if c))


def texto_ou_none(valor) -> str | None:
    """Texto limpo ou ``None``. Campos vazios do pandas chegam como ``NaN`` (float) ou ``pd.NA``."""
    if valor is None or valor is pd.NA or (isinstance(valor, float) and np.isnan(valor)):
        return None
    texto = str(valor).strip()
    return texto if texto and texto.lower() not in {"nan", "none"} else None


def bairro_entre_parenteses(texto: str | None) -> str | None:
    """"R DO PARAISO (VILA MARIANA)" -> "VILA MARIANA"."""
    m = RE_PARENTESES.search(texto or "")
    return m.group(1).strip() if m else None


def _geometria(resultado: dict):
    """Geometria shapely de um resultado do Nominatim (linha da via) ou o ponto central."""
    from shapely.geometry import Point, shape

    geo = resultado.get("geojson")
    if geo:
        try:
            return shape(geo)
        except Exception:  # noqa: BLE001 - geometria malformada: usa o ponto
            pass
    return Point(float(resultado["lon"]), float(resultado["lat"]))


def estimar_cruzamento(resultados_a: list[dict], resultados_b: list[dict],
                       dist_max_m: float = 250) -> tuple[float, float] | None:
    """Estima o cruzamento de duas vias: os pontos mais próximos entre as duas geometrias.

    Usa ``shapely.ops.nearest_points`` (funciona mesmo com segmentos longos, com poucos
    vértices). Se a menor distância passar de ``dist_max_m``, as vias não se cruzam ali.
    """
    if not resultados_a or not resultados_b:
        return None
    from shapely.ops import nearest_points, unary_union

    a = unary_union([_geometria(r) for r in resultados_a])
    b = unary_union([_geometria(r) for r in resultados_b])
    pa, pb = nearest_points(a, b)
    distancia = float(haversine_m(pa.y, pa.x, pb.y, pb.x))
    if distancia > dist_max_m:
        return None
    return (pa.y + pb.y) / 2, (pa.x + pb.x) / 2


class Geocodificador:
    """Cliente do Nominatim respeitando a política de uso: 1 req/s, cache e User-Agent com contato.

    Estratégia para um registro do CGE (logradouro + referência), em ordem:

    1. Referência é um **número** ("461") → procura "via, 461" e aceita se o resultado for um
       endereço (não a via inteira) a até 1 km da via (precisão ``numero``).
    2. Referência é uma **via** ("AV OTAVIANO ALVES DE LIMA") → estima o **cruzamento**
       (precisão ``cruzamento``).
    3. Outra referência (shopping, estação...) → aceita o lugar se estiver a até 1 km da via
       (precisão ``referencia``).
    4. Senão, centro do trecho da via (precisão ``logradouro`` — BAIXA para avenidas longas;
       fica fora dos pontos e rótulos por padrão).
    """

    URL = "https://nominatim.openstreetmap.org/search"
    SUFIXO = ", São Paulo, SP"

    def __init__(self, arquivo_cache: Path = DIR_RAW / "_cache_nominatim.json", intervalo_s: float = 1.1):
        from src.collect.http import ControleTaxa, carregar_env, criar_sessao

        carregar_env()
        self.arquivo_cache = Path(arquivo_cache)
        self.cache: dict[str, list[dict]] = (
            json.loads(self.arquivo_cache.read_text(encoding="utf-8")) if self.arquivo_cache.exists() else {}
        )
        self.sessao = criar_sessao(usar_cache=False)
        self.taxa = ControleTaxa(intervalo_s)
        self._novas = 0

    def salvar_cache(self) -> None:
        self.arquivo_cache.parent.mkdir(parents=True, exist_ok=True)
        self.arquivo_cache.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def buscar(self, consulta: str, limite: int = 10) -> list[dict]:
        if consulta in self.cache:
            return self.cache[consulta]
        self.taxa.aguardar()
        params = {
            "q": consulta, "format": "jsonv2", "limit": limite, "countrycodes": "br",
            "viewbox": f"{LIMITES_SP['lon_min']},{LIMITES_SP['lat_max']},{LIMITES_SP['lon_max']},{LIMITES_SP['lat_min']}",
            "bounded": 1, "polygon_geojson": 1, "polygon_threshold": 0.00005,
        }
        try:
            resp = self.sessao.get(self.URL, params=params, timeout=30)
            resp.raise_for_status()
            resultados = resp.json()
        except Exception as erro:  # noqa: BLE001
            log.warning("Nominatim falhou para %r: %s", consulta, erro)
            resultados = []
        finally:
            self.taxa.registrar()
        self.cache[consulta] = resultados
        self._novas += 1
        if self._novas % 50 == 0:
            self.salvar_cache()
        time.sleep(0)  # ponto de cancelamento amigável
        return resultados

    def _primeira_via(self, texto: str | None, bairro: str | None) -> tuple[str | None, list[dict]]:
        """Primeiro candidato de nome que o Nominatim encontra (com e sem o bairro)."""
        for candidato in candidatos_via(texto):
            resultados = self.buscar(candidato + self.SUFIXO)
            if not resultados and bairro:
                resultados = self.buscar(f"{candidato}, {bairro}{self.SUFIXO}")
            if resultados:
                return candidato, resultados
        return None, []

    @staticmethod
    def _eh_endereco(resultado: dict) -> bool:
        """True se o resultado é um endereço/edificação, e não a via inteira."""
        return (resultado.get("category") not in ("highway",)
                and resultado.get("addresstype") not in ("road", "suburb", "city", "neighbourhood"))

    def geocodificar_registro(self, logradouro: str | None, referencia: str | None,
                              bairro: str | None = None) -> tuple[float | None, float | None, str]:
        """Retorna (lat, lon, precisao) para um registro do CGE."""
        logradouro, referencia, bairro = (texto_ou_none(v) for v in (logradouro, referencia, bairro))
        if logradouro is None:
            return None, None, "falhou"
        bairro = bairro_entre_parenteses(logradouro) or bairro
        via, res_via = self._primeira_via(logradouro, bairro)
        if not res_via:
            return None, None, "falhou"
        ref_bruta = referencia or ""

        numero = RE_NUMERO.match(ref_bruta)
        if numero:
            achados = [r for r in self.buscar(f"{via}, {numero.group(1)}{self.SUFIXO}", limite=3)
                       if self._eh_endereco(r)]
            if achados and estimar_cruzamento(res_via, achados[:1], dist_max_m=1000):
                return float(achados[0]["lat"]), float(achados[0]["lon"]), "numero"
        elif ref_bruta and PREFIXOS_VIA.match(ref_bruta):
            _, res_ref = self._primeira_via(ref_bruta, bairro)
            cruz = estimar_cruzamento(res_via, res_ref)
            if cruz:
                return cruz[0], cruz[1], "cruzamento"
        elif ref_bruta:
            res_ref = self.buscar(expandir_abreviacoes(ref_bruta) + self.SUFIXO, limite=3)
            if res_ref and estimar_cruzamento(res_via, res_ref[:1], dist_max_m=1000):
                return float(res_ref[0]["lat"]), float(res_ref[0]["lon"]), "referencia"

        if bairro:
            res_bairro = self.buscar(f"{via}, {bairro}{self.SUFIXO}", limite=3) or res_via
        else:
            res_bairro = res_via
        return float(res_bairro[0]["lat"]), float(res_bairro[0]["lon"]), "logradouro"
