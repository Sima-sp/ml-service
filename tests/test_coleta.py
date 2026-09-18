"""Coletores e etapas geoespaciais, sem internet (HTML e GeoJSON falsos).

⚠️ O HTML abaixo imita a estrutura CONHECIDA da página do CGE. Se a página real for diferente,
ajuste os seletores em ``src/collect/cge_alagamentos.py`` e atualize este teste com um trecho
real (salvo em data/raw/cge_alagamentos/html/).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.collect.cge_alagamentos import extrair_dia

# Trecho REAL da página (busca de 14/09/2026), com um ponto intransitável acrescentado
HTML_CGE = """
<html><body><div class="content">
<h1 class="tit-bairros">Zona Norte</h1>
<table class="tb-pontos-de-alagamentos"><tbody>
 <tr><td class="bairro arial-bairros-alag linha-pontilhada"> Pirituba/Jaraguá <hr> </td>
     <td class="total-pts arial-bairros-alag">1 pts.</td></tr>
 <tr><td colspan="3"><div class="ponto-de-alagamento"><ul>
   <li class="ativo-transitavel" title="Ativo Transitável"></li><li class="espaco"></li>
   <li class="arial-descr-alag col-local">De 14:31 a <br>AV CARD MOTTA</li><li class="ln-vert"></li>
   <li class="arial-descr-alag">Sentido: CENTRO/BAIRRO<br>Referência: AV OTAVIANO ALVES DE LIMA</li>
 </ul></div></td></tr>
</tbody></table>
<table class="tb-pontos-de-alagamentos"><tbody>
 <tr><td class="bairro arial-bairros-alag linha-pontilhada"> Santana <hr> </td>
     <td class="total-pts arial-bairros-alag">2 pts.</td></tr>
 <tr><td colspan="3"><div class="ponto-de-alagamento"><ul>
   <li class="inativo-transitavel" title="Inativo Transitável"></li><li class="espaco"></li>
   <li class="arial-descr-alag col-local">De 19:28 a 19:34<br>R CAP RABELO</li><li class="ln-vert"></li>
   <li class="arial-descr-alag">Sentido: AMBOS<br>Referência: 461</li>
 </ul></div></td></tr>
 <tr><td colspan="3"><div class="ponto-de-alagamento"><ul>
   <li class="inativo-intransitavel" title="Inativo Intransitável"></li><li class="espaco"></li>
   <li class="arial-descr-alag col-local">De 23:41 a 00:06<br>AV CRUZEIRO DO SUL</li><li class="ln-vert"></li>
   <li class="arial-descr-alag">Sentido: IPIRANGA/SANTANA<br>Referência: 1800</li>
 </ul></div></td></tr>
</tbody></table>
<h1 class="tit-bairros">Zona Sudeste</h1>
<table class="tb-pontos-de-alagamentos"><tbody>
 <tr><td class="bairro arial-bairros-alag linha-pontilhada"> Vila Mariana <hr> </td>
     <td class="total-pts arial-bairros-alag">1 pts.</td></tr>
 <tr><td colspan="3"><div class="ponto-de-alagamento"><ul>
   <li class="inativo-transitavel" title="Inativo Transitável"></li><li class="espaco"></li>
   <li class="arial-descr-alag col-local">De 19:26 a 20:42<br>R DO PARAISO (VILA MARIANA)</li><li class="ln-vert"></li>
   <li class="arial-descr-alag">Sentido: BAIRRO/CENTRO<br>Referência: 800</li>
 </ul></div></td></tr>
</tbody></table>
</div></body></html>
"""


def test_extrair_dia_cge(caplog):
    registros = extrair_dia(HTML_CGE, pd.Timestamp("2026-09-14"))
    assert len(registros) == 4  # um por div.ponto-de-alagamento (não um por <li>)
    assert "anuncia" not in caplog.text  # bate com o "N pts." da página
    r0, r1, r2, r3 = registros
    assert r0.inicio_local == "2026-09-14 14:31" and r0.fim_local is None  # ainda ativo
    assert r0.ativo is True and r0.transitavel is True
    assert r0.logradouro == "AV CARD MOTTA"
    assert r0.sentido == "CENTRO/BAIRRO" and r0.referencia == "AV OTAVIANO ALVES DE LIMA"
    assert r0.regiao_cge == "Pirituba/Jaraguá" and r0.zona_cge == "Zona Norte"
    assert r1.referencia == "461" and r1.ativo is False and r1.regiao_cge == "Santana"
    assert r2.transitavel is False
    assert r2.fim_local == "2026-09-15 00:06"  # atravessou a meia-noite
    assert r3.zona_cge == "Zona Sudeste" and r3.logradouro == "R DO PARAISO (VILA MARIANA)"


def test_aviso_quando_contagem_nao_bate(caplog):
    html = HTML_CGE.replace("1 pts.", "5 pts.", 1)
    extrair_dia(html, pd.Timestamp("2026-09-14"))
    assert "anuncia 8 pontos" in caplog.text


def test_extrair_dia_reserva_por_texto():
    html = "<html><body><p>De 10:05 a 11:00</p><p>Av. Tucuruvi</p><p>Sentido: Centro</p></body></html>"
    registros = extrair_dia(html, pd.Timestamp("2024-02-01"))
    assert len(registros) == 1
    assert registros[0].inicio_local == "2024-02-01 10:05"
    assert registros[0].logradouro == "Av. Tucuruvi"


def test_extrair_periodo_converte_para_utc(tmp_path, monkeypatch):
    from src.collect import cge_alagamentos as mod

    monkeypatch.setattr(mod, "DIR_HTML", tmp_path)
    arq = mod.caminho_html(pd.Timestamp("2018-01-10"))
    arq.parent.mkdir(parents=True)
    arq.write_text(HTML_CGE, encoding="utf-8")
    df = mod.extrair_periodo("2018-01-09", "2018-01-10", destino=tmp_path / "eventos.parquet")
    assert len(df) == 4 and df["evento_id"].is_unique
    # janeiro/2018 tinha horário de verão (UTC-2): 14:31 local = 16:31 UTC
    assert df["inicio_utc"].iloc[0] == pd.Timestamp("2018-01-10 16:31", tz="UTC")
    assert pd.isna(df["fim_utc"].iloc[0])


def test_estimar_cruzamento():
    pytest.importorskip("shapely")
    from src.geo import estimar_cruzamento

    via_a = [{"lat": "-23.5", "lon": "-46.6", "geojson": {"type": "LineString",
              "coordinates": [[-46.600, -23.500], [-46.590, -23.500]]}}]
    via_b = [{"lat": "-23.5", "lon": "-46.6", "geojson": {"type": "LineString",
              "coordinates": [[-46.595, -23.505], [-46.595, -23.5001]]}}]
    lat, lon = estimar_cruzamento(via_a, via_b)
    assert abs(lat + 23.50005) < 0.0001 and abs(lon + 46.595) < 0.0001
    longe = [{"lat": "-23.6", "lon": "-46.7", "geojson": None}]
    assert estimar_cruzamento(via_a, longe) is None


def test_pontos_e_zonas_com_geopandas(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, box

    from src.collect.geosampa_wfs import extrair_ocorrencias
    from src.geo import atribuir_poligono, distancia_a_linhas_m
    from src.pontos import (
        atribuir_zona,
        contar_eventos_por_ano,
        identificar_pontos,
        selecionar_pontos,
    )

    # subprefeituras falsas: metade norte e metade sul de um retângulo
    subpref = gpd.GeoDataFrame({"nome": ["SANTANA-TUCURUVI", "SANTO AMARO"], "nm_regiao_05": ["Norte", "Sul"]},
                               geometry=[box(-46.7, -23.50, -46.5, -23.40), box(-46.7, -23.70, -46.5, -23.50)],
                               crs=4326)
    rio = gpd.GeoDataFrame(geometry=[LineString([(-46.7, -23.45), (-46.5, -23.45)])], crs=4326)
    pontos = pd.DataFrame({"lat": [-23.46, -23.60], "lon": [-46.60, -46.60]})
    assert list(atribuir_poligono(pontos, subpref, "nome")) == ["SANTANA-TUCURUVI", "SANTO AMARO"]
    dist = distancia_a_linhas_m(pontos, rio)
    assert 1000 < dist.iloc[0] < 1200  # 0,01° de latitude ≈ 1,1 km

    # ocorrências do GeoSampa em UTM (EPSG:31983), como o WFS entrega
    rng = np.random.default_rng(0)
    lats = np.r_[-23.46 + rng.normal(0, 0.0003, 6), -23.60 + rng.normal(0, 0.0003, 4)]
    lons = np.r_[-46.60 + rng.normal(0, 0.0003, 6), -46.60 + rng.normal(0, 0.0003, 4)]
    datas = pd.date_range("2019-01-05 17:00", periods=10, freq="37D").strftime("%Y-%m-%d %H:%M:%S")
    gdf = gpd.GeoDataFrame({"dt_ocorrencia": datas, "cd_ocorrencia": range(10)},
                           geometry=gpd.points_from_xy(lons, lats), crs=4326).to_crs(31983)
    arq = tmp_path / "ocorrencias.geojson"
    arq.write_text(gdf.to_json(), encoding="utf-8")
    eventos = extrair_ocorrencias(arq, destino=tmp_path / "eventos_geo.parquet")
    assert len(eventos) == 10
    assert eventos["lat"].between(-23.61, -23.45).all()  # voltou para WGS84
    assert json.loads(eventos["evento_id"].to_json())  # ids gerados

    eventos["inicio_utc"] = pd.to_datetime(eventos["inicio_utc"], utc=True)
    eventos = atribuir_zona(eventos, subpref, "nome")
    assert set(eventos["zona"]) == {"NORTE", "SUL"}
    contagem = contar_eventos_por_ano(eventos)
    assert contagem.filter(like="cidade_").to_numpy().sum() == 10

    candidatos = identificar_pontos(eventos, raio_m=200, min_eventos=3, ano_limite=None)
    assert sorted(candidatos["n_eventos"]) == [4, 6]
    candidatos = atribuir_zona(candidatos, subpref, "nome")
    so_top = selecionar_pontos(candidatos, n_cidade=1)
    assert len(so_top) == 1 and so_top["zona"].iloc[0] == "NORTE"
    # o ponto do sul (4 eventos) só entra quando cabe no top N; os da zona norte entram sempre
    candidatos.loc[candidatos["zona"] == "NORTE", "n_eventos"] = 3
    escolhidos = selecionar_pontos(candidatos, n_cidade=1)
    assert sorted(escolhidos["zona"]) == ["NORTE", "SUL"]
    assert escolhidos["ponto_id"].tolist() == ["P0001", "P0002"]


def test_geocodificador_prefere_cruzamento(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    from src.geo import Geocodificador

    linha_a = {"lat": "-23.5", "lon": "-46.595", "geojson": {
        "type": "LineString", "coordinates": [[-46.600, -23.500], [-46.590, -23.500]]}}
    linha_b = {"lat": "-23.502", "lon": "-46.595", "geojson": {
        "type": "LineString", "coordinates": [[-46.595, -23.505], [-46.595, -23.4999]]}}
    respostas = {
        "Avenida Guilherme Cotching, São Paulo, SP": [linha_a],
        "Rua Dona Santa Veloso, São Paulo, SP": [linha_b],
        "Shopping Center Norte, São Paulo, SP": [{"lat": "-23.5004", "lon": "-46.5920"}],
    }
    geo = Geocodificador(arquivo_cache=tmp_path / "cache.json")
    monkeypatch.setattr(geo, "buscar", lambda consulta, limite=10: respostas.get(consulta, []))

    lat, lon, precisao = geo.geocodificar_registro("Av. Guilherme Cotching", "R. Dona Santa Veloso")
    assert precisao == "cruzamento" and abs(lon + 46.595) < 1e-6

    lat, lon, precisao = geo.geocodificar_registro("Av. Guilherme Cotching", "Shopping Center Norte")
    assert precisao == "referencia" and lat == -23.5004

    _, _, precisao = geo.geocodificar_registro("Av. Guilherme Cotching", "altura do número 500")
    assert precisao == "logradouro"
    assert geo.geocodificar_registro("Rua Inexistente", None)[2] == "falhou"


@pytest.mark.parametrize("nome,zona", [
    ("VP - VILA PRUDENTE", "LESTE"), ("MB - MBoi Mirim", "SUL"), ("SE - SE", "CENTRO"),
    ("MG - VILA MARIA/VILA GUILHERME", "NORTE"), ("Pirituba/Jaraguá", "NORTE"),
    ("FO - Freguesia do O/Brasilandia", "NORTE"), ("G - GUAIANASES", "LESTE"),
    ("Vila Mariana", "SUL"), ("Senador Qualquer", None), (None, None),
])
def test_zona_por_nome(nome, zona):
    from src.geo import zona_da_subprefeitura

    assert zona_da_subprefeitura(nome) == zona


@pytest.mark.parametrize("texto,esperado", [
    ("AV CARD MOTTA", ["Avenida Cardeal MOTTA"]),
    ("R CAP RABELO", ["Rua Capitão RABELO"]),
    ("AL VICENTE PINZON", ["Alameda VICENTE PINZON"]),
    ("AV STO AMARO", ["Avenida Santo AMARO"]),
    ("R DO PARAISO (VILA MARIANA)", ["Rua DO PARAISO"]),
    ("AV JACU PESSEGO N TRABALHADORES", ["Avenida JACU PESSEGO N TRABALHADORES", "Avenida JACU PESSEGO"]),
    ("PA PASCOAL GIANNETTI", ["PA PASCOAL GIANNETTI", "PASCOAL GIANNETTI"]),
    ("Av. Eng. Caetano Álvares", ["Avenida Engenheiro Caetano Álvares"]),
])
def test_candidatos_de_via(texto, esperado):
    from src.geo import candidatos_via

    assert candidatos_via(texto) == esperado


def test_geocodificador_numero_da_referencia(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    from src.geo import Geocodificador

    via = {"lat": "-23.49", "lon": "-46.62", "category": "highway", "addresstype": "road", "geojson": {
        "type": "LineString", "coordinates": [[-46.625, -23.490], [-46.615, -23.490]]}}
    casa = {"lat": "-23.4902", "lon": "-46.6201", "category": "building", "addresstype": "building"}
    respostas = {
        "Rua Capitão RABELO, São Paulo, SP": [via],
        "Rua Capitão RABELO, 461, São Paulo, SP": [casa],
        "Rua Capitão RABELO, 9999, São Paulo, SP": [via],  # número não existe: volta a via
    }
    geo = Geocodificador(arquivo_cache=tmp_path / "cache.json")
    monkeypatch.setattr(geo, "buscar", lambda consulta, limite=10: respostas.get(consulta, []))
    assert geo.geocodificar_registro("R CAP RABELO", "461") == (-23.4902, -46.6201, "numero")
    assert geo.geocodificar_registro("R CAP RABELO", "9999")[2] == "logradouro"


def test_geosampa_so_data_nao_vira_rotulo(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from src.collect.geosampa_wfs import extrair_ocorrencias
    from src.pontos import atribuir_zona

    gdf = gpd.GeoDataFrame(
        {"cd_identificador": ["1458", "1459"], "dc_tipo_ocorrencia": ["ALAGAMENTO", "INUNDACAO"],
         "dt_ocorrencia": ["2026-04-01Z", "2026-04-01Z"],
         "nm_subprefeitura": ["VP - VILA PRUDENTE", "SB - SAPOPEMBA"]},
        geometry=gpd.points_from_xy([-46.58, -46.52], [-23.58, -23.60]), crs=4326).to_crs(31983)
    arq_a, arq_i = tmp_path / "alag.geojson", tmp_path / "inund.geojson"
    arq_a.write_text(gdf.iloc[[0]].to_json(), encoding="utf-8")
    arq_i.write_text(gdf.iloc[[1]].to_json(), encoding="utf-8")
    ev = extrair_ocorrencias([arq_a, arq_i], destino=tmp_path / "ev.parquet")
    assert len(ev) == 2 and not ev["hora_conhecida"].any()
    assert set(ev["tipo_ocorrencia"]) == {"ALAGAMENTO", "INUNDACAO"}
    assert list(atribuir_zona(ev)["zona"]) == ["LESTE", "LESTE"]


def test_dataset_ignora_eventos_sem_hora():
    from src.dataset import montar_dataset

    horas = pd.date_range("2022-12-25", "2023-01-05", freq="h", tz="UTC")
    chuva = pd.DataFrame({"celula_id": "c1", "data_hora_utc": horas, "chuva_mm": 1.0})
    pontos = pd.DataFrame([{"ponto_id": "P0001", "lat": -23.5, "lon": -46.6, "elevacao": 750.0,
                            "dist_corrego_m": 80.0, "zona": "NORTE", "celula_id": "c1"}])
    eventos = pd.DataFrame({
        "evento_id": ["com_hora", "sem_hora"],
        "inicio_utc": pd.to_datetime(["2023-01-02 15:00", "2023-01-03 03:00"], utc=True),
        "fim_utc": pd.to_datetime(["2023-01-02 16:00", None], utc=True),
        "lat": [-23.5, -23.5], "lon": [-46.6, -46.6], "hora_conhecida": [True, False],
    })
    dataset, _, _ = montar_dataset(pontos, chuva, eventos, inicio="2023-01-01", fim="2023-01-04")
    assert set(dataset.loc[dataset["alagou_3h"] == 1, "evento_alvo_id"]) == {"com_hora"}
    assert dataset["alagou_3h"].sum() == 3


@pytest.mark.parametrize("texto,numero", [
    ("461", "461"), ("Nº 461", "461"), ("ALTURA DO N. 66", "66"), ("ALTURA DO NÚMERO 196", "196"),
    ("AV OTAVIANO ALVES DE LIMA", None), ("PC ENGA NAIR FIRMINO DA SILVA", None),
])
def test_referencia_numerica(texto, numero):
    from src.geo import RE_NUMERO

    m = RE_NUMERO.match(texto)
    assert (m.group(1) if m else None) == numero


def test_geocodificar_cge_com_campos_vazios(monkeypatch):
    """Referência vazia chega como NaN (float); não pode quebrar nem sumir no dicionário."""
    from src.pontos import geocodificar_cge

    class Falso:
        def __init__(self):
            self.chamadas = []

        def geocodificar_registro(self, logradouro, referencia, bairro=None):
            self.chamadas.append((logradouro, referencia, bairro))
            if logradouro == "EXPLODE":
                raise RuntimeError("falha simulada")
            return -23.5, -46.6, "logradouro" if referencia is None else "cruzamento"

        def salvar_cache(self):
            pass

    cge = pd.DataFrame({
        "logradouro": ["AV TUCURUVI", "AV TUCURUVI", "AV TUCURUVI", "EXPLODE"],
        "referencia": [np.nan, float("nan"), "R X", None],
        "regiao_cge": ["Santana", "Santana", np.nan, "Santana"],
    })
    falso = Falso()
    saida = geocodificar_cge(cge, falso)
    assert len(falso.chamadas) == 3  # os dois NaN viram a mesma chave
    assert list(saida["precisao_geocod"]) == ["logradouro", "logradouro", "cruzamento", "falhou"]
    assert falso.chamadas[0] == ("AV TUCURUVI", None, "Santana")


def test_geocodificar_registro_aceita_nan(tmp_path, monkeypatch):
    from src.geo import Geocodificador

    geo = Geocodificador(arquivo_cache=tmp_path / "cache.json")
    monkeypatch.setattr(geo, "buscar", lambda consulta, limite=10: [])
    assert geo.geocodificar_registro("AV X", float("nan"), float("nan")) == (None, None, "falhou")
    assert geo.geocodificar_registro(float("nan"), "461") == (None, None, "falhou")


@pytest.mark.parametrize("texto,primeiro", [
    ("AV CD.DE FRONTIN", "Avenida Conde DE FRONTIN"),
    ("AV MARQ.DE SAO VICENTE", "Avenida Marquês DE SAO VICENTE"),
    ("AV NSRA.DO O", "Avenida Nossa Senhora DO O"),
    ("AV N.SRA. DO SABARA", "Avenida Nossa Senhora DO SABARA"),
    ("R. BR. DO. BANANAL", "Rua Barão DO BANANAL"),
    ("R PDE VIEGAS DE MENEZES", "Rua Padre VIEGAS DE MENEZES"),
    ("TN JORN FERNANDO VIEIRA DE MELLO", "Túnel Jornalista FERNANDO VIEIRA DE MELLO"),
    ("ES DE ITAPECERICA", "Estrada DE ITAPECERICA"),
    ("PÇ. DA BANDEIRA", "Praça DA BANDEIRA"),
    ("*AV. ARICANDUVA", "Avenida ARICANDUVA"),
])
def test_normalizacao_dos_nomes_reais(texto, primeiro):
    from src.geo import candidatos_via

    assert candidatos_via(texto)[0] == primeiro


def test_variantes_de_nomes():
    from src.geo import candidatos_via

    assert "MARGINAL TIETE" in candidatos_via("MARGINAL TIETE CBAS CENTRAL")
    assert "MARGINAL TIETE" in candidatos_via("MARGINAL TIETE - EXPRESSA")
    ponte = candidatos_via("PTE DA CASA VERDE-JORN.WALTER ABRAHAO", frouxo=True)
    assert "Ponte DA CASA VERDE" in ponte and "Ponte WALTER ABRAHAO" in ponte
    assert "Ponte ROSSI ZUCCOLO" in candidatos_via("PTE ENG ROBERTO ROSSI ZUCCOLO", frouxo=True)
    # nomes com número não perdem o número nas variantes
    assert candidatos_via("AVENIDA 9 DE JULHO", frouxo=True) == ["AVENIDA 9 DE JULHO"]
    # "PAN - AMERICANA" é um nome só: não vira "Praça PAN"
    assert "Praça PAN" not in candidatos_via("PC PAN - AMERICANA", frouxo=True)


@pytest.mark.parametrize("texto,esperado", [
    ("ALTURA DO VD GUADALAJARA", "VD GUADALAJARA"),
    ("PROX AO NUMERO 1100", "NUMERO 1100"),
    ("100M ANTES DA R JACURICÍ", "R JACURICÍ"),
    ("ALTURA DO N. 66", "ALTURA DO N. 66"),
    ("AV OTAVIANO ALVES DE LIMA", "AV OTAVIANO ALVES DE LIMA"),
])
def test_limpar_referencia(texto, esperado):
    from src.geo import limpar_referencia

    assert limpar_referencia(texto) == esperado


def _via(lon_ini, lon_fim, lat=-23.5):
    return {"lat": str(lat), "lon": str((lon_ini + lon_fim) / 2), "category": "highway", "addresstype": "road",
            "geojson": {"type": "LineString", "coordinates": [[lon_ini, lat], [lon_fim, lat]]}}


def test_via_curta_vira_logradouro_curto(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    from src.geo import Geocodificador

    respostas = {
        "Rua CURTA, São Paulo, SP": [_via(-46.600, -46.597)],  # ~300 m
        "Avenida LONGA, São Paulo, SP": [_via(-46.60, -46.58)],  # ~2 km
        "Rua CORTADA, São Paulo, SP": [_via(-46.600 + i * 1e-4, -46.5999 + i * 1e-4) for i in range(10)],
    }
    geo = Geocodificador(arquivo_cache=tmp_path / "c.json")
    monkeypatch.setattr(geo, "buscar", lambda consulta, limite=10: respostas.get(consulta, []))
    lat, lon, precisao = geo.geocodificar_registro("R CURTA", None)
    assert precisao == "logradouro_curto" and abs(lon + 46.5985) < 1e-6
    assert geo.geocodificar_registro("AV LONGA", None)[2] == "logradouro"
    # veio cortada no limite de 10 trechos: pode ser maior do que parece
    assert geo.geocodificar_registro("R CORTADA", None)[2] == "logradouro"


def test_cruzamento_amplia_busca_quando_via_veio_cortada(tmp_path, monkeypatch):
    pytest.importorskip("shapely")
    from src.geo import Geocodificador

    trechos = [_via(-46.70 + i * 0.001, -46.699 + i * 0.001) for i in range(10)]  # longe do viaduto
    viaduto = {"lat": "-23.5", "lon": "-46.60", "category": "highway", "addresstype": "road",
               "geojson": {"type": "LineString", "coordinates": [[-46.60, -23.501], [-46.60, -23.499]]}}
    pedidos = []

    def buscar(consulta, limite=10):
        pedidos.append((consulta, limite))
        if consulta.startswith("Avenida GRANDE"):
            return trechos + ([_via(-46.601, -46.599)] if limite == 40 else [])
        if consulta.startswith("Viaduto X"):
            return [viaduto]
        return []

    geo = Geocodificador(arquivo_cache=tmp_path / "c.json")
    monkeypatch.setattr(geo, "buscar", buscar)
    lat, lon, precisao = geo.geocodificar_registro("AV GRANDE", "VD X")
    assert precisao == "cruzamento" and abs(lon + 46.60) < 1e-6
    assert ("Avenida GRANDE, São Paulo, SP", 40) in pedidos


def test_cache_separa_limite_ampliado(tmp_path):
    from src.geo import Geocodificador

    geo = Geocodificador(arquivo_cache=tmp_path / "c.json")
    geo.cache["Rua A, São Paulo, SP"] = [{"x": 1}]
    geo.cache["Rua A, São Paulo, SP|limite=40"] = [{"x": 1}, {"x": 2}]
    assert len(geo.buscar("Rua A, São Paulo, SP")) == 1
    assert len(geo.buscar("Rua A, São Paulo, SP", limite=40)) == 2


def test_bairro_entre_parenteses_tolera_vazios():
    from src.geo import bairro_entre_parenteses

    assert bairro_entre_parenteses("R DO PARAISO (VILA MARIANA)") == "VILA MARIANA"
    assert bairro_entre_parenteses(float("nan")) is None
    assert bairro_entre_parenteses(pd.NA) is None
    assert bairro_entre_parenteses(None) is None
    assert bairro_entre_parenteses("R SEM BAIRRO ()") is None
