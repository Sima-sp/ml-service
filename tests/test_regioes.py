"""Alerta por subprefeitura: nomes, agregação, cobertura e o endpoint."""

import json
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.configuracao import Configuracoes
from app.fonte_chuva import ServicoChuva
from app.main import criar_app
from app.predictor import Preditor
from src.regioes import (
    MIN_PONTOS_REGIAO,
    agregar_por_regiao,
    anotar_regiao,
    avaliar_regioes,
    identificar_regiao,
    series_por_regiao,
)


def test_nomes_diferentes_da_mesma_regiao():
    """O CGE escreve 'Freguesia do Ó'; o GeoSampa, 'FREGUESIA-BRASILANDIA'."""
    cge = identificar_regiao("Freguesia do Ó")
    geosampa = identificar_regiao("FREGUESIA-BRASILANDIA")
    assert cge == geosampa
    assert cge[1] == "Freguesia do Ó / Brasilândia" and cge[2] == "NORTE"


def test_vila_mariana_nao_vira_vila_maria():
    assert identificar_regiao("VILA MARIANA")[0] == "VILA-MARIANA"
    assert identificar_regiao("VILA MARIA-VILA GUILHERME")[0] == "VILA-MARIA-VILA-GUILHERME"
    assert identificar_regiao("Vila Guilherme")[0] == "VILA-MARIA-VILA-GUILHERME"


def test_nome_desconhecido_nao_quebra():
    assert identificar_regiao("Bairro Inventado") == (None, None, None)
    assert identificar_regiao(None) == (None, None, None)
    assert identificar_regiao(float("nan")) == (None, None, None)


def _pontos(subprefeituras):
    return pd.DataFrame({
        "ponto_id": [f"P{i:04d}" for i in range(1, len(subprefeituras) + 1)],
        "subprefeitura": subprefeituras,
        "lat": [-23.5] * len(subprefeituras),
        "lon": [-46.6] * len(subprefeituras),
    })


def test_regiao_fica_com_o_maior_risco_dos_seus_pontos():
    pontos = anotar_regiao(_pontos(["SANTANA-TUCURUVI"] * 3 + ["MOOCA"] * 3))
    resumo = agregar_por_regiao(pontos, [0.01, 0.05, 0.02, 0.003, 0.001, 0.002])

    santana = resumo[resumo.regiao_id == "SANTANA-TUCURUVI"].iloc[0]
    assert santana.probabilidade == 0.05
    assert santana.ponto_id == "P0002"  # o ponto que puxou o risco vai junto
    assert resumo.iloc[0].regiao_id == "SANTANA-TUCURUVI"  # ordenado pelo risco


def test_regiao_com_poucos_pontos_fica_sem_cobertura():
    """Com 1 ponto, o 'alerta da região' seria o alerta daquele ponto disfarçado."""
    pontos = anotar_regiao(_pontos(["SANTANA-TUCURUVI"] * MIN_PONTOS_REGIAO + ["ITAQUERA"]))
    resumo = agregar_por_regiao(pontos, [0.01] * MIN_PONTOS_REGIAO + [0.9])

    por_id = resumo.set_index("regiao_id")
    assert por_id.loc["SANTANA-TUCURUVI", "cobertura"] == "COBERTA"
    assert por_id.loc["ITAQUERA", "cobertura"] == "SEM_COBERTURA"


def test_serie_e_metricas_por_regiao():
    """A série região × hora é o que o alerta promete: alagou ALGUM ponto da região?"""
    horas = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    dataset = pd.DataFrame({
        "regiao_id": ["MOOCA", "MOOCA", "MOOCA", "MOOCA"],
        "data_hora_utc": [horas[0], horas[0], horas[1], horas[1]],
        "alagou_3h": [0, 1, 0, 0],
    })
    serie = series_por_regiao(dataset, [0.2, 0.4, 0.01, 0.02])

    assert len(serie) == 2
    assert serie.sort_values("data_hora_utc").iloc[0].tolist()[2:] == [0.4, 1]

    metricas = avaliar_regioes(serie, {"MEDIO": 0.05, "ALTO": 0.3, "CRITICO": 0.9})
    total = metricas[metricas.regiao_id == "TOTAL"].iloc[0]
    assert total.horas_com_alerta == 1 and total.precisao == 1.0 and total.recall == 1.0


@pytest.fixture
def config_regioes(diretorio_modelo, tmp_path):
    """Copia o modelo de teste e acrescenta subprefeitura nos pontos + limiares de região."""
    destino = tmp_path / "models"
    shutil.copytree(diretorio_modelo, destino)

    pontos = pd.read_csv(destino / "pontos_teste.csv")
    regioes = ["SANTANA-TUCURUVI", "VILA MARIA-VILA GUILHERME", "MOOCA"]
    pontos["subprefeitura"] = [regioes[i % 3] for i in range(len(pontos))]
    pontos.to_csv(destino / "pontos_teste.csv", index=False)

    (destino / "thresholds_regiao_teste.json").write_text(
        json.dumps({"limiares": {"MEDIO": 0.01, "ALTO": 0.05, "CRITICO": 0.2}}), encoding="utf-8")
    return Configuracoes(versao_modelo="teste", dir_modelos=destino)


def test_endpoint_de_regioes(config_regioes, buscador_falso):
    preditor = Preditor(config_regioes, servico_chuva=ServicoChuva(buscador=buscador_falso([5.0, 15.0, 30.0])))
    with TestClient(criar_app(preditor)) as cliente:
        corpo = cliente.get("/predict/regioes").json()

    assert corpo["origem"] == "MODELO" and corpo["minPontosRegiao"] == MIN_PONTOS_REGIAO
    assert corpo["regioes"], "deveria devolver pelo menos uma região"
    riscos = [r["probabilidadeMaxima"] for r in corpo["regioes"] if r["probabilidadeMaxima"] is not None]
    assert riscos == sorted(riscos, reverse=True), "da mais crítica para a menos"
    for regiao in corpo["regioes"]:
        assert regiao["regiao"] and regiao["zona"]
        if regiao["cobertura"] == "SEM_COBERTURA":
            assert regiao["nivelRisco"] is None  # sem cobertura nunca vira nível
        else:
            assert regiao["nivelRisco"] in {"BAIXO", "MEDIO", "ALTO", "CRITICO"}
            assert regiao["pontoId"]


def test_sem_limiares_de_regiao_o_alerta_fica_desligado(config_com_modelo, buscador_falso):
    """Sem o arquivo de limiares, responde 503 em vez de usar limiar de outra escala."""
    preditor = Preditor(config_com_modelo, servico_chuva=ServicoChuva(buscador=buscador_falso([5.0])))
    with TestClient(criar_app(preditor)) as cliente:
        resposta = cliente.get("/predict/regioes")

    assert resposta.status_code == 503
    assert resposta.json()["codigo"] == "SEM_LIMIARES_REGIAO"
