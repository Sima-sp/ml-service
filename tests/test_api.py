"""Contrato HTTP do serviço (schema, modos, fallback)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.fonte_chuva import ServicoChuva
from app.main import criar_app
from app.predictor import Preditor
from src.tempo import agora_utc

CAMPOS_RESPOSTA = {
    "probabilidadeAlagamento", "nivelRisco", "janelaHoras", "geradaEm", "validaAte",
    "medicaoTransbordando", "ajusteSensorAplicado", "fonteChuva", "modeloVersao",
    "nivelRiscoModelo", "origem", "horaReferencia", "semLeituraSensor", "avisos",
}


def _cliente(config, buscador):
    preditor = Preditor(config, servico_chuva=ServicoChuva(buscador=buscador))
    return TestClient(criar_app(preditor)), preditor


@pytest.fixture
def cliente_modelo(config_com_modelo, buscador_falso):
    cliente, preditor = _cliente(config_com_modelo, buscador_falso(valores_finais=[2.0, 8.0, 20.0]))
    with cliente:
        yield cliente, preditor


def _ponto(preditor):
    p = preditor.resolvedor.pontos.iloc[0]
    return float(p["lat"]), float(p["lon"])


def test_health_e_model_info(cliente_modelo):
    cliente, _ = cliente_modelo
    saude = cliente.get("/health").json()
    assert saude["status"] == "ok" and saude["modeloVersao"] == "teste" and saude["origem"] == "MODELO"
    info = cliente.get("/model-info").json()
    assert "features" in info and "limiares" in info


def test_payload_antigo_continua_valido(cliente_modelo):
    cliente, preditor = cliente_modelo
    lat, lon = _ponto(preditor)
    r = cliente.post("/predict", json={"nivelAgua": 50.0, "porcentagemLixo": 10.0, "chuvaMm": 3.0,
                                       "latitude": lat, "longitude": lon})
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert CAMPOS_RESPOSTA <= corpo.keys()
    assert 0 <= corpo["probabilidadeAlagamento"] <= 1
    assert corpo["nivelRisco"] in {"BAIXO", "MEDIO", "ALTO", "CRITICO"}
    assert corpo["janelaHoras"] == 3
    assert corpo["fonteChuva"] == "OPEN_METEO"
    assert corpo["pontoId"] is not None
    assert pd.Timestamp(corpo["validaAte"]) - pd.Timestamp(corpo["geradaEm"]) == pd.Timedelta(minutes=30)
    assert corpo["chuvaRecente3hMm"] == 30.0


def test_payload_novo_do_briefing(cliente_modelo):
    cliente, preditor = cliente_modelo
    lat, lon = _ponto(preditor)
    r = cliente.post("/predict", json={
        "sensorId": 12, "latitude": lat, "longitude": lon,
        "timestamp": agora_utc().tz_convert("America/Sao_Paulo").isoformat(),
        "nivelAgua": 82.0, "porcentagemLixo": 64.0, "chuvaMm": 30.0,
        "chuva": {"chuva_1h": 12.4, "chuva_3h": 30.0, "chuva_24h": 41.2},
    })
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["sensorId"] == 12
    assert corpo["fonteChuva"] == "INFORMADA"
    assert corpo["ajusteSensorAplicado"] is True
    assert len(corpo["motivosAjuste"]) == 2


def test_transbordando_e_medicao(cliente_modelo):
    cliente, preditor = cliente_modelo
    lat, lon = _ponto(preditor)
    corpo = cliente.post("/predict", json={"latitude": lat, "longitude": lon, "nivelAgua": 100}).json()
    assert corpo["medicaoTransbordando"] is True and corpo["nivelRisco"] == "CRITICO"


def test_sem_leitura_do_sensor(cliente_modelo):
    cliente, preditor = cliente_modelo
    lat, lon = _ponto(preditor)
    corpo = cliente.post("/predict", json={"latitude": lat, "longitude": lon}).json()
    assert corpo["semLeituraSensor"] is True and corpo["ajusteSensorAplicado"] is False


def test_chuva_horaria_informada(cliente_modelo):
    cliente, preditor = cliente_modelo
    lat, lon = _ponto(preditor)
    serie = [0.0] * 69 + [5.0, 20.0, 45.0]
    corpo = cliente.post("/predict", json={"latitude": lat, "longitude": lon, "chuvaHoraria": serie}).json()
    assert corpo["fonteChuva"] == "INFORMADA" and corpo["chuvaRecente3hMm"] == 70.0


def test_chuva_horaria_curta_e_rejeitada(cliente_modelo):
    cliente, _ = cliente_modelo
    r = cliente.post("/predict", json={"latitude": -23.5, "longitude": -46.6, "chuvaHoraria": [1.0] * 10})
    assert r.status_code == 422


def test_validacao_de_entrada(cliente_modelo):
    cliente, _ = cliente_modelo
    assert cliente.post("/predict", json={"latitude": 200, "longitude": -46.6}).status_code == 422
    assert cliente.post("/predict", json={"latitude": -23.5, "longitude": -46.6,
                                          "porcentagemLixo": 120}).status_code == 422


def test_lote_mantem_ordem_e_isola_erros(config_com_modelo, buscador_falso):
    cliente, preditor = _cliente(config_com_modelo, buscador_falso(falhar=True))
    lat, lon = _ponto(preditor)
    with cliente:
        r = cliente.post("/predict/batch", json=[
            {"sensorId": 1, "latitude": lat, "longitude": lon},  # sem chuva -> erro
            {"sensorId": 2, "latitude": lat, "longitude": lon, "chuva": {"chuva_1h": 10}},
            {"sensorId": 3, "latitude": lat, "longitude": lon, "chuvaHoraria": [0.0] * 72},
        ])
    assert r.status_code == 200
    itens = r.json()
    assert [i["sensorId"] for i in itens] == [1, 2, 3]
    assert itens[0]["sucesso"] is False and itens[0]["codigoErro"] == "CHUVA_INDISPONIVEL"
    assert itens[1]["sucesso"] and itens[1]["previsao"]["fonteChuva"] == "INFORMADA"
    assert itens[2]["sucesso"]


def test_sem_chuva_devolve_503(config_com_modelo, buscador_falso):
    cliente, preditor = _cliente(config_com_modelo, buscador_falso(falhar=True))
    lat, lon = _ponto(preditor)
    with cliente:
        r = cliente.post("/predict", json={"latitude": lat, "longitude": lon})
    assert r.status_code == 503
    assert r.json()["codigo"] == "CHUVA_INDISPONIVEL"


def test_chuva_mm_legado_quando_open_meteo_falha(config_com_modelo, buscador_falso):
    cliente, preditor = _cliente(config_com_modelo, buscador_falso(falhar=True))
    lat, lon = _ponto(preditor)
    with cliente:
        corpo = cliente.post("/predict", json={"latitude": lat, "longitude": lon, "chuvaMm": 12}).json()
    assert corpo["fonteChuva"] == "INFORMADA_LEGADO"


def test_timestamp_fora_da_janela(cliente_modelo):
    cliente, preditor = cliente_modelo
    lat, lon = _ponto(preditor)
    r = cliente.post("/predict", json={"latitude": lat, "longitude": lon, "timestamp": "2020-01-01T10:00:00-03:00"})
    assert r.status_code == 422 and r.json()["codigo"] == "CHUVA_FORA_DA_JANELA"


def test_modo_regras_sem_modelo(config_sem_modelo, buscador_falso):
    cliente, _ = _cliente(config_sem_modelo, buscador_falso(valores_finais=[10.0, 15.0, 20.0]))
    with cliente:
        assert cliente.get("/health").json()["origem"] == "REGRAS"
        corpo = cliente.post("/predict", json={"latitude": -23.5, "longitude": -46.6}).json()
    assert corpo["origem"] == "REGRAS"
    assert corpo["probabilidadeAlagamento"] is None
    assert corpo["modeloVersao"] == "regras-v0"
    assert corpo["nivelRisco"] == "ALTO"  # 20 mm/1h e 45 mm/3h -> ALTO pelas regras provisórias
    assert corpo["avisos"]


def test_sensor_longe_de_ponto_conhecido(cliente_modelo):
    cliente, _ = cliente_modelo
    corpo = cliente.post("/predict", json={"latitude": -23.70, "longitude": -46.40}).json()
    assert corpo["pontoId"] is None and any("ponto conhecido" in a for a in corpo["avisos"])


def test_nivel_com_acento_configuravel(config_com_modelo, buscador_falso):
    config = config_com_modelo.model_copy(update={"nivel_risco_com_acento": True})
    cliente, preditor = _cliente(config, buscador_falso())
    lat, lon = _ponto(preditor)
    with cliente:
        corpo = cliente.post("/predict", json={"latitude": lat, "longitude": lon, "nivelAgua": 100}).json()
    assert corpo["nivelRisco"] == "CRÍTICO"
