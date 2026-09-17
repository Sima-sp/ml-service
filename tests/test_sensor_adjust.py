"""Casos-limite da regra do sensor (seção 7.5)."""

import pytest

from src.niveis import NivelRisco as N
from src.sensor_adjust import ajustar_por_sensor


@pytest.mark.parametrize("agua,esperado", [(79.9, N.BAIXO), (80.0, N.MEDIO), (99.9, N.MEDIO)])
def test_limiar_de_agua(agua, esperado):
    assert ajustar_por_sensor(N.BAIXO, agua, 0).nivel_final == esperado


def test_transbordando_e_medicao_e_critico():
    r = ajustar_por_sensor(N.BAIXO, 100.0, 0)
    assert r.nivel_final == N.CRITICO
    assert r.medicao_transbordando and r.ajuste_aplicado
    assert r.nivel_modelo == N.BAIXO


def test_transbordando_com_modelo_ja_critico_nao_conta_como_ajuste():
    r = ajustar_por_sensor(N.CRITICO, 120.0, 0)
    assert r.medicao_transbordando and not r.ajuste_aplicado


def test_lixo_sem_chuva_nao_sobe():
    assert ajustar_por_sensor(N.MEDIO, 10, 90, chuva_recente_mm=0.0).nivel_final == N.MEDIO


def test_lixo_com_chuva_recente_ou_prevista_sobe():
    assert ajustar_por_sensor(N.MEDIO, 10, 60, chuva_recente_mm=1.0).nivel_final == N.ALTO
    assert ajustar_por_sensor(N.MEDIO, 10, 60, chuva_prevista_mm=2.0).nivel_final == N.ALTO
    assert ajustar_por_sensor(N.MEDIO, 10, 59.9, chuva_recente_mm=5.0).nivel_final == N.MEDIO


def test_regras_acumulam_e_respeitam_teto():
    assert ajustar_por_sensor(N.BAIXO, 85, 70, chuva_recente_mm=3).nivel_final == N.ALTO
    assert ajustar_por_sensor(N.ALTO, 85, 70, chuva_recente_mm=3).nivel_final == N.CRITICO
    regras = {"nivel_agua_transbordando": 100, "nivel_agua_alerta": 80, "lixo_alerta": 60,
              "chuva_minima_mm": 0.2, "acumular": False}
    assert ajustar_por_sensor(N.BAIXO, 85, 70, 3, regras=regras).nivel_final == N.MEDIO


def test_nunca_desce():
    for nivel in N:
        assert ajustar_por_sensor(nivel, 0, 0).nivel_final == nivel


def test_sem_leitura():
    r = ajustar_por_sensor(N.ALTO, None, None)
    assert r.sem_leitura and not r.ajuste_aplicado and r.nivel_final == N.ALTO


def test_so_um_campo_do_sensor():
    r = ajustar_por_sensor(N.BAIXO, 90, None)
    assert not r.sem_leitura and r.nivel_final == N.MEDIO
