"""Rótulo alagou_3h: janela (t, t+3h], exclusão de horas alagadas e raio."""

import numpy as np
import pandas as pd

from src.labels import eventos_por_ponto, rotular_ponto, taxa_historica_anual

HORAS = pd.date_range("2024-01-01 00:00", periods=12, freq="h", tz="UTC")


def _eventos(inicios, fins=None):
    fins = fins or [None] * len(inicios)
    return pd.DataFrame({
        "evento_id": [f"e{i}" for i in range(len(inicios))],
        "inicio_utc": pd.to_datetime(inicios, utc=True),
        "fim_utc": pd.to_datetime(fins, utc=True),
    })


def test_janela_aberta_a_esquerda_fechada_a_direita():
    r = rotular_ponto(HORAS, _eventos(["2024-01-01 05:00"], ["2024-01-01 06:00"]))
    # início às 05:00 -> positivo para t = 02:00, 03:00, 04:00 (05:00 ∈ (t, t+3h])
    assert list(r["alagou_3h"].iloc[:6]) == [0, 0, 1, 1, 1, 0]
    assert r["evento_alvo_id"].iloc[2] == "e0"


def test_horas_alagadas_sao_marcadas():
    r = rotular_ponto(HORAS, _eventos(["2024-01-01 05:30"], ["2024-01-01 07:10"]))
    assert list(r["ja_alagado"].iloc[4:9]) == [False, False, True, True, False]


def test_sem_fim_usa_duracao_padrao():
    r = rotular_ponto(HORAS, _eventos(["2024-01-01 05:00"]), duracao_padrao_h=2)
    assert list(r["ja_alagado"].iloc[4:8]) == [False, True, True, False]


def test_eventos_sobrepostos():
    r = rotular_ponto(HORAS, _eventos(["2024-01-01 02:00", "2024-01-01 03:00"],
                                      ["2024-01-01 09:00", "2024-01-01 04:00"]))
    assert r["ja_alagado"].iloc[2:9].all()
    assert not r["ja_alagado"].iloc[9]


def test_ponto_sem_eventos():
    r = rotular_ponto(HORAS, None)
    assert r["alagou_3h"].sum() == 0 and not r["ja_alagado"].any()


def test_raio_do_rotulo():
    pontos = pd.DataFrame({"ponto_id": ["A"], "lat": [-23.5], "lon": [-46.6]})
    eventos = _eventos(["2024-01-01", "2024-01-02"])
    eventos["lat"] = [-23.5 + 0.0025, -23.5 + 0.004]  # ~280 m e ~445 m
    eventos["lon"] = [-46.6, -46.6]
    assert list(eventos_por_ponto(pontos, eventos, raio_m=300)["A"]["evento_id"]) == ["e0"]


def test_frequencia_historica_so_usa_anos_anteriores():
    ev = {"A": _eventos(["2016-02-01", "2017-03-01", "2017-04-01", "2019-01-10"])}
    freq = taxa_historica_anual(ev, [2016, 2017, 2018, 2020], ano_inicio_dados=2016).set_index("ano")
    assert np.isnan(freq.loc[2016, "freq_hist_alagamento"])
    assert freq.loc[2017, "freq_hist_alagamento"] == 1.0  # 1 evento em 1 ano
    assert freq.loc[2018, "freq_hist_alagamento"] == 1.5  # 3 eventos em 2 anos
    assert freq.loc[2020, "freq_hist_alagamento"] == 1.0  # 4 eventos em 4 anos


def test_duracao_absurda_vira_padrao():
    # "De 10:00 a 09:59" vira 23h59 no parser; aqui é tratado como erro e vale a duração padrão (2 h)
    r = rotular_ponto(HORAS, _eventos(["2024-01-01 01:30"], ["2024-01-02 01:29"]),
                      duracao_padrao_h=2, duracao_maxima_h=12)
    assert list(r["ja_alagado"].iloc[1:5]) == [False, True, True, False]
