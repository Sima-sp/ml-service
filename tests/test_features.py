"""Features: janelas corretas, sem olhar o futuro e iguais no treino e ao vivo."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    COLUNAS_CHUVA,
    calcular_features_chuva,
    calcular_features_tempo,
    completar_resumo_chuva,
    features_ao_vivo,
    horas_desde_ultima_chuva,
    lista_features,
    preparar_matriz,
)

ESTATICAS = {"lat": -23.5, "lon": -46.6, "elevacao": 740.0, "dist_corrego_m": 120.0,
             "zona": "NORTE", "freq_hist_alagamento": 1.5}


def _serie(valores, inicio="2024-01-01"):
    return pd.Series(valores, index=pd.date_range(inicio, periods=len(valores), freq="h", tz="UTC"))


def test_somas_moveis_e_pico():
    valores = np.zeros(100)
    valores[-3:] = [1.0, 2.0, 3.0]
    valores[50] = 10.0
    f = calcular_features_chuva(_serie(valores)).iloc[-1]
    assert f["chuva_1h"] == 3.0
    assert f["chuva_3h"] == 6.0
    assert f["chuva_6h"] == 6.0
    assert f["chuva_24h"] == 6.0
    assert f["chuva_72h"] == 16.0
    assert f["max_chuva_1h_24h"] == 3.0
    assert f["horas_desde_ultima_chuva"] == 0


def test_janela_incompleta_vira_nan():
    f = calcular_features_chuva(_serie(np.ones(30)))
    assert f["chuva_72h"].isna().all()
    assert f["chuva_24h"].iloc[:23].isna().all()
    assert f["chuva_24h"].iloc[23] == 24


def test_hora_faltante_nao_subestima_soma():
    s = _serie(np.ones(80)).drop(pd.Timestamp("2024-01-03 10:00", tz="UTC"))
    f = calcular_features_chuva(s)
    assert np.isnan(f.loc[pd.Timestamp("2024-01-03 11:00", tz="UTC"), "chuva_3h"])


def test_sem_vazamento_do_futuro():
    """Mudar a chuva DEPOIS de t não pode mudar nenhuma feature do modelo em t."""
    valores = np.random.default_rng(0).gamma(0.5, 2.0, 200)
    t = pd.Timestamp("2024-01-05 12:00", tz="UTC")
    base = calcular_features_chuva(_serie(valores)).loc[t, COLUNAS_CHUVA]
    alterado = valores.copy()
    posicao = _serie(valores).index.get_loc(t)
    alterado[posicao + 1:] = 99.0
    novo = calcular_features_chuva(_serie(alterado)).loc[t, COLUNAS_CHUVA]
    pd.testing.assert_series_equal(base, novo)


def test_chuva_prevista_usa_so_as_proximas_3_horas():
    valores = np.zeros(10)
    valores[5:8] = [1, 2, 3]
    valores[8] = 50
    f = calcular_features_chuva(_serie(valores))
    assert f["chuva_prevista_3h"].iloc[4] == 6
    assert "chuva_prevista_3h" not in lista_features(False)


def test_horas_desde_ultima_chuva_com_teto():
    r = horas_desde_ultima_chuva(np.array([0, 0, 0, 0, 1, 0, 0]), teto=3)
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert r[2] == r[3] == 3  # sem chuva desde o início, mas a série já cobre 3 h secas
    assert list(r[4:]) == [0, 1, 2]
    assert horas_desde_ultima_chuva(np.array([1] + [0] * 10), teto=3)[-1] == 3


def test_tempo_em_hora_local_com_horario_de_verao():
    # 2018-11-04 foi o início do horário de verão (UTC-2) em São Paulo
    indice = pd.DatetimeIndex(["2018-11-04 12:00", "2018-06-01 12:00"], tz="UTC")
    t = calcular_features_tempo(indice)
    assert list(t["hora_dia"]) == [10, 9]
    assert list(t["estacao_chuvosa"]) == [1, 0]
    assert t["dia_semana"].iloc[0] == 6  # domingo


def test_ao_vivo_igual_ao_treino():
    """features_ao_vivo precisa dar exatamente a mesma linha que o dataset histórico."""
    valores = np.random.default_rng(1).gamma(0.4, 3.0, 150)
    serie = _serie(valores)
    t = serie.index[120]
    historico = calcular_features_chuva(serie).loc[t]
    linha, t_ref, avisos = features_ao_vivo(serie, t + pd.Timedelta(minutes=37), ESTATICAS)
    assert t_ref == t and not avisos
    for coluna in COLUNAS_CHUVA:
        assert linha.iloc[0][coluna] == pytest.approx(historico[coluna])


def test_ao_vivo_usa_hora_anterior_se_ultima_falta():
    serie = _serie(np.ones(80))
    serie.iloc[-1] = np.nan
    linha, t_ref, avisos = features_ao_vivo(serie, serie.index[-1], ESTATICAS)
    assert t_ref == serie.index[-2]
    assert avisos


def test_ao_vivo_serie_curta_falha():
    with pytest.raises(ValueError):
        features_ao_vivo(_serie(np.ones(20)), _serie(np.ones(20)).index[-1], ESTATICAS)


def test_resumo_parcial_fica_coerente():
    r = completar_resumo_chuva({"chuva_1h": 12.4, "chuva_3h": 30.0, "chuva_24h": 41.2})
    assert r["chuva_1h"] <= r["chuva_3h"] <= r["chuva_6h"] <= r["chuva_24h"] <= r["chuva_72h"]
    assert r["horas_desde_ultima_chuva"] == 0
    assert r["max_chuva_1h_24h"] >= 12.4
    seco = completar_resumo_chuva({})
    assert seco["chuva_72h"] == 0 and seco["horas_desde_ultima_chuva"] == 72


def test_preparar_matriz_tipos_fixos():
    df = pd.DataFrame([{**ESTATICAS, **dict.fromkeys(COLUNAS_CHUVA, 1.0),
                        "hora_dia": 1, "mes": 1, "dia_semana": 1, "estacao_chuvosa": 1}])
    X = preparar_matriz(df, lista_features(False))
    assert list(X["zona"].cat.categories) == ["CENTRO", "LESTE", "NORTE", "OESTE", "SUL"]
    assert X["chuva_1h"].dtype == "float32"
    with pytest.raises(KeyError):
        preparar_matriz(df.drop(columns="chuva_1h"), lista_features(False))
