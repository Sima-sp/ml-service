"""Fixtures compartilhadas.

Nada aqui acessa a internet: a chuva "ao vivo" é simulada por ``buscador_falso`` e o modelo de
teste é treinado em segundos sobre dados sintéticos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.configuracao import Configuracoes
from src.collect.open_meteo import SerieChuva
from src.tempo import agora_utc


def serie_chuva(valores_finais: list[float], horas_totais: int = 102, horas_futuras: int = 6,
                fim_passado: pd.Timestamp | None = None) -> pd.Series:
    """Série horária cujo trecho que termina na hora atual contém ``valores_finais``."""
    fim_passado = fim_passado if fim_passado is not None else agora_utc().floor("h")
    indice = pd.date_range(end=fim_passado + pd.Timedelta(hours=horas_futuras), periods=horas_totais, freq="h")
    valores = np.zeros(horas_totais)
    if valores_finais:
        fim = horas_totais - horas_futuras
        valores[fim - len(valores_finais):fim] = valores_finais
    return pd.Series(valores, index=indice)


class BuscadorFalso:
    """Substitui o Open-Meteo. Conta chamadas e pode simular falha."""

    def __init__(self, valores_finais=(0.0,), falhar: bool = False):
        self.valores_finais = list(valores_finais)
        self.falhar = falhar
        self.chamadas = 0
        self.coordenadas_pedidas: list = []

    def __call__(self, coordenadas, timeout=5):
        self.chamadas += 1
        self.coordenadas_pedidas.append(list(coordenadas))
        if self.falhar:
            raise ConnectionError("Open-Meteo fora do ar (simulado)")
        return [SerieChuva(lat, lon, serie_chuva(self.valores_finais)) for lat, lon in coordenadas]


@pytest.fixture
def buscador_falso():
    return BuscadorFalso


@pytest.fixture(scope="session")
def dados_sinteticos():
    from src.sintetico import gerar_entradas

    return gerar_entradas(inicio="2020-01-01", fim="2025-12-31", n_pontos=8, semente=7)


@pytest.fixture(scope="session")
def dataset_sintetico(dados_sinteticos):
    from src.dataset import montar_dataset

    pontos, chuva, eventos = dados_sinteticos
    return montar_dataset(pontos, chuva, eventos, inicio="2020-01-01", fim="2025-12-31")


@pytest.fixture(scope="session")
def diretorio_modelo(tmp_path_factory, dataset_sintetico):
    """Treina um LightGBM pequeno e salva o pacote como versão 'teste'."""
    from src.artefatos import PacoteModelo, salvar_pacote
    from src.calibracao import Calibrador
    from src.features import lista_features, preparar_matriz
    from src.modelos import criar_lightgbm, pontuacao

    dataset, _, freq = dataset_sintetico
    features = lista_features(False)
    treino = dataset[dataset["split"] == "treino"]
    val = dataset[dataset["split"] == "validacao"]
    modelo = criar_lightgbm(n_estimators=60)
    modelo.fit(preparar_matriz(treino, features), treino["alagou_3h"], sample_weight=treino["peso_amostra"])
    calibrador = Calibrador("sigmoide").ajustar(pontuacao(modelo, preparar_matriz(val, features)), val["alagou_3h"])
    pontos = dataset.groupby("ponto_id", observed=True)[["lat", "lon", "elevacao", "dist_corrego_m", "zona"]].first()
    pontos = pontos.reset_index()
    pontos["ponto_id"] = pontos["ponto_id"].astype(str)
    producao = freq[freq["ano"] == freq["ano"].max()][["ponto_id", "freq_hist_alagamento"]]
    pontos = pontos.merge(producao, on="ponto_id")
    pacote = PacoteModelo("teste", "lightgbm", modelo, calibrador, features,
                          {"MEDIO": 0.01, "ALTO": 0.05, "CRITICO": 0.2}, {"fonte_chuva": "sintetica"})
    diretorio = tmp_path_factory.mktemp("models")
    salvar_pacote(pacote, pontos, diretorio)
    return diretorio


@pytest.fixture
def config_com_modelo(diretorio_modelo):
    return Configuracoes(versao_modelo="teste", dir_modelos=diretorio_modelo)


@pytest.fixture
def config_sem_modelo(tmp_path):
    return Configuracoes(versao_modelo="inexistente", dir_modelos=tmp_path)
