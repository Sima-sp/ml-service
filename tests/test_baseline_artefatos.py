"""Baseline de regras e ida-e-volta do pacote do modelo."""

import numpy as np
import pandas as pd

from src.artefatos import carregar_pacote, existe_pacote
from src.baseline import BaselineRegras
from src.features import lista_features, preparar_matriz
from src.niveis import NivelRisco


def test_baseline_maior_nivel_entre_janelas():
    base = BaselineRegras({"chuva_1h": (5, 15, 30), "chuva_3h": (10, 30, 50)})
    df = pd.DataFrame({"chuva_1h": [0, 6, 2, 31], "chuva_3h": [0, 6, 35, 0]})
    assert list(base.niveis(df)) == [0, 1, 2, 3]
    assert np.allclose(base.pontuacao(df), [0, 1 / 3, 2 / 3, 1])
    assert base.nivel({"chuva_1h": 16, "chuva_3h": 16}) == NivelRisco.ALTO


def test_pacote_ida_e_volta(diretorio_modelo, dataset_sintetico):
    assert existe_pacote("teste", diretorio_modelo)
    pacote = carregar_pacote("teste", diretorio_modelo)
    dataset, _, _ = dataset_sintetico
    amostra = dataset[dataset["split"] == "teste"].head(500)
    p = pacote.prever_probabilidade(amostra)
    assert p.shape == (500,) and np.all((p >= 0) & (p <= 1))
    assert pacote.features == lista_features(False)
    assert "bibliotecas" in pacote.metadados
    # a mesma matriz preparada dá a mesma pontuação
    bruto = pacote.estimador.predict_proba(preparar_matriz(amostra, pacote.features))[:, 1]
    assert np.allclose(bruto, pacote.pontuacao_bruta(amostra))
