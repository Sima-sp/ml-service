"""Integração do pipeline de dataset com dados sintéticos."""

import numpy as np

from src.features import COLUNAS_CHUVA


def test_estrutura_e_divisao_temporal(dataset_sintetico):
    dataset, resumo, _ = dataset_sintetico
    assert set(dataset["split"].unique()) == {"treino", "validacao", "teste"}
    assert dataset.loc[dataset["split"] == "treino", "ano"].max() <= 2022
    assert set(dataset.loc[dataset["split"] == "validacao", "ano"]) == {2023}
    assert set(dataset.loc[dataset["split"] == "teste", "ano"]) <= {2024, 2025}
    assert not dataset[COLUNAS_CHUVA].isna().any().any()
    assert resumo["por_split"]["treino"]["positivos"] > 0


def test_subamostragem_so_no_treino(dataset_sintetico):
    dataset, _, _ = dataset_sintetico
    fora_treino = dataset[dataset["split"] != "treino"]
    assert (fora_treino["peso_amostra"] == 1).all()
    treino = dataset[dataset["split"] == "treino"]
    assert (treino["peso_amostra"] > 1).any()
    # horas com peso > 1 são sempre secas e negativas
    pesadas = treino[treino["peso_amostra"] > 1]
    assert (pesadas["alagou_3h"] == 0).all()
    assert (pesadas["chuva_24h"] < 0.2).all()


def test_validacao_e_teste_completos(dataset_sintetico):
    """Sem subamostragem, cada ponto tem (quase) todas as horas do ano de validação."""
    dataset, _, _ = dataset_sintetico
    val = dataset[dataset["split"] == "validacao"]
    por_ponto = val.groupby("ponto_id", observed=True).size()
    assert (por_ponto > 8700).all()


def test_positivos_tem_evento_alvo(dataset_sintetico):
    dataset, _, _ = dataset_sintetico
    positivos = dataset[dataset["alagou_3h"] == 1]
    assert positivos["evento_alvo_id"].notna().all()
    assert dataset.loc[dataset["alagou_3h"] == 0, "evento_alvo_id"].isna().all()


def test_frequencia_primeiro_ano_vazia(dataset_sintetico):
    dataset, resumo, freq = dataset_sintetico
    primeiro = resumo["ano_inicio_eventos"]
    assert dataset.loc[dataset["ano"] == primeiro, "freq_hist_alagamento"].isna().all()
    assert freq["ano"].max() == 2026  # valor de produção
    assert np.isfinite(dataset.loc[dataset["ano"] > primeiro, "freq_hist_alagamento"]).all()
