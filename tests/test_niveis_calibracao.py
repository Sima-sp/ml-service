"""Níveis de risco, calibração e escolha de limiares."""

import numpy as np
import pytest

from src.calibracao import Calibrador, escolher_limiares
from src.niveis import (
    NivelRisco,
    carregar_limiares,
    niveis_por_probabilidade,
    nivel_por_probabilidade,
    salvar_limiares,
    subir_nivel,
    validar_limiares,
    valor_contrato,
)

LIM = {"MEDIO": 0.15, "ALTO": 0.40, "CRITICO": 0.70}


@pytest.mark.parametrize("p,esperado", [
    (0.0, "BAIXO"), (0.1499, "BAIXO"), (0.15, "MEDIO"), (0.3999, "MEDIO"),
    (0.40, "ALTO"), (0.70, "CRITICO"), (1.0, "CRITICO"),
])
def test_limites_dos_niveis(p, esperado):
    assert nivel_por_probabilidade(p, LIM).value == esperado


def test_versao_vetorizada_bate_com_escalar():
    probs = np.linspace(0, 1, 101)
    vet = niveis_por_probabilidade(probs, LIM)
    esc = [nivel_por_probabilidade(p, LIM).ordem for p in probs]
    assert list(vet) == esc


def test_subir_nivel_tem_teto():
    assert subir_nivel(NivelRisco.ALTO, 5) == NivelRisco.CRITICO


def test_valor_no_contrato():
    assert valor_contrato(NivelRisco.MEDIO) == "MEDIO"
    assert valor_contrato(NivelRisco.CRITICO, com_acento=True) == "CRÍTICO"


def test_limiares_invalidos():
    with pytest.raises(ValueError):
        validar_limiares({"MEDIO": 0.5, "ALTO": 0.4, "CRITICO": 0.7})
    with pytest.raises(ValueError):
        validar_limiares({"MEDIO": 0.1, "ALTO": 0.4})


def test_salvar_e_carregar_limiares(tmp_path):
    arq = tmp_path / "thresholds.json"
    salvar_limiares(LIM, arq, {"estrategia": "fixos"})
    assert carregar_limiares(arq) == LIM


def _dados(n=20000, semente=0):
    rng = np.random.default_rng(semente)
    x = rng.normal(size=n)
    p_real = 1 / (1 + np.exp(-(x * 1.5 - 4)))
    y = (rng.random(n) < p_real).astype(int)
    pontuacao = 1 / (1 + np.exp(-(x * 3)))  # pontuação descalibrada, mas monotônica
    return pontuacao, y, p_real


@pytest.mark.parametrize("metodo", ["sigmoide", "isotonica"])
def test_calibracao_melhora_brier(metodo):
    pont, y, _ = _dados()
    cal = Calibrador(metodo).ajustar(pont, y)
    p = cal.transformar(pont)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.mean((p - y) ** 2) < np.mean((pont - y) ** 2)
    assert abs(p.mean() - y.mean()) < 0.01


def test_calibracao_monotonica_e_auto():
    pont, y, _ = _dados()
    cal = Calibrador("auto").ajustar(pont, y)
    ordem = np.argsort(pont)
    assert np.all(np.diff(cal.transformar(pont[ordem])) >= -1e-12)
    assert cal.metodo in {"sigmoide", "isotonica"}


def test_calibracao_exige_duas_classes():
    with pytest.raises(ValueError):
        Calibrador().ajustar([0.1, 0.2], [0, 0])


def test_escolha_de_limiares_ordenada():
    pont, y, _ = _dados()
    p = Calibrador("sigmoide").ajustar(pont, y).transformar(pont)
    sem_orcamento = {"MEDIO": {"recall_min": 0.8}, "ALTO": {"recall_min": 0.5}, "CRITICO": {"precisao_min": 0.5}}
    limiares, diag = escolher_limiares(y, p, sem_orcamento)
    assert 0 < limiares["MEDIO"] < limiares["ALTO"] < limiares["CRITICO"] < 1
    assert diag["MEDIO"]["recall"] >= 0.8 - 1e-9
    # com o orçamento padrão (5 % das horas), o recall pedido não cabe e o orçamento manda
    _, diag_padrao = escolher_limiares(y, p)
    assert diag_padrao["MEDIO"]["limitado_por"] == "orcamento_de_alertas"
    assert diag_padrao["MEDIO"]["fracao_horas"] <= 0.0501


def test_orcamento_de_alertas_limita_modelo_fraco():
    rng = np.random.default_rng(3)
    y = (rng.random(50000) < 0.002).astype(int)
    p = np.clip(0.002 + 0.001 * y + rng.normal(0, 0.001, 50000), 0, 1)  # quase sem sinal
    limiares, diag = escolher_limiares(y, p)
    assert (p >= limiares["ALTO"]).mean() <= 0.0101
    assert (p >= limiares["MEDIO"]).mean() <= 0.0501
    assert diag["ALTO"]["limitado_por"] == "orcamento_de_alertas"


def test_orcamento_respeitado_com_degraus_da_isotonica():
    """Probabilidades em degraus (isotônica) não podem estourar o orçamento de alertas."""
    import numpy as np

    from src.calibracao import escolher_limiares, piso_orcamento

    rng = np.random.default_rng(0)
    # 90 % num degrau baixo, 8 % num degrau médio, 2 % em valores altos distintos
    p = np.concatenate([np.full(9000, 0.001), np.full(800, 0.003), rng.uniform(0.01, 0.3, 200)])
    y = (rng.random(len(p)) < p * 5).astype(int)
    assert (p >= piso_orcamento(p, 0.05)).mean() <= 0.05
    assert (p >= piso_orcamento(p, 0.10)).mean() <= 0.10
    limiares, diag = escolher_limiares(y, p)
    assert diag["MEDIO"]["fracao_horas"] <= 0.05
    assert diag["ALTO"]["fracao_horas"] <= 0.01
