"""Calibração das probabilidades e escolha dos limiares de nível (seção 7.4).

Por que calibrar? O app mostra "72 %". Esse número precisa significar "em situações parecidas,
alagou ~72 % das vezes". Modelos de árvores e a subamostragem do treino distorcem a escala; a
calibração corrige isso usando a **validação**, que mantém a prevalência real (não é
subamostrada — ver docs/adr/0003). Se um dia a validação for subamostrada, passe ``pesos``.

Métodos:
- ``sigmoide`` (Platt): regressão logística sobre o logit da pontuação. Poucos parâmetros,
  estável com poucos positivos.
- ``isotonica``: função crescente em degraus. Mais flexível, mas precisa de mais positivos.
- ``auto``: isotônica se houver pelo menos ``MIN_POSITIVOS_ISOTONICA`` positivos; senão sigmoide.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from src.config import METAS_LIMIARES
from src.niveis import validar_limiares

MIN_POSITIVOS_ISOTONICA = 300
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrador:
    """Mapeia a pontuação bruta do modelo para probabilidade calibrada."""

    def __init__(self, metodo: str = "auto"):
        if metodo not in {"auto", "sigmoide", "isotonica", "nenhum"}:
            raise ValueError(f"método inválido: {metodo}")
        self.metodo_pedido = metodo
        self.metodo = metodo
        self._modelo = None

    def ajustar(self, pontuacao, y, pesos=None) -> Calibrador:
        y = np.asarray(y, dtype=int)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == len(y):
            raise ValueError("A calibração precisa de positivos e negativos")
        if self.metodo_pedido == "auto":
            self.metodo = "isotonica" if n_pos >= MIN_POSITIVOS_ISOTONICA else "sigmoide"
        if self.metodo == "sigmoide":
            self._modelo = LogisticRegression(C=1e6, max_iter=1000)
            self._modelo.fit(_logit(pontuacao).reshape(-1, 1), y, sample_weight=pesos)
        elif self.metodo == "isotonica":
            self._modelo = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._modelo.fit(np.asarray(pontuacao, dtype="float64"), y, sample_weight=pesos)
        self.n_positivos = n_pos
        return self

    def transformar(self, pontuacao) -> np.ndarray:
        pontuacao = np.asarray(pontuacao, dtype="float64")
        if self.metodo == "nenhum":
            return pontuacao
        if self._modelo is None:
            raise RuntimeError("Calibrador não ajustado")
        if self.metodo == "sigmoide":
            return self._modelo.predict_proba(_logit(pontuacao).reshape(-1, 1))[:, 1]
        return self._modelo.predict(pontuacao)

    def descricao(self) -> dict:
        return {"metodo": self.metodo, "metodo_pedido": self.metodo_pedido,
                "n_positivos_calibracao": getattr(self, "n_positivos", None)}


def piso_orcamento(probabilidades, fracao_max: float) -> float:
    """Menor limiar que deixa no máximo ``fracao_max`` das horas em ``p >= limiar``.

    A calibração isotônica devolve degraus: milhares de horas com exatamente a mesma
    probabilidade. O quantil puro cairia dentro de um degrau e o ``>=`` puxaria o degrau
    inteiro (ex.: orçamento de 5 % virando 9,6 %). Aqui, se o quantil estoura o orçamento,
    sobe para o próximo valor distinto.
    """
    ordenadas = np.sort(np.asarray(probabilidades, dtype="float64"))
    n = len(ordenadas)
    maximo = int(np.floor(fracao_max * n))
    if maximo <= 0:
        return float(np.nextafter(ordenadas[-1], np.inf))
    candidato = ordenadas[n - maximo]
    if n - np.searchsorted(ordenadas, candidato, side="left") > maximo:
        pos = np.searchsorted(ordenadas, candidato, side="right")
        candidato = ordenadas[pos] if pos < n else np.nextafter(ordenadas[-1], np.inf)
    return float(candidato)


def escolher_limiares(y, probabilidades, metas: dict = METAS_LIMIARES) -> tuple[dict, dict]:
    """Limiares a partir de metas de precisão/recall na validação.

    Metas padrão (``config.METAS_LIMIARES``):
    - MEDIO: maior limiar que ainda captura >= 80 % das horas positivas (recall), mas sem
      passar de 5 % das horas em MEDIO ou acima (orçamento de alertas).
    - ALTO: maior limiar com recall >= 50 %, sem passar de 1 % das horas.
    - CRITICO: menor limiar com precisão >= 50 % (metade dos CRITICO viram alagamento).

    Quando o orçamento fala mais alto que o recall, o diagnóstico registra ``limitado_por``.

    Retorna (limiares, diagnóstico com precisão/recall obtidos). Se as metas forem
    inalcançáveis ou gerarem limiares fora de ordem, ajusta para manter
    ``MEDIO < ALTO < CRITICO`` e registra o ajuste no diagnóstico.
    """
    probabilidades = np.asarray(probabilidades, dtype="float64")
    precisao, recall, cortes = precision_recall_curve(np.asarray(y), probabilidades)
    precisao, recall = precisao[:-1], recall[:-1]  # alinhados com "cortes"
    diag: dict = {}
    escolhidos: dict = {}
    for nivel in ("MEDIO", "ALTO", "CRITICO"):
        meta = metas[nivel]
        limitado_por = None
        if "recall_min" in meta:
            ok = np.flatnonzero(recall >= meta["recall_min"])
            idx = ok[-1] if len(ok) else 0  # maior corte que ainda atinge o recall
            limitado_por = "recall"
        else:
            ok = np.flatnonzero(precisao >= meta["precisao_min"])
            idx = ok[0] if len(ok) else len(cortes) - 1  # menor corte com a precisão
            limitado_por = "precisao"
            if not len(ok):
                diag.setdefault("avisos", []).append(f"{nivel}: precisão-alvo inalcançável; usado o maior corte")
        limiar = float(cortes[idx])
        if "fracao_max" in meta:
            piso = piso_orcamento(probabilidades, meta["fracao_max"])
            if piso > limiar:
                limiar, limitado_por = piso, "orcamento_de_alertas"
                idx = min(int(np.searchsorted(cortes, limiar)), len(cortes) - 1)
        escolhidos[nivel] = limiar
        diag[nivel] = {"limiar": limiar, "precisao": float(precisao[idx]), "recall": float(recall[idx]),
                       "fracao_horas": float((probabilidades >= limiar).mean()),
                       "limitado_por": limitado_por, "meta": meta}

    # Garante ordem estrita e intervalo (0, 1)
    m = min(max(escolhidos["MEDIO"], 1e-4), 0.97)
    a = min(max(escolhidos["ALTO"], m * 1.5, m + 1e-4), 0.98)
    c = min(max(escolhidos["CRITICO"], a * 1.5, a + 1e-4), 0.99)
    limiares = {"MEDIO": m, "ALTO": a, "CRITICO": c}  # sem arredondar: limiares podem ser minúsculos
    if any(not np.isclose(limiares[k], escolhidos[k], rtol=1e-9, atol=0) for k in limiares):
        diag.setdefault("avisos", []).append(f"limiares ajustados para manter a ordem: {escolhidos} -> {limiares}")
    return validar_limiares(limiares), diag
