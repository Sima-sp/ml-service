"""Chuva ao vivo para o serviço: Open-Meteo com cache por célula e fallback (seção 8.2).

Comportamento:

1. A coordenada é convertida na **célula** de ``PASSO_GRADE_CACHE_GRAUS`` (a mesma usada no
   treino) e a chuva é pedida no centro da célula.
2. Se a célula foi baixada há menos de ``CACHE_CHUVA_TTL_MIN`` (15 min), usa o cache
   (``fonteChuva = OPEN_METEO``). O Open-Meteo atualiza de hora em hora, então não há ganho em
   baixar a cada 10 min do agendador — e isso protege a cota diária da API.
3. Células vencidas são baixadas numa **única chamada** (várias coordenadas de uma vez).
4. Se a chamada falhar, usa a última série conhecida com até ``IDADE_MAX_CHUVA_MIN`` (60 min)
   (``fonteChuva = CACHE``). Mais velha que isso: sem chuva → o serviço não gera previsão e o
   backend usa as regras (``fonte = REGRAS``).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from app.schemas import FonteChuva
from src.collect.open_meteo import buscar_recente, celula_grade, id_celula
from src.tempo import agora_utc

log = logging.getLogger(__name__)


@dataclass
class EntradaCache:
    serie: pd.Series
    obtida_em: pd.Timestamp


@dataclass
class ChuvaObtida:
    serie: pd.Series | None
    fonte: FonteChuva | None
    aviso: str | None = None


class ServicoChuva:
    def __init__(
        self,
        ttl_min: float = 15,
        idade_max_min: float = 60,
        timeout_s: float = 5,
        buscador: Callable = buscar_recente,
        relogio: Callable[[], pd.Timestamp] = agora_utc,
    ):
        self.ttl = pd.Timedelta(minutes=ttl_min)
        self.idade_max = pd.Timedelta(minutes=idade_max_min)
        self.timeout_s = timeout_s
        self.buscador = buscador
        self.relogio = relogio
        self._cache: dict[str, EntradaCache] = {}
        self._trava = threading.Lock()
        self.ultima_falha: str | None = None

    @property
    def celulas_em_cache(self) -> int:
        return len(self._cache)

    def obter(self, coordenadas: list[tuple[float, float]]) -> list[ChuvaObtida]:
        """Série de chuva para cada coordenada (na mesma ordem)."""
        chaves = [id_celula(lat, lon) for lat, lon in coordenadas]
        with self._trava:
            agora = self.relogio()
            vencidas = sorted({
                chave for chave in chaves
                if chave not in self._cache or agora - self._cache[chave].obtida_em > self.ttl
            })
            if vencidas:
                centros = {chave: celula_grade(lat, lon) for chave, (lat, lon) in zip(chaves, coordenadas, strict=True)}
                try:
                    resultados = self.buscador([centros[c] for c in vencidas], timeout=self.timeout_s)
                    for chave, resultado in zip(vencidas, resultados, strict=True):
                        self._cache[chave] = EntradaCache(resultado.serie, agora)
                    self.ultima_falha = None
                except Exception as erro:  # noqa: BLE001 - qualquer falha da API vira fallback
                    self.ultima_falha = f"{type(erro).__name__}: {erro}"
                    log.warning("Open-Meteo indisponível (%s); tentando cache", self.ultima_falha)

            saida = []
            for chave in chaves:
                entrada = self._cache.get(chave)
                if entrada is None:
                    saida.append(ChuvaObtida(None, None, "chuva indisponível (Open-Meteo falhou e não há cache)"))
                    continue
                idade = agora - entrada.obtida_em
                if idade <= self.ttl:
                    saida.append(ChuvaObtida(entrada.serie, FonteChuva.OPEN_METEO))
                elif idade <= self.idade_max:
                    minutos = int(idade.total_seconds() // 60)
                    saida.append(ChuvaObtida(entrada.serie, FonteChuva.CACHE,
                                             f"Open-Meteo indisponível; usada chuva obtida há {minutos} min"))
                else:
                    saida.append(ChuvaObtida(None, None, "chuva em cache mais velha que o limite; previsão não gerada"))
            return saida
