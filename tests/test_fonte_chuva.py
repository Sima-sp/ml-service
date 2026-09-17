"""Cache e fallback da chuva ao vivo (seção 8.2)."""

import pandas as pd

from app.fonte_chuva import ServicoChuva
from app.schemas import FonteChuva


class Relogio:
    def __init__(self):
        self.agora = pd.Timestamp("2026-09-16 17:00", tz="UTC")

    def __call__(self):
        return self.agora

    def avancar(self, minutos):
        self.agora += pd.Timedelta(minutes=minutos)


def test_cache_por_celula_e_uma_chamada_por_lote(buscador_falso):
    buscador = buscador_falso()
    servico = ServicoChuva(ttl_min=15, buscador=buscador, relogio=Relogio())
    coords = [(-23.5152, -46.5841), (-23.5160, -46.5850), (-23.46, -46.60)]
    r = servico.obter(coords)
    assert buscador.chamadas == 1
    assert len(buscador.coordenadas_pedidas[0]) == 2  # duas células distintas
    assert all(x.fonte == FonteChuva.OPEN_METEO for x in r)
    servico.obter(coords)
    assert buscador.chamadas == 1  # cache válido


def test_cache_vence_e_rebaixa(buscador_falso):
    buscador = buscador_falso()
    relogio = Relogio()
    servico = ServicoChuva(ttl_min=15, buscador=buscador, relogio=relogio)
    servico.obter([(-23.5, -46.6)])
    relogio.avancar(16)
    servico.obter([(-23.5, -46.6)])
    assert buscador.chamadas == 2


def test_falha_usa_cache_com_menos_de_1h(buscador_falso):
    buscador = buscador_falso()
    relogio = Relogio()
    servico = ServicoChuva(ttl_min=15, idade_max_min=60, buscador=buscador, relogio=relogio)
    servico.obter([(-23.5, -46.6)])
    buscador.falhar = True
    relogio.avancar(40)
    r = servico.obter([(-23.5, -46.6)])[0]
    assert r.fonte == FonteChuva.CACHE and r.serie is not None and "40 min" in r.aviso
    relogio.avancar(30)  # 70 min
    r = servico.obter([(-23.5, -46.6)])[0]
    assert r.serie is None and r.fonte is None


def test_falha_sem_cache(buscador_falso):
    servico = ServicoChuva(buscador=buscador_falso(falhar=True), relogio=Relogio())
    r = servico.obter([(-23.5, -46.6)])[0]
    assert r.serie is None and r.aviso
