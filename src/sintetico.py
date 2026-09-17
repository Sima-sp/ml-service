"""Dados SINTÉTICOS para testar o pipeline de ponta a ponta.

⚠️ Nada gerado aqui pode aparecer como resultado do TCC. O objetivo é só verificar que coleta →
dataset → treino → serviço funcionam antes de os dados reais chegarem (e nos testes
automatizados). Só viraria "Plano C" (seção 13) se as fontes reais falharem — e, nesse caso,
calibrado com os eventos reais disponíveis e declarado como limitação.

Como os dados são gerados:
- **Chuva**: tempestades sorteadas hora a hora, mais frequentes de outubro a março e à tarde;
  intensidade Gama; cada célula recebe a chuva regional multiplicada por um fator aleatório.
- **Pontos**: coordenadas na cidade (metade na Zona Norte), vulnerabilidade entre 0 e 1.
- **Alagamentos**: a chance de começar um alagamento na hora ``h`` cresce com a chuva da
  própria hora ``h`` e das anteriores e com a vulnerabilidade do ponto. Assim a chuva que já
  caiu até ``t`` carrega informação (parcial) sobre alagamentos em ``(t, t+3h]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.collect.open_meteo import id_celula
from src.config import FUSO_LOCAL, MESES_ESTACAO_CHUVOSA, SEMENTE

ZONAS_SINTETICAS = {
    "NORTE": (-23.49, -46.60), "SUL": (-23.62, -46.68), "LESTE": (-23.55, -46.48),
    "OESTE": (-23.56, -46.72), "CENTRO": (-23.55, -46.635),
}


def gerar_chuva_regional(horas: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Série horária regional (mm) com sazonalidade e ciclo diurno."""
    local = horas.tz_convert(FUSO_LOCAL)
    chuvoso = np.isin(local.month, MESES_ESTACAO_CHUVOSA)
    tarde = (local.hour >= 14) & (local.hour <= 20)
    p_inicio = np.where(chuvoso, 0.010, 0.0035) * np.where(tarde, 2.5, 0.6)
    inicios = rng.random(len(horas)) < p_inicio
    chuva = np.zeros(len(horas))
    for i in np.flatnonzero(inicios):
        duracao = int(rng.geometric(0.35))
        pico = rng.gamma(1.3, 5.0)
        perfil = pico * np.exp(-0.5 * np.arange(duracao))
        fim = min(i + duracao, len(horas))
        chuva[i:fim] += perfil[: fim - i]
    return chuva


def gerar_entradas(inicio: str = "2016-01-01", fim: str = "2025-12-31", n_pontos: int = 30,
                   semente: int = SEMENTE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna ``(pontos, chuva_horaria, eventos)`` no mesmo formato das etapas reais."""
    rng = np.random.default_rng(semente)
    horas = pd.date_range(
        (pd.Timestamp(inicio) - pd.Timedelta(days=4)).tz_localize(FUSO_LOCAL).tz_convert("UTC"),
        (pd.Timestamp(fim) + pd.Timedelta(days=1)).tz_localize(FUSO_LOCAL).tz_convert("UTC"),
        freq="h",
    )

    # --- pontos
    zonas = ["NORTE"] * (n_pontos // 2) + list(rng.choice(
        [z for z in ZONAS_SINTETICAS if z != "NORTE"], n_pontos - n_pontos // 2))
    linhas = []
    for i, zona in enumerate(zonas, start=1):
        lat0, lon0 = ZONAS_SINTETICAS[zona]
        lat, lon = lat0 + rng.normal(0, 0.03), lon0 + rng.normal(0, 0.03)
        linhas.append({
            "ponto_id": f"P{i:04d}", "lat": round(lat, 6), "lon": round(lon, 6),
            "bairro": f"Bairro {i}", "subprefeitura": f"Sub {zona.title()}", "zona": zona,
            "dist_corrego_m": round(float(rng.gamma(2.0, 150.0)), 1),
            "elevacao": round(float(rng.normal(760, 25)), 1),
            "vulnerabilidade": float(rng.beta(2, 3)),
        })
    pontos = pd.DataFrame(linhas)
    pontos["celula_id"] = [id_celula(a, b) for a, b in zip(pontos.lat, pontos.lon, strict=True)]

    # --- chuva por célula
    regional = gerar_chuva_regional(horas, rng)
    partes = []
    series = {}
    for celula in pontos["celula_id"].unique():
        fator = rng.lognormal(0.0, 0.45, len(horas))
        extra = gerar_chuva_regional(horas, rng) * 0.3
        valores = np.round(regional * fator + extra, 1)
        series[celula] = valores
        partes.append(pd.DataFrame({"celula_id": celula, "data_hora_utc": horas,
                                    "chuva_mm": valores.astype("float32")}))
    chuva = pd.concat(partes, ignore_index=True)

    # --- eventos (dependem da chuva da própria hora e das 2 anteriores)
    eventos = []
    for p in pontos.itertuples(index=False):
        s = pd.Series(series[p.celula_id])
        c1 = s.to_numpy()
        c3 = s.rolling(3, min_periods=1).sum().to_numpy()
        c24 = s.rolling(24, min_periods=1).sum().to_numpy()
        logito = (-10.5 + 0.16 * c1 + 0.05 * c3 + 0.01 * c24 + 3.0 * p.vulnerabilidade
                  - 0.002 * p.dist_corrego_m)
        prob = 1 / (1 + np.exp(-logito))
        sorteio = np.flatnonzero(rng.random(len(horas)) < prob)
        ultimo_fim = pd.Timestamp.min.tz_localize("UTC")
        for i in sorteio:
            inicio_ev = horas[i] - pd.Timedelta(minutes=int(rng.integers(0, 60)))
            if inicio_ev <= ultimo_fim:
                continue
            fim_ev = inicio_ev + pd.Timedelta(minutes=int(rng.integers(40, 180)))
            ultimo_fim = fim_ev
            eventos.append({
                "evento_id": f"sin_{p.ponto_id}_{i}",
                "fonte": "CGE" if rng.random() < 0.6 else "GEOSAMPA",
                "inicio_utc": inicio_ev, "fim_utc": fim_ev if rng.random() < 0.7 else pd.NaT,
                "lat": p.lat + rng.normal(0, 0.0006), "lon": p.lon + rng.normal(0, 0.0006),
                "precisao_geocod": "sintetico", "zona": p.zona,
            })
    eventos = pd.DataFrame(eventos)
    eventos["inicio_utc"] = pd.to_datetime(eventos["inicio_utc"], utc=True)
    eventos["fim_utc"] = pd.to_datetime(eventos["fim_utc"], utc=True)
    contagem = eventos.groupby(eventos["evento_id"].str.split("_").str[1]).size()
    pontos["n_eventos"] = pontos["ponto_id"].map(contagem).fillna(0).astype(int)
    return pontos.drop(columns="vulnerabilidade"), chuva, eventos
