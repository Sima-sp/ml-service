"""Características fixas do ponto para uma coordenada qualquer.

O modelo conhece ~50–150 pontos (``models/pontos_vN.csv``). Um sensor do backend é associado
ao **ponto mais próximo**:

- a até ``RAIO_PONTO_CONHECIDO_M`` (500 m): usa todas as características daquele ponto;
- mais longe: usa lat/lon do sensor e a zona do ponto mais próximo, mas deixa ``elevacao``,
  ``dist_corrego_m`` e ``freq_hist_alagamento`` vazios (o modelo trata ausência) e devolve um
  aviso. Recomendação: posicionar os sensores simulados sobre pontos conhecidos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.geo import mais_proximo

log = logging.getLogger(__name__)


class ResolvedorPontos:
    def __init__(self, caminho_csv: Path | None, raio_conhecido_m: float = 500.0):
        self.raio = raio_conhecido_m
        self.pontos = pd.DataFrame()
        if caminho_csv is not None and Path(caminho_csv).exists():
            self.pontos = pd.read_csv(caminho_csv)
            log.info("%d pontos carregados de %s", len(self.pontos), caminho_csv)

    def __len__(self) -> int:
        return len(self.pontos)

    def resolver(self, lat: float, lon: float) -> tuple[dict, str | None, float | None, list[str]]:
        """Retorna (características, ponto_id, distância em m, avisos)."""
        vazio = {"lat": lat, "lon": lon, "elevacao": np.nan, "dist_corrego_m": np.nan,
                 "zona": None, "freq_hist_alagamento": np.nan}
        if self.pontos.empty:
            return vazio, None, None, ["sem base de pontos carregada; características do ponto ausentes"]

        i, dist = mais_proximo(lat, lon, self.pontos["lat"].to_numpy(), self.pontos["lon"].to_numpy())
        ponto = self.pontos.iloc[i]
        if dist <= self.raio:
            caracteristicas = {
                "lat": float(ponto["lat"]), "lon": float(ponto["lon"]),
                "elevacao": float(ponto.get("elevacao", np.nan)),
                "dist_corrego_m": float(ponto.get("dist_corrego_m", np.nan)),
                "zona": ponto.get("zona"),
                "freq_hist_alagamento": float(ponto.get("freq_hist_alagamento", np.nan)),
            }
            return caracteristicas, str(ponto["ponto_id"]), round(dist, 1), []
        caracteristicas = {**vazio, "zona": ponto.get("zona")}
        aviso = (f"sensor a {dist:.0f} m do ponto conhecido mais próximo ({ponto['ponto_id']}); "
                 "previsão com características do ponto incompletas")
        return caracteristicas, None, round(dist, 1), [aviso]
