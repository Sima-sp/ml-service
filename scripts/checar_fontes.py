"""Verifica se as fontes públicas respondem a partir da SUA rede (dia 1 do cronograma).

Uso::

    python scripts/checar_fontes.py

Não baixa nada grande: só confirma que cada endereço responde. Se alguma fonte falhar,
veja docs/coleta-dados.md ("Rede bloqueada").
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.collect.http import user_agent  # noqa: E402

FONTES = {
    "CGE alagamentos": ("https://www.cgesp.org/v3/alagamentos.jsp", {}),
    "GeoSampa WFS": ("http://wfs.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wfs",
                     {"service": "WFS", "request": "GetCapabilities"}),
    "Open-Meteo arquivo": ("https://archive-api.open-meteo.com/v1/archive",
                           {"latitude": -23.5152, "longitude": -46.5841, "start_date": "2024-01-01",
                            "end_date": "2024-01-02", "hourly": "precipitation"}),
    "Open-Meteo previsões históricas": ("https://historical-forecast-api.open-meteo.com/v1/forecast",
                                        {"latitude": -23.5152, "longitude": -46.5841, "start_date": "2024-01-01",
                                         "end_date": "2024-01-02", "hourly": "precipitation"}),
    "Open-Meteo ao vivo": ("https://api.open-meteo.com/v1/forecast",
                           {"latitude": -23.5152, "longitude": -46.5841, "hourly": "precipitation",
                            "past_hours": 3, "forecast_hours": 3}),
    "Nominatim": ("https://nominatim.openstreetmap.org/search",
                  {"q": "Avenida Tucuruvi, São Paulo", "format": "jsonv2", "limit": 1}),
    "Nextcloud CGE (boletins)": ("https://arquivos.saisp.br", {}),
}


def main() -> int:
    falhas = 0
    for nome, (url, params) in FONTES.items():
        try:
            resp = requests.get(url, params=params, timeout=20, headers={"User-Agent": user_agent()})
            ok = resp.status_code < 400
            print(f"[{'OK ' if ok else 'ERRO'}] {nome}: HTTP {resp.status_code} ({len(resp.content)} bytes)")
            falhas += not ok
        except requests.RequestException as erro:
            print(f"[ERRO] {nome}: {type(erro).__name__}: {erro}")
            falhas += 1
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
