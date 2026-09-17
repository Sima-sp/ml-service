"""Cenários da demo ponta a ponta (dia 17): envia requisições ao serviço e mostra o resultado.

Uso::

    python scripts/demo_cenarios.py --url http://localhost:5000
    python scripts/demo_cenarios.py --url http://localhost:5000 --lat -23.5152 --lon -46.5841

Cenários (seção 12, dia 17):
1. seco                   — sem chuva, sensor normal
2. chuva forte            — 72 h de série com pancada de 60 mm nas últimas 3 h
3. bueiro entupido        — chuva moderada + lixo 75 %
4. água alta              — nivelAgua 85 %
5. transbordando          — nivelAgua 100 % (medição, não previsão)
6. sensor em manutenção   — sem nivelAgua/porcentagemLixo
7. chuva real agora       — sem série informada: o serviço busca no Open-Meteo

O cenário "IA fora" é testado no backend (desligue o container e veja o fallback para regras).
"""

from __future__ import annotations

import argparse
import json

import requests

SECA = [0.0] * 72
FORTE = [0.0] * 66 + [0.5, 2.0, 6.0, 15.0, 25.0, 20.0]
MODERADA = [0.0] * 66 + [0.0, 0.5, 1.0, 3.0, 4.0, 2.0]


def cenarios(lat: float, lon: float) -> dict[str, dict]:
    base = {"latitude": lat, "longitude": lon}
    return {
        "1. seco": {**base, "sensorId": 1, "nivelAgua": 20, "porcentagemLixo": 10, "chuvaHoraria": SECA},
        "2. chuva forte": {**base, "sensorId": 1, "nivelAgua": 45, "porcentagemLixo": 10, "chuvaHoraria": FORTE},
        "3. bueiro entupido": {**base, "sensorId": 1, "nivelAgua": 40, "porcentagemLixo": 75, "chuvaHoraria": MODERADA},
        "4. água alta": {**base, "sensorId": 1, "nivelAgua": 85, "porcentagemLixo": 20, "chuvaHoraria": MODERADA},
        "5. transbordando": {**base, "sensorId": 1, "nivelAgua": 100, "porcentagemLixo": 30, "chuvaHoraria": FORTE},
        "6. sensor em manutenção": {**base, "sensorId": 1, "chuvaHoraria": MODERADA},
        "7. chuva real agora": {**base, "sensorId": 1, "nivelAgua": 30, "porcentagemLixo": 20},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:5000")
    parser.add_argument("--lat", type=float, default=-23.5152, help="padrão: Vila Maria")
    parser.add_argument("--lon", type=float, default=-46.5841)
    args = parser.parse_args()
    print(json.dumps(requests.get(f"{args.url}/health", timeout=5).json(), ensure_ascii=False))
    for nome, corpo in cenarios(args.lat, args.lon).items():
        resp = requests.post(f"{args.url}/predict", json=corpo, timeout=10)
        dados = resp.json()
        if resp.status_code != 200:
            print(f"{nome:26s} HTTP {resp.status_code}: {dados}")
            continue
        prob = dados["probabilidadeAlagamento"]
        prob_txt = "  —  " if prob is None else f"{prob:6.1%}"
        medido = " [MEDIDO: transbordando]" if dados["medicaoTransbordando"] else ""
        if dados["semLeituraSensor"]:
            medido += " [sem leitura do sensor]"
        print(f"{nome:26s} {prob_txt}  {dados['nivelRiscoModelo']:>7s} -> {dados['nivelRisco']:<7s}"
              f" chuva3h={dados['chuvaRecente3hMm']} fonte={dados['fonteChuva']}{medido}")
        for motivo in dados["motivosAjuste"] + dados["avisos"]:
            print(f"{'':28s}· {motivo}")


if __name__ == "__main__":
    main()
