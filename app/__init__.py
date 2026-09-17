"""Serviço HTTP (FastAPI) de previsão de alagamentos do SIMA-SP.

- ``main``                  aplicação e rotas (/health, /model-info, /predict, /predict/batch)
- ``schemas``               contrato JSON (Pydantic) — espelha os DTOs do backend Java
- ``configuracao``          variáveis de ambiente
- ``predictor``             orquestra: chuva -> features -> modelo -> níveis -> ajuste do sensor
- ``fonte_chuva``           Open-Meteo ao vivo com cache por célula e fallback
- ``caracteristicas_ponto`` características fixas do ponto mais próximo
"""
