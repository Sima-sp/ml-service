"""Pacote do pipeline de IA do SIMA-SP.

Organização (ver README.md, seção "Módulos"):

- ``config``         parâmetros centrais do problema (janela, raio, períodos, limiares).
- ``tempo``          conversões de fuso (UTC <-> America/Sao_Paulo) e hora de referência.
- ``collect/``       coletores das fontes públicas (CGE, GeoSampa, Open-Meteo, pluviômetros).
- ``geo``            distâncias, agrupamento espacial e geocodificação.
- ``pontos``         unificação dos eventos e escolha dos pontos monitorados.
- ``labels``         rótulo ``alagou_3h`` e exclusão de horas já alagadas.
- ``features``       cálculo das features — MESMO código no treino e no serviço ao vivo.
- ``dataset``        montagem da tabela ponto × hora (Parquet).
- ``baseline``       régua de regras (equivalente ao RiskClassificationService do backend).
- ``modelos``        fábricas dos modelos (regressão logística, LightGBM, Random Forest).
- ``calibracao``     calibração de probabilidades e escolha dos limiares de nível.
- ``niveis``         escala única de risco BAIXO/MEDIO/ALTO/CRITICO.
- ``sensor_adjust``  ajuste por regra usando a leitura do sensor (não treinado).
- ``artefatos``      salvar/carregar o pacote do modelo (joblib + JSONs).
- ``train``          script de treino e comparação dos modelos.
- ``evaluate``       métricas, recortes (Zona Norte) e gráficos.
- ``sintetico``      dados sintéticos SÓ para testar o pipeline (nunca para resultados do TCC).

Os módulos importados pelo serviço (``features``, ``niveis``, ``sensor_adjust``, ``baseline``,
``artefatos``, ``calibracao``, ``collect.open_meteo``, ``tempo``, ``config``, ``geo``) não podem
importar bibliotecas pesadas (geopandas, matplotlib) no topo do arquivo.
"""
