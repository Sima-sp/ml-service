"""Coletores das fontes públicas de dados.

Cada coletor segue o mesmo padrão em duas etapas, para nunca precisar baixar de novo:

1. **baixar**  -> salva a resposta original em ``data/raw/<fonte>/`` (HTML, GeoJSON, JSON...).
2. **extrair** -> lê o que está em ``data/raw`` e grava uma tabela limpa em ``data/interim/``.

Módulos:
- ``http``             sessão HTTP com cache em disco, novas tentativas e pausa entre chamadas.
- ``cge_alagamentos``  histórico de alagamentos do CGE (rótulo principal, tem hora de início).
- ``geosampa_wfs``     camadas do GeoSampa via WFS (ocorrências de alagamento, hidrografia,
                       subprefeituras).
- ``open_meteo``       chuva horária: histórico (treino) e recente (serviço ao vivo).
- ``cge_pluvio``       boletins pluviométricos do CGE (melhoria opcional).

Todos podem ser executados pela linha de comando: ``python -m src.collect.<modulo> --help``.
"""
