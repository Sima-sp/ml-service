"""Parâmetros centrais do problema de previsão de alagamentos.

Tudo que é "decisão de projeto" fica aqui, num lugar só, para ficar fácil de citar no TCC e
de ajustar depois do portão de decisão (seção 13 do briefing). Os valores são os iniciais do
briefing; os marcados com ``# DECISÃO EM ABERTO`` ainda dependem da análise dos dados.

Convenções:
- Datas/horas são guardadas em **UTC** internamente. O horário local (America/Sao_Paulo)
  só é usado para features de calendário e para ler as fontes que publicam em hora local.
- Distâncias em metros; chuva em milímetros.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "data"
DIR_RAW = DIR_DADOS / "raw"
DIR_INTERIM = DIR_DADOS / "interim"
DIR_PROCESSED = DIR_DADOS / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_RELATORIOS = RAIZ / "reports"

# Arquivos intermediários (contrato entre as etapas do pipeline)
ARQ_EVENTOS_CGE = DIR_INTERIM / "eventos_cge.parquet"
ARQ_EVENTOS_GEOSAMPA = DIR_INTERIM / "eventos_geosampa.parquet"
ARQ_EVENTOS = DIR_INTERIM / "eventos.parquet"  # união das fontes, com coordenadas
ARQ_PONTOS = DIR_PROCESSED / "pontos.csv"
ARQ_CHUVA_HORARIA = DIR_INTERIM / "chuva_horaria.parquet"  # ponto_id × hora

# ---------------------------------------------------------------------------
# Definição do problema (seção 3 do briefing)
# ---------------------------------------------------------------------------
FUSO_LOCAL = "America/Sao_Paulo"
JANELA_PREVISAO_HORAS = 3  # fixa; fora de escopo aumentar
RAIO_ROTULO_M = 300  # DECISÃO EM ABERTO: raio R do rótulo proxy
# Duração assumida de um alagamento sem horário de término (para excluir horas "já alagadas")
DURACAO_PADRAO_EVENTO_H = 2
# Durações acima disso são tratadas como erro de digitação e trocadas pela duração padrão
# (no CGE 2011–2025: mediana de 76 min, p75 de 166 min, máximo de 1439 min = 23h59).
DURACAO_MAXIMA_EVENTO_H = 12

# ---------------------------------------------------------------------------
# Pontos monitorados (seção 6.1)
# ---------------------------------------------------------------------------
RAIO_AGRUPAMENTO_M = 200  # raio do DBSCAN (briefing: 150–300 m)
MIN_EVENTOS_PONTO = 3  # mínimo de registros para um agrupamento virar "ponto recorrente"
N_PONTOS_CIDADE = 100  # DECISÃO EM ABERTO: os N mais recorrentes da cidade (~50–150)
# Pontos escolhidos só com eventos até este ano (evita viés de seleção pelo período de teste)
ANO_LIMITE_SELECAO_PONTOS = 2022

# Zona de cada subprefeitura (chave = nome normalizado, sem acento, maiúsculo).
# A comparação é por prefixo, testando as chaves mais longas primeiro
# ("VILA MARIANA" antes de "VILA MARIA").
ZONA_POR_SUBPREFEITURA: dict[str, str] = {
    # Centro
    "SE": "CENTRO",
    # Norte (Norte 1 e Norte 2)
    "CASA VERDE": "NORTE",
    "FREGUESIA": "NORTE",
    "JACANA": "NORTE",
    "PERUS": "NORTE",
    "PIRITUBA": "NORTE",
    "SANTANA": "NORTE",
    "VILA MARIA": "NORTE",
    # Oeste
    "BUTANTA": "OESTE",
    "LAPA": "OESTE",
    "PINHEIROS": "OESTE",
    # Sul
    "CAMPO LIMPO": "SUL",
    "CAPELA DO SOCORRO": "SUL",
    "CIDADE ADEMAR": "SUL",
    "IPIRANGA": "SUL",
    "JABAQUARA": "SUL",
    "M BOI MIRIM": "SUL",
    "MBOI MIRIM": "SUL",
    "PARELHEIROS": "SUL",
    "SANTO AMARO": "SUL",
    "VILA MARIANA": "SUL",
    # Leste
    "ARICANDUVA": "LESTE",
    "CIDADE TIRADENTES": "LESTE",
    "ERMELINO MATARAZZO": "LESTE",
    "GUAIANASES": "LESTE",
    "ITAIM PAULISTA": "LESTE",
    "ITAQUERA": "LESTE",
    "MOOCA": "LESTE",
    "PENHA": "LESTE",
    "SAO MATEUS": "LESTE",
    "SAO MIGUEL": "LESTE",
    "SAPOPEMBA": "LESTE",
    "VILA PRUDENTE": "LESTE",
}
ZONAS = ["CENTRO", "LESTE", "NORTE", "OESTE", "SUL"]  # ordem fixa das categorias
ZONA_FOCO = "NORTE"  # recorte de avaliação e da demo
# Bairros de interesse da demo (sensores simulados)
BAIRROS_DEMO_ZONA_NORTE = [
    "Tucuruvi", "Vila Maria", "Vila Guilherme", "Santana",
    "Jaçanã", "Tremembé", "Casa Verde",
]

# Caixa aproximada do município (validação de coordenadas)
LIMITES_SP = {"lat_min": -24.01, "lat_max": -23.35, "lon_min": -46.83, "lon_max": -46.36}

# ---------------------------------------------------------------------------
# Dataset (seção 6.2)
# ---------------------------------------------------------------------------
# 2017: é quando começa o ECMWF IFS no arquivo do Open-Meteo (antes disso o arquivo é ERA5,
# outro modelo e outra resolução). Conferido em 16/09/2026 — ver docs/recomendacoes.md (R17).
PERIODO_INICIO = "2017-01-01"
PERIODO_FIM = "2025-12-31"
AQUECIMENTO_HORAS = 72  # horas de chuva antes do início para as janelas móveis

# Uma hora conta como "seca" se choveu menos que isso nas últimas 24 h
LIMIAR_HORA_SECA_24H_MM = 0.2
# Fração das horas secas negativas mantida no TREINO (as de validação/teste ficam completas)
TAXA_AMOSTRAGEM_SECAS = 0.07
SUBAMOSTRAR_APENAS_TREINO = True
SEMENTE = 42

# Divisão temporal (seção 7.1) — ajustar à cobertura real
ANO_FIM_TREINO = 2022
ANOS_VALIDACAO = (2023,)
ANOS_TESTE = (2024, 2025)

# ---------------------------------------------------------------------------
# Features (ver src/features.py)
# ---------------------------------------------------------------------------
LIMIAR_CHUVA_MM = 0.2  # abaixo disso a hora é considerada sem chuva (ruído/garoa)
MAX_HORAS_SEM_CHUVA = 72  # teto de "horas_desde_ultima_chuva" (= tamanho da série ao vivo)
MESES_ESTACAO_CHUVOSA = (10, 11, 12, 1, 2, 3)
# Chuva PREVISTA para (t, t+3h] como feature. Desligada no modelo principal: ver
# docs/recomendacoes.md (R1). Só ligar com a API "Historical Forecast" no treino.
USAR_CHUVA_PREVISTA = False

# ---------------------------------------------------------------------------
# Níveis de risco (seção 7.4) — proposta inicial; os finais saem da curva P/R
# ---------------------------------------------------------------------------
LIMIARES_NIVEIS_PADRAO = {"MEDIO": 0.15, "ALTO": 0.40, "CRITICO": 0.70}
# Estratégia alternativa: limiares escolhidos por metas de precisão/recall na validação.
# "fracao_max" é o ORÇAMENTO de alertas: fração máxima das horas que pode ficar naquele nível
# ou acima. Sem ele, um modelo fraco atinge o recall pedido colocando metade das horas em ALTO.
METAS_LIMIARES = {
    "MEDIO": {"recall_min": 0.80, "fracao_max": 0.05},
    "ALTO": {"recall_min": 0.50, "fracao_max": 0.01},
    "CRITICO": {"precisao_min": 0.50},
}

# ---------------------------------------------------------------------------
# Ajuste pelo sensor (seção 7.5) — regra, não treinada
# ---------------------------------------------------------------------------
AJUSTE_SENSOR = {
    "nivel_agua_transbordando": 100.0,  # >= : medição "transbordando agora" => CRITICO
    "nivel_agua_alerta": 80.0,  # >= : sobe um nível
    "lixo_alerta": 60.0,  # >= (com chuva) : sobe um nível
    "chuva_minima_mm": LIMIAR_CHUVA_MM,  # "chuva prevista/recente > 0"
    "acumular": True,  # água alta + lixo alto sobem dois níveis
}

# ---------------------------------------------------------------------------
# Tempo real (seção 8)
# ---------------------------------------------------------------------------
VALIDADE_PREVISAO_MIN = 30
IDADE_MAX_CHUVA_MIN = 60
HORAS_PASSADAS_AO_VIVO = 96  # > 72 para sobrar margem nas janelas
HORAS_FUTURAS_AO_VIVO = 6
PASSO_GRADE_CACHE_GRAUS = 0.05  # célula de cache (~5,5 km); menor que a grade dos modelos

# ---------------------------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------------------------
URL_OPEN_METEO_ARQUIVO = "https://archive-api.open-meteo.com/v1/archive"
URL_OPEN_METEO_PREVISOES_HISTORICAS = "https://historical-forecast-api.open-meteo.com/v1/forecast"
URL_OPEN_METEO_AO_VIVO = "https://api.open-meteo.com/v1/forecast"
# Modelo meteorológico FIXO no treino e ao vivo (portão de decisão, 16/09/2026):
# para São Paulo o "best_match" ao vivo É o ECMWF IFS 9 km (mesma grade e mesmos valores), e o
# arquivo tem esse modelo desde 01/01/2017. Fixar o nome evita trocas silenciosas do best_match.
OPEN_METEO_MODELO: str | None = "ecmwf_ifs"

# ---------------------------------------------------------------------------
# Artefatos
# ---------------------------------------------------------------------------
VERSAO_MODELO = "v1"


def caminho_modelo(versao: str = VERSAO_MODELO, diretorio: Path = DIR_MODELOS) -> Path:
    """Arquivo joblib com estimador + calibrador."""
    return diretorio / f"flood_lgbm_{versao}.joblib"


def caminho_limiares(versao: str = VERSAO_MODELO, diretorio: Path = DIR_MODELOS) -> Path:
    """JSON com os limiares de probabilidade de cada nível."""
    return diretorio / f"thresholds_{versao}.json"


def caminho_metadados(versao: str = VERSAO_MODELO, diretorio: Path = DIR_MODELOS) -> Path:
    """JSON com features, período, métricas e data do treino."""
    return diretorio / f"metadata_{versao}.json"


def caminho_pontos_modelo(versao: str = VERSAO_MODELO, diretorio: Path = DIR_MODELOS) -> Path:
    """CSV com as características fixas dos pontos, usado pelo serviço ao vivo."""
    return diretorio / f"pontos_{versao}.csv"
