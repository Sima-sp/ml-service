"""Contrato JSON do serviço (seção 9 do briefing), em Pydantic.

Os nomes dos campos são os mesmos dos DTOs do backend Java (camelCase em português).
Campos novos são **opcionais**, então o payload antigo (``nivelAgua``, ``porcentagemLixo``,
``chuvaMm``, ``latitude``, ``longitude``) continua válido. Mudanças aqui são decisão de
arquitetura: registrar em ``docs/adr/0001-contrato-predict.md``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FonteChuva(str, Enum):
    OPEN_METEO = "OPEN_METEO"  # baixada agora (ou cache ainda válido)
    CACHE = "CACHE"  # Open-Meteo falhou; usada a última chuva conhecida (< 1 h)
    INFORMADA = "INFORMADA"  # enviada no request (chuvaHoraria ou chuva)
    INFORMADA_LEGADO = "INFORMADA_LEGADO"  # só chuvaMm (campo antigo), tratado como chuva_1h


class OrigemPrevisao(str, Enum):
    MODELO = "MODELO"  # modelo treinado + calibração
    REGRAS = "REGRAS"  # sem modelo carregado: baseline de regras (sem probabilidade)


class ResumoChuva(BaseModel):
    """Acumulados de chuva informados pelo cliente (todos opcionais, em mm)."""

    model_config = ConfigDict(extra="forbid")

    chuva_1h: float | None = Field(None, ge=0)
    chuva_3h: float | None = Field(None, ge=0)
    chuva_6h: float | None = Field(None, ge=0)
    chuva_24h: float | None = Field(None, ge=0)
    chuva_72h: float | None = Field(None, ge=0)
    max_chuva_1h_24h: float | None = Field(None, ge=0)
    horas_desde_ultima_chuva: float | None = Field(None, ge=0)
    chuva_prevista_3h: float | None = Field(None, ge=0)


class RequisicaoPrevisao(BaseModel):
    """Entrada de ``POST /predict``."""

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "sensorId": 12, "latitude": -23.5152, "longitude": -46.5841,
        "timestamp": "2026-09-16T14:20:00-03:00", "nivelAgua": 82.0, "porcentagemLixo": 64.0,
        "chuvaMm": 30.0, "chuva": {"chuva_1h": 12.4, "chuva_3h": 30.0, "chuva_24h": 41.2},
    }]})

    sensorId: int | None = Field(None, description="ID do sensor no backend (novo, opcional)")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime | None = Field(None, description="Instante da previsão; padrão = agora")
    nivelAgua: float | None = Field(None, ge=0, le=200, description="% da capacidade (>=100 = transbordando)")
    porcentagemLixo: float | None = Field(None, ge=0, le=100)
    chuvaMm: float | None = Field(None, ge=0, description="LEGADO: usado como chuva_1h só se nada mais vier")
    chuva: ResumoChuva | None = Field(None, description="Acumulados informados (sobrescrevem os calculados)")
    chuvaHoraria: list[float] | None = Field(
        None, min_length=72, max_length=240,
        description="Série horária (mm), a última posição é a hora de referência. Para cenários de demo.",
    )

    @field_validator("chuvaHoraria")
    @classmethod
    def _sem_negativos(cls, valores):
        if valores is not None and any(v < 0 for v in valores):
            raise ValueError("chuvaHoraria não aceita valores negativos")
        return valores


class RespostaPrevisao(BaseModel):
    """Saída de ``POST /predict``."""

    sensorId: int | None = None
    probabilidadeAlagamento: float | None = Field(
        None, description="0–1, calibrada. null quando origem=REGRAS (não há probabilidade honesta)")
    nivelRisco: str = Field(..., description="Nível final (após ajuste do sensor)")
    nivelRiscoModelo: str = Field(..., description="Nível só pela probabilidade/regras, antes do sensor")
    janelaHoras: int = 3
    geradaEm: str
    validaAte: str
    horaReferencia: str = Field(..., description="Hora cheia usada nas features (fim do último intervalo de chuva)")
    medicaoTransbordando: bool = False
    ajusteSensorAplicado: bool = False
    motivosAjuste: list[str] = []
    semLeituraSensor: bool = False
    fonteChuva: FonteChuva
    chuvaRecente3hMm: float | None = None
    chuvaPrevista3hMm: float | None = Field(None, description="Informativo (Open-Meteo); não entra no modelo v1")
    origem: OrigemPrevisao
    modeloVersao: str
    pontoId: str | None = None
    distanciaPontoM: float | None = None
    avisos: list[str] = []


class RegiaoRisco(BaseModel):
    """Risco de uma subprefeitura: o maior risco entre os pontos monitorados dela."""

    regiaoId: str
    regiao: str = Field(..., description="Nome para a tela, ex.: 'Vila Maria / Vila Guilherme'")
    zona: str
    nivelRisco: str | None = Field(None, description="null quando a região não tem cobertura")
    probabilidadeMaxima: float | None = None
    pontoId: str | None = Field(None, description="Ponto que puxou o risco da região")
    logradouro: str | None = None
    pontosMonitorados: int
    cobertura: str = Field(..., description="COBERTA | SEM_COBERTURA (poucos pontos monitorados)")


class RespostaRegioes(BaseModel):
    """Saída de ``GET /predict/regioes``: uma linha por subprefeitura, da mais crítica para a menos."""

    geradaEm: str
    validaAte: str
    horaReferencia: str
    janelaHoras: int = 3
    origem: OrigemPrevisao
    modeloVersao: str
    minPontosRegiao: int = Field(..., description="Abaixo disso a região fica SEM_COBERTURA")
    regioes: list[RegiaoRisco]
    avisos: list[str] = []


class ItemLote(BaseModel):
    """Um resultado de ``POST /predict/batch`` (a lista mantém a ordem da entrada)."""

    sensorId: int | None = None
    sucesso: bool
    previsao: RespostaPrevisao | None = None
    erro: str | None = None
    codigoErro: str | None = None


class RespostaSaude(BaseModel):
    status: str
    modeloVersao: str
    origem: OrigemPrevisao
    pontosCarregados: int
    celulasChuvaEmCache: int


class ErroResposta(BaseModel):
    codigo: str
    mensagem: str
