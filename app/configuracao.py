"""Configurações do serviço, lidas de variáveis de ambiente (ou do arquivo ``.env``).

Veja ``.env.example`` para a lista comentada. Todas têm valor padrão, então o serviço sobe
sem nenhuma configuração.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import (
    DIR_MODELOS,
    IDADE_MAX_CHUVA_MIN,
    VALIDADE_PREVISAO_MIN,
    VERSAO_MODELO,
)


class Configuracoes(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    versao_modelo: str = VERSAO_MODELO
    dir_modelos: Path = DIR_MODELOS
    porta: int = 5000
    log_level: str = "INFO"

    validade_previsao_min: int = VALIDADE_PREVISAO_MIN
    cache_chuva_ttl_min: int = 15
    idade_max_chuva_min: int = IDADE_MAX_CHUVA_MIN
    timeout_open_meteo_s: float = 5.0
    raio_ponto_conhecido_m: float = 500.0
    nivel_risco_com_acento: bool = False
    max_itens_lote: int = 500


@lru_cache
def obter_configuracoes() -> Configuracoes:
    return Configuracoes()
