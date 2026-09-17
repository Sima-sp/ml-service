"""Pacote do modelo: o que o treino salva e o serviço carrega (seção 7.6).

Arquivos em ``models/`` para a versão ``vN``:

- ``flood_lgbm_vN.joblib``  estimador + calibrador + lista de features (binário)
- ``thresholds_vN.json``    limiares BAIXO/MEDIO/ALTO/CRITICO + como foram escolhidos
- ``metadata_vN.json``      features, período, fonte da chuva, métricas, data do treino
- ``pontos_vN.csv``         características fixas dos pontos (lat, lon, elevação, córrego,
                            zona, ``freq_hist_alagamento`` de produção)

O joblib referencia classes de ``src/`` (ex.: ``Calibrador``), por isso a imagem do serviço
copia a pasta ``src/`` junto com ``app/``. Treinar e servir com as MESMAS versões de
scikit-learn/LightGBM (ver ``metadata.bibliotecas``).
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DIR_MODELOS,
    caminho_limiares,
    caminho_metadados,
    caminho_modelo,
    caminho_pontos_modelo,
)
from src.features import preparar_matriz
from src.niveis import carregar_limiares, salvar_limiares


@dataclass
class PacoteModelo:
    versao: str
    nome_modelo: str
    estimador: Any
    calibrador: Any
    features: list[str]
    limiares: dict
    metadados: dict = field(default_factory=dict)

    def pontuacao_bruta(self, df: pd.DataFrame) -> np.ndarray:
        X = preparar_matriz(df, self.features)
        return self.estimador.predict_proba(X)[:, 1]

    def prever_probabilidade(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidade CALIBRADA de alagamento nas próximas 3 h."""
        return np.clip(self.calibrador.transformar(self.pontuacao_bruta(df)), 0.0, 1.0)


def versoes_bibliotecas() -> dict:
    import sklearn

    versoes = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__}
    try:
        import lightgbm

        versoes["lightgbm"] = lightgbm.__version__
    except ImportError:  # pragma: no cover
        pass
    return versoes


def salvar_pacote(pacote: PacoteModelo, pontos: pd.DataFrame, diretorio: Path = DIR_MODELOS,
                  extras_limiares: dict | None = None) -> dict[str, Path]:
    diretorio.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "modelo": caminho_modelo(pacote.versao, diretorio),
        "limiares": caminho_limiares(pacote.versao, diretorio),
        "metadados": caminho_metadados(pacote.versao, diretorio),
        "pontos": caminho_pontos_modelo(pacote.versao, diretorio),
    }
    joblib.dump({
        "versao": pacote.versao, "nome_modelo": pacote.nome_modelo,
        "estimador": pacote.estimador, "calibrador": pacote.calibrador, "features": pacote.features,
    }, caminhos["modelo"], compress=3)
    salvar_limiares(pacote.limiares, caminhos["limiares"], extras_limiares)
    metadados = {**pacote.metadados, "versao": pacote.versao, "nome_modelo": pacote.nome_modelo,
                 "features": pacote.features, "bibliotecas": versoes_bibliotecas()}
    caminhos["metadados"].write_text(json.dumps(metadados, indent=2, ensure_ascii=False, default=str),
                                     encoding="utf-8")
    pontos.to_csv(caminhos["pontos"], index=False)
    return caminhos


def existe_pacote(versao: str, diretorio: Path = DIR_MODELOS) -> bool:
    return caminho_modelo(versao, diretorio).exists() and caminho_limiares(versao, diretorio).exists()


def carregar_pacote(versao: str, diretorio: Path = DIR_MODELOS) -> PacoteModelo:
    conteudo = joblib.load(caminho_modelo(versao, diretorio))
    arq_meta = caminho_metadados(versao, diretorio)
    metadados = json.loads(arq_meta.read_text(encoding="utf-8")) if arq_meta.exists() else {}
    return PacoteModelo(
        versao=conteudo["versao"], nome_modelo=conteudo["nome_modelo"],
        estimador=conteudo["estimador"], calibrador=conteudo["calibrador"],
        features=list(conteudo["features"]),
        limiares=carregar_limiares(caminho_limiares(versao, diretorio)),
        metadados=metadados,
    )
