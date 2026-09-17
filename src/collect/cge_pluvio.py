"""Boletins pluviométricos do CGE — **melhoria opcional** (fonte D do briefing).

Os boletins ficam num Nextcloud público (``arquivos.saisp.br`` → ``cgesp-publico/
BOLETIM_PLUVIOMETRICO``), um arquivo por ano (~1,2–1,5 MB). O formato ainda não foi
inspecionado; por isso este módulo:

1. **não baixa automaticamente** — baixe os anos pelo navegador e salve em
   ``data/raw/cge_pluvio/`` (ex.: ``2023.xlsx``, ``2024.xlsx``);
2. oferece um leitor genérico (CSV/XLSX) que tenta chegar ao formato longo
   ``estacao, data_hora_local, chuva_mm``; o mapeamento das colunas deve ser ajustado depois
   da inspeção no notebook ``01_inspecao_fontes.ipynb``.

Regra de uso (seção 5.2): a chuva do CGE só entra no modelo como ``chuva_cge_*`` se houver uma
versão AO VIVO equivalente (raspagem das estações), sempre com Open-Meteo como reserva.
Enquanto isso não existir, serve apenas para validar o Open-Meteo e para o texto do TCC.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DIR_INTERIM, DIR_RAW

log = logging.getLogger(__name__)

DIR_PLUVIO = DIR_RAW / "cge_pluvio"
ARQ_PLUVIO = DIR_INTERIM / "chuva_cge_estacoes.parquet"

# Preencher depois da inspeção
COLUNA_ESTACAO = ["estacao", "posto", "nome_estacao"]
COLUNA_DATA = ["data", "data_hora", "dia"]
COLUNA_CHUVA = ["chuva", "precipitacao", "chuva_mm", "mm"]


def ler_arquivo(caminho: Path) -> pd.DataFrame:
    """Lê um boletim (CSV ou XLSX, todas as abas) sem interpretar: útil para inspecionar."""
    if caminho.suffix.lower() in {".xlsx", ".xls"}:
        abas = pd.read_excel(caminho, sheet_name=None)
        partes = [df.assign(_aba=nome) for nome, df in abas.items()]
        return pd.concat(partes, ignore_index=True)
    if caminho.suffix.lower() == ".csv":
        return pd.read_csv(caminho, sep=None, engine="python", encoding_errors="replace")
    raise ValueError(f"Formato não suportado ainda: {caminho.suffix} (inspecionar e adaptar)")


def _achar(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    minusculas = {str(c).strip().lower(): c for c in df.columns}
    return next((minusculas[c] for c in candidatos if c in minusculas), None)


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Tenta levar o boletim ao formato longo. Falha com mensagem clara se não reconhecer."""
    c_est, c_data, c_chuva = _achar(df, COLUNA_ESTACAO), _achar(df, COLUNA_DATA), _achar(df, COLUNA_CHUVA)
    if not all([c_est, c_data, c_chuva]):
        raise ValueError(
            "Formato do boletim não reconhecido. Colunas encontradas: "
            f"{list(df.columns)[:30]}. Ajuste COLUNA_* em src/collect/cge_pluvio.py."
        )
    saida = pd.DataFrame({
        "estacao": df[c_est].astype(str).str.strip(),
        "data_hora_local": pd.to_datetime(df[c_data], errors="coerce", dayfirst=True),
        "chuva_mm": pd.to_numeric(df[c_chuva], errors="coerce"),
    }).dropna(subset=["data_hora_local"])
    return saida


def processar_pasta(pasta: Path = DIR_PLUVIO) -> pd.DataFrame:
    partes = []
    for arq in sorted(pasta.glob("*")):
        if arq.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        log.info("Lendo %s", arq.name)
        partes.append(normalizar(ler_arquivo(arq)))
    if not partes:
        raise FileNotFoundError(f"Nenhum boletim em {pasta}")
    df = pd.concat(partes, ignore_index=True)
    ARQ_PLUVIO.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARQ_PLUVIO, index=False)
    return df


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etapa", choices=["inspecionar", "processar"])
    parser.add_argument("--arquivo", type=Path, help="para 'inspecionar'")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.etapa == "inspecionar":
        df = ler_arquivo(args.arquivo)
        print(df.head(20).to_string())
        print(df.dtypes)
    else:
        processar_pasta()


if __name__ == "__main__":
    _principal()
