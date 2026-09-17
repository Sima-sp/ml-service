"""Montagem da tabela **ponto × hora** (seção 6.2) — ``data/processed/dataset_<versao>.parquet``.

Entradas (geradas pelas etapas anteriores):
- ``data/processed/pontos.csv``           pontos escolhidos + características fixas
- ``data/interim/chuva_horaria.parquet``  chuva por célula × hora (Open-Meteo)
- ``data/interim/eventos.parquet``        alagamentos unificados com coordenadas

Passo a passo para cada ponto:
1. features de chuva da célula do ponto (``src.features``, calculadas uma vez por célula);
2. features de calendário;
3. rótulo ``alagou_3h`` e exclusão das horas já alagadas (``src.labels``);
4. características fixas do ponto + ``freq_hist_alagamento`` do ano (sem vazamento);
5. remoção de horas sem chuva completa (buracos na série);
6. divisão temporal (``split``) pelo ano LOCAL;
7. subamostragem das horas secas negativas **só no treino**, com ``peso_amostra = 1/taxa``
   (validação e teste ficam com a prevalência real — métricas e calibração honestas).

Saídas:
- ``dataset_<versao>.parquet``
- ``dataset_<versao>_resumo.json`` (contagens usadas no ``DATASET.md``)

Linha de comando::

    python -m src.dataset --versao v1
    python -m src.dataset --versao sintetico --sintetico     # só para testar o pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANO_FIM_TREINO,
    ANOS_TESTE,
    ANOS_VALIDACAO,
    ARQ_CHUVA_HORARIA,
    ARQ_EVENTOS,
    ARQ_PONTOS,
    DIR_PROCESSED,
    FUSO_LOCAL,
    JANELA_PREVISAO_HORAS,
    LIMIAR_HORA_SECA_24H_MM,
    PERIODO_FIM,
    PERIODO_INICIO,
    RAIO_ROTULO_M,
    SEMENTE,
    SUBAMOSTRAR_APENAS_TREINO,
    TAXA_AMOSTRAGEM_SECAS,
    ZONA_FOCO,
)
from src.features import COLUNAS_CHUVA, calcular_features_chuva, calcular_features_tempo
from src.labels import eventos_por_ponto, rotular_ponto, taxa_historica_anual
from src.tempo import utc_para_local

log = logging.getLogger(__name__)

COLUNAS_FIXAS_PONTO = ["lat", "lon", "elevacao", "dist_corrego_m", "zona"]


def caminho_dataset(versao: str) -> Path:
    return DIR_PROCESSED / f"dataset_{versao}.parquet"


def caminho_resumo(versao: str) -> Path:
    return DIR_PROCESSED / f"dataset_{versao}_resumo.json"


def caminho_freq_hist(versao: str) -> Path:
    return DIR_PROCESSED / f"dataset_{versao}_freq_hist.parquet"


def caminho_pontos_dataset(versao: str) -> Path:
    """Cópia dos pontos usados nesta versão (o serviço precisa deles junto do modelo)."""
    return DIR_PROCESSED / f"dataset_{versao}_pontos.csv"


def definir_split(anos: pd.Series) -> pd.Series:
    """treino (<= ANO_FIM_TREINO), validacao, teste ou fora (descartado)."""
    split = np.select(
        [anos <= ANO_FIM_TREINO, anos.isin(ANOS_VALIDACAO), anos.isin(ANOS_TESTE)],
        ["treino", "validacao", "teste"], default="fora",
    )
    return pd.Series(split, index=anos.index, dtype="object")


def subamostrar(df: pd.DataFrame, taxa: float, rng: np.random.Generator,
                apenas_treino: bool = SUBAMOSTRAR_APENAS_TREINO) -> pd.DataFrame:
    """Mantém todos os positivos e todas as horas com chuva; das horas secas negativas, só ``taxa``.

    ``peso_amostra`` compensa a remoção (1/taxa nas secas mantidas), então somas ponderadas
    reproduzem a distribuição original.
    """
    seca_negativa = (df["chuva_24h"] < LIMIAR_HORA_SECA_24H_MM) & (df["alagou_3h"] == 0)
    elegivel = seca_negativa & ((df["split"] == "treino") if apenas_treino else True)
    sorteio = rng.random(len(df)) < taxa
    manter = ~elegivel | sorteio
    saida = df.loc[manter].copy()
    saida["peso_amostra"] = np.where(elegivel[manter], 1.0 / taxa, 1.0).astype("float32")
    return saida


def montar_dataset(
    pontos: pd.DataFrame,
    chuva: pd.DataFrame,
    eventos: pd.DataFrame,
    inicio: str = PERIODO_INICIO,
    fim: str = PERIODO_FIM,
    raio_m: float = RAIO_ROTULO_M,
    taxa_secas: float = TAXA_AMOSTRAGEM_SECAS,
    semente: int = SEMENTE,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Gera o dataset, um dicionário de contagens (para o DATASET.md) e a tabela
    ``freq_hist_alagamento`` por ponto × ano (inclui o ano seguinte ao fim = valor de produção)."""
    horas = pd.date_range(
        pd.Timestamp(inicio).tz_localize(FUSO_LOCAL).tz_convert("UTC"),
        (pd.Timestamp(fim) + pd.Timedelta(hours=23)).tz_localize(FUSO_LOCAL).tz_convert("UTC"),
        freq="h",
    )
    if "hora_conhecida" in eventos:
        sem_hora = ~eventos["hora_conhecida"].fillna(True).astype(bool)
        if sem_hora.any():
            log.info("%d eventos sem hora (só data) ficam fora do rótulo e da frequência", int(sem_hora.sum()))
        eventos = eventos.loc[~sem_hora]
    anos_local = pd.Series(utc_para_local(horas).year, index=horas)
    tempo = calcular_features_tempo(horas)

    ev_ponto = eventos_por_ponto(pontos, eventos, raio_m)
    ano_inicio_dados = int(utc_para_local(eventos["inicio_utc"]).dt.year.min())
    anos = sorted(set(anos_local.unique()) | {int(anos_local.max()) + 1})
    freq = taxa_historica_anual(ev_ponto, anos, ano_inicio_dados)

    contagem = {"linhas_brutas": 0, "horas_ja_alagadas_excluidas": 0,
                "horas_sem_chuva_completa_excluidas": 0, "linhas_fora_do_periodo": 0}
    partes = []
    chuva = chuva.copy()
    chuva["data_hora_utc"] = pd.to_datetime(chuva["data_hora_utc"], utc=True)
    series_por_celula = {c: g.set_index("data_hora_utc")["chuva_mm"] for c, g in chuva.groupby("celula_id")}

    for n_celula, (celula_id, pontos_celula) in enumerate(pontos.groupby("celula_id"), start=1):
        if celula_id not in series_por_celula:
            log.warning("Célula %s sem chuva baixada — %d pontos ignorados", celula_id, len(pontos_celula))
            continue
        feats_chuva = calcular_features_chuva(series_por_celula[celula_id]).reindex(horas)
        base = pd.concat([feats_chuva, tempo], axis=1)
        for ponto in pontos_celula.itertuples(index=False):
            rotulos = rotular_ponto(horas, ev_ponto.get(ponto.ponto_id))
            df = pd.concat([base, rotulos], axis=1)
            df["ano"] = anos_local.to_numpy()
            contagem["linhas_brutas"] += len(df)

            ja = df["ja_alagado"].to_numpy()
            contagem["horas_ja_alagadas_excluidas"] += int(ja.sum())
            df = df.loc[~ja]
            sem_chuva = df[COLUNAS_CHUVA].isna().any(axis=1).to_numpy()
            contagem["horas_sem_chuva_completa_excluidas"] += int(sem_chuva.sum())
            df = df.loc[~sem_chuva]

            df["split"] = definir_split(df["ano"])
            fora = (df["split"] == "fora").to_numpy()
            contagem["linhas_fora_do_periodo"] += int(fora.sum())
            df = df.loc[~fora]

            for coluna in COLUNAS_FIXAS_PONTO:
                df[coluna] = getattr(ponto, coluna)
            df["ponto_id"] = ponto.ponto_id
            # semente estável por ponto (hash() do Python muda a cada execução; crc32 não)
            rng = np.random.default_rng([semente, zlib.crc32(str(ponto.ponto_id).encode())])
            partes.append(subamostrar(df.drop(columns="ja_alagado"), taxa_secas, rng))
        log.info("Célula %d/%d (%s): %d pontos", n_celula, pontos["celula_id"].nunique(),
                 celula_id, len(pontos_celula))

    if not partes:
        raise RuntimeError("Nenhuma linha gerada — confira pontos, células e chuva")
    dataset = pd.concat(partes).rename_axis("data_hora_utc").reset_index()
    dataset = dataset.merge(freq, on=["ponto_id", "ano"], how="left")

    dataset["ponto_id"] = dataset["ponto_id"].astype("category")
    dataset["split"] = pd.Categorical(dataset["split"], categories=["treino", "validacao", "teste"])
    dataset["zona"] = dataset["zona"].astype("object")
    for coluna in ["lat", "lon", "elevacao", "dist_corrego_m", "freq_hist_alagamento"]:
        dataset[coluna] = pd.to_numeric(dataset[coluna], errors="coerce").astype("float32")
    for coluna in COLUNAS_CHUVA + ["chuva_prevista_3h"]:
        dataset[coluna] = dataset[coluna].astype("float32")
    dataset["ano"] = dataset["ano"].astype("int16")

    ordem = ["ponto_id", "data_hora_utc", "ano", "split", "zona", *COLUNAS_CHUVA, "chuva_prevista_3h",
             "hora_dia", "mes", "dia_semana", "estacao_chuvosa",
             "lat", "lon", "elevacao", "dist_corrego_m", "freq_hist_alagamento",
             "alagou_3h", "evento_alvo_id", "peso_amostra"]
    dataset = dataset[ordem].sort_values(["data_hora_utc", "ponto_id"]).reset_index(drop=True)

    resumo = resumir(dataset, pontos, eventos, contagem, inicio, fim, raio_m, taxa_secas)
    resumo["ano_inicio_eventos"] = ano_inicio_dados
    return dataset, resumo, freq


def resumir(dataset: pd.DataFrame, pontos: pd.DataFrame, eventos: pd.DataFrame, contagem: dict,
            inicio: str, fim: str, raio_m: float, taxa_secas: float) -> dict:
    """Contagens para documentar o dataset (DATASET.md) e conferir o balanceamento."""
    por_split = {}
    for split, g in dataset.groupby("split", observed=True):
        norte = g[g["zona"] == ZONA_FOCO]
        por_split[str(split)] = {
            "linhas": int(len(g)),
            "positivos": int(g["alagou_3h"].sum()),
            "prevalencia_amostra": round(float(g["alagou_3h"].mean()), 6),
            "prevalencia_real_ponderada": round(float(np.average(g["alagou_3h"], weights=g["peso_amostra"])), 6),
            "eventos_distintos": int(g["evento_alvo_id"].nunique()),
            "anos": sorted(int(a) for a in g["ano"].unique()),
            "linhas_zona_norte": int(len(norte)),
            "positivos_zona_norte": int(norte["alagou_3h"].sum()),
        }
    return {
        "periodo": {"inicio": inicio, "fim": fim},
        "janela_horas": JANELA_PREVISAO_HORAS,
        "raio_rotulo_m": raio_m,
        "taxa_amostragem_secas": taxa_secas,
        "subamostragem_apenas_treino": SUBAMOSTRAR_APENAS_TREINO,
        "limiar_hora_seca_24h_mm": LIMIAR_HORA_SECA_24H_MM,
        "pontos": int(len(pontos)),
        "pontos_zona_norte": int((pontos["zona"] == ZONA_FOCO).sum()),
        "celulas_chuva": int(pontos["celula_id"].nunique()),
        "eventos_total": int(len(eventos)),
        "eventos_por_fonte": ({str(k): int(v) for k, v in eventos["fonte"].value_counts().items()}
                              if "fonte" in eventos else {}),
        "exclusoes": contagem,
        "por_split": por_split,
        "linhas_total": int(len(dataset)),
        "gerado_em": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def salvar(dataset: pd.DataFrame, resumo: dict, freq: pd.DataFrame, pontos: pd.DataFrame,
           versao: str, fonte_chuva: str) -> None:
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    resumo = {**resumo, "versao": versao, "fonte_chuva": fonte_chuva}
    dataset.to_parquet(caminho_dataset(versao), index=False)
    freq.to_parquet(caminho_freq_hist(versao), index=False)
    pontos.to_csv(caminho_pontos_dataset(versao), index=False)
    caminho_resumo(versao).write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Dataset salvo: %s (%d linhas)", caminho_dataset(versao), len(dataset))


def carregar(versao: str) -> tuple[pd.DataFrame, dict]:
    dataset = pd.read_parquet(caminho_dataset(versao))
    resumo = json.loads(caminho_resumo(versao).read_text(encoding="utf-8"))
    return dataset, resumo


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--versao", default="v1")
    parser.add_argument("--fonte-chuva", default="arquivo", choices=["arquivo", "previsoes_historicas", "sintetica"],
                        help="de onde veio a chuva (registrado no resumo; protege contra vazamento)")
    parser.add_argument("--sintetico", action="store_true",
                        help="gera entradas sintéticas (SÓ para testar o pipeline)")
    parser.add_argument("--inicio", default=PERIODO_INICIO)
    parser.add_argument("--fim", default=PERIODO_FIM)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.sintetico:
        from src.sintetico import gerar_entradas

        pontos, chuva, eventos = gerar_entradas(inicio=args.inicio, fim=args.fim)
        fonte = "sintetica"
    else:
        pontos = pd.read_csv(ARQ_PONTOS)
        chuva = pd.read_parquet(ARQ_CHUVA_HORARIA)
        eventos = pd.read_parquet(ARQ_EVENTOS)
        fonte = args.fonte_chuva
    dataset, resumo, freq = montar_dataset(pontos, chuva, eventos, args.inicio, args.fim)
    salvar(dataset, resumo, freq, pontos, args.versao, fonte)
    print(json.dumps(resumo["por_split"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _principal()
