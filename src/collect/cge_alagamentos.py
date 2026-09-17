"""Histórico de alagamentos do CGE (Centro de Gerenciamento de Emergências Climáticas).

Fonte A do briefing — **rótulo principal**, porque traz a hora de INÍCIO de cada alagamento.

Página: https://www.cgesp.org/v3/alagamentos.jsp (campo de data + botão Buscar).

Formato conferido em 16/09/2026 (busca de 14/09/2026):
    ``GET alagamentos.jsp?dataBusca=DD/MM/AAAA&enviaBusca=Buscar``. A página agrupa por
    ``h1.tit-bairros`` (zona do CGE, ex.: "Zona Norte") → ``td.bairro`` (subprefeitura, ex.:
    "Pirituba/Jaraguá") → ``div.ponto-de-alagamento`` (um alagamento). Dentro de cada ponto:
    ``li[class=ativo|inativo-transitavel|intransitavel]`` (situação), ``li.col-local``
    ("De 14:31 a 15:02" + logradouro; sem horário final = ainda ativo) e outro
    ``li.arial-descr-alag`` ("Sentido: ..." + "Referência: ..."). A referência pode ser uma
    via ("AV OTAVIANO ALVES DE LIMA") ou um número ("461").

Fluxo em duas etapas:

1. ``baixar``  -> um HTML por dia em ``data/raw/cge_alagamentos/html/AAAA/AAAA-MM-DD.html``
   (dias já baixados são pulados; pausa entre requisições; nada é refeito se cair no meio).
2. ``extrair`` -> lê os HTMLs e grava ``data/interim/eventos_cge.parquet``.

Linha de comando::

    python -m src.collect.cge_alagamentos baixar  --inicio 2024-01-01 --fim 2024-01-31
    python -m src.collect.cge_alagamentos extrair --inicio 2024-01-01 --fim 2024-01-31

Cuidados registrados no briefing:
- Os dados vêm de agentes de trânsito (CET) e cobrem sobretudo vias principais e corredores.
- Há endereço (logradouro + referência), mas não coordenada: geocodificar em ``src/geo.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config import ARQ_EVENTOS_CGE, DIR_RAW
from src.tempo import local_para_utc

log = logging.getLogger(__name__)

URL_CGE_ALAGAMENTOS = "https://www.cgesp.org/v3/alagamentos.jsp"
DIR_HTML = DIR_RAW / "cge_alagamentos" / "html"

# --- CONFERIR NA ABA REDE DO NAVEGADOR -------------------------------------------------
METODO_HTTP = "GET"


def montar_parametros(data: pd.Timestamp) -> dict:
    """Parâmetros da busca por data. CONFERIR nome do campo e formato (dd/mm/aaaa)."""
    return {"dataBusca": data.strftime("%d/%m/%Y"), "enviaBusca": "Buscar"}


# Seletores da página (conferidos em 16/09/2026)
SELETOR_PONTO = "div.ponto-de-alagamento"  # um alagamento
SELETOR_TOTAL = "td.total-pts"  # "2 pts." — usado para conferir a extração
# Subprefeitura: td.bairro · Zona: h1.tit-bairros (buscados com find_previous)
# ----------------------------------------------------------------------------------------

RE_HORARIO = re.compile(
    r"De\s+(\d{1,2})[:h](\d{2})\s*(?:a|às|as|até|ate)?\s*(?:(\d{1,2})[:h](\d{2}))?",
    re.IGNORECASE,
)
RE_SENTIDO = re.compile(r"Sentido\s*:\s*(.+)", re.IGNORECASE)
RE_REFERENCIA = re.compile(r"Refer[êe]ncia\s*:\s*(.+)", re.IGNORECASE)


@dataclass
class RegistroCGE:
    """Um alagamento como aparece na página do CGE (ainda sem coordenadas)."""

    data: str  # AAAA-MM-DD (dia pesquisado)
    inicio_local: str | None  # AAAA-MM-DD HH:MM
    fim_local: str | None
    regiao_cge: str | None  # subprefeitura como o CGE escreve (ex.: "Pirituba/Jaraguá")
    zona_cge: str | None  # zona como o CGE agrupa (ex.: "Zona Norte")
    logradouro: str | None
    sentido: str | None
    referencia: str | None
    transitavel: bool | None
    ativo: bool | None  # ainda alagado no momento em que a página foi gerada
    texto_bruto: str


# ---------------------------------------------------------------------------
# Etapa 1: download
# ---------------------------------------------------------------------------
def caminho_html(data: pd.Timestamp) -> Path:
    return DIR_HTML / f"{data.year}" / f"{data:%Y-%m-%d}.html"


def baixar_periodo(inicio: str, fim: str, pausa_s: float | None = None, refazer: bool = False) -> int:
    """Baixa o HTML de cada dia do período. Retorna quantos dias foram baixados agora."""
    from src.collect.http import ControleTaxa, carregar_env, criar_sessao

    carregar_env()
    pausa = float(os.getenv("PAUSA_CGE_S", "1.5")) if pausa_s is None else pausa_s
    sessao = criar_sessao(usar_cache=False)
    taxa = ControleTaxa(pausa)
    baixados = 0
    for data in pd.date_range(inicio, fim, freq="D"):
        destino = caminho_html(data)
        if destino.exists() and not refazer:
            continue
        taxa.aguardar()
        params = montar_parametros(data)
        if METODO_HTTP == "GET":
            resp = sessao.get(URL_CGE_ALAGAMENTOS, params=params, timeout=30)
        else:
            resp = sessao.post(URL_CGE_ALAGAMENTOS, data=params, timeout=30)
        taxa.registrar()
        if resp.status_code != 200:
            log.warning("%s: HTTP %s — dia pulado", data.date(), resp.status_code)
            continue
        if "alag" not in resp.text.lower():
            log.warning("%s: resposta sem a palavra 'alag' — conferir parâmetros da busca", data.date())
        destino.parent.mkdir(parents=True, exist_ok=True)
        # O site costuma usar ISO-8859-1; guardamos em UTF-8 para facilitar a leitura depois
        resp.encoding = resp.apparent_encoding or resp.encoding
        destino.write_text(resp.text, encoding="utf-8")
        baixados += 1
        if baixados % 30 == 0:
            log.info("... %d dias baixados (último: %s)", baixados, data.date())
    return baixados


# ---------------------------------------------------------------------------
# Etapa 2: extração
# ---------------------------------------------------------------------------
def _hora(data: pd.Timestamp, h: str | None, m: str | None) -> pd.Timestamp | None:
    if h is None or m is None:
        return None
    return data.normalize() + pd.Timedelta(hours=int(h), minutes=int(m))


def _interpretar_texto(texto: str, data: pd.Timestamp) -> dict:
    """Extrai horários, logradouro, sentido e referência de um bloco de texto."""
    linhas = [ln.strip() for ln in texto.splitlines() if ln.strip()]
    inicio = fim = None
    logradouro = sentido = referencia = None
    for i, linha in enumerate(linhas):
        m = RE_HORARIO.search(linha)
        if m and inicio is None:
            inicio = _hora(data, m.group(1), m.group(2))
            fim = _hora(data, m.group(3), m.group(4))
            if inicio is not None and fim is not None and fim < inicio:
                fim += pd.Timedelta(days=1)  # atravessou a meia-noite
            resto = [ln for ln in linhas[i + 1 :] if not RE_SENTIDO.match(ln) and not RE_REFERENCIA.match(ln)]
            logradouro = resto[0] if resto else None
            continue
        if (ms := RE_SENTIDO.match(linha)) is not None:
            sentido = ms.group(1).strip()
        if (mr := RE_REFERENCIA.match(linha)) is not None:
            referencia = mr.group(1).strip()
    fmt = "%Y-%m-%d %H:%M"
    return {
        "inicio_local": inicio.strftime(fmt) if inicio is not None else None,
        "fim_local": fim.strftime(fmt) if fim is not None else None,
        "logradouro": logradouro, "sentido": sentido, "referencia": referencia,
    }


def _situacao(ponto) -> tuple[bool | None, bool | None]:
    """(transitavel, ativo) a partir da classe do ``li`` de situação, ex.: ``inativo-intransitavel``."""
    for li in ponto.find_all("li"):
        for classe in li.get("class") or []:
            classe = classe.lower()
            if classe.endswith("transitavel"):
                return ("intransitavel" not in classe), (not classe.startswith("inativo"))
    titulo = " ".join((li.get("title") or "") for li in ponto.find_all("li")).lower()
    if "transit" in titulo:
        return ("intransit" not in titulo), ("inativo" not in titulo)
    return None, None


def extrair_dia(html: str, data: pd.Timestamp) -> list[RegistroCGE]:
    """Converte o HTML de um dia em registros (um por ``div.ponto-de-alagamento``).

    Se a página mudar e nenhum ponto for encontrado, cai numa estratégia de reserva que
    procura "De hh:mm a hh:mm" no texto inteiro (sem subprefeitura/situação).
    """
    from bs4 import BeautifulSoup

    sopa = BeautifulSoup(html, "html.parser")
    registros: list[RegistroCGE] = []
    dia = f"{data:%Y-%m-%d}"

    for ponto in sopa.select(SELETOR_PONTO):
        texto = ponto.get_text("\n", strip=True)
        sub = ponto.find_previous("td", class_="bairro")
        zona = ponto.find_previous("h1", class_="tit-bairros")
        transitavel, ativo = _situacao(ponto)
        registros.append(RegistroCGE(
            data=dia,
            regiao_cge=sub.get_text(" ", strip=True) if sub else None,
            zona_cge=zona.get_text(" ", strip=True) if zona else None,
            transitavel=transitavel, ativo=ativo, texto_bruto=texto,
            **_interpretar_texto(texto, data),
        ))

    esperado = sum(int(m.group(1)) for td in sopa.select(SELETOR_TOTAL)
                   if (m := re.search(r"(\d+)", td.get_text())))
    if esperado and esperado != len(registros):
        log.warning("%s: a página anuncia %d pontos, mas %d foram extraídos", dia, esperado, len(registros))

    if not registros and RE_HORARIO.search(sopa.get_text(" ")):
        log.warning("%s: estrutura não reconhecida; usando extração por texto (conferir seletores)", dia)
        linhas = sopa.get_text("\n", strip=True).splitlines()
        for i, linha in enumerate(linhas):
            if RE_HORARIO.search(linha):
                trecho = "\n".join(linhas[i : i + 4])
                registros.append(RegistroCGE(
                    data=dia, regiao_cge=None, zona_cge=None, transitavel=None, ativo=None,
                    texto_bruto=trecho, **_interpretar_texto(trecho, data),
                ))
    return registros


def _id_evento(reg: RegistroCGE) -> str:
    chave = f"CGE|{reg.inicio_local}|{reg.logradouro}|{reg.sentido}|{reg.referencia}"
    return "cge_" + hashlib.sha1(chave.encode("utf-8")).hexdigest()[:12]


def extrair_periodo(inicio: str, fim: str, destino: Path = ARQ_EVENTOS_CGE) -> pd.DataFrame:
    """Lê os HTMLs do período e grava a tabela de eventos do CGE (sem coordenadas)."""
    registros = []
    dias_sem_arquivo = 0
    for data in pd.date_range(inicio, fim, freq="D"):
        arq = caminho_html(data)
        if not arq.exists():
            dias_sem_arquivo += 1
            continue
        for reg in extrair_dia(arq.read_text(encoding="utf-8"), data):
            registros.append({"evento_id": _id_evento(reg), "fonte": "CGE", **asdict(reg)})
    if dias_sem_arquivo:
        log.warning("%d dias do período ainda não foram baixados", dias_sem_arquivo)

    colunas = ["evento_id", "fonte", *RegistroCGE.__dataclass_fields__]
    df = pd.DataFrame(registros, columns=colunas).drop_duplicates("evento_id")
    df["inicio_utc"] = local_para_utc(df["inicio_local"]) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
    df["fim_utc"] = local_para_utc(df["fim_local"]) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    log.info("%d eventos do CGE salvos em %s (%d sem horário de início)",
             len(df), destino, int(df["inicio_local"].isna().sum()))
    return df


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("etapa", choices=["baixar", "extrair"])
    parser.add_argument("--inicio", required=True)
    parser.add_argument("--fim", required=True)
    parser.add_argument("--pausa", type=float, default=None, help="segundos entre requisições")
    parser.add_argument("--refazer", action="store_true", help="baixa de novo dias já salvos")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.etapa == "baixar":
        n = baixar_periodo(args.inicio, args.fim, args.pausa, args.refazer)
        log.info("%d dias baixados", n)
    else:
        extrair_periodo(args.inicio, args.fim)


if __name__ == "__main__":
    _principal()
