"""Sessão HTTP compartilhada pelos coletores.

Recursos:
- **Cache em disco** (``requests-cache``, SQLite em ``data/raw/_cache_http``): repetir uma
  coleta não refaz as requisições — importante para não sobrecarregar sites públicos e para
  sobreviver a quedas no meio de uma raspagem longa.
- **Novas tentativas** com espera exponencial para erros 429/5xx.
- **Pausa mínima** entre requisições reais (não em respostas vindas do cache).
- **User-Agent identificável**, com e-mail de contato (exigido pelo Nominatim).

O serviço ao vivo NÃO usa este módulo (tem cache próprio em memória, ver app/fonte_chuva.py).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import DIR_RAW

log = logging.getLogger(__name__)

ARQ_CACHE = DIR_RAW / "_cache_http"


def user_agent() -> str:
    """User-Agent com contato, lido de ``CONTATO_EMAIL`` (arquivo .env ou ambiente)."""
    contato = os.getenv("CONTATO_EMAIL", "sem-contato")
    return f"SIMA-SP-TCC/0.1 (pesquisa academica; contato: {contato})"


def criar_sessao(usar_cache: bool = True, expira_dias: int | None = None) -> requests.Session:
    """Cria uma sessão com cache (se ``requests-cache`` estiver instalado) e novas tentativas.

    Args:
        usar_cache: guarda as respostas em disco. Desligue para dados "ao vivo".
        expira_dias: validade do cache; ``None`` = nunca expira (dados históricos não mudam).
    """
    sessao: requests.Session
    if usar_cache:
        try:
            import requests_cache

            Path(ARQ_CACHE).parent.mkdir(parents=True, exist_ok=True)
            expira = -1 if expira_dias is None else expira_dias * 86400
            sessao = requests_cache.CachedSession(
                str(ARQ_CACHE), backend="sqlite", expire_after=expira,
                allowable_methods=("GET", "POST"),
            )
        except ImportError:  # pragma: no cover - só sem requests-cache
            log.warning("requests-cache não instalado; seguindo sem cache em disco.")
            sessao = requests.Session()
    else:
        sessao = requests.Session()

    tentativas = Retry(
        total=5, backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adaptador = HTTPAdapter(max_retries=tentativas)
    sessao.mount("http://", adaptador)
    sessao.mount("https://", adaptador)
    sessao.headers.update({"User-Agent": user_agent()})
    return sessao


class ControleTaxa:
    """Garante um intervalo mínimo entre requisições REAIS (respostas do cache não esperam).

    Uso::

        taxa = ControleTaxa(1.0)
        resp = sessao.get(url)
        taxa.registrar(resp)   # dorme só se a resposta veio da rede
    """

    def __init__(self, intervalo_s: float):
        self.intervalo_s = intervalo_s
        self._ultima = 0.0

    def aguardar(self) -> None:
        """Dorme o necessário antes da próxima requisição real."""
        falta = self.intervalo_s - (time.monotonic() - self._ultima)
        if falta > 0:
            time.sleep(falta)

    def registrar(self, resposta: requests.Response | None = None) -> None:
        """Marca o instante da última requisição real."""
        if resposta is not None and getattr(resposta, "from_cache", False):
            return
        self._ultima = time.monotonic()


def carregar_env(caminho: str | Path = ".env") -> None:
    """Carrega variáveis de um arquivo .env simples (CHAVE=valor), sem sobrescrever o ambiente."""
    caminho = Path(caminho)
    if not caminho.exists():
        return
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip())
