"""Aplicação FastAPI do serviço de IA do SIMA-SP (porta 5000).

Rotas:
- ``GET  /health``         vivo? qual modelo? (usado pelo Docker e pelo backend)
- ``GET  /model-info``     features, período de treino, métricas, limiares
- ``POST /predict``        previsão para um sensor/ponto
- ``POST /predict/batch``  lista de sensores (usado pelo agendador do backend a cada 10 min)
- ``GET  /predict/regioes`` risco por subprefeitura (aviso de região do app)
- ``POST /predict/regioes`` o mesmo, com chuva informada (simulação para a demonstração)
- ``GET  /docs``           Swagger gerado automaticamente

Subir localmente::

    uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

O modelo é carregado UMA vez, na inicialização (``lifespan``). Sem modelo em ``models/``, o
serviço sobe em **modo regras** (baseline), o que permite integrar o backend antes do treino.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.configuracao import obter_configuracoes
from app.predictor import ErroPrevisao, Preditor
from app.schemas import (
    ErroResposta,
    ItemLote,
    RequisicaoPrevisao,
    RequisicaoRegioes,
    RespostaPrevisao,
    RespostaRegioes,
    RespostaSaude,
)

log = logging.getLogger("sima.ia")


def criar_app(preditor: Preditor | None = None) -> FastAPI:
    """Fábrica da aplicação (os testes injetam um ``Preditor`` com chuva falsa)."""
    config = obter_configuracoes()
    logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        app.state.preditor = preditor or Preditor(config)
        log.info("Serviço pronto — modelo %s (%s)", app.state.preditor.modelo_versao,
                 app.state.preditor.origem.value)
        yield

    app = FastAPI(
        title="SIMA-SP — Serviço de previsão de alagamentos",
        description="Probabilidade de alagamento nas próximas 3 h por ponto monitorado.",
        version="0.1.0",
        lifespan=ciclo_de_vida,
    )

    @app.exception_handler(ErroPrevisao)
    async def _erro_previsao(_: Request, erro: ErroPrevisao):
        return JSONResponse(status_code=erro.status_http,
                            content=ErroResposta(codigo=erro.codigo, mensagem=erro.mensagem).model_dump())

    @app.middleware("http")
    async def _medir_tempo(request: Request, chamar):
        inicio = time.perf_counter()
        resposta = await chamar(request)
        resposta.headers["X-Tempo-Ms"] = f"{(time.perf_counter() - inicio) * 1000:.1f}"
        return resposta

    @app.get("/health", response_model=RespostaSaude, tags=["operação"])
    def saude(request: Request) -> RespostaSaude:
        p: Preditor = request.app.state.preditor
        return RespostaSaude(status="ok", modeloVersao=p.modelo_versao, origem=p.origem,
                             pontosCarregados=len(p.resolvedor),
                             celulasChuvaEmCache=p.servico_chuva.celulas_em_cache)

    @app.get("/model-info", tags=["operação"])
    def info_modelo(request: Request) -> dict:
        return request.app.state.preditor.info_modelo()

    @app.post("/predict", response_model=RespostaPrevisao, tags=["previsão"],
              responses={422: {"model": ErroResposta}, 503: {"model": ErroResposta}})
    def prever(req: RequisicaoPrevisao, request: Request) -> RespostaPrevisao:
        return request.app.state.preditor.prever(req)

    @app.get("/predict/regioes", response_model=RespostaRegioes, tags=["previsão"],
             responses={503: {"model": ErroResposta}})
    def prever_regioes(request: Request) -> RespostaRegioes:
        """Risco por subprefeitura, do mais crítico para o menos.

        Regiões com menos de ``minPontosRegiao`` pontos monitorados vêm com
        ``cobertura = SEM_COBERTURA`` e sem nível: não dá para falar pela região inteira com um
        ponto só. "Sem cobertura" não é "sem risco", e a tela precisa deixar isso claro.
        """
        return request.app.state.preditor.prever_regioes()

    @app.post("/predict/regioes", response_model=RespostaRegioes, tags=["previsão"],
              responses={503: {"model": ErroResposta}})
    def prever_regioes_simulado(req: RequisicaoRegioes, request: Request) -> RespostaRegioes:
        """Risco por subprefeitura com chuva **informada** — para simulação e demonstração.

        A série de ``chuvaHoraria`` vale para todos os pontos, no lugar da chuva real. A resposta
        vem com ``fonteChuva = "INFORMADA"``; quem exibe deve avisar que é simulação. Sem
        ``chuvaHoraria``, equivale ao ``GET``.
        """
        return request.app.state.preditor.prever_regioes(chuva_horaria=req.chuvaHoraria)

    @app.post("/predict/batch", response_model=list[ItemLote], tags=["previsão"],
              responses={413: {"model": ErroResposta}})
    def prever_lote(itens: list[RequisicaoPrevisao], request: Request):
        if len(itens) > config.max_itens_lote:
            raise ErroPrevisao("LOTE_GRANDE_DEMAIS", f"Máximo de {config.max_itens_lote} itens por lote", 413)
        return request.app.state.preditor.prever_lote(itens)

    return app


app = criar_app()
