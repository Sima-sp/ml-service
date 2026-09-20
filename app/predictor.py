"""Orquestração de uma previsão (o "caso de uso" do serviço).

Fluxo de ``Preditor.prever``::

    request ──► hora de referência (hora cheia anterior, UTC)
            ──► chuva: chuvaHoraria (informada) | Open-Meteo/cache | resumo informado | chuvaMm (legado)
            ──► características do ponto mais próximo
            ──► src.features.features_ao_vivo  (MESMO código do treino)
            ──► modelo + calibração ─► probabilidade ─► nível (limiares do modelo)
                (sem modelo: baseline de regras ─► nível, probabilidade = null)
            ──► src.sensor_adjust (água/lixo) ─► nível final
            ──► resposta com geradaEm/validaAte, fonte da chuva e avisos

Falhas viram ``ErroPrevisao`` com um código estável, que o backend usa para cair nas regras.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from app.caracteristicas_ponto import ResolvedorPontos
from app.configuracao import Configuracoes
from app.fonte_chuva import ChuvaObtida, ServicoChuva
from app.schemas import (
    FonteChuva,
    ItemLote,
    OrigemPrevisao,
    RegiaoRisco,
    RequisicaoPrevisao,
    RespostaPrevisao,
    RespostaRegioes,
)
from src.artefatos import PacoteModelo, carregar_pacote, existe_pacote
from src.baseline import BaselineRegras
from src.config import JANELA_PREVISAO_HORAS, caminho_pontos_modelo
from src.regioes import MIN_PONTOS_REGIAO, agregar_por_regiao, anotar_regiao
from src.features import features_ao_vivo
from src.niveis import nivel_por_probabilidade, valor_contrato, validar_limiares
from src.sensor_adjust import ajustar_por_sensor
from src.tempo import agora_utc, garantir_utc, hora_referencia, para_iso_local

log = logging.getLogger(__name__)


class ErroPrevisao(Exception):
    def __init__(self, codigo: str, mensagem: str, status_http: int = 422):
        super().__init__(mensagem)
        self.codigo = codigo
        self.mensagem = mensagem
        self.status_http = status_http


def _float_ou_none(valor) -> float | None:
    return None if valor is None or pd.isna(valor) else round(float(valor), 2)


class Preditor:
    def __init__(self, config: Configuracoes, servico_chuva: ServicoChuva | None = None,
                 pacote: PacoteModelo | None = None, resolvedor: ResolvedorPontos | None = None,
                 carregar_do_disco: bool = True):
        self.config = config
        self.servico_chuva = servico_chuva or ServicoChuva(
            ttl_min=config.cache_chuva_ttl_min, idade_max_min=config.idade_max_chuva_min,
            timeout_s=config.timeout_open_meteo_s)
        self.pacote = pacote
        if self.pacote is None and carregar_do_disco and existe_pacote(config.versao_modelo, config.dir_modelos):
            self.pacote = carregar_pacote(config.versao_modelo, config.dir_modelos)
            log.info("Modelo %s (%s) carregado", self.pacote.versao, self.pacote.nome_modelo)
        if self.pacote is None:
            log.warning("Nenhum modelo em %s para a versão %s — servindo em MODO REGRAS",
                        config.dir_modelos, config.versao_modelo)
        self.resolvedor = resolvedor or ResolvedorPontos(
            caminho_pontos_modelo(config.versao_modelo, config.dir_modelos), config.raio_ponto_conhecido_m)
        self.baseline = BaselineRegras()
        self.pontos_regiao = anotar_regiao(self.resolvedor.pontos) if len(self.resolvedor) else pd.DataFrame()
        self.limiares_regiao = self._carregar_limiares_regiao()

    def _carregar_limiares_regiao(self) -> dict | None:
        """Limiares do alerta por região (``models/thresholds_regiao_<versao>.json``).

        São diferentes dos limiares por ponto: o alerta de região usa o MAIOR risco entre os
        pontos da região, então a escala muda. Sem o arquivo, o alerta por região fica
        desligado — é melhor não alertar do que alertar com limiar de outra escala.
        """
        caminho = self.config.dir_modelos / f"thresholds_regiao_{self.config.versao_modelo}.json"
        if not caminho.exists():
            log.warning("Sem %s — alerta por região desligado", caminho.name)
            return None
        try:
            return validar_limiares(json.loads(caminho.read_text(encoding="utf-8"))["limiares"])
        except Exception as erro:  # noqa: BLE001
            log.warning("Limiares de região inválidos em %s: %s", caminho.name, erro)
            return None

    # ------------------------------------------------------------------ info
    @property
    def origem(self) -> OrigemPrevisao:
        return OrigemPrevisao.MODELO if self.pacote else OrigemPrevisao.REGRAS

    @property
    def modelo_versao(self) -> str:
        return self.pacote.versao if self.pacote else "regras-v0"

    def info_modelo(self) -> dict:
        if not self.pacote:
            return {"modeloVersao": self.modelo_versao, "origem": self.origem.value,
                    "descricao": "Sem modelo treinado: baseline de regras de chuva acumulada.",
                    "limiaresRegras": self.baseline.limiares}
        meta = self.pacote.metadados
        return {
            "modeloVersao": self.pacote.versao, "origem": self.origem.value,
            "nomeModelo": self.pacote.nome_modelo, "features": self.pacote.features,
            "limiares": self.pacote.limiares, "janelaHoras": JANELA_PREVISAO_HORAS,
            "treinadoEm": meta.get("treinado_em"), "periodo": meta.get("periodo"),
            "divisao": meta.get("divisao"), "fonteChuva": meta.get("fonte_chuva"),
            "raioRotuloM": meta.get("raio_rotulo_m"), "calibracao": meta.get("calibracao"),
            "metricasTeste": meta.get("metricas_teste"), "bibliotecas": meta.get("bibliotecas"),
            "pontosConhecidos": len(self.resolvedor),
        }

    # ------------------------------------------------------------------ previsão
    def prever(self, req: RequisicaoPrevisao, agora: pd.Timestamp | None = None) -> RespostaPrevisao:
        chuva = None
        if not req.chuvaHoraria:
            chuva = self.servico_chuva.obter([(req.latitude, req.longitude)])[0]
        return self._prever(req, chuva, agora or agora_utc())

    def prever_lote(self, reqs: list[RequisicaoPrevisao], agora: pd.Timestamp | None = None) -> list[ItemLote]:
        agora = agora or agora_utc()
        precisam = [i for i, r in enumerate(reqs) if not r.chuvaHoraria]
        obtidas = self.servico_chuva.obter([(reqs[i].latitude, reqs[i].longitude) for i in precisam]) if precisam else []
        chuva_por_indice = dict(zip(precisam, obtidas, strict=True))
        itens = []
        for i, req in enumerate(reqs):
            try:
                previsao = self._prever(req, chuva_por_indice.get(i), agora)
                itens.append(ItemLote(sensorId=req.sensorId, sucesso=True, previsao=previsao))
            except ErroPrevisao as erro:
                itens.append(ItemLote(sensorId=req.sensorId, sucesso=False, erro=erro.mensagem,
                                      codigoErro=erro.codigo))
            except Exception as erro:  # noqa: BLE001 - um item ruim não derruba o lote
                log.exception("Falha inesperada no item %d do lote", i)
                itens.append(ItemLote(sensorId=req.sensorId, sucesso=False, erro=str(erro),
                                      codigoErro="ERRO_INTERNO"))
        return itens

    def prever_regioes(self, agora: pd.Timestamp | None = None) -> RespostaRegioes:
        """Risco por subprefeitura: o maior risco entre os pontos monitorados de cada uma.

        Serve ao aviso de região do app ("risco elevado na Vila Maria nas próximas 3 h"). O
        modelo é o mesmo dos pontos; muda só a escala em que a resposta é lida — e, nessa
        escala, ele acerta várias vezes mais (ver ``docs/recomendacoes.md``, R18).
        """
        agora = agora or agora_utc()
        if self.pontos_regiao.empty:
            raise ErroPrevisao("SEM_PONTOS", "Nenhum ponto monitorado carregado", 503)
        if self.limiares_regiao is None:
            raise ErroPrevisao("SEM_LIMIARES_REGIAO",
                               "Limiares de região não encontrados para esta versão do modelo", 503)

        pontos = self.pontos_regiao
        coordenadas = list(zip(pontos["lat"], pontos["lon"], strict=True))
        chuvas = self.servico_chuva.obter(coordenadas)

        linhas, probabilidades, avisos, validos = [], [], [], []
        t_ref = None
        for (_, ponto), chuva in zip(pontos.iterrows(), chuvas, strict=True):
            if chuva is None or chuva.serie is None:
                continue
            estaticas = {
                "lat": float(ponto["lat"]), "lon": float(ponto["lon"]),
                "elevacao": float(ponto.get("elevacao", float("nan"))),
                "dist_corrego_m": float(ponto.get("dist_corrego_m", float("nan"))),
                "zona": ponto.get("zona"),
                "freq_hist_alagamento": float(ponto.get("freq_hist_alagamento", float("nan"))),
            }
            try:
                linha, t_ref, _ = features_ao_vivo(chuva.serie, agora, estaticas, None)
            except ValueError:
                continue
            linhas.append(linha)
            validos.append(ponto["ponto_id"])

        if not linhas:
            raise ErroPrevisao("CHUVA_INDISPONIVEL", "Sem chuva recente para os pontos monitorados", 503)
        if len(linhas) < len(pontos):
            avisos.append(f"{len(pontos) - len(linhas)} ponto(s) sem chuva disponível ficaram de fora")

        matriz = pd.concat(linhas, ignore_index=True)
        if self.pacote:
            probabilidades = self.pacote.prever_probabilidade(matriz)
        else:
            # Modo regras: sem probabilidade honesta, o alerta por região não é emitido.
            raise ErroPrevisao("SEM_MODELO", "Alerta por região exige modelo treinado", 503)

        usados = pontos[pontos["ponto_id"].isin(validos)]
        resumo = agregar_por_regiao(usados, probabilidades, MIN_PONTOS_REGIAO)
        acento = self.config.nivel_risco_com_acento
        gerada = agora.floor("s")

        regioes = []
        for linha in resumo.itertuples(index=False):
            coberta = linha.cobertura == "COBERTA"
            nivel = nivel_por_probabilidade(float(linha.probabilidade), self.limiares_regiao)
            regioes.append(RegiaoRisco(
                regiaoId=linha.regiao_id,
                regiao=linha.regiao_nome,
                zona=linha.regiao_zona,
                nivelRisco=valor_contrato(nivel, acento) if coberta else None,
                probabilidadeMaxima=round(float(linha.probabilidade), 4) if coberta else None,
                pontoId=str(linha.ponto_id) if coberta else None,
                logradouro=getattr(linha, "logradouro", None) if coberta else None,
                pontosMonitorados=int(linha.pontos_monitorados),
                cobertura=linha.cobertura,
            ))

        return RespostaRegioes(
            geradaEm=para_iso_local(gerada),
            validaAte=para_iso_local(gerada + pd.Timedelta(minutes=self.config.validade_previsao_min)),
            horaReferencia=para_iso_local(t_ref),
            janelaHoras=JANELA_PREVISAO_HORAS,
            origem=self.origem,
            modeloVersao=self.modelo_versao,
            minPontosRegiao=MIN_PONTOS_REGIAO,
            regioes=regioes,
            avisos=avisos,
        )

    def _resolver_chuva(self, req: RequisicaoPrevisao, chuva: ChuvaObtida | None, instante: pd.Timestamp):
        """Decide de onde vem a chuva. Retorna (série | None, resumo | None, fonte, avisos)."""
        avisos: list[str] = []
        resumo = req.chuva.model_dump(exclude_none=True) if req.chuva else {}

        if req.chuvaHoraria:
            fim = hora_referencia(instante)
            indice = pd.date_range(end=fim, periods=len(req.chuvaHoraria), freq="h")
            serie = pd.Series(req.chuvaHoraria, index=indice, dtype="float64")
            return serie, (resumo or None), FonteChuva.INFORMADA, avisos

        serie = chuva.serie if chuva else None
        fonte = chuva.fonte if chuva else None
        if chuva and chuva.aviso:
            avisos.append(chuva.aviso)

        if serie is None:
            if resumo:
                return None, resumo, FonteChuva.INFORMADA, avisos
            if req.chuvaMm is not None:
                avisos.append("apenas chuvaMm informado (legado); tratado como chuva da última hora")
                return None, {"chuva_1h": req.chuvaMm}, FonteChuva.INFORMADA_LEGADO, avisos
            raise ErroPrevisao("CHUVA_INDISPONIVEL",
                               "Sem chuva recente (Open-Meteo e cache indisponíveis) e nenhuma chuva informada",
                               503)
        if resumo:
            avisos.append("acumulados informados em 'chuva' sobrescreveram os do Open-Meteo")
            fonte = FonteChuva.INFORMADA
        return serie, (resumo or None), fonte, avisos

    def _prever(self, req: RequisicaoPrevisao, chuva: ChuvaObtida | None, agora: pd.Timestamp) -> RespostaPrevisao:
        instante = garantir_utc(req.timestamp) if req.timestamp else agora
        serie, resumo, fonte, avisos = self._resolver_chuva(req, chuva, instante)

        estaticas, ponto_id, distancia, avisos_ponto = self.resolvedor.resolver(req.latitude, req.longitude)
        avisos += avisos_ponto
        try:
            linha, t_ref, avisos_feat = features_ao_vivo(serie, instante, estaticas, resumo)
        except ValueError as erro:
            raise ErroPrevisao("CHUVA_FORA_DA_JANELA", str(erro), 422) from erro
        avisos += avisos_feat
        registro = linha.iloc[0].to_dict()

        if self.pacote:
            probabilidade = float(self.pacote.prever_probabilidade(linha)[0])
            nivel_modelo = nivel_por_probabilidade(probabilidade, self.pacote.limiares)
        else:
            probabilidade = None
            nivel_modelo = self.baseline.nivel(registro)

        chuva_3h = _float_ou_none(registro.get("chuva_3h"))
        chuva_prevista = _float_ou_none(registro.get("chuva_prevista_3h"))
        chuva_recente = max(chuva_3h or 0.0, req.chuvaMm or 0.0)
        ajuste = ajustar_por_sensor(nivel_modelo, req.nivelAgua, req.porcentagemLixo,
                                    chuva_recente, chuva_prevista)

        acento = self.config.nivel_risco_com_acento
        gerada = agora.floor("s")
        return RespostaPrevisao(
            sensorId=req.sensorId,
            probabilidadeAlagamento=None if probabilidade is None else round(probabilidade, 4),
            nivelRisco=valor_contrato(ajuste.nivel_final, acento),
            nivelRiscoModelo=valor_contrato(ajuste.nivel_modelo, acento),
            janelaHoras=JANELA_PREVISAO_HORAS,
            geradaEm=para_iso_local(gerada),
            validaAte=para_iso_local(gerada + pd.Timedelta(minutes=self.config.validade_previsao_min)),
            horaReferencia=para_iso_local(t_ref),
            medicaoTransbordando=ajuste.medicao_transbordando,
            ajusteSensorAplicado=ajuste.ajuste_aplicado,
            motivosAjuste=ajuste.motivos,
            semLeituraSensor=ajuste.sem_leitura,
            fonteChuva=fonte,
            chuvaRecente3hMm=chuva_3h,
            chuvaPrevista3hMm=chuva_prevista,
            origem=self.origem,
            modeloVersao=self.modelo_versao,
            pontoId=ponto_id,
            distanciaPontoM=distancia,
            avisos=avisos,
        )
