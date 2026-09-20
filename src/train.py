"""Treino, calibração, escolha de limiares e comparação dos modelos (seção 7).

Linha de comando::

    python -m src.train --dataset v1 --versao v1
    python -m src.train --dataset v1 --versao v1 --modelos baseline,logistica,lightgbm,random_forest
    python -m src.train --dataset v1 --versao v1 --limiares metas      # limiares pela curva P/R

O que acontece, em ordem:

1. Carrega ``data/processed/dataset_<dataset>.parquet`` e separa treino/validação/teste pela
   coluna ``split`` (divisão TEMPORAL feita no dataset).
2. **Baseline de regras**: só avalia (não treina).
3. Para cada modelo treinável:
   a. ajusta no treino com ``peso_amostra`` (desfaz a subamostragem das horas secas);
   b. (LightGBM) para de adicionar árvores quando a PR-AUC da validação para de melhorar;
   c. **calibra** as probabilidades na validação (``src/calibracao.py``);
   d. avalia na validação e no **teste** (geral e Zona Norte).
4. **Limiares** dos níveis: fixos do briefing (``--limiares fixos``, padrão) ou escolhidos na
   validação por metas de precisão/recall (``--limiares metas``). Nunca olhando o teste.
5. Salva o modelo principal (``--principal``, padrão lightgbm) em ``models/`` e os relatórios
   em ``reports/`` (métricas JSON, tabela Markdown, curvas P/R e de calibração, importâncias).

O teste é usado UMA vez, no fim, só para reportar. Qualquer ajuste feito depois de olhar o
teste deve ser declarado no TCC.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from src.artefatos import PacoteModelo, salvar_pacote
from src.baseline import BaselineRegras
from src.calibracao import Calibrador, escolher_limiares
from src.config import (
    ANO_FIM_TREINO,
    ANOS_TESTE,
    ANOS_VALIDACAO,
    ARQ_PONTOS,
    DIR_MODELOS,
    DIR_RELATORIOS,
    LIMIARES_NIVEIS_PADRAO,
    METAS_LIMIARES,
    USAR_CHUVA_PREVISTA,
)
from src.dataset import caminho_freq_hist, caminho_pontos_dataset, carregar
from src.evaluate import (
    avaliar,
    curva_calibracao,
    grafico_calibracao,
    grafico_precisao_recall,
    tabela_comparacao,
)
from src.features import lista_features, preparar_matriz
from src.modelos import (
    criar_lightgbm,
    criar_random_forest,
    criar_regressao_logistica,
    importancia_lightgbm,
    pontuacao,
    treinar_lightgbm,
)
from src.niveis import niveis_por_probabilidade

log = logging.getLogger(__name__)

MODELOS_TREINAVEIS = ("logistica", "lightgbm", "random_forest")


def _markdown(df: pd.DataFrame) -> str:
    """Tabela Markdown sem depender do pacote ``tabulate``."""
    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    linhas = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in linha) + " |"
              for linha in df.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def montar_pontos_modelo(dataset: pd.DataFrame, versao_dataset: str) -> pd.DataFrame:
    """Características fixas por ponto + ``freq_hist_alagamento`` de produção (todo o histórico)."""
    fixas = (dataset.groupby("ponto_id", observed=True)[["lat", "lon", "elevacao", "dist_corrego_m", "zona"]]
             .first().reset_index())
    freq = pd.read_parquet(caminho_freq_hist(versao_dataset))
    producao = freq[freq["ano"] == freq["ano"].max()][["ponto_id", "freq_hist_alagamento"]]
    fixas["ponto_id"] = fixas["ponto_id"].astype(str)
    pontos = fixas.merge(producao, on="ponto_id", how="left")
    for arq in (caminho_pontos_dataset(versao_dataset), ARQ_PONTOS):
        if arq.exists():
            extra = pd.read_csv(arq)
            # logradouro entra para o alerta por região poder nomear o ponto mais crítico
            colunas = [c for c in ("ponto_id", "bairro", "subprefeitura", "celula_id", "logradouro") if c in extra]
            pontos = pontos.merge(extra[colunas], on="ponto_id", how="left")
            break
    return pontos


def treinar(
    versao_dataset: str,
    versao_modelo: str,
    modelos: tuple[str, ...] = ("baseline", "logistica", "lightgbm"),
    principal: str = "lightgbm",
    metodo_calibracao: str = "auto",
    estrategia_limiares: str = "fixos",
    usar_chuva_prevista: bool = USAR_CHUVA_PREVISTA,
    permitir_oraculo: bool = False,
    salvar_modelo: bool = True,
) -> dict:
    inicio_relogio = time.time()
    dataset, resumo = carregar(versao_dataset)
    if usar_chuva_prevista and resumo.get("fonte_chuva") == "arquivo" and not permitir_oraculo:
        raise ValueError(
            "chuva_prevista_3h com a fonte 'arquivo' é chuva OBSERVADA no futuro (vazamento). "
            "Use o dataset com previsões históricas, ou --permitir-oraculo só como teto teórico."
        )
    features = lista_features(usar_chuva_prevista)
    partes = {nome: dataset[dataset["split"] == nome].reset_index(drop=True)
              for nome in ("treino", "validacao", "teste")}
    for nome, parte in partes.items():
        log.info("%s: %d linhas, %d positivos", nome, len(parte), int(parte["alagou_3h"].sum()))
        if parte["alagou_3h"].sum() == 0:
            raise ValueError(f"O conjunto '{nome}' não tem positivos — revise período/raio (portão de decisão)")

    X = {nome: preparar_matriz(parte, features) for nome, parte in partes.items()}
    y = {nome: parte["alagou_3h"].to_numpy() for nome, parte in partes.items()}
    w_treino = partes["treino"]["peso_amostra"].to_numpy()

    resultados_val: dict = {}
    resultados_teste: dict = {}
    p_val_por_modelo: dict = {}
    p_teste_por_modelo: dict = {}
    probs_teste: dict = {}
    pacotes: dict = {}
    extras: dict = {}
    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)

    if "baseline" in modelos:
        base = BaselineRegras()
        for nome, destino in (("validacao", resultados_val), ("teste", resultados_teste)):
            niv = base.niveis(partes[nome])
            destino["baseline_regras"] = avaliar(partes[nome], base.pontuacao(partes[nome]), niv, calibrada=False)
        probs_teste["baseline_regras"] = base.pontuacao(partes["teste"])

    for nome_modelo in modelos:
        if nome_modelo == "baseline":
            continue
        if nome_modelo not in MODELOS_TREINAVEIS:
            raise ValueError(f"Modelo desconhecido: {nome_modelo}")
        log.info("Treinando %s ...", nome_modelo)
        t0 = time.time()
        if nome_modelo == "lightgbm":
            modelo = treinar_lightgbm(criar_lightgbm(), X["treino"], y["treino"], w_treino,
                                      X["validacao"], y["validacao"])
            extras["lightgbm_melhor_iteracao"] = int(modelo.best_iteration_ or modelo.n_estimators)
            importancia_lightgbm(modelo, features).to_csv(
                DIR_RELATORIOS / f"importancia_features_{versao_modelo}.csv", index=False)
        else:
            fabrica = criar_regressao_logistica if nome_modelo == "logistica" else criar_random_forest
            modelo = fabrica(features)
            modelo.fit(X["treino"], y["treino"], modelo__sample_weight=w_treino)
        log.info("%s treinado em %.1fs", nome_modelo, time.time() - t0)

        calibrador = Calibrador(metodo_calibracao).ajustar(pontuacao(modelo, X["validacao"]), y["validacao"])
        p_val = calibrador.transformar(pontuacao(modelo, X["validacao"]))
        p_teste = calibrador.transformar(pontuacao(modelo, X["teste"]))

        if estrategia_limiares == "metas":
            limiares, diag = escolher_limiares(y["validacao"], p_val, METAS_LIMIARES)
        else:
            limiares, diag = dict(LIMIARES_NIVEIS_PADRAO), {"estrategia": "fixos do briefing"}
        resultados_val[nome_modelo] = avaliar(partes["validacao"], p_val, niveis_por_probabilidade(p_val, limiares))
        resultados_teste[nome_modelo] = avaliar(partes["teste"], p_teste, niveis_por_probabilidade(p_teste, limiares))
        probs_teste[nome_modelo] = p_teste
        p_val_por_modelo[nome_modelo] = p_val
        p_teste_por_modelo[nome_modelo] = p_teste
        pacotes[nome_modelo] = (modelo, calibrador, limiares, diag)

    # --- relatórios
    comparacao = {
        "validacao": tabela_comparacao(resultados_val),
        "teste": tabela_comparacao(resultados_teste),
        "teste_zona_norte": tabela_comparacao(resultados_teste, "zona_norte"),
    }
    prevalencia = float(y["teste"].mean())
    grafico_precisao_recall({k: (y["teste"], v) for k, v in probs_teste.items()},
                            DIR_RELATORIOS / f"curva_pr_{versao_modelo}.png", prevalencia)
    curvas = {k: curva_calibracao(y["teste"], v) for k, v in probs_teste.items() if k != "baseline_regras"}
    if curvas:
        grafico_calibracao(curvas, DIR_RELATORIOS / f"calibracao_{versao_modelo}.png")

    metricas = {"validacao": resultados_val, "teste": resultados_teste,
                "limiares": {k: v[2] for k, v in pacotes.items()},
                "diagnostico_limiares": {k: v[3] for k, v in pacotes.items()},
                "calibracao": {k: v[1].descricao() for k, v in pacotes.items()}, **extras}
    (DIR_RELATORIOS / f"metricas_{versao_modelo}.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    texto = [f"# Comparação de modelos — {versao_modelo}", "",
             f"Dataset `{versao_dataset}` (chuva: {resumo.get('fonte_chuva')}). "
             f"Treino ≤ {ANO_FIM_TREINO} · validação {list(ANOS_VALIDACAO)} · teste {list(ANOS_TESTE)}.", ""]
    for titulo, tabela in comparacao.items():
        texto += [f"## {titulo}", "", _markdown(tabela), ""]
    (DIR_RELATORIOS / f"comparacao_{versao_modelo}.md").write_text("\n".join(texto), encoding="utf-8")

    melhor_val = comparacao["validacao"].dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False)
    if len(melhor_val) and melhor_val.iloc[0]["modelo"] != principal:
        log.warning("Na validação, %s superou o principal (%s) em PR-AUC — registrar no TCC.",
                    melhor_val.iloc[0]["modelo"], principal)

    # --- modelo principal
    if salvar_modelo and principal in pacotes:
        modelo, calibrador, limiares, diag = pacotes[principal]
        teste_principal = resultados_teste[principal]
        pacote = PacoteModelo(
            versao=versao_modelo, nome_modelo=principal, estimador=modelo, calibrador=calibrador,
            features=features, limiares=limiares,
            metadados={
                "treinado_em": pd.Timestamp.now(tz="UTC").isoformat(),
                "dataset": versao_dataset, "fonte_chuva": resumo.get("fonte_chuva"),
                "periodo": resumo.get("periodo"), "raio_rotulo_m": resumo.get("raio_rotulo_m"),
                "janela_horas": resumo.get("janela_horas"),
                "divisao": {"treino_ate": ANO_FIM_TREINO, "validacao": list(ANOS_VALIDACAO),
                            "teste": list(ANOS_TESTE)},
                "calibracao": calibrador.descricao(),
                "metricas_validacao": resultados_val[principal],
                "metricas_teste": teste_principal,
                "contagens": resumo.get("por_split"),
                **extras,
            },
        )
        pontos_modelo = montar_pontos_modelo(dataset, versao_dataset)
        caminhos = salvar_pacote(pacote, pontos_modelo, DIR_MODELOS,
                                 extras_limiares={"estrategia": estrategia_limiares, "diagnostico": diag})
        log.info("Modelo salvo: %s", {k: str(v) for k, v in caminhos.items()})
        _limiares_de_regiao(dataset, pontos_modelo, p_val_por_modelo[principal],
                            p_teste_por_modelo[principal], versao_modelo)

    log.info("Concluído em %.1fs", time.time() - inicio_relogio)
    for titulo, tabela in comparacao.items():
        print(f"\n== {titulo} ==\n{tabela.to_string(index=False)}")
    return metricas


def _limiares_de_regiao(dataset: pd.DataFrame, pontos: pd.DataFrame, p_val, p_teste,
                        versao_modelo: str) -> None:
    """Escolhe e salva os limiares do alerta por subprefeitura (ADR 0005).

    O alerta de região usa o MAIOR risco entre os pontos da região, então a escala das
    probabilidades muda e os limiares por ponto não servem. Regiões com menos de
    ``MIN_PONTOS_REGIAO`` pontos ficam de fora (cobertura insuficiente).
    """
    from src.regioes import (
        METAS_LIMIARES_REGIAO,
        MIN_PONTOS_REGIAO,
        anotar_regiao,
        avaliar_regioes,
        series_por_regiao,
    )

    if "subprefeitura" not in pontos:
        log.warning("Pontos sem coluna 'subprefeitura' — limiares de região não gerados")
        return

    pontos = anotar_regiao(pontos)
    cobertura = pontos.groupby("regiao_id").size()
    cobertas = set(cobertura[cobertura >= MIN_PONTOS_REGIAO].index)
    if not cobertas:
        log.warning("Nenhuma região com %d+ pontos — limiares de região não gerados", MIN_PONTOS_REGIAO)
        return

    base = dataset.assign(ponto_id=dataset["ponto_id"].astype(str)).merge(
        pontos[["ponto_id", "regiao_id", "regiao_nome"]], on="ponto_id", how="left")
    val = base[base["split"] == "validacao"].assign(p=p_val)
    teste = base[base["split"] == "teste"].assign(p=p_teste)
    val = val[val["regiao_id"].isin(cobertas)]
    teste = teste[teste["regiao_id"].isin(cobertas)]

    serie_val = series_por_regiao(val, val["p"])
    limiares, diag = escolher_limiares(serie_val["y"], serie_val["p"], METAS_LIMIARES_REGIAO)

    serie_teste = series_por_regiao(teste, teste["p"])
    nomes = pontos.drop_duplicates("regiao_id").set_index("regiao_id")["regiao_nome"]
    for nivel in ("MEDIO", "ALTO"):
        aval = avaliar_regioes(serie_teste, limiares, nivel_minimo=nivel)
        aval.insert(1, "regiao", aval["regiao_id"].map(nomes).fillna("TOTAL"))
        aval.to_csv(DIR_RELATORIOS / f"regioes_{versao_modelo}_{nivel.lower()}.csv", index=False)
        total = aval[aval["regiao_id"] == "TOTAL"].iloc[0]
        log.info("Região (corte %s): precisão %.3f, recall %.3f, %.2f%% das horas em alerta",
                 nivel, total["precisao"], total["recall"], 100 * total["fracao_horas_em_alerta"])

    (DIR_MODELOS / f"thresholds_regiao_{versao_modelo}.json").write_text(
        json.dumps({"limiares": limiares, "estrategia": "metas_regiao",
                    "min_pontos_regiao": MIN_PONTOS_REGIAO, "regioes_cobertas": sorted(cobertas),
                    "diagnostico": diag}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info("Limiares de região salvos: %s", limiares)


def _principal() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="v1", help="versão do dataset em data/processed")
    parser.add_argument("--versao", default="v1", help="versão do modelo (vira modeloVersao)")
    parser.add_argument("--modelos", default="baseline,logistica,lightgbm")
    parser.add_argument("--principal", default="lightgbm")
    parser.add_argument("--calibracao", default="auto", choices=["auto", "sigmoide", "isotonica"])
    parser.add_argument("--limiares", default="fixos", choices=["fixos", "metas"])
    parser.add_argument("--usar-chuva-prevista", action="store_true")
    parser.add_argument("--permitir-oraculo", action="store_true",
                        help="aceita chuva futura observada (só para um teto teórico; nunca em produção)")
    parser.add_argument("--nao-salvar", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    treinar(args.dataset, args.versao, tuple(m.strip() for m in args.modelos.split(",")),
            args.principal, args.calibracao, args.limiares, args.usar_chuva_prevista,
            args.permitir_oraculo, not args.nao_salvar)


if __name__ == "__main__":
    _principal()
