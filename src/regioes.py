"""Alerta por região (subprefeitura) — agregação, nomes e limiares.

**Por que região?** O modelo ordena bem o risco (ROC-AUC 0,93 no teste), mas errar por poucos
quarteirões já estraga o acerto no ponto: no nível ALTO, cerca de 2 % dos ponto-horas alertados
terminam em alagamento registrado. Agregando o mesmo modelo por subprefeitura, o acerto sobe
para ~4,5 % e o alerta passa a pegar ~36 % dos alagamentos (ver ``docs/recomendacoes.md``, R18).

**Por que subprefeitura, e não zona nem distrito?**

- é a divisão oficial (32 regiões, camada ``subprefeitura`` do GeoSampa) e é como o **próprio
  CGE** agrupa os registros, então o rótulo e o alerta falam da mesma coisa;
- os nomes são os que as pessoas usam ("Vila Maria / Vila Guilherme", "Santana / Tucuruvi");
- por zona o acerto seria maior (~10 %), mas "Zona Norte" é grande demais para ser acionável;
- por distrito (96) a cobertura fica rala: 137 pontos não cobrem 96 regiões.

**Cobertura.** Uma região com menos de ``MIN_PONTOS_REGIAO`` pontos monitorados **não** recebe
alerta: seria o alerta de um ponto disfarçado de região. Ela aparece como ``SEM_COBERTURA``,
que é diferente de "sem risco" — a tela precisa dizer isso com todas as letras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geo import normalizar_texto

# Mínimo de pontos monitorados para uma região poder receber alerta.
MIN_PONTOS_REGIAO = 3

# Metas dos limiares de região (mesma ideia de config.METAS_LIMIARES, com orçamento próprio:
# alerta de região aparece para muita gente, então o orçamento é mais apertado).
METAS_LIMIARES_REGIAO = {
    "MEDIO": {"recall_min": 0.80, "fracao_max": 0.05},
    "ALTO": {"recall_min": 0.50, "fracao_max": 0.02},
    "CRITICO": {"precisao_min": 0.30},
}

# chave de busca -> (id da região, nome para a tela, zona)
# A chave é procurada dentro do nome normalizado; a ordem de busca é da mais longa para a mais
# curta, para "VILA MARIANA" não cair em "VILA MARIA".
REGIOES: dict[str, tuple[str, str, str]] = {
    "SE": ("SE", "Sé", "CENTRO"),
    # Norte
    "CASA VERDE": ("CASA-VERDE-CACHOEIRINHA", "Casa Verde / Limão / Cachoeirinha", "NORTE"),
    "FREGUESIA": ("FREGUESIA-BRASILANDIA", "Freguesia do Ó / Brasilândia", "NORTE"),
    "BRASILANDIA": ("FREGUESIA-BRASILANDIA", "Freguesia do Ó / Brasilândia", "NORTE"),
    "JACANA": ("JACANA-TREMEMBE", "Jaçanã / Tremembé", "NORTE"),
    "TREMEMBE": ("JACANA-TREMEMBE", "Jaçanã / Tremembé", "NORTE"),
    "PERUS": ("PERUS-ANHANGUERA", "Perus / Anhanguera", "NORTE"),
    "PIRITUBA": ("PIRITUBA-JARAGUA", "Pirituba / Jaraguá", "NORTE"),
    "SANTANA": ("SANTANA-TUCURUVI", "Santana / Tucuruvi", "NORTE"),
    "TUCURUVI": ("SANTANA-TUCURUVI", "Santana / Tucuruvi", "NORTE"),
    "VILA MARIA": ("VILA-MARIA-VILA-GUILHERME", "Vila Maria / Vila Guilherme", "NORTE"),
    "VILA GUILHERME": ("VILA-MARIA-VILA-GUILHERME", "Vila Maria / Vila Guilherme", "NORTE"),
    # Oeste
    "BUTANTA": ("BUTANTA", "Butantã", "OESTE"),
    "LAPA": ("LAPA", "Lapa", "OESTE"),
    "PINHEIROS": ("PINHEIROS", "Pinheiros", "OESTE"),
    # Sul
    "CAMPO LIMPO": ("CAMPO-LIMPO", "Campo Limpo", "SUL"),
    "CAPELA DO SOCORRO": ("CAPELA-DO-SOCORRO", "Capela do Socorro", "SUL"),
    "CIDADE ADEMAR": ("CIDADE-ADEMAR", "Cidade Ademar", "SUL"),
    "IPIRANGA": ("IPIRANGA", "Ipiranga", "SUL"),
    "JABAQUARA": ("JABAQUARA", "Jabaquara", "SUL"),
    "M BOI MIRIM": ("MBOI-MIRIM", "M'Boi Mirim", "SUL"),
    "MBOI MIRIM": ("MBOI-MIRIM", "M'Boi Mirim", "SUL"),
    "PARELHEIROS": ("PARELHEIROS", "Parelheiros", "SUL"),
    "SANTO AMARO": ("SANTO-AMARO", "Santo Amaro", "SUL"),
    "VILA MARIANA": ("VILA-MARIANA", "Vila Mariana", "SUL"),
    # Leste
    "ARICANDUVA": ("ARICANDUVA-FORMOSA-CARRAO", "Aricanduva / Formosa / Carrão", "LESTE"),
    "CIDADE TIRADENTES": ("CIDADE-TIRADENTES", "Cidade Tiradentes", "LESTE"),
    "ERMELINO MATARAZZO": ("ERMELINO-MATARAZZO", "Ermelino Matarazzo", "LESTE"),
    "GUAIANASES": ("GUAIANASES", "Guaianases", "LESTE"),
    "ITAIM PAULISTA": ("ITAIM-PAULISTA", "Itaim Paulista", "LESTE"),
    "ITAQUERA": ("ITAQUERA", "Itaquera", "LESTE"),
    "MOOCA": ("MOOCA", "Mooca", "LESTE"),
    "PENHA": ("PENHA", "Penha", "LESTE"),
    "SAO MATEUS": ("SAO-MATEUS", "São Mateus", "LESTE"),
    "SAO MIGUEL": ("SAO-MIGUEL", "São Miguel Paulista", "LESTE"),
    "SAPOPEMBA": ("SAPOPEMBA", "Sapopemba", "LESTE"),
    "VILA PRUDENTE": ("VILA-PRUDENTE", "Vila Prudente", "LESTE"),
}

_CHAVES = sorted(REGIOES, key=len, reverse=True)

COBERTA = "COBERTA"
SEM_COBERTURA = "SEM_COBERTURA"


def identificar_regiao(nome) -> tuple[str | None, str | None, str | None]:
    """Nome escrito de qualquer jeito -> (id, nome de exibição, zona).

    Resolve as variações entre fontes: o CGE escreve "Freguesia do Ó" e o GeoSampa
    "FREGUESIA-BRASILANDIA"; ambos viram a mesma região.
    """
    texto = normalizar_texto(nome)
    if not texto:
        return None, None, None
    for chave in _CHAVES:
        if chave in texto:
            return REGIOES[chave]
    return None, None, None


def anotar_regiao(pontos: pd.DataFrame, coluna: str = "subprefeitura") -> pd.DataFrame:
    """Acrescenta ``regiao_id``, ``regiao_nome`` e ``regiao_zona`` a uma tabela de pontos."""
    saida = pontos.copy()
    trio = [identificar_regiao(v) for v in saida.get(coluna, pd.Series([None] * len(saida)))]
    saida["regiao_id"] = [t[0] for t in trio]
    saida["regiao_nome"] = [t[1] for t in trio]
    saida["regiao_zona"] = [t[2] for t in trio]
    return saida


def agregar_por_regiao(pontos: pd.DataFrame, probabilidades, min_pontos: int = MIN_PONTOS_REGIAO) -> pd.DataFrame:
    """Risco de cada região = maior risco entre seus pontos.

    ``pontos`` precisa ter ``ponto_id`` e ``regiao_id`` (use :func:`anotar_regiao`).
    Devolve uma linha por região, com o ponto que puxou o risco e a situação da cobertura.
    """
    tabela = pontos.copy()
    tabela["probabilidade"] = np.asarray(probabilidades, dtype="float64")
    tabela = tabela[tabela["regiao_id"].notna()]
    if tabela.empty:
        return pd.DataFrame(columns=["regiao_id", "regiao_nome", "regiao_zona", "probabilidade",
                                     "ponto_id", "pontos_monitorados", "cobertura"])

    indices = tabela.groupby("regiao_id", observed=True)["probabilidade"].idxmax()
    resumo = tabela.loc[indices, ["regiao_id", "regiao_nome", "regiao_zona", "probabilidade", "ponto_id"]]
    contagem = tabela.groupby("regiao_id", observed=True).size().rename("pontos_monitorados")
    if "logradouro" in tabela:
        resumo = resumo.join(tabela.loc[indices, ["logradouro"]])
    resumo = resumo.merge(contagem, on="regiao_id")
    resumo["cobertura"] = np.where(resumo["pontos_monitorados"] >= min_pontos, COBERTA, SEM_COBERTURA)
    return resumo.sort_values("probabilidade", ascending=False).reset_index(drop=True)


def series_por_regiao(dataset: pd.DataFrame, probabilidades, coluna_hora: str = "data_hora_utc",
                      rotulo: str = "alagou_3h") -> pd.DataFrame:
    """Série região × hora para avaliar e escolher limiares.

    Cada linha é uma região numa hora: ``p`` é o maior risco entre os pontos dela e ``y`` diz se
    **algum** ponto da região alagou na janela. É exatamente o que o alerta promete.
    """
    base = dataset[["regiao_id", coluna_hora, rotulo]].copy()
    base["p"] = np.asarray(probabilidades, dtype="float64")
    base = base[base["regiao_id"].notna()]
    return (base.groupby(["regiao_id", coluna_hora], observed=True)
            .agg(p=("p", "max"), y=(rotulo, "max"))
            .reset_index())


def avaliar_regioes(serie: pd.DataFrame, limiares: dict, nivel_minimo: str = "ALTO") -> pd.DataFrame:
    """Precisão, recall e fração de horas em alerta, por região e no total."""
    corte = float(limiares[nivel_minimo])
    serie = serie.copy()
    serie["alerta"] = serie["p"] >= corte

    def _metricas(g: pd.DataFrame) -> pd.Series:
        alertas = int(g["alerta"].sum())
        positivos = int(g["y"].sum())
        acertos = int((g["alerta"] & (g["y"] == 1)).sum())
        return pd.Series({
            "horas": len(g),
            "horas_com_alerta": alertas,
            "fracao_horas_em_alerta": round(alertas / max(len(g), 1), 4),
            "horas_com_alagamento": positivos,
            "precisao": round(acertos / alertas, 4) if alertas else 0.0,
            "recall": round(acertos / positivos, 4) if positivos else float("nan"),
        })

    por_regiao = serie.groupby("regiao_id", observed=True).apply(_metricas, include_groups=False)
    total = _metricas(serie).rename("TOTAL")
    return pd.concat([por_regiao, total.to_frame().T]).reset_index(names="regiao_id")
