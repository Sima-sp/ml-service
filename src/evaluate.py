"""Métricas, recortes e gráficos (seção 7.3).

Por que não acurácia? Com ~0,05 % de horas positivas, um modelo que diz sempre "não alaga"
teria 99,95 % de acurácia e seria inútil. Usamos:

- **PR-AUC** (average precision) — principal. Referência: a PR-AUC de um palpite aleatório é
  igual à prevalência (``prevalencia`` no relatório).
- **Precisão e recall** da classe positiva nos limiares dos níveis (≥MEDIO, ≥ALTO, CRITICO).
- **Brier score** e **curva de calibração** — o app mostra a porcentagem, ela precisa ser fiel.
  ``brier_referencia`` = Brier de sempre prever a prevalência; o modelo deve ficar abaixo.
- **ROC-AUC** como complemento.
- **Métricas por evento** — para a banca, mais intuitivas que "por hora": de todos os
  alagamentos do período, quantos tiveram aviso ≥ALTO antes de começar, e com que antecedência;
  e quantos "ponto-dias" tiveram alerta sem alagamento (alarme falso).
- Tudo também **recortado para a Zona Norte**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import ZONA_FOCO
from src.niveis import ORDEM_NIVEIS

NIVEL_ALERTA = 2  # ALTO


def metricas_probabilisticas(y, p, pesos=None, calibrada: bool = True) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype="float64")
    n_pos = int(y.sum())
    prevalencia = float(np.average(y, weights=pesos)) if len(y) else float("nan")
    saida = {"n": int(len(y)), "n_positivos": n_pos, "prevalencia": round(prevalencia, 6)}
    if n_pos == 0 or n_pos == len(y):
        saida["aviso"] = "sem as duas classes: PR-AUC/ROC-AUC indefinidas"
        return saida
    saida["pr_auc"] = round(float(average_precision_score(y, p, sample_weight=pesos)), 4)
    saida["roc_auc"] = round(float(roc_auc_score(y, p, sample_weight=pesos)), 4)
    if calibrada:
        saida["brier"] = round(float(brier_score_loss(y, p, sample_weight=pesos)), 6)
        saida["brier_referencia"] = round(prevalencia * (1 - prevalencia), 6)
    return saida


def metricas_por_nivel(y, niveis, pesos=None) -> dict:
    """Precisão/recall tratando "nível >= k" como alerta positivo."""
    y = np.asarray(y, dtype=int)
    niveis = np.asarray(niveis, dtype=int)
    w = np.ones(len(y)) if pesos is None else np.asarray(pesos, dtype="float64")
    saida = {}
    for k in (1, 2, 3):
        nome = f">={ORDEM_NIVEIS[k].value}" if k < 3 else ORDEM_NIVEIS[k].value
        alerta = niveis >= k
        vp = float(w[alerta & (y == 1)].sum())
        fp = float(w[alerta & (y == 0)].sum())
        fn = float(w[~alerta & (y == 1)].sum())
        precisao = vp / (vp + fp) if vp + fp else 0.0
        recall = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        saida[nome] = {"precisao": round(precisao, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                       "fracao_horas_em_alerta": round(float(w[alerta].sum() / w.sum()), 6)}
    return saida


def metricas_por_evento(df: pd.DataFrame, niveis, nivel_min: int = NIVEL_ALERTA) -> dict:
    """Detecção de eventos e alarmes falsos por ponto-dia.

    Um evento é "detectado" se alguma linha positiva ligada a ele (qualquer ponto a até R
    metros, nas 3 h anteriores ao início) teve nível >= ``nivel_min``.
    """
    niveis = np.asarray(niveis, dtype=int)
    tab = pd.DataFrame({
        "evento": df["evento_alvo_id"].to_numpy(), "t": pd.to_datetime(df["data_hora_utc"], utc=True).to_numpy(),
        "alerta": niveis >= nivel_min, "y": df["alagou_3h"].to_numpy(),
        "ponto": df["ponto_id"].astype(str).to_numpy(),
    })
    pos = tab[tab["y"] == 1]
    saida: dict = {"nivel_minimo": ORDEM_NIVEIS[nivel_min].value}
    if len(pos):
        por_evento = pos.groupby("evento").agg(detectado=("alerta", "any"), t_max=("t", "max"))
        primeiro = pos[pos["alerta"]].groupby("evento")["t"].min()
        por_evento["antecedencia_h"] = ((por_evento["t_max"] - primeiro).dt.total_seconds() / 3600 + 1)
        saida.update({
            "eventos": int(len(por_evento)),
            "eventos_detectados": int(por_evento["detectado"].sum()),
            "recall_eventos": round(float(por_evento["detectado"].mean()), 4),
            "antecedencia_media_h": (round(float(por_evento["antecedencia_h"].mean()), 2)
                                     if por_evento["detectado"].any() else None),
        })
    dia = pd.to_datetime(tab["t"]).dt.tz_convert("America/Sao_Paulo").dt.date
    ponto_dia = tab.assign(dia=dia).groupby(["ponto", "dia"]).agg(alerta=("alerta", "any"), y=("y", "any"))
    alertados = ponto_dia[ponto_dia["alerta"]]
    saida.update({
        "ponto_dias": int(len(ponto_dia)),
        "ponto_dias_com_alerta": int(len(alertados)),
        "ponto_dias_alarme_falso": int((~alertados["y"]).sum()),
        "taxa_alarme_falso": round(float((~alertados["y"]).mean()), 4) if len(alertados) else None,
    })
    return saida


def avaliar(df: pd.DataFrame, p, niveis, calibrada: bool = True) -> dict:
    """Relatório completo, geral e Zona Norte."""
    resultado = {}
    p = np.asarray(p)
    niveis = np.asarray(niveis)
    pesos = df["peso_amostra"].to_numpy() if "peso_amostra" in df else None
    for nome, mascara in (("geral", np.ones(len(df), dtype=bool)),
                          ("zona_norte", (df["zona"] == ZONA_FOCO).to_numpy())):
        if not mascara.any():
            continue
        sub = df.loc[mascara]
        w = pesos[mascara] if pesos is not None else None
        resultado[nome] = {
            "probabilistico": metricas_probabilisticas(sub["alagou_3h"], p[mascara], w, calibrada),
            "por_nivel": metricas_por_nivel(sub["alagou_3h"], niveis[mascara], w),
            "por_evento": metricas_por_evento(sub, niveis[mascara]),
        }
    return resultado


def curva_calibracao(y, p, n_faixas: int = 10) -> pd.DataFrame:
    """Probabilidade média prevista × frequência observada, por faixas de mesmo tamanho (quantis)."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype="float64")})
    df["faixa"] = pd.qcut(df["p"].rank(method="first"), q=n_faixas, labels=False)
    return df.groupby("faixa").agg(p_media=("p", "mean"), freq_observada=("y", "mean"),
                                   n=("y", "size")).reset_index(drop=True)


def grafico_calibracao(curvas: dict[str, pd.DataFrame], caminho: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    maximo = max(max(c["p_media"].max(), c["freq_observada"].max()) for c in curvas.values())
    ax.plot([0, maximo], [0, maximo], linestyle="--", color="gray", linewidth=1, label="calibração perfeita")
    for nome, curva in curvas.items():
        ax.plot(curva["p_media"], curva["freq_observada"], marker="o", linewidth=1.5, label=nome)
    ax.set_xlabel("Probabilidade prevista (média da faixa)")
    ax.set_ylabel("Frequência observada de alagamento")
    ax.set_title("Curva de calibração — teste")
    ax.legend(frameon=False)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def grafico_precisao_recall(dados: dict[str, tuple], caminho: Path, prevalencia: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, (y, p) in dados.items():
        precisao, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precisao, linewidth=1.5, label=f"{nome} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(prevalencia, linestyle="--", color="gray", linewidth=1, label=f"aleatório ({prevalencia:.4f})")
    ax.set_xlabel("Recall (alagamentos capturados)")
    ax.set_ylabel("Precisão (alertas corretos)")
    ax.set_title("Precisão × Recall — teste")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def tabela_comparacao(resultados: dict[str, dict], recorte: str = "geral") -> pd.DataFrame:
    """Uma linha por modelo com as métricas principais (para o README/TCC)."""
    linhas = []
    for modelo, res in resultados.items():
        r = res.get(recorte, {})
        prob = r.get("probabilistico", {})
        alto = r.get("por_nivel", {}).get(">=ALTO", {})
        ev = r.get("por_evento", {})
        linhas.append({
            "modelo": modelo, "pr_auc": prob.get("pr_auc"), "roc_auc": prob.get("roc_auc"),
            "brier": prob.get("brier"), "prevalencia": prob.get("prevalencia"),
            "precisao_ALTO": alto.get("precisao"), "recall_ALTO": alto.get("recall"),
            "recall_eventos": ev.get("recall_eventos"), "antecedencia_h": ev.get("antecedencia_media_h"),
            "taxa_alarme_falso": ev.get("taxa_alarme_falso"),
        })
    return pd.DataFrame(linhas)
